"""
StemLab – lokaler Server.

Läuft nur auf 127.0.0.1, nimmt Audio- und Videodateien entgegen, trennt sie,
analysiert sie und bietet Nachbearbeitung an. Fortschritt per Server-Sent-Events.
Nichts verlässt den Mac – und fremde Webseiten kommen nicht an die API.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ["PATH"] = ":".join(
    dict.fromkeys(["/opt/homebrew/bin", "/usr/local/bin", *os.environ.get("PATH", "").split(":")])
)

import asyncio  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import queue  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import engine  # noqa: E402
import postprocess  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
LOG = logging.getLogger("stemlab")

HERE = Path(__file__).resolve().parent
HOME = Path.home()
DEFAULT_OUTPUT = HOME / "Music" / "StemLab"
CONFIG_PATH = engine.APP_SUPPORT / "config.json"
MAX_UPLOAD_BYTES = 4 * 1024**3
MAX_JOBS_KEPT = 100
ACCEPTED_SUFFIXES = engine.AUDIO_SUFFIXES + engine.VIDEO_SUFFIXES
TEXT_SUFFIXES = (".txt", ".lrc", ".srt", ".json")
ACTION_KINDS = {"refine", "loops", "shift", "mix", "lyrics"}
ACTION_TITLES = {"refine": "Vocal-Veredelung", "loops": "Loop-Export", "shift": "Pitch/Tempo", "mix": "Mix-Export", "lyrics": "Lyrics"}

app = FastAPI(title="StemLab", docs_url=None, redoc_url=None)


# --------------------------------------------------------------------------- #
# Nur die eigene Oberfläche darf die API benutzen
# --------------------------------------------------------------------------- #

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


def _host_only(value: str) -> str:
    value = value.strip().lower()
    if value.startswith("["):
        return value.split("]")[0] + "]"
    return value.split(":")[0]


@app.middleware("http")
async def only_local_callers(request: Request, call_next):
    if _host_only(request.headers.get("host", "")) not in _LOCAL_HOSTS:
        return JSONResponse({"detail": "Nur lokal erreichbar."}, status_code=403)
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin") or request.headers.get("referer")
        if origin:
            origin_host = origin.split("://", 1)[-1].split("/", 1)[0]
            if _host_only(origin_host) not in _LOCAL_HOSTS:
                return JSONResponse({"detail": "Fremder Ursprung abgelehnt."}, status_code=403)
    return await call_next(request)


# --------------------------------------------------------------------------- #
# Einstellungen
# --------------------------------------------------------------------------- #


def load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text())
    except Exception:
        return {}


def save_config(data: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(data, indent=2))


def output_root() -> Path:
    raw = load_config().get("output_dir") or str(DEFAULT_OUTPUT)
    return Path(os.path.expanduser(raw)).resolve()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #


@dataclass
class Job:
    id: str
    kind: str                      # separate | refine | loops | shift | mix | lyrics
    display_name: str
    source: Path | None = None
    model_key: str = ""
    output_format: str = "wav"
    options: dict = field(default_factory=dict)
    folder: str = ""
    created: float = field(default_factory=time.time)
    status: str = "queued"
    percent: int = 0
    pass_no: int = 1
    passes: int = 1
    messages: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    extras: list[str] = field(default_factory=list)
    seconds: float = 0.0
    device: str = ""
    analysis: dict | None = None
    error: str = ""
    finished: float = 0.0

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": ACTION_TITLES.get(self.kind, ""),
            "name": self.display_name,
            "model": self.model_key,
            "format": self.output_format,
            "status": self.status,
            "percent": self.percent,
            "pass": self.pass_no,
            "passes": self.passes,
            "message": self.messages[-1] if self.messages else "",
            "files": self.files,
            "extras": self.extras,
            "folder": self.folder,
            "seconds": round(self.seconds, 1),
            "device": self.device,
            "analysis": self.analysis,
            "error": self.error,
            "created": self.created,
        }


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
WORK: "queue.Queue[str]" = queue.Queue()

_subscribers: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
_sub_lock = threading.Lock()
_sub_counter = 0


def publish(job: Job) -> None:
    payload = json.dumps(job.as_dict())
    with _sub_lock:
        targets = list(_subscribers.values())
    for loop, q in targets:
        try:
            loop.call_soon_threadsafe(q.put_nowait, payload)
        except RuntimeError:
            pass


def prune_jobs() -> None:
    with JOBS_LOCK:
        finished = sorted((j for j in JOBS.values() if j.status in ("done", "error")), key=lambda j: j.finished)
        if len(finished) > MAX_JOBS_KEPT:
            for old in finished[:-MAX_JOBS_KEPT]:
                JOBS.pop(old.id, None)


def _job_callbacks(job: Job):
    def on_progress(pct: int, pass_no: int = 1, passes: int = 1) -> None:
        if (pct, pass_no) != (job.percent, job.pass_no):
            job.percent, job.pass_no, job.passes = pct, pass_no, passes
            publish(job)

    def on_log(message: str) -> None:
        message = message.strip()
        if not message:
            return
        job.messages.append(message)
        del job.messages[:-40]
        publish(job)

    def on_analysis(data: dict) -> None:
        job.analysis = data
        publish(job)

    return on_progress, on_log, on_analysis


def run_job(job: Job) -> None:
    job.status = "running"
    job.percent = 0
    publish(job)
    on_progress, on_log, on_analysis = _job_callbacks(job)
    try:
        if job.kind == "separate":
            result = engine.separate(
                source=job.source, model_key=job.model_key, output_root=output_root(),
                output_format=job.output_format, display_name=job.display_name, options=job.options,
                on_progress=on_progress, on_log=on_log, on_analysis=on_analysis,
            )
            job.files = [str(p) for p in result.files]
            job.extras = [str(p) for p in result.extras]
            job.folder = str(result.folder)
            job.seconds = result.seconds
            job.device = result.device
            job.analysis = result.analysis
        else:
            started = time.time()
            folder = Path(job.folder)
            p = job.options
            if job.kind == "refine":
                steps = [s for s in p.get("steps", []) if s in postprocess.REFINE_STEPS] or ["dereverb"]
                files = postprocess.refine_vocals(folder, steps, on_log, lambda pct, n: on_progress(min(99, pct), n, len(steps)))
                job.device = engine.run_model.last_device
            elif job.kind == "loops":
                a = json.loads((folder / "analysis.json").read_text())
                files = postprocess.export_loops(folder, a.get("downbeats", []), int(p.get("bars", 4)), float(a.get("bpm") or 0), on_log)
            elif job.kind == "shift":
                files = postprocess.shift_folder(folder, list(p.get("stems", [])), float(p.get("semitones", 0)), float(p.get("tempo", 1.0)), on_log)
            elif job.kind == "mix":
                files = [postprocess.export_mix(folder, {k: float(v) for k, v in p.get("gains", {}).items()}, str(p.get("label", "mix")), on_log)]
            elif job.kind == "lyrics":
                files = postprocess.transcribe(folder, p.get("language") or None, on_log)
            else:
                raise RuntimeError("Unbekannte Aktion.")
            job.files = [str(f) for f in files]
            job.seconds = time.time() - started
            postprocess.write_waveforms(folder, [f for f in files if f.suffix.lower() in postprocess.AUDIO_OUT])
        job.percent = 100
        job.status = "done"
    except Exception as exc:
        LOG.exception("Job %s fehlgeschlagen", job.id)
        job.error = str(exc).strip() or exc.__class__.__name__
        job.status = "error"
    finally:
        job.finished = time.time()
        if job.source:
            job.source.unlink(missing_ok=True)
        publish(job)
        prune_jobs()


def worker_loop() -> None:
    while True:
        job_id = WORK.get()
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job:
            run_job(job)
        WORK.task_done()


# --------------------------------------------------------------------------- #
# Bibliothek: fertige Ordner im Zielverzeichnis
# --------------------------------------------------------------------------- #


def folder_summary(folder: Path) -> dict | None:
    stems = postprocess.stem_files(folder)
    if not stems:
        return None
    analysis = None
    try:
        a = json.loads((folder / "analysis.json").read_text())
        analysis = {k: (len(v) if isinstance(v, list) else v) for k, v in a.items()}
    except Exception:
        pass
    extras = [str(p) for p in folder.iterdir() if p.is_file() and (p.name.startswith("original") or p.suffix.lower() in TEXT_SUFFIXES + (".mid",))]
    return {
        "id": "lib-" + uuid.uuid5(uuid.NAMESPACE_URL, str(folder)).hex[:10],
        "kind": "library",
        "title": "",
        "name": folder.name,
        "model": "",
        "format": stems[0].suffix.lstrip(".").lower(),
        "status": "done",
        "percent": 100,
        "pass": 1,
        "passes": 1,
        "message": "",
        "files": [str(p) for p in stems],
        "extras": extras,
        "folder": str(folder),
        "seconds": 0,
        "device": "",
        "analysis": analysis,
        "error": "",
        "created": folder.stat().st_mtime,
    }


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #


@app.get("/api/ping")
def ping() -> JSONResponse:
    return JSONResponse({"app": "stemlab"})


@app.get("/api/state")
def state() -> JSONResponse:
    with JOBS_LOCK:
        jobs = sorted(JOBS.values(), key=lambda j: j.created)
    return JSONResponse(
        {
            "models": [c.as_dict() for c in engine.catalog()],
            "catalog": engine.catalog_status(),
            "device": engine.detect_device(),
            "ffmpeg": bool(engine.ffmpeg_path()),
            "output_dir": str(output_root()),
            "formats": list(engine.OUTPUT_FORMATS),
            "accepts": list(ACCEPTED_SUFFIXES),
            "refine_steps": {k: v["title"] for k, v in postprocess.REFINE_STEPS.items()},
            "jobs": [j.as_dict() for j in jobs],
        }
    )


@app.get("/api/library")
def library() -> JSONResponse:
    root = output_root()
    items = []
    if root.is_dir():
        for folder in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:60]:
            if folder.is_dir():
                summary = folder_summary(folder)
                if summary:
                    items.append(summary)
    return JSONResponse({"items": items})


@app.post("/api/output-dir")
async def set_output_dir(request: Request) -> JSONResponse:
    payload = await request.json()
    raw = str(payload.get("path") or "").strip()
    if not raw:
        raise HTTPException(400, "Kein Pfad angegeben.")
    path = Path(os.path.expanduser(raw)).resolve()
    if not _inside(path, HOME) or path == HOME.resolve():
        raise HTTPException(400, "Der Zielordner muss in deinem Benutzerordner liegen.")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise HTTPException(400, f"Ordner nicht nutzbar: {exc}")
    config = load_config()
    config["output_dir"] = str(path)
    save_config(config)
    return JSONResponse({"output_dir": str(path)})


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    model: str = Form(...),
    output_format: str = Form("wav"),
    tags: str = Form("1"),
    rename: str = Form("0"),
) -> JSONResponse:
    try:
        choice = engine.get_choice(model)
    except KeyError:
        raise HTTPException(400, "Unbekanntes Modell.")
    if choice.resolved is None:
        raise HTTPException(400, f"{choice.title} ist derzeit nicht verfügbar.")
    if output_format not in engine.OUTPUT_FORMATS:
        raise HTTPException(400, "Unbekanntes Format.")

    original = Path(file.filename or "audio").name
    suffix = Path(original).suffix.lower()
    if suffix not in ACCEPTED_SUFFIXES:
        raise HTTPException(400, f"Dateityp {suffix or '(ohne Endung)'} wird nicht unterstützt.")

    handle = tempfile.NamedTemporaryFile(prefix="stemlab_", suffix=suffix, delete=False)
    written = 0
    try:
        with handle:
            while chunk := await file.read(4 << 20):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Datei ist zu groß.")
                await run_in_threadpool(handle.write, chunk)
    except Exception:
        Path(handle.name).unlink(missing_ok=True)
        raise
    if written == 0:
        Path(handle.name).unlink(missing_ok=True)
        raise HTTPException(400, "Die Datei ist leer – wurde vielleicht ein Ordner gezogen?")

    job = Job(
        id=uuid.uuid4().hex[:8], kind="separate", display_name=original, source=Path(handle.name),
        model_key=model, output_format=output_format, passes=choice.passes,
        options={"tags": tags == "1", "rename": rename == "1"},
    )
    with JOBS_LOCK:
        JOBS[job.id] = job
    WORK.put(job.id)
    publish(job)
    return JSONResponse({"id": job.id})


@app.post("/api/actions")
async def create_action(request: Request) -> JSONResponse:
    payload = await request.json()
    kind = str(payload.get("kind", ""))
    if kind not in ACTION_KINDS:
        raise HTTPException(400, "Unbekannte Aktion.")
    folder = _allowed_result_path(str(payload.get("folder", "")))
    if not folder.is_dir():
        raise HTTPException(400, "Kein Ergebnisordner.")
    params = payload.get("params") or {}
    if not isinstance(params, dict):
        raise HTTPException(400, "Ungültige Parameter.")
    job = Job(id=uuid.uuid4().hex[:8], kind=kind, display_name=folder.name, folder=str(folder), options=params)
    with JOBS_LOCK:
        JOBS[job.id] = job
    WORK.put(job.id)
    publish(job)
    return JSONResponse({"id": job.id})


@app.get("/api/events")
async def events() -> StreamingResponse:
    global _sub_counter
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    with _sub_lock:
        _sub_counter += 1
        token = _sub_counter
        _subscribers[token] = (loop, q)

    async def stream():
        try:
            yield ": verbunden\n\n"
            with JOBS_LOCK:
                snapshot = sorted(JOBS.values(), key=lambda j: j.created)
            for job in snapshot:
                yield f"data: {json.dumps(job.as_dict())}\n\n"
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=20)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            with _sub_lock:
                _subscribers.pop(token, None)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _allowed_result_path(raw: str) -> Path:
    target = Path(os.path.expanduser(raw)).resolve()
    if not target.exists():
        raise HTTPException(404, "Pfad existiert nicht.")
    if _inside(target, output_root()) and target != output_root():
        return target
    raise HTTPException(403, "Pfad liegt außerhalb des Zielordners.")


@app.post("/api/reveal")
async def reveal(request: Request) -> JSONResponse:
    payload = await request.json()
    raw = Path(os.path.expanduser(str(payload.get("path", "")))).resolve()
    target = raw if raw == output_root() else _allowed_result_path(str(raw))
    subprocess.Popen(["open", "-R", str(target)] if target.is_file() else ["open", str(target)])
    return JSONResponse({"ok": True})


@app.get("/api/audio")
def audio(path: str) -> FileResponse:
    target = _allowed_result_path(path)
    if not target.is_file() or target.suffix.lower() not in postprocess.AUDIO_OUT:
        raise HTTPException(404, "Keine Audiodatei.")
    return FileResponse(target)


@app.get("/api/text")
def text(path: str) -> PlainTextResponse:
    target = _allowed_result_path(path)
    if not target.is_file() or target.suffix.lower() not in TEXT_SUFFIXES:
        raise HTTPException(404, "Keine Textdatei.")
    media = "application/json" if target.suffix.lower() == ".json" else "text/plain; charset=utf-8"
    return PlainTextResponse(target.read_text(), media_type=media)


@app.get("/api/folder")
def folder_info(path: str) -> JSONResponse:
    target = _allowed_result_path(path)
    if not target.is_dir():
        raise HTTPException(400, "Kein Ordner.")
    summary = folder_summary(target) or {}
    subs = {}
    for name in ("refined", "loops", "shifted", "mixes"):
        sub = target / name
        if sub.is_dir():
            subs[name] = [str(p) for p in sorted(sub.iterdir()) if p.suffix.lower() in postprocess.AUDIO_OUT]
    summary["sub"] = subs
    return JSONResponse(summary)


def _taggable(raw: str) -> Path:
    """Pfad einer Audiodatei im Zielordner – für den Tag-Editor."""
    target = _allowed_result_path(raw)
    if not target.is_file() or target.suffix.lower() not in postprocess.AUDIO_OUT:
        raise HTTPException(404, "Keine Audiodatei.")
    return target


@app.get("/api/tags")
def get_tags(path: str) -> JSONResponse:
    target = _taggable(path)
    try:
        return JSONResponse(postprocess.read_tags(target))
    except Exception as exc:
        LOG.warning("Tags nicht lesbar (%s): %s", target.name, exc)
        raise HTTPException(400, f"Tags nicht lesbar: {exc}")


@app.post("/api/tags")
async def set_tags(request: Request) -> JSONResponse:
    payload = await request.json()
    target = _taggable(str(payload.get("path", "")))
    values = payload.get("values") or {}
    if not isinstance(values, dict):
        raise HTTPException(400, "Ungültige Felder.")
    try:
        await run_in_threadpool(postprocess.write_tags, target, values)
        return JSONResponse(postprocess.read_tags(target))
    except Exception as exc:
        LOG.warning("Tags nicht geschrieben (%s): %s", target.name, exc)
        raise HTTPException(400, f"Tags nicht geschrieben: {exc}")


@app.get("/api/cover")
def get_cover(path: str) -> Response:
    target = _taggable(path)
    try:
        found = postprocess.read_cover(target)
    except Exception as exc:
        raise HTTPException(400, f"Cover nicht lesbar: {exc}")
    if not found:
        raise HTTPException(404, "Kein Cover.")
    data, mime = found
    return Response(content=data, media_type=mime or "image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.post("/api/cover")
async def set_cover(path: str = Form(...), file: UploadFile = File(...)) -> JSONResponse:
    target = _taggable(path)
    mime = (file.content_type or "").split(";")[0].strip().lower()
    if mime not in postprocess.COVER_TYPES:
        raise HTTPException(400, "Cover muss JPEG oder PNG sein.")
    data = await file.read()
    if len(data) > postprocess.MAX_COVER_BYTES:
        raise HTTPException(400, "Cover ist größer als 8 MB.")
    try:
        await run_in_threadpool(postprocess.write_cover, target, data, mime)
        return JSONResponse(postprocess.read_tags(target))
    except Exception as exc:
        LOG.warning("Cover nicht geschrieben (%s): %s", target.name, exc)
        raise HTTPException(400, f"Cover nicht geschrieben: {exc}")


@app.post("/api/cover/delete")
async def delete_cover(request: Request) -> JSONResponse:
    payload = await request.json()
    target = _taggable(str(payload.get("path", "")))
    try:
        await run_in_threadpool(postprocess.remove_cover, target)
        return JSONResponse(postprocess.read_tags(target))
    except Exception as exc:
        raise HTTPException(400, f"Cover nicht entfernt: {exc}")


@app.post("/api/quit")
def quit_app() -> JSONResponse:
    def _bye() -> None:
        time.sleep(0.4)
        os._exit(0)

    threading.Thread(target=_bye, daemon=True).start()
    return JSONResponse({"ok": True})


class _FreshStatic(StaticFiles):
    """Oberfläche immer neu validieren lassen – sonst zeigt der Browser nach einem
    Update wochenlang die alte Datei aus seinem Heuristik-Cache."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response


app.mount("/", _FreshStatic(directory=str(HERE / "static"), html=True), name="static")


@app.on_event("startup")
def _startup() -> None:
    engine.clear_scratch()
    engine.warm_catalog_in_background()
    output_root().mkdir(parents=True, exist_ok=True)
    threading.Thread(target=worker_loop, name="stemlab-worker", daemon=True).start()
    LOG.info("StemLab bereit · ffmpeg: %s · Gerät: %s", engine.ffmpeg_path() or "FEHLT", engine.detect_device())


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("STEMLAB_PORT") or "8765")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
