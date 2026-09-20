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
import profil  # noqa: E402

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
        if job.kind == "analyze":
            result = engine.analyze_only(
                source=job.source, output_root=output_root(),
                display_name=job.display_name, options=job.options,
                on_progress=on_progress, on_log=on_log, on_analysis=on_analysis,
            )
            job.files = []
            job.extras = [str(p) for p in result.extras]
            job.folder = str(result.folder)
            job.seconds = result.seconds
            job.analysis = result.analysis
        elif job.kind == "separate":
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
        # Die hochgeladene Zwischendatei wegräumen – aber nur, wenn sie
        # wirklich eine ist. Zwei Fälle, in denen die Quelle bleiben muss:
        # Beim nachträglichen Trennen ist sie die original.wav im
        # Ergebnisordner, und beim Sammellauf liegt sie in der
        # Musiksammlung des Nutzers. Dort etwas zu löschen wäre
        # unverzeihlich.
        if job.source and not job.options.get("reuse") and not job.options.get("sammellauf"):
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
    # Ein Ordner ohne Stems kann eine reine Analyse sein – die gehört genauso
    # in die Bibliothek, sonst verschwindet sie nach dem Neuladen.
    if not stems and not (folder / "analysis.json").exists():
        return None
    analysis = None
    try:
        a = json.loads((folder / "analysis.json").read_text())
        # Lange Listen werden zu ihrer Anzahl: Niemand braucht 800
        # Beat-Zeitpunkte in der Oberfläche. Die Abschnitte sind die
        # Ausnahme – ohne sie kann die Arrangement-Ansicht nichts
        # zeichnen, und "7" sagt nichts über den Aufbau.
        durchreichen = {"segments"}
        analysis = {k: (v if k in durchreichen or not isinstance(v, list) else len(v))
                    for k, v in a.items()}
    except Exception:
        pass
    extras = [str(p) for p in folder.iterdir() if p.is_file() and (p.name.startswith("original") or p.suffix.lower() in TEXT_SUFFIXES + (".mid",))]
    return {
        "id": "lib-" + uuid.uuid5(uuid.NAMESPACE_URL, str(folder)).hex[:10],
        "kind": "library",
        "title": "",
        "name": folder.name,
        "model": "",
        "format": stems[0].suffix.lstrip(".").lower() if stems else "wav",
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


@app.get("/api/analyses")
def analyses() -> JSONResponse:
    """Schlanke Übersicht aller bisherigen Analysen – ohne die Stems selbst zu laden."""
    root = output_root()
    items = []
    if root.is_dir():
        for folder in root.iterdir():
            if not folder.is_dir():
                continue
            stems = postprocess.stem_files(folder)
            data = {}
            try:
                data = json.loads((folder / "analysis.json").read_text())
            except Exception:
                pass
            # Ein Ordner ohne Stems kann eine reine Analyse sein. Ohne Analyse
            # und ohne Stems ist er nichts.
            if not stems and not data:
                continue
            items.append({
                "id": "lib-" + uuid.uuid5(uuid.NAMESPACE_URL, str(folder)).hex[:10],
                "name": folder.name,
                "folder": str(folder),
                "bpm": data.get("bpm") or 0,
                "bpm_alt": data.get("bpm_alt") or 0,
                "bpm_confidence": data.get("bpm_confidence") or 0,
                "key": data.get("key") or "",
                "key_de": data.get("key_de") or "",
                "key_alt": data.get("key_alt") or "",
                "camelot": data.get("camelot") or "",
                "key_confidence": data.get("key_confidence") or 0,
                "energy": data.get("energy") or 0,
                # Ältere Analysen kennen das Feld nicht – dann aus den
                # vorhandenen Werten nachrechnen statt eine Lücke zu zeigen.
                "danceability": data.get("danceability") or 0,
                "seconds": data.get("seconds_analyzed") or 0,
                "bars": len(data.get("downbeats") or []),
                "stems": [postprocess.stem_name(p) for p in stems],
                "format": stems[0].suffix.lstrip(".").lower() if stems else "",
                "created": folder.stat().st_mtime,
                "has_lyrics": (folder / "lyrics.json").exists(),
                "has_chords": bool(data.get("chords")),
                "has_loops": (folder / "loops").is_dir(),
                "has_refined": (folder / "refined").is_dir(),
            })
    items.sort(key=lambda x: x["created"], reverse=True)
    return JSONResponse({"items": items[:500], "total": len(items)})


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
    model: str = Form(""),
    output_format: str = Form("wav"),
    tags: str = Form("1"),
    rename: str = Form("0"),
    mode: str = Form("separate"),
) -> JSONResponse:
    if mode not in ("separate", "analyze"):
        raise HTTPException(400, "Unbekannter Modus.")

    # Beim reinen Analysieren spielt das Modell keine Rolle – es wird nie
    # geladen, also darf es auch fehlen oder nicht verfügbar sein.
    choice = None
    if mode == "separate":
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
        id=uuid.uuid4().hex[:8], kind=mode, display_name=original, source=Path(handle.name),
        model_key=model, output_format=output_format, passes=choice.passes if choice else 1,
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


@app.get("/api/settings")
def get_settings() -> JSONResponse:
    """Aktuelle Einstellungen plus alles, was die Oberfläche zum Aufbau braucht."""
    config = load_config()
    return JSONResponse({
        "values": postprocess.settings(config),
        "presets": postprocess.PRESETS,
        "patterns": postprocess.TAG_PATTERNS,
        "targets": postprocess.TAG_TARGETS,
        "notations": postprocess.KEY_NOTATIONS,
        "output_dir": str(output_root()),
    })


@app.get("/api/notes")
def get_notes(path: str) -> JSONResponse:
    """Notizen liegen als notes.txt im Ergebnisordner.

    Nicht in einer zentralen Datei: Wer den Ordner kopiert, nimmt die
    Notiz mit; wer ihn löscht, wird sie los. Das entspricht dem, was das
    Programm sonst auch tut – alles zu einem Song bleibt beieinander.
    """
    ordner = _allowed_result_path(path)
    datei = ordner / "notes.txt"
    text = ""
    if datei.is_file():
        try:
            text = datei.read_text(encoding="utf-8")
        except Exception as exc:
            LOG.warning("Notiz nicht lesbar (%s): %s", datei, exc)
    return JSONResponse({"text": text})


@app.post("/api/notes")
async def post_notes(request: Request) -> JSONResponse:
    payload = await request.json()
    ordner = _allowed_result_path(str(payload.get("folder", "")))
    text = str(payload.get("text", ""))
    datei = ordner / "notes.txt"
    if text.strip():
        datei.write_text(text, encoding="utf-8")
    else:
        # Kein leeres Blatt hinterlassen.
        datei.unlink(missing_ok=True)
    return JSONResponse({"text": text if text.strip() else ""})


@app.get("/api/profil")
def get_profil(playlist: str = "") -> JSONResponse:
    """Das Profil einer Playlist – oder des ganzen Zielordners.

    Gelesen werden die analysis.json der beteiligten Ordner. Eine
    unlesbare Datei übergeht das Profil, statt daran zu scheitern: Bei
    fünfzig Tracks soll eine kaputte nicht alles verhindern.
    """
    wurzel = output_root()
    if playlist:
        listen = load_config().get("playlists") or {}
        if playlist not in listen:
            raise HTTPException(404, f"Keine Playlist namens {playlist!r}.")
        ordner = [Path(p) for p in listen[playlist]]
    else:
        ordner = [p for p in wurzel.iterdir() if p.is_dir()] if wurzel.is_dir() else []

    analysen = []
    for o in ordner:
        datei = o / "analysis.json"
        if not datei.is_file():
            continue
        try:
            analysen.append(json.loads(datei.read_text()))
        except Exception as exc:
            LOG.warning("Analyse nicht lesbar (%s): %s", datei, exc)

    return JSONResponse({"profil": profil.erstelle(analysen),
                         "playlist": playlist or None})


@app.get("/api/playlists")
def get_playlists() -> JSONResponse:
    """Playlists sind gespeicherte Filter, keine zweite Dateiverwaltung.

    Gemerkt werden Ordnerpfade; die Dateien bleiben, wo sie sind. Ordner,
    die es nicht mehr gibt, fallen beim Lesen heraus – wer einen
    Ergebnisordner im Finder löscht, soll ihn nicht weiter in seinen
    Listen finden.
    """
    config = load_config()
    roh = config.get("playlists") or {}
    sauber = {name: [p for p in pfade if Path(p).is_dir()]
              for name, pfade in roh.items() if isinstance(pfade, list)}
    if sauber != roh:
        config["playlists"] = sauber
        save_config(config)
    return JSONResponse({"playlists": sauber})


@app.post("/api/playlists")
async def post_playlists(request: Request) -> JSONResponse:
    payload = await request.json()
    aktion = str(payload.get("aktion", ""))
    name = str(payload.get("name", "")).strip()
    if not name:
        raise HTTPException(400, "Die Liste braucht einen Namen.")

    config = load_config()
    listen = dict(config.get("playlists") or {})

    if aktion == "anlegen":
        # Ein vorhandener Name darf die Liste nicht leeren.
        listen.setdefault(name, [])
    elif aktion == "loeschen":
        listen.pop(name, None)
    elif aktion in ("hinzufuegen", "entfernen"):
        roh = str(payload.get("folder", ""))
        if not roh:
            raise HTTPException(400, "Kein Ordner angegeben.")
        ordner = str(_allowed_result_path(roh))
        eintraege = list(listen.get(name) or [])
        if aktion == "hinzufuegen":
            if ordner not in eintraege:      # kein Song doppelt im Set
                eintraege.append(ordner)
        else:
            eintraege = [p for p in eintraege if p != ordner]
        listen[name] = eintraege
    else:
        raise HTTPException(400, f"Unbekannte Aktion: {aktion}")

    config["playlists"] = listen
    save_config(config)
    return JSONResponse({"playlists": listen})


@app.post("/api/settings")
async def set_settings(request: Request) -> JSONResponse:
    payload = await request.json()
    values = payload.get("values")
    if not isinstance(values, dict):
        raise HTTPException(400, "Ungültige Einstellungen.")

    # Nur bekannte Schlüssel übernehmen und gegen die erlaubten Werte prüfen –
    # ein Tippfehler in der Oberfläche soll nicht stillschweigend landen.
    erlaubt = {
        "tag_pattern": set(postprocess.TAG_PATTERNS),
        "tag_target": set(postprocess.TAG_TARGETS),
        "key_notation": set(postprocess.KEY_NOTATIONS),
    }
    geprueft = {}
    for key, default in postprocess.DEFAULT_SETTINGS.items():
        if key not in values:
            continue
        wert = values[key]
        if key in erlaubt and wert not in erlaubt[key]:
            raise HTTPException(400, f"Unbekannter Wert für {key}: {wert}")
        if isinstance(default, bool):
            wert = bool(wert)
        elif isinstance(default, int) and not isinstance(wert, bool):
            try:
                wert = int(wert)
            except (TypeError, ValueError):
                raise HTTPException(400, f"{key} braucht eine Zahl.")
        geprueft[key] = wert

    if geprueft.get("tempo_min", 0) and geprueft.get("tempo_max", 0):
        if geprueft["tempo_min"] >= geprueft["tempo_max"]:
            raise HTTPException(400, "Das untere Tempo muss kleiner sein als das obere.")

    config = load_config()
    config["tagging"] = {**postprocess.settings(config), **geprueft}
    save_config(config)
    return JSONResponse({"values": postprocess.settings(config)})


@app.post("/api/settings/preview")
async def preview_settings(request: Request) -> JSONResponse:
    """Zeigt an einem Beispiel, was die Einstellungen bewirken.

    Ohne diese Vorschau muss man raten, was "Tonart und Energie" im
    Kommentarfeld bedeutet – mit ihr steht es direkt daneben.
    """
    payload = await request.json()
    opts = postprocess.settings({"tagging": payload.get("values") or {}})
    beispiel = {"camelot": "6A", "key_tonic": "G", "key_mode": "minor",
                "bpm": 124.0, "energy": 5}
    return JSONResponse({
        "tag": postprocess.build_tag_text(beispiel, opts),
        "filename": postprocess.build_filename("vocals", beispiel, opts) + ".wav",
        "key": postprocess.format_key("6A", "G", "minor", opts),
        "tempo": postprocess.format_tempo(124.0, opts),
    })


@app.post("/api/export")
async def export_dj(request: Request) -> JSONResponse:
    """Export für Rekordbox und Traktor in den Ergebnisordner schreiben."""
    payload = await request.json()
    folder = _allowed_result_path(str(payload.get("folder", "")))
    if not folder.is_dir():
        raise HTTPException(400, "Kein Ergebnisordner.")
    try:
        daten = json.loads((folder / "analysis.json").read_text())
    except Exception:
        raise HTTPException(400, "Keine Analyse in diesem Ordner.")

    try:
        beats = json.loads((folder / "beats.json").read_text())
        daten["beats_json_downbeats"] = beats.get("downbeats") or []
    except Exception:
        daten["beats_json_downbeats"] = daten.get("downbeats") or []

    meldungen: list[str] = []
    try:
        dateien = await run_in_threadpool(
            postprocess.export_dj, folder, daten, folder.name, load_config(), meldungen.append)
    except Exception as exc:
        LOG.warning("Export fehlgeschlagen (%s): %s", folder.name, exc)
        raise HTTPException(400, f"Export fehlgeschlagen: {exc}")
    return JSONResponse({"files": [str(p) for p in dateien], "log": meldungen})


@app.post("/api/analyze-folder")
async def analyze_folder(request: Request) -> JSONResponse:
    """Einen ganzen Ordner analysieren, ohne Stems zu trennen.

    Für ein Genre-Profil braucht es dreißig und mehr Tracks. Die einzeln
    hineinzuziehen ist keine Arbeitsweise. Getrennt wird dabei nicht:
    Analysieren dauert Sekunden, Trennen Minuten, und für Tempo, Tonart
    und Aufbau reicht die Analyse.

    Der Quellordner darf überall liegen – gelesen wird nur, geschrieben
    ausschließlich in den Zielordner.
    """
    payload = await request.json()
    roh = str(payload.get("folder", "")).strip()
    if not roh:
        raise HTTPException(400, "Kein Ordner angegeben.")
    quelle = Path(roh).expanduser()
    if not quelle.is_dir():
        raise HTTPException(400, "Das ist kein Ordner.")

    # Eine Ebene tiefer genügt: Sammlungen sind nach Genre sortiert, nicht
    # beliebig verschachtelt.
    dateien = sorted(
        p for p in [*quelle.iterdir(), *(k for d in quelle.iterdir() if d.is_dir() for k in d.iterdir())]
        if p.is_file() and p.suffix.lower() in ACCEPTED_SUFFIXES
    )
    if not dateien:
        raise HTTPException(400, "In diesem Ordner liegen keine Audiodateien.")

    neu_rechnen = bool(payload.get("neu"))
    ziel = output_root()
    eingereiht, uebersprungen = [], 0

    for datei in dateien:
        if not neu_rechnen:
            # Derselbe Name wie beim Einzellauf – so findet sich wieder,
            # was schon analysiert wurde.
            vorhanden = ziel / engine.safe_song_name(datei.name)
            if (vorhanden / "analysis.json").is_file():
                uebersprungen += 1
                continue
        job = Job(id=uuid.uuid4().hex[:8], kind="analyze", display_name=datei.name,
                  source=datei, options={"tags": False, "rename": False, "sammellauf": True})
        with JOBS_LOCK:
            JOBS[job.id] = job
        WORK.put(job.id)
        publish(job)
        eingereiht.append(job.id)

    return JSONResponse({"eingereiht": len(eingereiht), "uebersprungen": uebersprungen,
                         "gesamt": len(dateien), "ids": eingereiht})


@app.post("/api/separate-folder")
async def separate_folder(request: Request) -> JSONResponse:
    """Aus einer fertigen Analyse nachträglich Stems erzeugen.

    Die Analyse hat `original.wav` bereits abgelegt – getrennt wird direkt
    daraus, der Song muss also nicht erneut hochgeladen werden.
    """
    payload = await request.json()
    folder = _allowed_result_path(str(payload.get("folder", "")))
    original = folder / "original.wav"
    if not original.is_file():
        raise HTTPException(400, "In diesem Ordner liegt keine Aufnahme.")
    if postprocess.stem_files(folder):
        raise HTTPException(400, "Dieser Ordner enthält bereits Stems.")

    model = str(payload.get("model") or "")
    try:
        choice = engine.get_choice(model)
    except KeyError:
        raise HTTPException(400, "Unbekanntes Modell.")
    if choice.resolved is None:
        raise HTTPException(400, f"{choice.title} ist derzeit nicht verfügbar.")

    output_format = str(payload.get("format") or "wav")
    if output_format not in engine.OUTPUT_FORMATS:
        raise HTTPException(400, "Unbekanntes Format.")

    job = Job(
        id=uuid.uuid4().hex[:8], kind="separate", display_name=folder.name,
        source=original, model_key=model, output_format=output_format,
        passes=choice.passes, options={"tags": True, "rename": False, "reuse": str(folder)},
    )
    with JOBS_LOCK:
        JOBS[job.id] = job
    WORK.put(job.id)
    publish(job)
    return JSONResponse({"id": job.id})


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
