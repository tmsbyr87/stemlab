"""
StemLab – Trenn-Engine.

Kapselt `audio-separator` (Roformer, MDX, Demucs, Ensembles) hinter einer kleinen,
stabilen API:
  * Modell-Katalog mit Fallback-Kandidaten, im Hintergrund gegen die echte
    Modellliste geprüft (robust gegen umbenannte Modelldateien und fehlendes Netz)
  * ein wiederverwendeter Separator pro Modell – spart bei Batches das Nachladen
  * Video-Eingaben: ffmpeg zieht vorher die Tonspur
  * Fortschritt als Callback, aus den tqdm-Balken der Bibliothek geparst und
    über mehrere Durchgänge hinweg auf 0–100 normiert
  * aufgeräumte Ausgabe: <Ziel>/<Songname>/<stem>.<format> plus original.wav,
    analysis.json, beats.json, chords.txt, click.mid, waveform.json
"""

from __future__ import annotations

import io
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

LOG = logging.getLogger("stemlab.engine")

APP_SUPPORT = Path.home() / "Library" / "Application Support" / "StemLab"
MODEL_CACHE = APP_SUPPORT / "models"
SCRATCH = APP_SUPPORT / "scratch"

AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".aiff", ".aif", ".wma", ".wv", ".ape")
VIDEO_SUFFIXES = (".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".mpg", ".mpeg", ".ts", ".wmv", ".flv", ".3gp")
OUTPUT_FORMATS = ("wav", "flac", "mp3")

ProgressFn = Callable[[int, int, int], None]  # (gesamt_prozent, durchgang, durchgaenge_erwartet)
LogFn = Callable[[str], None]
AnalysisFn = Callable[[dict], None]


# --------------------------------------------------------------------------- #
# Modell-Katalog
# --------------------------------------------------------------------------- #


@dataclass
class ModelChoice:
    key: str
    title: str
    subtitle: str
    stems: list[str]
    candidates: list[str]
    speed: str
    passes: int                     # tqdm-Durchgänge, die die Bibliothek fährt
    preset: str | None = None       # Ensemble-Preset statt Einzelmodell
    resolved: str | None = None
    verified: bool = False

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "subtitle": self.subtitle,
            "stems": self.stems,
            "speed": self.speed,
            "resolved": self.resolved,
            "verified": self.verified,
            "ensemble": bool(self.preset),
            "available": self.resolved is not None,
        }


CATALOG: list[ModelChoice] = [
    ModelChoice(
        key="roformer",
        title="Roformer – beste Qualität",
        subtitle="Vocals / Instrumental. Der aktuelle Stand der Technik, rund 2 dB SDR über Demucs.",
        stems=["vocals", "instrumental"], speed="langsam", passes=1,
        candidates=[
            "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
            "melband_roformer_big_beta4.ckpt",
            "mel_band_roformer_kim_ft_unwa.ckpt",
            "MDX23C-8KFFT-InstVoc_HQ.ckpt",
        ],
    ),
    ModelChoice(
        key="ensemble_vocal",
        title="Ensemble – Vocal Balanced",
        subtitle="Zwei Roformer-Modelle, im Spektrum gemittelt. Sauberer als jedes Einzelmodell, doppelte Rechenzeit.",
        stems=["vocals", "instrumental"], speed="sehr langsam", passes=2,
        candidates=["bs_roformer_vocals_resurrection_unwa.ckpt", "melband_roformer_big_beta6x.ckpt"],
        preset="vocal_balanced",
    ),
    ModelChoice(
        key="ensemble_inst",
        title="Ensemble – Instrumental Clean",
        subtitle="Zwei Instrumental-Modelle per Max-Spec kombiniert – minimaler Vocal-Rest im Playback.",
        stems=["vocals", "instrumental"], speed="sehr langsam", passes=2,
        candidates=["mel_band_roformer_instrumental_fv7z_gabox.ckpt", "bs_roformer_instrumental_resurrection_unwa.ckpt"],
        preset="instrumental_clean",
    ),
    ModelChoice(
        key="demucs6",
        title="Demucs 6 Stems",
        subtitle="Vocals, Drums, Bass, Gitarre, Klavier, Rest – das volle Besteck.",
        stems=["vocals", "drums", "bass", "guitar", "piano", "other"], speed="mittel", passes=2,
        candidates=["htdemucs_6s.yaml"],
    ),
    ModelChoice(
        key="demucs4",
        title="Demucs 4 Stems",
        subtitle="Vocals, Drums, Bass, Rest. Schnell und sehr robust.",
        stems=["vocals", "drums", "bass", "other"], speed="schnell", passes=2,
        candidates=["htdemucs.yaml"],
    ),
    ModelChoice(
        key="demucs4ft",
        title="Demucs 4 Stems – fine-tuned",
        subtitle="Wie oben, aber vier spezialisierte Netze. Rund viermal so lange, etwas sauberer.",
        stems=["vocals", "drums", "bass", "other"], speed="sehr langsam", passes=8,
        candidates=["htdemucs_ft.yaml"],
    ),
    ModelChoice(
        key="mdx_inst",
        title="MDX-Net Instrumental HQ",
        subtitle="Schneller Zweispur-Klassiker. Guter Kompromiss, wenn Roformer zu langsam ist.",
        stems=["vocals", "instrumental"], speed="schnell", passes=1,
        candidates=["UVR-MDX-NET-Inst_HQ_4.onnx", "UVR-MDX-NET-Inst_HQ_3.onnx", "Kim_Vocal_2.onnx"],
    ),
]

STEM_ALIASES = {
    "vocals": "vocals", "vocal": "vocals",
    "instrumental": "instrumental", "no vocals": "instrumental",
    "drums": "drums", "bass": "bass", "guitar": "guitar", "piano": "piano", "other": "other",
}

_MODEL_EXTS = (".ckpt", ".onnx", ".yaml", ".pth", ".th")


def _flatten_model_names(tree) -> Iterable[str]:
    if isinstance(tree, dict):
        for key, value in tree.items():
            if isinstance(key, str) and key.endswith(_MODEL_EXTS):
                yield key
            if isinstance(value, str) and value.endswith(_MODEL_EXTS):
                yield value
            yield from _flatten_model_names(value)
    elif isinstance(tree, (list, tuple, set)):
        for item in tree:
            yield from _flatten_model_names(item)


_catalog_lock = threading.Lock()
_catalog_state = {"ready": False, "verified": False, "error": ""}
_available_models: set[str] = set()


def catalog_status() -> dict:
    return dict(_catalog_state)


def _apply_default_resolution() -> None:
    for choice in CATALOG:
        if choice.resolved is None:
            choice.resolved = choice.candidates[0]


def verify_catalog() -> bool:
    """Gleicht die Kandidaten mit der echten Modellliste ab. Braucht Netz."""
    global _available_models
    try:
        from audio_separator.separator import Separator

        probe = Separator(log_level=logging.ERROR, info_only=True, model_file_dir=str(MODEL_CACHE))
        available = set(_flatten_model_names(probe.list_supported_model_files()))
    except Exception as exc:
        with _catalog_lock:
            _apply_default_resolution()
            _catalog_state.update(ready=True, verified=False, error=str(exc) or exc.__class__.__name__)
        LOG.warning("Modellliste nicht abrufbar (%s) – nehme die ersten Kandidaten.", exc)
        return False

    with _catalog_lock:
        _available_models = available
        for choice in CATALOG:
            if choice.preset:
                choice.resolved = choice.preset if all(c in available for c in choice.candidates) else None
            else:
                choice.resolved = next((c for c in choice.candidates if c in available), None)
            choice.verified = True
        _catalog_state.update(ready=True, verified=True, error="")
    return True


def warm_catalog_in_background() -> None:
    def run() -> None:
        for attempt in range(3):
            if verify_catalog():
                return
            time.sleep(10 * (attempt + 1))

    threading.Thread(target=run, name="stemlab-catalog", daemon=True).start()


def catalog() -> list[ModelChoice]:
    with _catalog_lock:
        if not _catalog_state["ready"]:
            _apply_default_resolution()
        return list(CATALOG)


def get_choice(key: str) -> ModelChoice:
    for choice in catalog():
        if choice.key == key:
            return choice
    raise KeyError(f"Unbekanntes Modell: {key}")


def first_available(candidates: list[str]) -> str:
    """Erster Kandidat, den die Modellliste kennt – oder der erste überhaupt."""
    with _catalog_lock:
        if _available_models:
            for c in candidates:
                if c in _available_models:
                    return c
    return candidates[0]


# --------------------------------------------------------------------------- #
# Fortschritt aus tqdm abgreifen
# --------------------------------------------------------------------------- #

_PCT = re.compile(r"(\d{1,3})%\|")


class _ProgressTap(io.TextIOBase):
    """Spiegelt stderr weiter und meldet nebenbei tqdm-Prozente samt Durchgang."""

    def __init__(self, original, on_percent: Callable[[int, int], None]):
        super().__init__()
        self._original = original
        self._on_percent = on_percent
        self._last = -1
        self._pass = 1

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        try:
            return self._original.isatty()
        except Exception:
            return False

    def fileno(self) -> int:
        return self._original.fileno()

    @property
    def encoding(self):  # type: ignore[override]
        return getattr(self._original, "encoding", "utf-8")

    def write(self, chunk: str) -> int:
        for raw in _PCT.findall(chunk):
            pct = int(raw)
            if pct + 5 < self._last:
                self._pass += 1
            self._last = pct
            try:
                self._on_percent(pct, self._pass)
            except Exception:
                pass
        try:
            self._original.write(chunk)
        except Exception:
            pass
        return len(chunk)

    def flush(self) -> None:
        try:
            self._original.flush()
        except Exception:
            pass


class _LogTap(logging.Handler):
    def __init__(self, on_message: LogFn):
        super().__init__(level=logging.INFO)
        self._on_message = on_message

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._on_message(record.getMessage())
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def _run_ffmpeg(args: list[str]) -> None:
    exe = ffmpeg_path()
    if not exe:
        raise RuntimeError("ffmpeg wurde nicht gefunden – bitte `brew install ffmpeg` ausführen.")
    proc = subprocess.run([exe, "-y", "-nostdin", "-loglevel", "error", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        raise RuntimeError("ffmpeg: " + (detail[-1] if detail else f"Exit-Code {proc.returncode}"))


def _convert(src: Path, dst: Path, fmt: str, bitrate: str = "320k") -> None:
    codec = {"mp3": ["-codec:a", "libmp3lame", "-b:a", bitrate], "flac": ["-codec:a", "flac"]}.get(fmt, [])
    _run_ffmpeg(["-i", str(src), *codec, str(dst)])


def _to_wav(src: Path, dst: Path) -> None:
    """Beliebige Audio-/Videodatei als 44,1-kHz-Stereo-WAV (erste Tonspur)."""
    _run_ffmpeg(["-i", str(src), "-vn", "-map", "0:a:0", "-codec:a", "pcm_s16le", "-ar", "44100", "-ac", "2", str(dst)])


_FORBIDDEN = re.compile(r'[\x00-\x1f<>:"/\\|?*]+')


def safe_song_name(name: str) -> str:
    stem = Path(name).stem if Path(name).suffix.lower() in AUDIO_SUFFIXES + VIDEO_SUFFIXES else Path(name).name
    cleaned = _FORBIDDEN.sub("-", stem).strip(" .-")
    if cleaned in ("", ".", ".."):
        cleaned = "output"
    return cleaned[:150]


def _holds_stems(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    return any(p.suffix.lower() in (".wav", ".mp3", ".flac") for p in folder.iterdir() if p.is_file())


def stem_label(filename: str) -> str | None:
    """`x_(Vocals)_model.wav` -> `Vocals` (letzte Klammer zählt)."""
    matches = re.findall(r"\(([^)]+)\)", filename)
    return matches[-1].strip() if matches else None


def _pretty_stem(filename: str) -> str | None:
    label = (stem_label(filename) or "").lower()
    if not label:
        return None
    return STEM_ALIASES.get(label, re.sub(r"[^a-z0-9]+", "_", label).strip("_") or None)


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES


# --------------------------------------------------------------------------- #
# Separator-Cache: ein geladenes Modell bleibt für den nächsten Job im Speicher
# --------------------------------------------------------------------------- #

_sep_lock = threading.Lock()
_sep_cache: dict = {"key": None, "separator": None}
_force_cpu = False


def _make_separator(preset: str | None):
    from audio_separator.separator import Separator

    if _force_cpu:
        import torch

        torch.backends.mps.is_available = lambda: False  # type: ignore[assignment]

    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    kwargs = dict(log_level=logging.INFO, model_file_dir=str(MODEL_CACHE), output_dir=str(SCRATCH), output_format="WAV")
    if preset:
        kwargs["ensemble_preset"] = preset
    return Separator(**kwargs)


def _get_separator(model_file: str, preset: str | None, on_log: LogFn, original_stderr):
    """Separator mit geladenem Modell (oder Ensemble), nach Möglichkeit aus dem Cache."""
    cache_key = f"preset:{preset}" if preset else model_file
    with _sep_lock:
        cached = _sep_cache["separator"]
        if cached is not None and _sep_cache["key"] == cache_key:
            on_log(f"Modell {cache_key} ist noch geladen.")
            return cached

        _sep_cache.update(key=None, separator=None)
        separator = _make_separator(preset)

        on_log(f"Lade {'Ensemble ' + preset if preset else 'Modell ' + model_file} …")
        last_shown = [-10]

        def download_note(pct: int, _pass: int) -> None:
            if pct >= last_shown[0] + 5 or pct == 100:
                last_shown[0] = pct
                on_log(f"Modell wird geladen … {pct} %")

        sys.stderr = _ProgressTap(original_stderr, download_note)
        try:
            if preset:
                separator.load_model()
            else:
                separator.load_model(model_filename=model_file)
        finally:
            sys.stderr = original_stderr

        _sep_cache.update(key=cache_key, separator=separator)
        return separator


def describe_device(separator) -> str:
    device = str(getattr(separator, "torch_device", "cpu"))
    provider = getattr(separator, "onnx_execution_provider", None)
    if provider and isinstance(provider, (list, tuple)):
        provider = provider[0]
    if provider and "CoreML" in str(provider):
        return "coreml"
    return "mps" if "mps" in device else "cpu"


def detect_device() -> str:
    try:
        import platform

        import torch

        if not _force_cpu and torch.backends.mps.is_available() and platform.uname().processor == "arm":
            return "mps"
    except Exception:
        pass
    return "cpu"


def run_model(model_file: str, src: Path, on_log: LogFn, on_progress: Callable[[int, int], None] | None = None,
              preset: str | None = None) -> list[Path]:
    """Ein beliebiges Modell auf eine Datei anwenden; liefert die erzeugten Dateien im Scratch."""
    global _force_cpu
    original_stderr = sys.stderr
    lib_logger = logging.getLogger("audio_separator")
    tap = _LogTap(on_log)
    lib_logger.addHandler(tap)
    tick = on_progress or (lambda _p, _n: None)
    try:
        separator = _get_separator(model_file, preset, on_log, original_stderr)
        sys.stderr = _ProgressTap(original_stderr, tick)
        try:
            produced = separator.separate(str(src))
        except Exception as exc:
            text = str(exc)
            if not _force_cpu and ("MPS" in text or "mps" in text):
                on_log("MPS-Operation nicht unterstützt – wiederhole auf der CPU …")
                sys.stderr = original_stderr
                _force_cpu = True
                _sep_cache.update(key=None, separator=None)
                separator = _get_separator(model_file, preset, on_log, original_stderr)
                sys.stderr = _ProgressTap(original_stderr, tick)
                produced = separator.separate(str(src))
            else:
                raise
        finally:
            sys.stderr = original_stderr
    finally:
        lib_logger.removeHandler(tap)

    paths = []
    for name in produced:
        p = Path(name)
        if not p.is_absolute():
            p = SCRATCH / p
        if p.exists():
            paths.append(p)
    run_model.last_device = describe_device(separator)  # type: ignore[attr-defined]
    return paths


run_model.last_device = "cpu"  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Trennung
# --------------------------------------------------------------------------- #


@dataclass
class JobResult:
    folder: Path
    files: list[Path] = field(default_factory=list)
    extras: list[Path] = field(default_factory=list)
    seconds: float = 0.0
    model_file: str = ""
    device: str = ""
    analysis: dict | None = None


def analyze_only(
    source: Path,
    output_root: Path,
    display_name: str | None = None,
    options: dict | None = None,
    on_progress: ProgressFn | None = None,
    on_log: LogFn | None = None,
    on_analysis: AnalysisFn | None = None,
) -> JobResult:
    """Nur analysieren, nicht trennen.

    Für die Vorbereitung eines Sets braucht man Tempo, Tonart und Cue-Points,
    aber keine Stems – das dauert Sekunden statt Minuten. Der Ordner entsteht
    genauso wie bei einer Trennung, sodass sich die Stems jederzeit
    nachträglich erzeugen lassen.
    """
    import analysis as _analysis
    import postprocess

    options = options or {}
    say = on_log or (lambda _msg: None)
    tick = on_progress or (lambda _pct, _pass, _total: None)

    song_name = safe_song_name(display_name or source.name)
    target = output_root / song_name
    counter = 2
    # Anders als beim Trennen zählen wir nur hoch, wenn dort schon eine
    # Analyse liegt: ein zweiter Lauf auf denselben Song soll ihn ersetzen,
    # nicht danebenlegen.
    while (target / "analysis.json").exists():
        target = output_root / f"{song_name} ({counter})"
        counter += 1

    if not ffmpeg_path():
        raise RuntimeError("ffmpeg wurde nicht gefunden – bitte `brew install ffmpeg` ausführen.")

    started = time.time()
    target.mkdir(parents=True, exist_ok=True)
    result = JobResult(folder=target, seconds=0.0, device="cpu")

    say("Bereite Audio vor …" if not is_video(source) else "Video erkannt – ziehe Tonspur heraus …")
    original = target / "original.wav"
    _to_wav(source, original)
    result.extras.append(original)
    tick(30, 1, 1)

    analysis_obj = _analysis.analyze(original, on_log=say)
    if on_analysis:
        on_analysis(analysis_obj.summary())
    result.analysis = analysis_obj.summary()
    result.extras += _analysis.write_sidecars(target, analysis_obj)
    tick(80, 1, 1)

    say("Berechne Wellenform …")
    result.extras.append(postprocess.write_waveforms(target, [original]))
    result.seconds = time.time() - started
    tick(100, 1, 1)
    say(f"Analyse fertig in {result.seconds:.1f} s.")
    return result


def separate(
    source: Path,
    model_key: str,
    output_root: Path,
    output_format: str = "wav",
    display_name: str | None = None,
    options: dict | None = None,
    on_progress: ProgressFn | None = None,
    on_log: LogFn | None = None,
    on_analysis: AnalysisFn | None = None,
) -> JobResult:
    """Trennt eine Datei und legt Stems samt Analyse in <output_root>/<Songname>/ ab."""
    import analysis as _analysis
    import postprocess

    options = options or {}
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"Unbekanntes Format: {output_format}")

    choice = get_choice(model_key)
    if choice.resolved is None:
        raise RuntimeError(f"{choice.title} ist in der aktuellen Modellliste nicht verfügbar.")

    say = on_log or (lambda _msg: None)
    tick = on_progress or (lambda _pct, _pass, _total: None)

    # "reuse" kommt vom nachträglichen Trennen einer vorhandenen Analyse: dann
    # wird in deren Ordner geschrieben, statt einen zweiten anzulegen.
    weiterverwenden = Path(options["reuse"]) if options.get("reuse") else None
    if weiterverwenden is not None:
        target = weiterverwenden
        song_name = target.name
    else:
        song_name = safe_song_name(display_name or source.name)
        target = output_root / song_name
        counter = 2
        while _holds_stems(target):
            target = output_root / f"{song_name} ({counter})"
            counter += 1

    if not ffmpeg_path():
        raise RuntimeError("ffmpeg wurde nicht gefunden – bitte `brew install ffmpeg` ausführen.")

    SCRATCH.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="job_", dir=SCRATCH))
    started = time.time()

    try:
        # Eine WAV-Kopie des Originals: Eingabe für die Modelle und später A/B-Referenz.
        say("Bereite Audio vor …" if not is_video(source) else "Video erkannt – ziehe Tonspur heraus …")
        audio_in = work / "original.wav"
        _to_wav(source, audio_in)

        # Tempo, Tonart, Takte, Akkorde – dauert Sekunden und steht dann schon in der Karte.
        analysis_obj = None
        vorhanden = weiterverwenden is not None and (target / "analysis.json").exists()
        try:
            if vorhanden:
                say("Analyse liegt schon vor – überspringe sie.")
            else:
                analysis_obj = _analysis.analyze(audio_in, on_log=say)
                if on_analysis:
                    on_analysis(analysis_obj.summary())
        except Exception as exc:
            LOG.warning("Analyse übersprungen: %s", exc)
            say("Analyse übersprungen.")

        def overall(pct: int, pass_no: int) -> None:
            total = max(choice.passes, pass_no)
            merged = int(((pass_no - 1) + pct / 100.0) / total * 100)
            tick(min(99, merged), pass_no, total)

        say("Starte Trennung …")
        produced = run_model(choice.resolved if not choice.preset else "", audio_in, say, overall, preset=choice.preset)

        result = JobResult(folder=target, seconds=0.0, model_file=choice.resolved or "", device=run_model.last_device)
        target.mkdir(parents=True, exist_ok=True)

        for produced_path in produced:
            stem = _pretty_stem(produced_path.name) or "stem"
            final = target / f"{stem}.{output_format}"
            if output_format == "wav":
                shutil.move(str(produced_path), final)
            else:
                say(f"Konvertiere {stem} nach {output_format.upper()} …")
                _convert(produced_path, final, output_format)
                produced_path.unlink(missing_ok=True)
            result.files.append(final)
        result.files.sort(key=lambda p: p.name)

        # Original für A/B und Mixer, Analyse-Dateien, Wellenformen, Tags.
        original = target / "original.wav"
        if audio_in.resolve() != original.resolve():
            shutil.move(str(audio_in), original)
        result.extras.append(original)
        if analysis_obj is not None:
            result.analysis = analysis_obj.summary()
            result.extras += _analysis.write_sidecars(target, analysis_obj)
            if options.get("tags", True):
                result.files = postprocess.tag_folder(target, analysis_obj.as_dict(), song_name, bool(options.get("rename")), say)
        say("Berechne Wellenformen …")
        result.extras.append(postprocess.write_waveforms(target, result.files + [original]))

        result.seconds = time.time() - started
        say(f"Fertig in {result.seconds:.0f} s – {len(result.files)} Stems.")
        return result
    finally:
        shutil.rmtree(work, ignore_errors=True)


def clear_scratch() -> None:
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
