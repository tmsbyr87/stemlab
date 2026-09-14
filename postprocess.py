"""
StemLab – Nachbearbeitung fertiger Trennungen.

  * Tags (BPM, Tonart) in WAV/FLAC/MP3 und optional in den Dateinamen
  * Wellenform-Peaks für die Oberfläche
  * Loop-Export am Downbeat (4 oder 8 Takte)
  * Pitch-/Tempo-Shift über Rubber Band (ffmpeg) mit librosa-Rückfall
  * Mix-Export mit Pegeln pro Stem
  * Vocal-Veredelung: De-Reverb, De-Noise, Lead/Backing
  * Lyrics per Whisper (mlx-whisper auf Apple Silicon, sonst faster-whisper)
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np

LOG = logging.getLogger("stemlab.post")
LogFn = Callable[[str], None]

STEM_ORDER = ["vocals", "instrumental", "drums", "bass", "guitar", "piano", "other"]
AUDIO_OUT = (".wav", ".flac", ".mp3")


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg wurde nicht gefunden – bitte `brew install ffmpeg` ausführen.")
    return exe


def _run(args: list[str]) -> str:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        raise RuntimeError(tail[-1] if tail else f"Exit-Code {proc.returncode}")
    return proc.stdout


def stem_files(folder: Path) -> list[Path]:
    """Die eigentlichen Stems eines Ergebnisordners, in musikalischer Reihenfolge."""
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_OUT and not p.name.startswith("original")]
    order = {n: i for i, n in enumerate(STEM_ORDER)}
    return sorted(files, key=lambda p: (order.get(stem_name(p), 99), p.name))


def stem_name(path: Path) -> str:
    return re.split(r"[ _\-–(]", path.stem, maxsplit=1)[0].lower()


# --------------------------------------------------------------------------- #
# Tags & Dateinamen
# --------------------------------------------------------------------------- #


def tag_file(path: Path, bpm: float, key_id3: str, camelot: str, song: str, stem: str) -> None:
    from mutagen.flac import FLAC
    from mutagen.id3 import ID3, TALB, TBPM, TIT2, TKEY, COMM, ID3NoHeaderError
    from mutagen.mp3 import MP3
    from mutagen.wave import WAVE

    suffix = path.suffix.lower()
    if suffix == ".flac":
        f = FLAC(str(path))
        if bpm:
            f["BPM"] = str(int(round(bpm)))
        if key_id3:
            f["INITIALKEY"] = key_id3
            f["KEY"] = key_id3
        f["TITLE"] = f"{song} – {stem}"
        f["ALBUM"] = song
        if camelot and camelot != "–":
            f["COMMENT"] = f"Camelot {camelot} · StemLab"
        f.save()
        return

    audio = MP3(str(path)) if suffix == ".mp3" else WAVE(str(path))
    try:
        tags = audio.tags if audio.tags is not None else None
    except Exception:
        tags = None
    if tags is None:
        audio.add_tags()
        tags = audio.tags
    if bpm:
        tags.add(TBPM(encoding=3, text=str(int(round(bpm)))))
    if key_id3:
        tags.add(TKEY(encoding=3, text=key_id3))
    tags.add(TIT2(encoding=3, text=f"{song} – {stem}"))
    tags.add(TALB(encoding=3, text=song))
    if camelot and camelot != "–":
        tags.add(COMM(encoding=3, lang="deu", desc="", text=f"Camelot {camelot} · StemLab"))
    audio.save()


# Felder, die der Tag-Editor lesen und schreiben kann. Links der Name in der
# Oberfläche, rechts der ID3-Rahmen bzw. der Vorbis-Schlüssel für FLAC.
TAG_FIELDS = {
    "title":   ("TIT2", "TITLE"),
    "artist":  ("TPE1", "ARTIST"),
    "album":   ("TALB", "ALBUM"),
    "label":   ("TPUB", "LABEL"),
    "remixer": ("TPE4", "REMIXER"),
    "composer": ("TCOM", "COMPOSER"),
    "grouping": ("TIT1", "GROUPING"),
    "genre":   ("TCON", "GENRE"),
    "year":    ("TDRC", "DATE"),
    "key":     ("TKEY", "INITIALKEY"),
    "bpm":     ("TBPM", "BPM"),
    "comment": ("COMM", "COMMENT"),
}

COVER_TYPES = {"image/jpeg", "image/png"}
MAX_COVER_BYTES = 8 * 1024 * 1024


def _id3_tags(path: Path):
    """ID3-Container einer WAV- oder MP3-Datei, notfalls frisch angelegt."""
    from mutagen.mp3 import MP3
    from mutagen.wave import WAVE

    audio = MP3(str(path)) if path.suffix.lower() == ".mp3" else WAVE(str(path))
    if audio.tags is None:
        audio.add_tags()
    return audio


def read_tags(path: Path) -> dict:
    """Alle editierbaren Felder plus Cover-Infos einer Datei."""
    from mutagen.flac import FLAC, Picture

    suffix = path.suffix.lower()
    out: dict = {name: "" for name in TAG_FIELDS}
    out["has_cover"] = False
    out["cover_mime"] = ""

    if suffix == ".flac":
        f = FLAC(str(path))
        for name, (_frame, vorbis) in TAG_FIELDS.items():
            value = f.get(vorbis)
            out[name] = str(value[0]) if value else ""
        pics = f.pictures
        out["has_cover"] = bool(pics)
        out["cover_mime"] = pics[0].mime if pics else ""
        return out

    audio = _id3_tags(path)
    tags = audio.tags
    for name, (frame, _vorbis) in TAG_FIELDS.items():
        if name == "comment":
            found = tags.getall("COMM")
            out[name] = str(found[0].text[0]) if found and found[0].text else ""
            continue
        found = tags.getall(frame)
        out[name] = str(found[0].text[0]) if found and getattr(found[0], "text", None) else ""
    pics = tags.getall("APIC")
    out["has_cover"] = bool(pics)
    out["cover_mime"] = pics[0].mime if pics else ""
    return out


def write_tags(path: Path, values: dict) -> None:
    """Nur die übergebenen Felder schreiben. Ein leerer Wert löscht das Feld."""
    from mutagen.flac import FLAC
    from mutagen.id3 import COMM

    suffix = path.suffix.lower()
    fields = {k: v for k, v in values.items() if k in TAG_FIELDS}
    if not fields:
        return

    if suffix == ".flac":
        f = FLAC(str(path))
        for name, value in fields.items():
            vorbis = TAG_FIELDS[name][1]
            text = str(value).strip()
            if text:
                f[vorbis] = text
            else:
                f.pop(vorbis, None)
        f.save()
        return

    import mutagen.id3 as id3

    audio = _id3_tags(path)
    tags = audio.tags
    for name, value in fields.items():
        frame = TAG_FIELDS[name][0]
        text = str(value).strip()
        tags.delall(frame)
        if not text:
            continue
        if name == "comment":
            tags.add(COMM(encoding=3, lang="deu", desc="", text=text))
        else:
            tags.add(getattr(id3, frame)(encoding=3, text=text))
    audio.save()


def write_cover(path: Path, data: bytes, mime: str) -> None:
    from mutagen.flac import FLAC, Picture
    from mutagen.id3 import APIC

    if mime not in COVER_TYPES:
        raise ValueError("Cover muss JPEG oder PNG sein.")
    if len(data) > MAX_COVER_BYTES:
        raise ValueError("Cover ist größer als 8 MB.")

    if path.suffix.lower() == ".flac":
        f = FLAC(str(path))
        f.clear_pictures()
        pic = Picture()
        pic.type, pic.mime, pic.data = 3, mime, data
        f.add_picture(pic)
        f.save()
        return

    audio = _id3_tags(path)
    audio.tags.delall("APIC")
    audio.tags.add(APIC(encoding=3, mime=mime, type=3, desc="Cover", data=data))
    audio.save()


def read_cover(path: Path) -> tuple[bytes, str] | None:
    from mutagen.flac import FLAC

    if path.suffix.lower() == ".flac":
        pics = FLAC(str(path)).pictures
        return (pics[0].data, pics[0].mime) if pics else None
    pics = _id3_tags(path).tags.getall("APIC")
    return (pics[0].data, pics[0].mime) if pics else None


def remove_cover(path: Path) -> None:
    from mutagen.flac import FLAC

    if path.suffix.lower() == ".flac":
        f = FLAC(str(path))
        f.clear_pictures()
        f.save()
        return
    audio = _id3_tags(path)
    audio.tags.delall("APIC")
    audio.save()


def tag_folder(folder: Path, analysis: dict, song: str, rename: bool, say: LogFn) -> list[Path]:
    bpm = float(analysis.get("bpm") or 0)
    key_id3 = analysis.get("key_id3") or ""
    camelot = analysis.get("camelot") or ""
    result = []
    for path in stem_files(folder):
        stem = stem_name(path)
        try:
            tag_file(path, bpm, key_id3, camelot, song, stem)
        except Exception as exc:
            LOG.warning("Tag für %s nicht geschrieben: %s", path.name, exc)
        if rename and bpm and not re.search(r"\d+bpm", path.stem):
            label = f"{stem} - {int(round(bpm))}bpm" + (f" - {camelot}" if camelot and camelot != "–" else "")
            new = path.with_name(label + path.suffix)
            if not new.exists():
                path.rename(new)
                path = new
        result.append(path)
    if result:
        say("Tempo und Tonart in die Tags geschrieben.")
    return result


# --------------------------------------------------------------------------- #
# Wellenformen
# --------------------------------------------------------------------------- #


def waveform_peaks(path: Path, bins: int = 1200) -> list[float]:
    import soundfile as sf

    with sf.SoundFile(str(path)) as f:
        total = len(f)
        if total == 0:
            return []
        per_bin = max(1, total // bins)
        out = []
        while True:
            block = f.read(per_bin, dtype="float32", always_2d=True)
            if len(block) == 0:
                break
            out.append(float(np.sqrt(np.mean(block ** 2))))
            if len(out) >= bins:
                break
    # Absolute RMS-Werte – die Oberfläche normiert über alle Spuren einer Karte,
    # damit ein leiser Stem auch leise aussieht.
    return [round(v, 4) for v in out]


def write_waveforms(folder: Path, files: list[Path]) -> Path:
    data = {}
    for p in files:
        try:
            data[p.name] = waveform_peaks(p)
        except Exception as exc:
            LOG.warning("Wellenform für %s fehlgeschlagen: %s", p.name, exc)
    target = folder / "waveform.json"
    existing = {}
    if target.exists():
        try:
            existing = json.loads(target.read_text())
        except Exception:
            existing = {}
    existing.update(data)
    target.write_text(json.dumps(existing))
    return target


# --------------------------------------------------------------------------- #
# Loops
# --------------------------------------------------------------------------- #


def export_loops(folder: Path, downbeats: list[float], bars: int, bpm: float, say: LogFn) -> list[Path]:
    import soundfile as sf

    if len(downbeats) < bars + 1:
        raise RuntimeError("Zu wenige Takte für Loops erkannt.")
    out_dir = folder / "loops"
    out_dir.mkdir(exist_ok=True)
    written = []
    files = stem_files(folder)
    bpm_label = f"{int(round(bpm))}bpm" if bpm else "loop"
    for path in files:
        with sf.SoundFile(str(path)) as f:
            sr = f.samplerate
            data = f.read(dtype="float32", always_2d=True)
        stem = stem_name(path)
        for i in range(0, len(downbeats) - bars, bars):
            start, end = int(downbeats[i] * sr), int(downbeats[i + bars] * sr)
            if end - start < sr * 0.5:
                continue
            chunk = data[start:end]
            # kurze Ein-/Ausblendung gegen Klicks an den Schnittstellen
            fade = min(256, len(chunk) // 4)
            if fade:
                ramp = np.linspace(0, 1, fade, dtype="float32")[:, None]
                chunk[:fade] *= ramp
                chunk[-fade:] *= ramp[::-1]
            name = f"{stem}_takte_{i + 1:03d}-{i + bars:03d}_{bpm_label}.wav"
            sf.write(str(out_dir / name), chunk, sr, subtype="PCM_24")
            written.append(out_dir / name)
    say(f"{len(written)} Loops à {bars} Takte exportiert.")
    return written


# --------------------------------------------------------------------------- #
# Pitch & Tempo
# --------------------------------------------------------------------------- #


def _has_rubberband() -> bool:
    try:
        return "rubberband" in _run([_ffmpeg(), "-hide_banner", "-filters"])
    except Exception:
        return False


def shift_file(src: Path, dst: Path, semitones: float, tempo_ratio: float, say: LogFn) -> Path:
    """Tonhöhe in Halbtönen, Tempo als Faktor (1.1 = 10 % schneller)."""
    pitch = 2 ** (semitones / 12.0)
    if _has_rubberband():
        _run([_ffmpeg(), "-y", "-nostdin", "-loglevel", "error", "-i", str(src),
              "-af", f"rubberband=pitch={pitch:.6f}:tempo={tempo_ratio:.6f}:pitchq=quality:transients=crisp",
              str(dst)])
        return dst
    say("Rubber Band fehlt in ffmpeg – nehme den Phasenvocoder (etwas weicher im Klang).")
    import librosa
    import soundfile as sf

    y, sr = librosa.load(str(src), sr=None, mono=False)
    if y.ndim == 1:
        y = y[None, :]
    out = []
    for ch in y:
        z = librosa.effects.pitch_shift(ch, sr=sr, n_steps=semitones) if semitones else ch
        z = librosa.effects.time_stretch(z, rate=tempo_ratio) if tempo_ratio != 1 else z
        out.append(z)
    sf.write(str(dst), np.stack(out, axis=1), sr, subtype="PCM_24")
    return dst


def shift_folder(folder: Path, stems: list[str], semitones: float, tempo_ratio: float, say: LogFn) -> list[Path]:
    out_dir = folder / "shifted"
    out_dir.mkdir(exist_ok=True)
    written = []
    for path in stem_files(folder):
        if stems and stem_name(path) not in stems:
            continue
        label = []
        if semitones:
            label.append(f"{semitones:+g}st")
        if tempo_ratio != 1:
            label.append(f"x{tempo_ratio:.3f}".rstrip("0").rstrip("."))
        dst = out_dir / f"{stem_name(path)}_{'_'.join(label) or 'copy'}.wav"
        say(f"Verschiebe {path.name} …")
        written.append(shift_file(path, dst, semitones, tempo_ratio, say))
    return written


# --------------------------------------------------------------------------- #
# Mix-Export
# --------------------------------------------------------------------------- #


def export_mix(folder: Path, gains_db: dict[str, float], label: str, say: LogFn) -> Path:
    """Stems mit Pegeln (dB) zu einer Datei mischen; ausgeblendete Stems bekommen -inf."""
    files = [p for p in stem_files(folder) if gains_db.get(stem_name(p), 0.0) > -60]
    if not files:
        raise RuntimeError("Kein Stem im Mix.")
    out_dir = folder / "mixes"
    out_dir.mkdir(exist_ok=True)
    safe = re.sub(r"[^\w\-. ]+", "", label).strip() or "mix"
    dst = out_dir / f"{safe}.wav"
    args = [_ffmpeg(), "-y", "-nostdin", "-loglevel", "error"]
    chain = []
    for i, p in enumerate(files):
        args += ["-i", str(p)]
        chain.append(f"[{i}:a]volume={gains_db.get(stem_name(p), 0.0):.2f}dB[a{i}]")
    inputs = "".join(f"[a{i}]" for i in range(len(files)))
    chain.append(f"{inputs}amix=inputs={len(files)}:normalize=0:dropout_transition=0[out]")
    args += ["-filter_complex", ";".join(chain), "-map", "[out]", "-c:a", "pcm_s24le", str(dst)]
    _run(args)
    say(f"Mix exportiert: {dst.name}")
    return dst


# --------------------------------------------------------------------------- #
# Vocal-Veredelung
# --------------------------------------------------------------------------- #

REFINE_STEPS = {
    "dereverb": {
        "title": "De-Reverb / De-Echo",
        "candidates": ["dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt", "dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt", "UVR-DeEcho-DeReverb.pth"],
        "keep": ["dry", "no reverb", "no echo"],
        "suffix": "dry",
    },
    "denoise": {
        "title": "De-Noise",
        "candidates": ["denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt", "denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt", "UVR-DeNoise.pth"],
        "keep": ["dry", "no noise"],
        "suffix": "clean",
    },
    "leadback": {
        "title": "Lead / Backing Vocals",
        "candidates": ["bs_roformer_karaoke_anvuew.ckpt", "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt", "UVR-BVE-4B_SN-44100-1.pth"],
        "keep": ["vocals", "lead"],          # Lead-Anteil
        "also": {"instrumental": "backing", "no vocals": "backing", "backing": "backing"},
        "suffix": "lead",
    },
}


def refine_vocals(folder: Path, steps: list[str], say: LogFn, on_progress=None) -> list[Path]:
    """Kette auf vocals.*: jeder Schritt arbeitet auf dem Ergebnis des vorigen."""
    import engine

    vocals = next((p for p in stem_files(folder) if stem_name(p) == "vocals"), None)
    if vocals is None:
        raise RuntimeError("Kein Vocal-Stem in diesem Ordner.")
    out_dir = folder / "refined"
    out_dir.mkdir(exist_ok=True)
    current = vocals
    written: list[Path] = []
    tag = []
    for step in steps:
        spec = REFINE_STEPS[step]
        model = engine.first_available(spec["candidates"])
        say(f"{spec['title']} mit {model} …")
        outputs = engine.run_model(model, current, say, on_progress)
        keep = None
        for produced in outputs:
            label = (engine.stem_label(produced.name) or "").lower()
            if label in spec["keep"] and keep is None:
                keep = produced
            elif label in spec.get("also", {}):
                dst = out_dir / f"vocals_{spec['also'][label]}.wav"
                shutil.move(str(produced), dst)
                written.append(dst)
        if keep is None:
            raise RuntimeError(f"{spec['title']}: unerwartete Ausgabe {[p.name for p in outputs]}")
        tag.append(spec["suffix"])
        dst = out_dir / f"vocals_{'_'.join(tag)}.wav"
        shutil.move(str(keep), dst)
        for leftover in outputs:
            if leftover.exists():
                leftover.unlink(missing_ok=True)
        written.append(dst)
        current = dst
    say(f"Vocal-Veredelung fertig: {len(written)} Dateien.")
    return written


# --------------------------------------------------------------------------- #
# Lyrics
# --------------------------------------------------------------------------- #


def _fmt_lrc(t: float) -> str:
    return f"[{int(t // 60):02d}:{t % 60:05.2f}]"


def _fmt_srt(t: float) -> str:
    ms = int(round((t - math.floor(t)) * 1000))
    s = int(t)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d},{ms:03d}"


# Floskeln, die Whisper über Stille legt – aus Untertiteln von YouTube-Videos
# angelernt. Sie tauchen wortgleich auf, wenn gar nichts gesungen wird.
HALLUCINATIONS = re.compile(
    r"^(thank you|thanks for watching|thanks|subscribe|please subscribe"
    r"|like and subscribe|bye|bye bye|goodbye|the end|you|okay|ok"
    r"|untertitel.*|.*amara\.org.*|copyright.*|© .*"
    r"|vielen dank|danke|tschüss|abonniert.*)[.!?\s]*$",
    re.IGNORECASE,
)

# Grenzen in dB unter dem Spitzenpegel des Stems.
SILENCE_DB = -40.0      # darunter ist nichts mehr zu hören: immer verwerfen
FILLER_DB = -20.0       # darüber darf auch eine Floskel echt sein


def _segment_level(y: np.ndarray, sr: int, peak: float, start: float, end: float) -> float:
    """Effektivpegel eines Abschnitts in dB unter dem Spitzenpegel."""
    a = max(0, int(start * sr))
    b = min(len(y), int(end * sr))
    if b <= a or peak <= 0:
        return -99.0
    chunk = y[a:b]
    rms = float(np.sqrt((chunk ** 2).mean()))
    return 20.0 * math.log10(max(rms, 1e-9) / peak)


def drop_hallucinations(segments: list[dict], audio: Path, say: LogFn) -> list[dict]:
    """Segmente entfernen, die über Stille erfunden wurden.

    Whisper füllt stille Passagen mit Floskeln aus seinen Trainingsdaten
    ("Thank you.", "Untertitel von …"). Die eigene Unsicherheit hilft dabei
    nicht: gemessen stand `no_speech_prob` bei genau diesen Segmenten auf
    0,000. Verlässlich ist nur das Audio selbst – ein Vocal-Stem ist in den
    Pausen wirklich still, während gesungene Zeilen 20 bis 30 dB darüber
    liegen. Sehr leise Segmente fliegen daher immer raus, bekannte Floskeln
    schon bei mäßig leisen.
    """
    if not segments:
        return segments
    try:
        import librosa

        y, sr = librosa.load(str(audio), sr=16000, mono=True)
    except Exception as exc:
        LOG.warning("Stille-Prüfung übersprungen: %s", exc)
        return segments
    peak = float(np.abs(y).max()) if len(y) else 0.0
    if peak <= 0:
        return []

    kept: list[dict] = []
    for seg in segments:
        level = _segment_level(y, sr, peak, seg["start"], seg["end"])
        filler = bool(HALLUCINATIONS.match(seg["text"].strip()))
        if level < SILENCE_DB or (filler and level < FILLER_DB):
            continue
        kept.append(seg)

    removed = len(segments) - len(kept)
    if removed:
        say(f"{removed} erfundene Zeile(n) über Stille verworfen.")
    return kept


def transcribe(folder: Path, language: str | None, say: LogFn) -> list[Path]:
    vocals = next((p for p in stem_files(folder) if stem_name(p) == "vocals"), None)
    src = vocals or next(iter(stem_files(folder)), None)
    if src is None:
        raise RuntimeError("Keine Audiodatei zum Transkribieren.")
    model_name = os.environ.get("STEMLAB_WHISPER_MODEL", "large-v3-turbo")
    segments: list[dict] = []
    detected = language or ""

    try:
        import mlx_whisper  # nur auf Apple Silicon vorhanden

        say(f"Whisper ({model_name}) auf der Apple-GPU …")
        repo = f"mlx-community/whisper-{model_name}" if "/" not in model_name else model_name
        # condition_on_previous_text=False ist entscheidend: sonst übernimmt
        # Whisper eine einmal erfundene Zeile als Kontext und wiederholt sie
        # über den ganzen Track. hallucination_silence_threshold überspringt
        # Passagen, in denen nichts gesungen wird.
        result = mlx_whisper.transcribe(
            str(src), path_or_hf_repo=repo, language=language or None,
            word_timestamps=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            hallucination_silence_threshold=2.0,
        )
        detected = result.get("language", detected)
        for s in result.get("segments", []):
            segments.append({"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()})
    except ImportError:
        from faster_whisper import WhisperModel

        fw_name = {"large-v3-turbo": "turbo"}.get(model_name, model_name)
        say(f"Whisper ({fw_name}, faster-whisper) …")
        model = WhisperModel(fw_name, device="cpu", compute_type="int8")
        segs, info = model.transcribe(
            str(src), language=language or None, vad_filter=True, beam_size=5,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            hallucination_silence_threshold=2.0,
        )
        detected = info.language
        for s in segs:
            segments.append({"start": float(s.start), "end": float(s.end), "text": s.text.strip()})

    segments = [s for s in segments if s["text"]]
    segments = drop_hallucinations(segments, src, say)
    txt = folder / "lyrics.txt"
    lrc = folder / "lyrics.lrc"
    srt = folder / "lyrics.srt"
    txt.write_text("\n".join(s["text"] for s in segments) + "\n")
    lrc.write_text("\n".join(f"{_fmt_lrc(s['start'])}{s['text']}" for s in segments) + "\n")
    srt.write_text("".join(f"{i + 1}\n{_fmt_srt(s['start'])} --> {_fmt_srt(s['end'])}\n{s['text']}\n\n" for i, s in enumerate(segments)))
    (folder / "lyrics.json").write_text(json.dumps({"language": detected, "segments": segments}, ensure_ascii=False, indent=1))
    say(f"Lyrics: {len(segments)} Zeilen ({detected or 'Sprache unbekannt'}).")
    return [txt, lrc, srt]
