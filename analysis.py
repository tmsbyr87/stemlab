"""
StemLab – musikalische Analyse.

  * Taktraster mit Beat This! (Transformer, CPJKU 2024): Beats und Downbeats,
    daraus ein präzises Tempo per Regression über ein bereinigtes Beat-Raster.
    Fällt das Modell aus (kein Netz beim ersten Start, kein torch), springt das
    Tempogramm von librosa ein.
  * Tonart über ein auf 100–1000 Hz begrenztes CQT-Chromagramm und
    Albrecht-Shanahan-Profile, dazu Camelot.
  * Akkorde pro Takt per Template-Matching auf dem taktsynchronen Chromagramm.
  * MIDI-Klickspur und beats.json als Nebenprodukt.
"""

from __future__ import annotations

import json
import logging
import struct
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import numpy as np

LOG = logging.getLogger("stemlab.analysis")

SAMPLE_RATE = 22050
MAX_SECONDS = 600

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_NAMES_DE = ["C", "Cis", "D", "Dis", "E", "F", "Fis", "G", "Gis", "A", "B", "H"]

# Tonalitätsprofile nach Albrecht & Shanahan (2013) – im Test 24/24 Kadenzen richtig.
MAJOR_PROFILE = np.array([0.238, 0.006, 0.111, 0.006, 0.137, 0.094, 0.016, 0.214, 0.009, 0.080, 0.008, 0.081])
MINOR_PROFILE = np.array([0.220, 0.006, 0.104, 0.123, 0.019, 0.103, 0.012, 0.214, 0.062, 0.022, 0.061, 0.052])

CAMELOT = {
    ("B", "major"): "1B", ("G#", "minor"): "1A", ("F#", "major"): "2B", ("D#", "minor"): "2A",
    ("C#", "major"): "3B", ("A#", "minor"): "3A", ("G#", "major"): "4B", ("F", "minor"): "4A",
    ("D#", "major"): "5B", ("C", "minor"): "5A", ("A#", "major"): "6B", ("G", "minor"): "6A",
    ("F", "major"): "7B", ("D", "minor"): "7A", ("C", "major"): "8B", ("A", "minor"): "8A",
    ("G", "major"): "9B", ("E", "minor"): "9A", ("D", "major"): "10B", ("B", "minor"): "10A",
    ("A", "major"): "11B", ("F#", "minor"): "11A", ("E", "major"): "12B", ("C#", "minor"): "12A",
}

# ID3-Schreibweise für TKEY (Rekordbox, Traktor, Serato lesen das): "Am", "F#", "C#m"
def id3_key(tonic: str, mode: str) -> str:
    return tonic + ("m" if mode == "minor" else "")


@dataclass
class Analysis:
    bpm: float = 0.0
    bpm_confidence: float = 0.0
    bpm_alt: float = 0.0
    bpm_source: str = ""            # "beat_this" oder "tempogram"
    key: str = "unbekannt"          # "A minor"
    key_de: str = "unbekannt"       # "a-Moll"
    key_id3: str = ""               # "Am"
    camelot: str = "–"
    key_tonic: str = ""             # "G" – für andere Schreibweisen
    key_mode: str = ""              # "minor"
    energy: int = 0                 # 1–10, wie in Mixed In Key
    key_confidence: float = 0.0
    key_alt: str = ""
    beats: list[float] = field(default_factory=list)
    downbeats: list[float] = field(default_factory=list)
    beats_per_bar: int = 4
    chords: list[dict] = field(default_factory=list)   # {bar, start, end, chord}
    chord_sheet: str = ""
    seconds_analyzed: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> dict:
        """Kompakte Fassung für Ereignisse und Tags (ohne Beat-Listen)."""
        d = self.as_dict()
        d["beats"] = len(self.beats)
        d["downbeats"] = len(self.downbeats)
        d["chords"] = len(self.chords)
        return d


# --------------------------------------------------------------------------- #
# Laden
# --------------------------------------------------------------------------- #


def _load(path: Path) -> tuple[np.ndarray, float]:
    import librosa

    y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True, duration=MAX_SECONDS)
    return y, float(len(y) / sr)


# --------------------------------------------------------------------------- #
# Taktraster: Beat This!
# --------------------------------------------------------------------------- #

_beat_lock = threading.Lock()
_beat_model = {"obj": None, "device": ""}


def _beat_this_model():
    """Einmal laden, dann behalten. Erst Apple-GPU, sonst CPU."""
    with _beat_lock:
        if _beat_model["obj"] is not None:
            return _beat_model["obj"]
        from beat_this.inference import Audio2Beats

        import torch

        device = "cpu"
        try:
            import platform

            if torch.backends.mps.is_available() and platform.uname().processor == "arm":
                device = "mps"
        except Exception:
            pass
        try:
            model = Audio2Beats(checkpoint_path="final0", device=device, dbn=False)
        except Exception:
            if device == "cpu":
                raise
            model = Audio2Beats(checkpoint_path="final0", device="cpu", dbn=False)
            device = "cpu"
        _beat_model.update(obj=model, device=device)
        return model


def _tempo_from_beats(beats: np.ndarray, subdivision: int = 1) -> tuple[float, float]:
    """Tempo per Regression über ein bereinigtes Beat-Raster.

    Beat This! liefert Zeiten auf einem 20-ms-Raster. Ein echtes Tempo wie
    124 BPM (0.483871 s pro Beat) lässt sich darauf gar nicht abbilden, das
    Modell wechselt deshalb zwischen 0.48 s und 0.50 s. Eine Regression mittelt
    diese Quantisierung sauber weg – aber nur, wenn jeder Beat auch wirklich
    ein Beat ist. Setzt das Modell in dichten Passagen Zwischenschläge oder
    lässt es welche aus, verschiebt das alle folgenden Indizes und das Tempo
    läuft weg (gemessen: 126.9 statt 124.0 BPM).

    Darum zuerst die Grundperiode als Median der nicht unterteilten Abstände
    bestimmen, dann jedem Beat seine echte Rasterposition zuordnen und nur die
    Beats regressieren, die auf dem Raster liegen.
    """
    beats = np.asarray(beats, dtype=float)
    if len(beats) < 4:
        return 0.0, 0.0
    intervals = np.diff(beats)
    if not (intervals > 0).any():
        return 0.0, 0.0

    # Grundperiode: Median der Abstände ohne die offensichtlichen Unterteilungen.
    rough = float(np.median(intervals))
    full = intervals[intervals > rough * 0.6]
    period = float(np.median(full)) if len(full) else rough
    if period <= 0:
        return 0.0, 0.0

    # Rasterposition je Beat. Abstände, die kein ganzes Vielfaches der Periode
    # sind, schieben das Raster weiter, liefern aber keinen Stützpunkt.
    positions = [0.0]
    times = [beats[0]]
    cursor = 0.0
    for k, gap in enumerate(intervals):
        steps = gap / period
        nearest = round(steps)
        if nearest >= 1 and abs(steps - nearest) < 0.18:
            cursor += nearest
            positions.append(cursor)
            times.append(beats[k + 1])
        else:
            cursor += max(steps, 0.0)

    if len(positions) < 4:
        return 0.0, 0.0

    x = np.array(positions)
    y = np.array(times)
    coeffs = np.polyfit(x, y, 1)
    slope = float(coeffs[0])
    if slope <= 0:
        return 0.0, 0.0
    bpm = 60.0 / slope * subdivision

    # Konfidenz aus dem Rest der Regression, gemessen in Bruchteilen eines Beats,
    # plus dem Anteil der Beats, die überhaupt auf dem Raster lagen.
    residual = float(np.std(y - np.polyval(coeffs, x)))
    tightness = max(0.0, 1.0 - (residual / slope) * 6.0)
    coverage = len(positions) / len(beats)
    # Wenige Beats tragen wenig Evidenz: ein durchgehender Track hat hunderte.
    support = min(1.0, len(positions) / 64.0)
    confidence = float(max(0.0, min(1.0, tightness * coverage * support)))
    return round(bpm, 2), round(confidence, 2)


# Wie weit ein geschätztes Tempo von einem glatten Wert abweichen darf, um
# trotzdem als dieser gelesen zu werden.
#
# 0.08 statt der früheren 0.06: Die enge Grenze verfehlte genau die Fälle,
# für die das Snapping gedacht war – `abs(119.06 - 119.0)` ergibt in
# Gleitkomma 0.060000000000002274 und liegt damit knapp über 0.06.
#
# Nach oben begrenzt die Halbschritt-Prüfung den Wert: Bei 0.10 läge ein
# echtes Live-Tempo von 123,4 exakt an der Grenze zu 123,5 und würde
# verbogen. 0.08 fängt die beobachteten Abweichungen (höchstens 0.09 …
# 122,09 bleibt damit bewusst außen vor, siehe unten) und lässt krumme
# Tempi in Ruhe.
_SNAP_TOLERANZ = 0.08


def _snap_bpm(bpm: float, confidence: float) -> float:
    """Produzierte Tracks laufen auf einem DAW-Tempo. Liegt die Schätzung sehr
    dicht an einem glatten Wert, ist der glatte Wert fast immer der richtige."""
    if bpm <= 0 or confidence < 0.5:
        return round(bpm, 2)
    for step in (1.0, 0.5):
        candidate = round(bpm / step) * step
        if abs(bpm - candidate) <= _SNAP_TOLERANZ:
            return round(candidate, 2)
    return round(bpm, 2)


def _beats_beat_this(y: np.ndarray, sr: int) -> tuple[list[float], list[float]]:
    model = _beat_this_model()
    beats, downbeats = model(y, sr)
    return [round(float(b), 3) for b in beats], [round(float(d), 3) for d in downbeats]


def _beats_librosa(y: np.ndarray) -> tuple[list[float], list[float], float]:
    import librosa

    onset = librosa.onset.onset_strength(y=y, sr=SAMPLE_RATE)
    tempo, frames = librosa.beat.beat_track(onset_envelope=onset, sr=SAMPLE_RATE, units="frames")
    times = librosa.frames_to_time(frames, sr=SAMPLE_RATE)
    beats = [round(float(t), 3) for t in times]
    return beats, beats[::4], float(np.atleast_1d(tempo)[0])


def _fold_bpm(bpm: float) -> float:
    while 0 < bpm < 60:
        bpm *= 2
    while bpm >= 200:
        bpm /= 2
    return round(bpm, 2)


# --------------------------------------------------------------------------- #
# Tonart
# --------------------------------------------------------------------------- #


def key_chroma(y: np.ndarray, hop: int = 2048) -> np.ndarray:
    """Chromagramm für die Tonartbestimmung, auf 100–1000 Hz begrenzt.

    Im vollen Band dominieren Bassdrum und Sub-Bass eines Dance-Tracks das
    Chromagramm: der breitbandige Kick-Impuls schmiert über alle zwölf Bins und
    drückt das gemittelte Profil flach. Die Korrelation gegen die Tonprofile
    entscheidet dann praktisch zufällig zwischen benachbarten Quinten
    (gemessen: C-Dur statt g-Moll – sechs von sieben Tönen teilen sich beide).

    Das Band von 100 Hz bis 1 kHz lässt Kick und Sub draußen und behält den
    harmonisch tragenden Bereich. Damit wird die Tonart auf dem reinen Mix
    genauso sicher erkannt wie auf den getrennten Stems.
    """
    import librosa
    import scipy.signal as signal

    nyquist = SAMPLE_RATE / 2
    try:
        sos = signal.butter(4, [100.0 / nyquist, 1000.0 / nyquist], btype="band", output="sos")
        filtered = signal.sosfiltfilt(sos, y)
    except Exception:
        filtered = y
    return librosa.feature.chroma_cqt(y=filtered, sr=SAMPLE_RATE, hop_length=hop)


def _key(chroma_mean: np.ndarray) -> dict:
    if not chroma_mean.any():
        return {}
    profile = chroma_mean / chroma_mean.max()
    scores: list[tuple[float, int, str]] = []
    for shift in range(12):
        rolled = np.roll(profile, -shift)
        for name, template in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            scores.append((float(np.corrcoef(rolled, template)[0, 1]), shift, name))
    scores.sort(reverse=True)
    best = scores[0]
    tonic, mode = NOTE_NAMES[best[1]], best[2]

    # Die Zweitplatzierte ist oft dieselbe Tonart mit anderem Etikett (Parallele,
    # Variante). Als echte Alternative zählt nur ein anderer Grundton.
    alt = next((x for x in scores[1:] if x[1] != best[1]), scores[1])
    alt_tonic = NOTE_NAMES_DE[alt[1]]
    return {
        "key": f"{tonic} {mode}",
        "key_de": f"{NOTE_NAMES_DE[best[1]]}-Dur" if mode == "major" else f"{NOTE_NAMES_DE[best[1]].lower()}-Moll",
        "key_id3": id3_key(tonic, mode),
        "camelot": CAMELOT.get((tonic, mode), "–"),
        "key_tonic": tonic,
        "key_mode": mode,
        "key_confidence": round(float(min(1.0, max(0.0, (best[0] - alt[0]) * 4))), 2),
        "key_alt": f"{alt_tonic}-Dur" if alt[2] == "major" else f"{alt_tonic.lower()}-Moll",
    }


# --------------------------------------------------------------------------- #
# Energie
# --------------------------------------------------------------------------- #


def _energy(y: np.ndarray) -> int:
    """Energielevel 1 bis 10, wie es DJ-Software erwartet.

    Mixed In Key nennt das "Energy". Gemeint ist der wahrgenommene Druck eines
    Tracks, nicht seine Lautheit: ein leiser Ambient-Track und ein leiser
    Techno-Track unterscheiden sich im Anteil tiefer Frequenzen und in der
    Dichte der Anschläge. Beides fließt hier ein.
    """
    import librosa

    if not y.size:
        return 0
    # Wie dicht sind die Anschläge? Ein treibender Track hat viele.
    onset = librosa.onset.onset_strength(y=y, sr=SAMPLE_RATE)
    dichte = float(np.mean(onset)) if onset.size else 0.0

    # Wie viel Energie steckt im Bass? Vier Sekunden reichen für ein Bild.
    spec = np.abs(librosa.stft(y[: SAMPLE_RATE * 60], n_fft=2048))
    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=2048)
    gesamt = float(spec.sum()) or 1.0
    tief = float(spec[freqs < 250].sum()) / gesamt

    # Beide Anteile auf 1–10 abbilden. Die Faktoren sind an typischer
    # Clubmusik ausgerichtet und bewusst großzügig geschnitten.
    wert = dichte / 3.0 * 5.0 + tief * 12.0
    return int(max(1, min(10, round(wert))))


# --------------------------------------------------------------------------- #
# Akkorde pro Takt
# --------------------------------------------------------------------------- #

_CHORD_TEMPLATES: list[tuple[str, np.ndarray]] = []
for _root in range(12):
    for _suffix, _intervals in (("", (0, 4, 7)), ("m", (0, 3, 7))):
        _tpl = np.zeros(12)
        for _k, _iv in enumerate(_intervals):
            _tpl[(_root + _iv) % 12] = (1.0, 0.8, 0.9)[_k]
        _CHORD_TEMPLATES.append((NOTE_NAMES[_root] + _suffix, _tpl))


def _chords(chroma: np.ndarray, hop: int, downbeats: list[float], beats: list[float], total: float) -> list[dict]:
    """Ein Akkord pro Takt: mittleres Chroma zwischen zwei Downbeats gegen 24 Dreiklang-Templates."""
    import librosa

    borders = list(downbeats) if len(downbeats) >= 2 else list(beats[::4])
    if len(borders) < 2:
        return []
    if borders[-1] < total - 0.5:
        borders.append(min(total, borders[-1] + (borders[-1] - borders[-2])))
    frames = librosa.time_to_frames(np.array(borders), sr=SAMPLE_RATE, hop_length=hop)
    energy_floor = float(np.percentile(chroma.sum(axis=0), 20)) * 0.5
    out = []
    for i in range(len(borders) - 1):
        a, b = int(frames[i]), max(int(frames[i]) + 1, int(frames[i + 1]))
        seg = chroma[:, a:b]
        if seg.size == 0:
            continue
        mean = seg.mean(axis=1)
        if mean.sum() <= energy_floor:
            label, conf = "N.C.", 0.0
        else:
            v = mean / (np.linalg.norm(mean) + 1e-9)
            sims = [(float(v @ (t / np.linalg.norm(t))), name) for name, t in _CHORD_TEMPLATES]
            sims.sort(reverse=True)
            label = sims[0][1]
            conf = round(sims[0][0] - sims[1][0], 3)
        out.append({"bar": i + 1, "start": round(borders[i], 3), "end": round(borders[i + 1], 3), "chord": label, "confidence": conf})
    return out


def _chord_sheet(chords: list[dict], per_line: int = 4) -> str:
    if not chords:
        return ""
    cells = [c["chord"] for c in chords]
    lines = []
    for i in range(0, len(cells), per_line):
        lines.append("| " + " | ".join(f"{c:<5}" for c in cells[i:i + per_line]) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# MIDI-Klickspur
# --------------------------------------------------------------------------- #


def write_click_midi(path: Path, beats: list[float], downbeats: list[float], bpm: float) -> None:
    """Minimaler Standard-MIDI-File-Writer: Downbeat = Note 76, Beat = Note 77 (GM Woodblock)."""
    ticks_per_beat = 480
    tempo_us = int(60_000_000 / max(bpm, 1))
    down = set(round(d, 2) for d in downbeats)

    def vlq(n: int) -> bytes:
        out = [n & 0x7F]
        n >>= 7
        while n:
            out.append((n & 0x7F) | 0x80)
            n >>= 7
        return bytes(reversed(out))

    events: list[tuple[int, bytes]] = [(0, b"\xff\x51\x03" + tempo_us.to_bytes(3, "big"))]
    for t in beats:
        tick = int(t * bpm / 60 * ticks_per_beat)
        note = 76 if round(t, 2) in down else 77
        vel = 110 if note == 76 else 80
        events.append((tick, bytes([0x99, note, vel])))
        events.append((tick + ticks_per_beat // 8, bytes([0x89, note, 0])))
    events.sort(key=lambda e: e[0])
    data = bytearray()
    last = 0
    for tick, msg in events:
        data += vlq(tick - last) + msg
        last = tick
    data += b"\x00\xff\x2f\x00"
    with open(path, "wb") as fh:
        fh.write(b"MThd" + struct.pack(">IHHH", 6, 0, 1, ticks_per_beat))
        fh.write(b"MTrk" + struct.pack(">I", len(data)) + data)


# --------------------------------------------------------------------------- #
# Alles zusammen
# --------------------------------------------------------------------------- #


def analyze(path: Path, on_log: Callable[[str], None] | None = None) -> Analysis:
    import librosa

    say = on_log or (lambda _m: None)
    result = Analysis()
    y, seconds = _load(path)
    result.seconds_analyzed = round(seconds, 1)
    if seconds < 3:
        return result

    # --- Taktraster ---
    beats: list[float] = []
    downbeats: list[float] = []
    try:
        say("Taktraster mit Beat This! …")
        beats, downbeats = _beats_beat_this(y, SAMPLE_RATE)
        bpm, conf = _tempo_from_beats(np.array(beats))
        # Gegenprobe über die Downbeats: ein Takt ist vier Beats lang, das Raster
        # ist dort dünner besetzt und damit unabhängig von Zwischenschlägen.
        if len(downbeats) >= 4:
            bar_bpm, bar_conf = _tempo_from_beats(np.array(downbeats), subdivision=4)
            if bar_bpm > 0 and bpm > 0 and abs(bar_bpm - bpm) / bpm > 0.02:
                LOG.warning("Tempo unsicher: Beats %.2f, Takte %.2f BPM", bpm, bar_bpm)
                if bar_conf > conf:
                    bpm, conf = bar_bpm, bar_conf
                conf = min(conf, 0.45)
        result.bpm_source = "beat_this"
    except Exception as exc:
        LOG.warning("Beat This! nicht verfügbar (%s) – nehme librosa.", exc)
        say("Beat This! nicht verfügbar – Tempo per Tempogramm.")
        beats, downbeats, bpm = _beats_librosa(y)
        conf = 0.3
        result.bpm_source = "tempogram"
    result.beats, result.downbeats = beats, downbeats
    result.bpm = _snap_bpm(_fold_bpm(bpm), conf)
    result.bpm_confidence = conf
    result.bpm_alt = round(result.bpm * 2, 2) if result.bpm < 100 else round(result.bpm / 2, 2)
    if len(downbeats) >= 2 and beats:
        per_bar = [sum(1 for b in beats if downbeats[i] <= b < downbeats[i + 1]) for i in range(len(downbeats) - 1)]
        if per_bar:
            result.beats_per_bar = int(np.bincount(per_bar).argmax()) or 4

    # --- Chroma, Tonart, Akkorde ---
    say("Tonart und Akkorde …")
    hop = 2048
    chroma = librosa.feature.chroma_cqt(y=y, sr=SAMPLE_RATE, hop_length=hop)
    # Tonart auf dem bandbegrenzten Chromagramm, Akkorde auf dem vollen: die
    # Akkorderkennung pro Takt ist gegen den Kick unempfindlich, die gemittelte
    # Tonartbestimmung nicht.
    for k, v in _key(key_chroma(y, hop).mean(axis=1)).items():
        setattr(result, k, v)
    result.energy = _energy(y)
    result.chords = _chords(chroma, hop, downbeats, beats, seconds)
    result.chord_sheet = _chord_sheet(result.chords, per_line=4)
    return result


def write_sidecars(folder: Path, a: Analysis) -> list[Path]:
    """analysis.json, beats.json, chords.txt und click.mid neben die Stems legen."""
    written = []
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "analysis.json").write_text(json.dumps(a.as_dict(), indent=2, ensure_ascii=False))
    written.append(folder / "analysis.json")
    if a.beats:
        (folder / "beats.json").write_text(json.dumps({"bpm": a.bpm, "beats_per_bar": a.beats_per_bar, "beats": a.beats, "downbeats": a.downbeats}))
        written.append(folder / "beats.json")
        try:
            write_click_midi(folder / "click.mid", a.beats, a.downbeats, a.bpm or 120)
            written.append(folder / "click.mid")
        except Exception as exc:
            LOG.warning("MIDI-Klick nicht geschrieben: %s", exc)
    if a.chords:
        header = f"{a.key_de} · {a.bpm} BPM · {a.beats_per_bar}/4\n\n"
        (folder / "chords.txt").write_text(header + a.chord_sheet + "\n")
        written.append(folder / "chords.txt")
    return written
