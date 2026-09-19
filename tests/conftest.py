"""Bausteine für synthetisches Testmaterial.

Die Tests kommen ohne Audiodateien aus. Das ist Absicht: Musik im Repo wäre
urheberrechtlich heikel und würde das Klonen aufblähen, und Tests, die an der
privaten Sammlung eines Entwicklers hängen, laufen bei niemand anderem.

Stattdessen wird das Material erzeugt, dessen Wahrheit dadurch per
Konstruktion feststeht: ein Raster aus exakt 124 BPM muss 124 BPM ergeben,
eine g-Moll-Kadenz muss g-Moll ergeben.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import analysis  # noqa: E402

SR = analysis.SAMPLE_RATE

# Beat This! liefert Zeiten auf einem 20-ms-Raster. Genau daran scheiterte die
# frühere Tempoberechnung, deshalb rastern die Tests ebenso.
FRAME = 0.02

# Gleichstufige Stimmung ab C4.
SEMITONE = {
    "C": 261.63, "C#": 277.18, "D": 293.66, "D#": 311.13,
    "E": 329.63, "F": 349.23, "F#": 369.99, "G": 392.00,
    "G#": 415.30, "A": 440.00, "A#": 466.16, "B": 493.88,
}


def beat_grid(bpm: float, seconds: float = 240.0, quantize: bool = True) -> np.ndarray:
    """Beat-Zeiten eines gleichmäßigen Rasters, wie ein Beat-Tracker sie liefert."""
    times = np.arange(0.0, seconds, 60.0 / bpm)
    return np.round(times / FRAME) * FRAME if quantize else times


def tone(freq: float, seconds: float, harmonics: int = 3) -> np.ndarray:
    """Ton mit abfallenden Obertönen – ein reiner Sinus gibt dem CQT zu wenig."""
    t = np.arange(int(seconds * SR)) / SR
    return sum(np.sin(2 * np.pi * freq * k * t) / k for k in range(1, harmonics + 1))


def triad(root: str, third: str, fifth: str, seconds: float = 1.0) -> np.ndarray:
    return tone(SEMITONE[root], seconds) + tone(SEMITONE[third], seconds) + tone(SEMITONE[fifth], seconds)


def progression(chords: list[tuple[str, str, str]], repeats: int = 4) -> np.ndarray:
    """Kadenz als normalisiertes Signal."""
    y = np.concatenate([triad(*c) for c in chords] * repeats)
    return y / np.abs(y).max() * 0.8


def kick_track(length: int, every: float = 0.5, gain: float = 20.0) -> np.ndarray:
    """Bassdrum: Sinus-Sweep von 120 auf 45 Hz, wie in elektronischer Musik.

    Der Pegel ist bewusst hoch. Ein dezenter Kick kippt die Tonart nicht, der
    Test wäre dann wertlos – gemessen: ab Faktor 12 liefert das ungefilterte
    Chromagramm eine falsche Tonart, mit Bandfilter bleibt sie richtig. In
    einem fertig gemasterten Dance-Track dominiert der Kick das Spektrum
    ähnlich deutlich.
    """
    out = np.zeros(length)
    dur = int(0.12 * SR)
    t = np.arange(dur) / SR
    impulse = np.sin(2 * np.pi * (120 * np.exp(-t * 18) + 45) * t) * np.exp(-t * 12)
    for start in range(0, length - dur, int(SR * every)):
        out[start:start + dur] += impulse
    return out * gain


def tiny_png() -> bytes:
    """Ein 1x1-PNG – kleinstmögliches gültiges Bild für Cover-Tests."""
    import struct
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    signatur = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
    return (signatur + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00")) + chunk(b"IEND", b""))


@pytest.fixture
def g_minor() -> np.ndarray:
    """Gm – Cm – D# – Gm, die Kadenz des Referenztracks."""
    return progression([("G", "A#", "D"), ("C", "D#", "G"), ("D#", "G", "A#"), ("G", "A#", "D")])


@pytest.fixture
def c_major() -> np.ndarray:
    """C – F – G – C."""
    return progression([("C", "E", "G"), ("F", "A", "C"), ("G", "B", "D"), ("C", "E", "G")])


# --------------------------------------------------------------------------- #
# Gerüst für die engine-Tests
# --------------------------------------------------------------------------- #
#
# engine hält Zustand auf Modulebene und spricht mit audio_separator. Beides
# braucht in jedem engine-Testmodul dieselbe Behandlung, deshalb steht es hier
# statt dreimal nebeneinander: die Fixture, die den Zustand zurücksetzt, und
# die Doppelgänger, die die Bibliothek ersetzen.

import logging  # noqa: E402

import engine  # noqa: E402



@pytest.fixture(autouse=True)
def engine_zustand():
    """Sichert die Modul-Globals von engine und stellt sie danach wieder her.

    _sep_cache wird flach kopiert, nicht per Referenz gehalten: engine ruft
    darauf .update(), was das Original sonst mitverändern würde.
    """
    cache = dict(engine._sep_cache)
    force_cpu = engine._force_cpu
    modelle = set(engine._available_models)
    katalog = {c.key: (c.resolved, c.verified) for c in engine.CATALOG}
    katalogstatus = dict(engine._catalog_state)
    try:
        yield
    finally:
        engine._catalog_state.clear()
        engine._catalog_state.update(katalogstatus)
        engine._sep_cache.clear()
        engine._sep_cache.update(cache)
        engine._force_cpu = force_cpu
        engine._available_models = modelle
        for choice in engine.CATALOG:
            choice.resolved, choice.verified = katalog[choice.key]


class FakeSeparator:
    """Doppelgänger für audio_separator.Separator.

    Hält fest, wie er gerufen wurde, und kann auf Wunsch beim Trennen eine
    Ausnahme werfen – damit lässt sich der MPS-Rückfall in run_model prüfen,
    ohne je ein Modell zu laden.
    """

    def __init__(self, preset=None, ausgaben=None, fehler=None,
                 torch_device="cpu", onnx_execution_provider=None,
                 fortschritt=None, logmeldungen=None):
        self.preset = preset
        self._ausgaben = list(ausgaben or [])
        # Geteilte Liste, absichtlich nicht kopiert: Beim MPS-Rückfall legt
        # engine einen zweiten Separator an. Die Fehlerfolge beschreibt den
        # Ablauf über beide hinweg ("erst MPS-Fehler, dann Erfolg"), nicht
        # das Verhalten je Instanz.
        #
        # Verbraucht wird sie in der Reihenfolge der separate()-Aufrufe,
        # nicht in der Reihenfolge der Erzeugung. Wer einen Test schreibt,
        # in dem beides auseinanderfällt, bekommt sonst den Fehler des
        # jeweils anderen Separators.
        self._fehler = fehler if fehler is not None else []
        self._fortschritt = list(fortschritt or [])
        self._logmeldungen = list(logmeldungen or [])
        self.torch_device = torch_device
        self.onnx_execution_provider = onnx_execution_provider
        self.load_model_aufrufe: list[dict] = []
        self.separate_aufrufe: list[str] = []

    def load_model(self, **kwargs):
        self.load_model_aufrufe.append(kwargs)

    def separate(self, pfad):
        self.separate_aufrufe.append(pfad)
        if self._fehler:
            fehler = self._fehler.pop(0)
            if fehler is not None:
                raise fehler
        # Erst nach der Fehlerprüfung: ein fehlschlagender Versuch meldet
        # keinen Fortschritt. Sonst wäre nicht unterscheidbar, ob der
        # ProgressTap nach dem MPS-Rückfall wieder an stderr hängt.
        for text in self._fortschritt:
            sys.stderr.write(text)
        for text in self._logmeldungen:
            logging.getLogger("audio_separator").info(text)
        return list(self._ausgaben)


class SeparatorFabrik:
    """Merkt sich jeden erzeugten Fake – _get_separator liefert ihn nur zurück."""

    def __init__(self, **vorgaben):
        # fehler wird von allen erzeugten Fakes geteilt (siehe FakeSeparator).
        self.vorgaben = dict(vorgaben)
        self.vorgaben["fehler"] = list(vorgaben.get("fehler") or [])
        self.erzeugte: list[FakeSeparator] = []

    def __call__(self, preset=None):
        sep = FakeSeparator(preset=preset, **self.vorgaben)
        self.erzeugte.append(sep)
        return sep

    @property
    def letzter(self) -> FakeSeparator:
        return self.erzeugte[-1]

    @property
    def ladevorgaenge(self) -> int:
        return len(self.erzeugte)


@pytest.fixture
def separator_fabrik(monkeypatch):
    """Schleust FakeSeparator an der Stelle ein, an der engine das echte Paket importiert."""
    def einrichten(**vorgaben) -> SeparatorFabrik:
        fabrik = SeparatorFabrik(**vorgaben)
        monkeypatch.setattr(engine, "_make_separator", fabrik)
        return fabrik
    return einrichten


@pytest.fixture
def stille():
    """on_log-Rückruf, der die Meldungen sammelt statt sie zu drucken."""
    meldungen: list[str] = []

    def sammeln(text: str) -> None:
        meldungen.append(text)

    sammeln.meldungen = meldungen  # type: ignore[attr-defined]
    return sammeln


# Die Fixture selbst braucht einen Beleg. Zwei aufeinander aufbauende Tests
# wären dafür untauglich: läuft der prüfende allein (per -k, als Einzelaufruf
# oder nach einer Umsortierung durch ein Plugin), ist nichts verstellt – er
# wäre dann auch ohne Fixture grün und würde nichts belegen. Deshalb prüft ein
