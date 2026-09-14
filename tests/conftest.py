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


def kick_track(length: int, every: float = 0.5, gain: float = 2.5) -> np.ndarray:
    """Bassdrum: Sinus-Sweep von 120 auf 45 Hz, wie in elektronischer Musik."""
    out = np.zeros(length)
    dur = int(0.12 * SR)
    t = np.arange(dur) / SR
    impulse = np.sin(2 * np.pi * (120 * np.exp(-t * 18) + 45) * t) * np.exp(-t * 12)
    for start in range(0, length - dur, int(SR * every)):
        out[start:start + dur] += impulse
    return out * gain


@pytest.fixture
def g_minor() -> np.ndarray:
    """Gm – Cm – D# – Gm, die Kadenz des Referenztracks."""
    return progression([("G", "A#", "D"), ("C", "D#", "G"), ("D#", "G", "A#"), ("G", "A#", "D")])


@pytest.fixture
def c_major() -> np.ndarray:
    """C – F – G – C."""
    return progression([("C", "E", "G"), ("F", "A", "C"), ("G", "B", "D"), ("C", "E", "G")])
