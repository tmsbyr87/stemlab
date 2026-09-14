"""Filterung erfundener Lyrics-Zeilen.

Whisper legt über stille Passagen Floskeln aus seinen Trainingsdaten. Am
Referenztrack kam ein 7:29-Stück als 21-mal "Thank you." zurück, davon eine
Zeile echt. Whispers eigenes `no_speech_prob` stand dabei auf 0,000 – nur der
Pegel im Vocal-Stem trennt zuverlässig.
"""

import numpy as np
import pytest
import soundfile as sf

import postprocess
from conftest import SR


@pytest.fixture
def stem(tmp_path):
    """Vocal-Stem: laut zwischen 10–20 s und 40–50 s, sonst still."""
    y = np.zeros(int(60 * SR))
    t = np.arange(int(10 * SR)) / SR
    voice = np.sin(2 * np.pi * 220 * t) * 0.8
    y[int(10 * SR):int(20 * SR)] = voice
    y[int(40 * SR):int(50 * SR)] = voice
    path = tmp_path / "vocals.wav"
    sf.write(path, y, SR)
    return path


def texts(segments):
    return [s["text"] for s in segments]


def test_floskel_ueber_stille_faellt_weg(stem):
    segments = [
        {"start": 0.0, "end": 3.0, "text": "Thank you."},
        {"start": 12.0, "end": 15.0, "text": "And you, and you, we are"},
    ]
    kept = postprocess.drop_hallucinations(segments, stem, lambda _m: None)
    assert texts(kept) == ["And you, and you, we are"]


def test_echte_zeile_ueber_stille_faellt_ebenfalls_weg(stem):
    """Wo nichts zu hören ist, kann auch nichts gesungen worden sein."""
    segments = [{"start": 25.0, "end": 28.0, "text": "Everywhere inside"}]
    assert postprocess.drop_hallucinations(segments, stem, lambda _m: None) == []


def test_floskel_im_lauten_teil_bleibt(stem):
    """Ein Lied darf 'Thank you' singen – dann ist es auch zu hören."""
    segments = [{"start": 12.0, "end": 15.0, "text": "Thank you."}]
    kept = postprocess.drop_hallucinations(segments, stem, lambda _m: None)
    assert texts(kept) == ["Thank you."]


@pytest.mark.parametrize("phrase", [
    "Thank you.", "Thanks for watching", "Please subscribe", "Bye",
    "Untertitel von der Amara.org Community", "Vielen Dank", "Tschüss",
])
def test_bekannte_floskeln_werden_erkannt(phrase):
    assert postprocess.HALLUCINATIONS.match(phrase)


@pytest.mark.parametrize("phrase", [
    "And you, and you, we are",
    "Everywhere inside",
    "Believe me, believe me",
    "We are",
])
def test_echte_zeilen_gelten_nicht_als_floskel(phrase):
    """Der Hook des Referenztracks darf nicht in die Floskelliste geraten."""
    assert not postprocess.HALLUCINATIONS.match(phrase)


def test_ohne_segmente_passiert_nichts(stem):
    assert postprocess.drop_hallucinations([], stem, lambda _m: None) == []


def test_unlesbare_datei_laesst_segmente_stehen(tmp_path):
    """Lieber ungefiltert als stillschweigend alles verwerfen."""
    segments = [{"start": 0.0, "end": 1.0, "text": "Thank you."}]
    kept = postprocess.drop_hallucinations(segments, tmp_path / "fehlt.wav", lambda _m: None)
    assert kept == segments


def test_komplett_stiller_stem_ergibt_nichts(tmp_path):
    path = tmp_path / "leer.wav"
    sf.write(path, np.zeros(int(5 * SR)), SR)
    segments = [{"start": 0.0, "end": 2.0, "text": "Thank you."}]
    assert postprocess.drop_hallucinations(segments, path, lambda _m: None) == []


def test_schwellen_haben_die_erwartete_ordnung():
    """Stille-Schwelle muss unter der Floskel-Schwelle liegen.

    Sonst würde die Floskelregel nie greifen: was unter SILENCE_DB liegt,
    fliegt ohnehin schon raus.
    """
    assert postprocess.SILENCE_DB < postprocess.FILLER_DB < 0
