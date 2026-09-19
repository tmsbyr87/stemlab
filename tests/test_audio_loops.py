"""Loops schneiden und Wellenformen berechnen.

Beides arbeitet auf echtem Audio – synthetisch erzeugt, aber ungefälscht:
soundfile und numpy liegen in der CI, ffmpeg wird hier nicht gebraucht.
Deshalb prüfen diese Tests nicht Argumente, sondern Ergebnisse.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

import postprocess
from conftest import SR


def _stem(ordner, name: str, sekunden: float, pegel: float = 0.5, sr: int = SR):
    """Schreibt einen Stem aus gleichmäßigem Rauschen fester Lautstärke."""
    rauschen = np.full(int(sekunden * sr), pegel, dtype="float32")
    pfad = ordner / f"{name}.wav"
    sf.write(str(pfad), rauschen, sr, subtype="PCM_24")
    return pfad


@pytest.fixture
def stemordner(tmp_path):
    """Ergebnisordner mit zwei Stems à 8 Sekunden."""
    _stem(tmp_path, "vocals", 8.0)
    _stem(tmp_path, "drums", 8.0)
    return tmp_path


def _takte(anzahl: int, sekunden_pro_takt: float = 2.0) -> list[float]:
    return [i * sekunden_pro_takt for i in range(anzahl)]


# --------------------------------------------------------------------------- #
# export_loops
# --------------------------------------------------------------------------- #

def test_loops_schneidet_auf_den_taktgrenzen(stemordner, stille):
    """Ein Loop über 1 Takt à 2 s muss exakt 2 s lang sein."""
    pfade = postprocess.export_loops(stemordner, _takte(5), bars=1, bpm=120, say=stille)

    assert pfade, "es müssen Loops entstehen"
    for p in pfade:
        with sf.SoundFile(str(p)) as f:
            assert abs(len(f) / f.samplerate - 2.0) < 0.01


def test_loops_ueber_mehrere_takte(stemordner, stille):
    pfade = postprocess.export_loops(stemordner, _takte(5), bars=2, bpm=120, say=stille)

    for p in pfade:
        with sf.SoundFile(str(p)) as f:
            assert abs(len(f) / f.samplerate - 4.0) < 0.01


def test_loops_zu_wenige_takte(stemordner, stille):
    """Für 4 Takte braucht es 5 Downbeats – mit 4 geht es nicht."""
    with pytest.raises(RuntimeError, match="Zu wenige Takte"):
        postprocess.export_loops(stemordner, _takte(4), bars=4, bpm=120, say=stille)


def test_loops_landen_im_unterordner(stemordner, stille):
    pfade = postprocess.export_loops(stemordner, _takte(3), bars=1, bpm=120, say=stille)

    assert all(p.parent == stemordner / "loops" for p in pfade)
    assert all(p.exists() for p in pfade)


def test_loops_dateiname_traegt_takte_und_tempo(stemordner, stille):
    pfade = postprocess.export_loops(stemordner, _takte(3), bars=1, bpm=123.6, say=stille)
    namen = sorted(p.name for p in pfade)

    # 123,6 wird auf 124 GERUNDET, nicht auf 123 abgeschnitten. Die
    # Taktnummern sind dreistellig und 1-basiert.
    assert namen[0] == "drums_takte_001-001_124bpm.wav"
    assert any("takte_002-002" in n for n in namen)


def test_loops_ohne_tempo_heisst_loop(stemordner, stille):
    pfade = postprocess.export_loops(stemordner, _takte(3), bars=1, bpm=0, say=stille)
    assert all("_loop.wav" in p.name for p in pfade)


def test_loops_ueberspringt_zu_kurze_abschnitte(tmp_path, stille):
    """Unter einer halben Sekunde lohnt kein Loop – solche Abschnitte fallen weg."""
    _stem(tmp_path, "vocals", 4.0)
    # Takte im Abstand von 0,2 s: jeder Abschnitt ist zu kurz.
    zu_kurz = [i * 0.2 for i in range(6)]

    assert postprocess.export_loops(tmp_path, zu_kurz, bars=1, bpm=120, say=stille) == []


def test_loops_blendet_die_raender_aus(tmp_path, stille):
    """Gegen Klicks an den Schnittstellen: Anfang und Ende laufen weich."""
    _stem(tmp_path, "vocals", 8.0, pegel=0.8)
    pfade = postprocess.export_loops(tmp_path, _takte(3), bars=1, bpm=120, say=stille)

    daten, _sr = sf.read(str(pfade[0]), dtype="float32")
    assert daten[0] == pytest.approx(0.0, abs=1e-6), "Anfang muss bei null starten"
    assert daten[-1] < 0.8, "Ende muss abfallen"
    assert daten[len(daten) // 2] == pytest.approx(0.8, abs=1e-3), "Mitte bleibt unangetastet"


def test_loops_je_stem_eigene_dateien(stemordner, stille):
    pfade = postprocess.export_loops(stemordner, _takte(3), bars=1, bpm=120, say=stille)
    stems = {p.name.split("_")[0] for p in pfade}

    assert stems == {"vocals", "drums"}


def test_loops_meldet_die_anzahl(stemordner, stille):
    postprocess.export_loops(stemordner, _takte(3), bars=1, bpm=120, say=stille)
    assert "4 Loops" in stille.meldungen[-1]


def test_loops_im_leeren_ordner(tmp_path, stille):
    assert postprocess.export_loops(tmp_path, _takte(3), bars=1, bpm=120, say=stille) == []


# --------------------------------------------------------------------------- #
# waveform_peaks / write_waveforms
# --------------------------------------------------------------------------- #

def test_wellenform_liefert_die_gewuenschte_zahl_stuetzstellen(tmp_path):
    """Die Oberfläche zeichnet ein Band fester Breite – zu viele Werte
    stauchen es, zu wenige lassen es stückeln."""
    pfad = _stem(tmp_path, "vocals", 10.0)
    assert len(postprocess.waveform_peaks(pfad, bins=100)) == 100
    assert len(postprocess.waveform_peaks(pfad, bins=50)) == 50


def test_wellenform_laenge_haengt_nicht_an_der_spieldauer(tmp_path):
    kurz = postprocess.waveform_peaks(_stem(tmp_path, "vocals", 4.0), bins=60)
    lang = postprocess.waveform_peaks(_stem(tmp_path, "drums", 40.0), bins=60)
    assert len(kurz) == len(lang) == 60


def test_wellenform_misst_die_lautstaerke(tmp_path):
    """RMS eines konstanten Pegels ist dieser Pegel."""
    pfad = _stem(tmp_path, "vocals", 4.0, pegel=0.5)
    werte = postprocess.waveform_peaks(pfad, bins=50)

    assert all(v == pytest.approx(0.5, abs=0.01) for v in werte)


def test_wellenform_misst_rms_nicht_den_mittelwert(tmp_path):
    """Bei einem Rechteck, das zwischen 0 und 1 springt, ist der RMS 0,707,
    der Betragsmittelwert aber 0,5. Nur so sind die beiden unterscheidbar."""
    sr = SR
    muster = np.tile(np.array([1.0] * 100 + [0.0] * 100, dtype="float32"), 40)
    pfad = tmp_path / "rechteck.wav"
    sf.write(str(pfad), muster, sr, subtype="PCM_24")

    werte = postprocess.waveform_peaks(pfad, bins=8)

    assert all(v == pytest.approx(0.707, abs=0.02) for v in werte)


def test_wellenform_bei_stille_ist_null(tmp_path):
    pfad = _stem(tmp_path, "vocals", 4.0, pegel=0.0)
    assert all(v == 0.0 for v in postprocess.waveform_peaks(pfad, bins=20))


def test_wellenform_bleibt_absolut(tmp_path):
    """Nicht normiert: ein leiser Stem muss auch leise aussehen."""
    laut = postprocess.waveform_peaks(_stem(tmp_path, "vocals", 4.0, pegel=0.8), bins=20)
    leise = postprocess.waveform_peaks(_stem(tmp_path, "drums", 4.0, pegel=0.1), bins=20)

    assert max(laut) > max(leise) * 5


def test_wellenform_leerer_datei(tmp_path):
    pfad = tmp_path / "leer.wav"
    sf.write(str(pfad), np.zeros(0, dtype="float32"), SR)
    assert postprocess.waveform_peaks(pfad) == []


def test_wellenform_rundet_auf_vier_stellen(tmp_path):
    """Vier Stellen, nicht zwei: ein leiser Stem mit RMS 0,0312 darf in
    der Oberfläche nicht als 0,03 alle Struktur verlieren."""
    pfad = _stem(tmp_path, "vocals", 2.0, pegel=0.03125)
    werte = postprocess.waveform_peaks(pfad, bins=10)

    assert all(w == pytest.approx(0.0312, abs=0.0002) for w in werte)
    assert any(round(w, 2) != w for w in werte), "zwei Stellen wären zu grob"


def test_write_waveforms_schreibt_json(tmp_path):
    a = _stem(tmp_path, "vocals", 2.0)
    b = _stem(tmp_path, "drums", 2.0)

    ziel = postprocess.write_waveforms(tmp_path, [a, b])

    daten = json.loads(ziel.read_text())
    assert ziel.name == "waveform.json"
    assert set(daten) == {"vocals.wav", "drums.wav"}
    assert all(isinstance(v, list) and v for v in daten.values())


def test_write_waveforms_ergaenzt_bestehende_datei(tmp_path):
    """Ein zweiter Lauf für einen weiteren Stem darf die anderen nicht löschen."""
    a = _stem(tmp_path, "vocals", 2.0)
    postprocess.write_waveforms(tmp_path, [a])

    b = _stem(tmp_path, "drums", 2.0)
    ziel = postprocess.write_waveforms(tmp_path, [b])

    assert set(json.loads(ziel.read_text())) == {"vocals.wav", "drums.wav"}


def test_write_waveforms_ueberlebt_kaputte_datei(tmp_path):
    """Eine unlesbare Spur darf die übrigen nicht mitreißen."""
    gut = _stem(tmp_path, "vocals", 2.0)
    kaputt = tmp_path / "kaputt.wav"
    kaputt.write_bytes(b"kein WAV")

    daten = json.loads(postprocess.write_waveforms(tmp_path, [gut, kaputt]).read_text())

    assert "vocals.wav" in daten
    assert "kaputt.wav" not in daten


def test_write_waveforms_ueberlebt_kaputtes_json(tmp_path):
    (tmp_path / "waveform.json").write_text("{kein json")
    a = _stem(tmp_path, "vocals", 2.0)

    daten = json.loads(postprocess.write_waveforms(tmp_path, [a]).read_text())
    assert "vocals.wav" in daten
