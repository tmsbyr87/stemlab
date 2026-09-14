"""Tempoberechnung aus einem Beat-Raster.

Die Fälle hier sind keine erfundenen Randfälle, sondern das, was Beat This!
tatsächlich liefert. Der Referenztrack wurde einmal mit 126,9 BPM statt 124,0
gemeldet – die Ursache steckt in `test_zwischenschlaege_verschieben_das_tempo_nicht`.
"""

import numpy as np
import pytest

import analysis
from conftest import FRAME, beat_grid


def estimate(beats: np.ndarray) -> float:
    """Tempo wie in analyze(): schätzen, falten, auf glatte Werte ziehen."""
    bpm, conf = analysis._tempo_from_beats(np.asarray(beats))
    return analysis._snap_bpm(analysis._fold_bpm(bpm), conf)


@pytest.mark.parametrize("bpm", [90.0, 100.0, 124.0, 128.0, 140.0, 174.0])
def test_sauberes_raster_trifft_genau(bpm):
    """Ein ungestörtes Raster muss exakt sein – auch bei 124 BPM.

    0,483871 s pro Beat lässt sich auf einem 20-ms-Raster gar nicht abbilden;
    der Tracker wechselt dort zwischen 0,48 s und 0,50 s. Die Regression muss
    diese Quantisierung wegmitteln.
    """
    assert estimate(beat_grid(bpm)) == pytest.approx(bpm, abs=0.05)


def test_zwischenschlaege_verschieben_das_tempo_nicht():
    """Setzt der Tracker Achtel zwischen die Viertel, darf das Tempo nicht steigen.

    Genau das ging schief: jeder zusätzliche Beat verschob alle folgenden
    Indizes, die Regression lief auf 126,9 BPM davon.
    """
    grid = beat_grid(124.0)
    period = 60.0 / 124.0
    extra = np.sort(np.concatenate([grid, grid[100:140] + period / 2]))
    assert estimate(extra) == pytest.approx(124.0, abs=0.6)


def test_fehlende_beats_verschieben_das_tempo_nicht():
    """Ein Breakdown ohne Drums lässt Beats ausfallen – das Tempo bleibt gleich."""
    grid = beat_grid(124.0)
    with_gap = np.concatenate([grid[:120], grid[145:]])
    assert estimate(with_gap) == pytest.approx(124.0, abs=0.6)


def test_konfidenz_faellt_bei_gestoertem_raster():
    """Ein gestörtes Raster darf nicht dieselbe Sicherheit melden wie ein sauberes."""
    grid = beat_grid(124.0)
    period = 60.0 / 124.0
    noisy = np.sort(np.concatenate([grid, grid[80:160] + period / 2]))
    _, clean_conf = analysis._tempo_from_beats(grid)
    _, noisy_conf = analysis._tempo_from_beats(noisy)
    assert noisy_conf < clean_conf


def test_wenige_beats_liefern_geringe_konfidenz():
    """18 Beats über ein paar Sekunden sind keine belastbare Grundlage.

    Ein Sprach-Reel mit etwas Hintergrundmusik darf kein selbstbewusstes
    Tempo melden.
    """
    _, conf = analysis._tempo_from_beats(beat_grid(86.0, seconds=13.0))
    assert conf < 0.4


def test_zu_wenige_beats_ergeben_kein_tempo():
    assert analysis._tempo_from_beats(np.array([0.0, 0.5, 1.0])) == (0.0, 0.0)


def test_leere_eingabe_stuerzt_nicht_ab():
    assert analysis._tempo_from_beats(np.array([])) == (0.0, 0.0)


def test_identische_zeitstempel_stuerzen_nicht_ab():
    assert analysis._tempo_from_beats(np.zeros(20)) == (0.0, 0.0)


@pytest.mark.parametrize("roh, erwartet", [
    (31.0, 62.0),     # unter 60 wird verdoppelt, bis es im Fenster liegt
    (248.0, 124.0),   # ab 200 wird halbiert
    (124.0, 124.0),   # dazwischen bleibt es stehen
    (62.0, 62.0),     # 62 liegt im Fenster, wird also nicht angefasst
])
def test_faltung_in_den_ueblichen_bereich(roh, erwartet):
    """Tempi werden in das Fenster 60 bis 200 BPM gefaltet."""
    assert analysis._fold_bpm(roh) == pytest.approx(erwartet, abs=0.01)


def test_snapping_nur_bei_hoher_konfidenz():
    """Produzierte Tracks laufen auf glatten DAW-Tempi – aber nur wenn wir sicher sind."""
    assert analysis._snap_bpm(123.98, 0.9) == 124.0
    assert analysis._snap_bpm(123.98, 0.2) == 123.98


def test_snapping_verbiegt_echte_werte_nicht():
    """Eine Live-Aufnahme mit 123,4 BPM darf nicht auf 123 gezogen werden."""
    assert analysis._snap_bpm(123.4, 0.9) == 123.4


def test_downbeats_ergeben_dasselbe_tempo():
    """Über die Takte gerechnet muss dasselbe herauskommen wie über die Beats."""
    grid = beat_grid(124.0)
    bars = grid[::4]
    bpm, _ = analysis._tempo_from_beats(bars, subdivision=4)
    assert bpm == pytest.approx(124.0, abs=0.3)


def test_rasterung_allein_verfaelscht_nicht():
    """Quantisierung auf 20 ms darf das Ergebnis nicht verschieben.

    Der Median der Abstände läge bei 125,0 BPM – die Regression muss besser sein.
    """
    exact = beat_grid(124.0, quantize=False)
    quantized = beat_grid(124.0, quantize=True)
    assert np.abs(np.diff(quantized) - 60.0 / 124.0).max() >= FRAME / 2 - 1e-9
    bpm_exact, _ = analysis._tempo_from_beats(exact)
    bpm_quant, _ = analysis._tempo_from_beats(quantized)
    assert bpm_quant == pytest.approx(bpm_exact, abs=0.05)
