"""Tonarterkennung auf synthetischen Kadenzen.

Die Tonart wird aus einem Chromagramm gegen Albrecht-Shanahan-Profile
korreliert. Testbar ist das mit erzeugten Dreiklängen: eine Gm-Cm-D#-Gm-Kadenz
muss g-Moll ergeben.
"""

import numpy as np
import pytest

import analysis
from conftest import SR, kick_track, progression


def key_of(y: np.ndarray) -> dict:
    return analysis._key(analysis.key_chroma(y).mean(axis=1))


def test_moll_kadenz(g_minor):
    """Die Kadenz des Referenztracks – Mixed In Key meldet dort Gm / 06A."""
    result = key_of(g_minor)
    assert result["key"] == "G minor"
    assert result["camelot"] == "6A"
    assert result["key_id3"] == "Gm"


def test_dur_kadenz(c_major):
    result = key_of(c_major)
    assert result["key"] == "C major"
    assert result["camelot"] == "8B"
    assert result["key_id3"] == "C"


def test_tonart_ueberlebt_eine_bassdrum(g_minor):
    """Mit Kick darf die Tonart nicht kippen.

    Das ist der Fehler, der am Referenztrack C-Dur statt g-Moll lieferte: der
    breitbandige Kick schmiert über alle zwölf Chroma-Bins und drückt das
    gemittelte Profil so flach, dass die Korrelation zwischen benachbarten
    Quinten praktisch würfelt.
    """
    mix = g_minor + kick_track(len(g_minor))
    mix = mix / np.abs(mix).max() * 0.9
    assert key_of(mix)["key"] == "G minor"


def test_ohne_bandfilter_kippt_die_tonart(g_minor):
    """Gegenprobe: ohne die Bandbegrenzung muss derselbe Mix falsch liegen.

    Ohne diesen Test wäre der vorige wertlos – er bliebe auch dann grün, wenn
    der Filter in `key_chroma` ersatzlos entfernt würde. Hier wird belegt,
    dass die Bandbegrenzung den Unterschied macht und nicht bloß mitläuft.
    """
    import librosa

    mix = g_minor + kick_track(len(g_minor))
    mix = mix / np.abs(mix).max() * 0.9
    voll = librosa.feature.chroma_cqt(y=mix, sr=SR, hop_length=2048)
    assert analysis._key(voll.mean(axis=1))["key"] != "G minor"


def test_key_chroma_ist_nicht_das_volle_chromagramm(g_minor):
    """`key_chroma` muss sich messbar vom ungefilterten Chromagramm unterscheiden.

    Ein direkter Pegeltest scheitert daran, dass `chroma_cqt` jeden Frame auf
    1,0 normiert und die Grundfrequenz auch nach der Dämpfung noch findet.
    Geprüft wird deshalb, dass überhaupt gefiltert wird – dass es das
    Richtige bewirkt, zeigt `test_ohne_bandfilter_kippt_die_tonart`.
    """
    import librosa

    mix = g_minor + kick_track(len(g_minor))
    mix = mix / np.abs(mix).max() * 0.9
    voll = librosa.feature.chroma_cqt(y=mix, sr=SR, hop_length=2048)
    gefiltert = analysis.key_chroma(mix)
    assert not np.allclose(voll, gefiltert)


def test_stille_ergibt_keine_tonart():
    assert analysis._key(np.zeros(12)) == {}


def test_konfidenz_vergleicht_verschiedene_grundtoene(g_minor):
    """Die Alternative muss ein anderer Grundton sein, nicht dieselbe Tonart.

    Verglichen wurde früher mit dem zweitbesten Ergebnis überhaupt – das ist
    meist die Parallele oder Variante mit fast gleichem Wert. Die Konfidenz
    war dadurch strukturell zu niedrig.
    """
    result = key_of(g_minor)
    tonic = result["key"].split()[0]
    assert not result["key_alt"].lower().startswith(tonic.lower())
    assert result["key_confidence"] > 0.0


def test_camelot_tabelle_ist_vollstaendig():
    """Alle 24 Tonarten brauchen einen Camelot-Code, und jeder kommt einmal vor."""
    assert len(analysis.CAMELOT) == 24
    assert len(set(analysis.CAMELOT.values())) == 24


@pytest.mark.parametrize("tonart, code", [
    (("G", "minor"), "6A"),
    (("C", "major"), "8B"),
    (("A", "minor"), "8A"),
    (("E", "major"), "12B"),
])
def test_camelot_einzelwerte(tonart, code):
    """Stichproben gegen die Standardtabelle des Camelot-Rads."""
    assert analysis.CAMELOT[tonart] == code


def test_id3_schreibweise():
    """Rekordbox und Traktor erwarten 'Gm' bzw. 'C', nicht 'G minor'."""
    assert analysis.id3_key("G", "minor") == "Gm"
    assert analysis.id3_key("C", "major") == "C"


# --------------------------------------------------------------------------- #
# Bandgrenzen des Tonart-Chromagramms
#
# Die Grenzen 150–1500 Hz stammen aus einem Benchmark über 495 veröffentlichte
# Tracks der Jahrgänge 2025 und 2026, gemessen gegen Mixed In Key und Beatport:
# Grundton 96,6 → 98,9 %, Übereinstimmung mit Beatport 68,8 → 72,9 %
# gegenüber 100–1000 Hz, in beiden Jahrgängen einzeln stabil.
# Die Tests prüfen die Grenzen selbst. Ein Pegeltest auf einem einzelnen Ton
# scheitert an der Normierung jedes Frames, darum stehen hier immer zwei Töne
# gleicher Lautstärke nebeneinander: einer mitten im Band, einer außerhalb
# der alten und innerhalb der neuen Grenzen (oder umgekehrt).
# --------------------------------------------------------------------------- #

NOTE = {name: i for i, name in enumerate(analysis.NOTE_NAMES)}


def _sinus_paar(f1: float, f2: float, seconds: float = 8.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.4 * np.sin(2 * np.pi * f1 * t) + 0.4 * np.sin(2 * np.pi * f2 * t)).astype("float32")


def test_tiefer_bereich_unter_150_hz_zaehlt_kaum():
    """A2 (110 Hz) liegt unter dem Band und muss deutlich hinter E5 (659 Hz) zurückfallen.

    Gemessen: mit der alten Untergrenze von 100 Hz das 1,77-Fache von E5,
    mit der neuen 0,13.
    """
    chroma = analysis.key_chroma(_sinus_paar(110.0, 659.26)).mean(axis=1)
    assert chroma[NOTE["A"]] < 0.5 * chroma[NOTE["E"]]


def test_bereich_bis_1500_hz_zaehlt_mit():
    """E6 (1319 Hz) liegt im Band und muss neben C5 (523 Hz) deutlich sichtbar bleiben.

    Gleichauf kommen die beiden nie: Auch mitten im Band erreicht ein zweiter,
    gleich lauter Ton in `chroma_cqt` nur etwa die Hälfte. Gemessen: mit der
    alten Obergrenze von 1000 Hz 0,05, mit der neuen 0,50.
    """
    chroma = analysis.key_chroma(_sinus_paar(523.25, 1318.51)).mean(axis=1)
    assert chroma[NOTE["E"]] > 0.25 * chroma[NOTE["C"]]
