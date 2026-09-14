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
