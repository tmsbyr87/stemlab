"""Export für Rekordbox und Traktor.

Die Formate sind an einer echten Traktor-Sammlung (Pro 4) abgelesen: der
Pfadtrenner "/:", die Angabe der Cue-Zeiten in Millisekunden und die
Zahlenkodierung der Tonart. Was hier falsch ist, merkt man erst, wenn die
Software den Import stillschweigend verweigert.
"""

import xml.etree.ElementTree as ET

import numpy as np
import pytest
import soundfile as sf

import postprocess
from conftest import SR, beat_grid

ANALYSE = {
    "bpm": 124.0, "camelot": "6A", "key_tonic": "G", "key_mode": "minor",
    "energy": 5, "seconds_analyzed": 240.0,
}

CUES = [
    {"index": 0, "bar": 0, "time": 0.14, "name": "Intro", "color": (48, 225, 235)},
    {"index": 1, "bar": 16, "time": 31.18, "name": "Drop", "color": (225, 142, 45)},
]


@pytest.fixture
def ordner(tmp_path):
    for name in ("vocals.wav", "drums.wav", "original.wav"):
        sf.write(tmp_path / name, np.zeros(SR), SR)
    return tmp_path


# --- Tonart-Kodierung ------------------------------------------------------

@pytest.mark.parametrize("tonic, mode, erwartet", [
    ("C", "major", 0),    # an der echten Sammlung überprüft
    ("D", "major", 2),
    ("F", "major", 5),
    ("C", "minor", 12),
    ("E", "minor", 16),
    ("G", "minor", 19),
])
def test_traktor_tonart_kodierung(tonic, mode, erwartet):
    """Chroma-Index, für Moll plus zwölf."""
    assert postprocess._traktor_key(tonic, mode) == erwartet


def test_unbekannte_tonart_ergibt_nichts():
    assert postprocess._traktor_key("", "") is None


def test_tonart_faellt_auf_camelot_zurueck():
    """Ältere Analysen haben kein key_tonic – der Camelot-Code rettet das."""
    tonic, mode = postprocess._tonart_aus_analyse({"camelot": "6A"})
    assert (tonic, mode) == ("G", "minor")


# --- Rekordbox -------------------------------------------------------------

def test_rekordbox_ist_gueltiges_xml(ordner):
    ziel = postprocess.export_rekordbox(ordner, ANALYSE, CUES, "Testsong", postprocess.settings({}))
    wurzel = ET.parse(ziel).getroot()
    assert wurzel.tag == "DJ_PLAYLISTS"
    assert len(wurzel.findall(".//TRACK[@Location]")) == 3


def test_rekordbox_pfade_sind_aufloesbar(tmp_path):
    """Leerzeichen und Klammern im Ordnernamen dürfen den Pfad nicht zerstören.

    Genau daran scheitern solche Exporte in der Praxis – "(Remix)" und
    Leerzeichen müssen in der file://-URL kodiert sein.
    """
    import urllib.parse

    ordner = tmp_path / "AM I RIGHT - We Are (Remix)"
    ordner.mkdir()
    sf.write(ordner / "vocals.wav", np.zeros(SR), SR)
    ziel = postprocess.export_rekordbox(ordner, ANALYSE, CUES, "Test", postprocess.settings({}))
    ort = ET.parse(ziel).getroot().find(".//TRACK[@Location]").get("Location")
    assert "%28" in ort, "Klammern müssen kodiert sein"
    from pathlib import Path
    assert Path(urllib.parse.unquote(ort.replace("file://localhost", ""))).exists()


def test_rekordbox_enthaelt_cues(ordner):
    ziel = postprocess.export_rekordbox(ordner, ANALYSE, CUES, "Testsong", postprocess.settings({}))
    marken = ET.parse(ziel).getroot().findall(".//POSITION_MARK")
    assert len(marken) == len(CUES) * 3
    assert marken[0].get("Name") == "Intro"


def test_rekordbox_ohne_cues(ordner):
    ziel = postprocess.export_rekordbox(ordner, ANALYSE, [], "Testsong", postprocess.settings({}))
    assert ET.parse(ziel).getroot().findall(".//POSITION_MARK") == []


# --- Traktor ---------------------------------------------------------------

def test_traktor_ist_gueltiges_xml(ordner):
    ziel = postprocess.export_traktor(ordner, ANALYSE, CUES, "Testsong", postprocess.settings({}))
    wurzel = ET.parse(ziel).getroot()
    assert wurzel.tag == "NML"
    assert wurzel.find("COLLECTION").get("ENTRIES") == "3"


def test_traktor_pfadtrenner(ordner):
    """Traktor trennt Verzeichnisse mit "/:" und führt den Datenträger extra."""
    ziel = postprocess.export_traktor(ordner, ANALYSE, CUES, "Testsong", postprocess.settings({}))
    ort = ET.parse(ziel).getroot().find(".//LOCATION")
    assert ort.get("DIR").startswith("/:")
    assert "/:" in ort.get("DIR")
    assert ort.get("VOLUME") == "Macintosh HD"


def test_traktor_cue_zeiten_in_millisekunden(ordner):
    """START ist in Millisekunden – in Sekunden läge jeder Cue am Anfang."""
    ziel = postprocess.export_traktor(ordner, ANALYSE, CUES, "Testsong", postprocess.settings({}))
    hotcues = [c for c in ET.parse(ziel).getroot().findall(".//CUE_V2") if c.get("TYPE") == "0"]
    assert float(hotcues[0].get("START")) == pytest.approx(140.0, abs=0.1)


def test_traktor_hat_ein_beatgrid(ordner):
    ziel = postprocess.export_traktor(ordner, ANALYSE, CUES, "Testsong", postprocess.settings({}))
    raster = [c for c in ET.parse(ziel).getroot().findall(".//CUE_V2") if c.get("TYPE") == "4"]
    assert len(raster) == 3


def test_traktor_playlist_ist_gefuellt(ordner):
    """Ohne Playlist zeigt Traktor die importierte Datei als leer."""
    ziel = postprocess.export_traktor(ordner, ANALYSE, CUES, "Testsong", postprocess.settings({}))
    wurzel = ET.parse(ziel).getroot()
    assert wurzel.find(".//PLAYLIST").get("ENTRIES") == "3"
    assert wurzel.find(".//PRIMARYKEY").get("KEY").startswith("Macintosh HD/:")


def test_traktor_tonart_als_zahl(ordner):
    ziel = postprocess.export_traktor(ordner, ANALYSE, CUES, "Testsong", postprocess.settings({}))
    assert ET.parse(ziel).getroot().find(".//MUSICAL_KEY").get("VALUE") == "19"


def test_sonderzeichen_werden_maskiert(ordner):
    """Ein Ampersand im Titel darf die Datei nicht unlesbar machen."""
    ziel = postprocess.export_traktor(ordner, ANALYSE, CUES, "Rock & Roll <Remix>", postprocess.settings({}))
    ET.parse(ziel)  # wirft, wenn die Maskierung fehlt
    assert "&amp;" in ziel.read_text()


# --- Cue-Erkennung ---------------------------------------------------------

def test_cues_liegen_auf_taktgrenzen(tmp_path):
    """Ein Cue zwischen zwei Takten wäre beim Mischen unbrauchbar."""
    pfad = tmp_path / "original.wav"
    laut = np.tile(np.sin(2 * np.pi * 110 * np.arange(SR * 4) / SR), 30) * 0.9
    leise = laut * 0.05
    y = np.concatenate([leise[:SR * 40], laut[:SR * 40], leise[:SR * 40]])
    sf.write(pfad, y, SR)
    downbeats = list(beat_grid(124.0, seconds=120.0))[::4]
    cues = postprocess.detect_cues(pfad, downbeats, 120.0)
    assert cues, "keine Cues gefunden"
    bar = np.median(np.diff(downbeats))
    for cue in cues:
        takte = (cue["time"] - downbeats[0]) / bar
        assert abs(takte - round(takte)) < 0.01


def test_cues_haben_hoechstens_acht(tmp_path):
    """CDJs und Traktor haben acht Hotcue-Plätze."""
    pfad = tmp_path / "original.wav"
    teile = [np.sin(2 * np.pi * 110 * np.arange(SR * 8) / SR) * (0.9 if i % 2 else 0.05)
             for i in range(20)]
    sf.write(pfad, np.concatenate(teile), SR)
    cues = postprocess.detect_cues(pfad, list(beat_grid(124.0, seconds=160.0))[::4], 160.0)
    assert len(cues) <= postprocess.MAX_CUES


def test_ohne_downbeats_keine_cues(tmp_path):
    pfad = tmp_path / "original.wav"
    sf.write(pfad, np.zeros(SR * 10), SR)
    assert postprocess.detect_cues(pfad, [], 10.0) == []


def test_fehlende_datei_stuerzt_nicht_ab(tmp_path):
    assert postprocess.detect_cues(tmp_path / "gibtesnicht.wav",
                                   list(beat_grid(124.0, seconds=120.0))[::4], 120.0) == []
