"""Einstellungen für Tags, Tonart, Tempo und Dateinamen.

Was hier schiefgeht, landet still in den Dateien des Nutzers: ein falsch
zusammengesetzter Kommentar fällt erst auf, wenn Rekordbox ihn anzeigt.
"""

import pytest

import postprocess

BEISPIEL = {"camelot": "6A", "key_tonic": "G", "key_mode": "minor",
            "bpm": 124.0, "energy": 5}


def opts(**kwargs):
    return {**postprocess.DEFAULT_SETTINGS, **kwargs}


@pytest.mark.parametrize("notation, erwartet", [
    ("camelot", "06A"),
    ("standard", "Gm"),
    ("sharps", "Gm"),
    ("flats", "Gm"),
])
def test_schreibweisen_g_moll(notation, erwartet):
    assert postprocess.format_key("6A", "G", "minor", opts(key_notation=notation)) == erwartet


@pytest.mark.parametrize("notation, erwartet", [
    ("standard", "A#m"),
    ("sharps", "A#m"),
    ("flats", "Bbm"),
])
def test_schreibweisen_mit_vorzeichen(notation, erwartet):
    """Nur hier unterscheiden sich Kreuze und B-Vorzeichen."""
    assert postprocess.format_key("3A", "A#", "minor", opts(key_notation=notation)) == erwartet


def test_fuehrende_null():
    assert postprocess.format_key("6A", "G", "minor", opts(key_leading_zero=True)) == "06A"
    assert postprocess.format_key("6A", "G", "minor", opts(key_leading_zero=False)) == "6A"


def test_zweistellige_camelot_bekommen_keine_null():
    assert postprocess.format_key("12B", "E", "major", opts(key_leading_zero=True)) == "12B"


def test_dur_ohne_m():
    assert postprocess.format_key("8B", "C", "major", opts(key_notation="standard")) == "C"


def test_unbekannte_tonart_bleibt_leer():
    assert postprocess.format_key("–", "", "", opts()) == ""


@pytest.mark.parametrize("muster, erwartet", [
    ("key", "06A"),
    ("energy", "5"),
    ("key_energy", "06A - 5"),
    ("key_tempo", "06A - 124"),
    ("key_tempo_energy", "06A - 124 - 5"),
    ("tempo_key_energy", "124 - 06A - 5"),
])
def test_tag_muster(muster, erwartet):
    assert postprocess.build_tag_text(BEISPIEL, opts(tag_pattern=muster)) == erwartet


def test_fehlende_werte_lassen_keine_luecken():
    """Ohne Energie darf kein '06A - 124 - ' mit offenem Ende entstehen."""
    ohne = {**BEISPIEL, "energy": 0}
    assert postprocess.build_tag_text(ohne, opts(tag_pattern="key_tempo_energy")) == "06A - 124"


@pytest.mark.parametrize("stellen, erwartet", [(0, "124"), (1, "124.0"), (2, "124.00")])
def test_tempo_nachkommastellen(stellen, erwartet):
    assert postprocess.format_tempo(124.0, opts(tempo_decimals=stellen)) == erwartet


def test_tempo_wird_in_den_bereich_gefaltet():
    """Halbes und doppeltes Tempo landen im eingestellten Fenster."""
    o = opts(tempo_min=70, tempo_max=190)
    assert postprocess.format_tempo(62.0, o) == "124"
    assert postprocess.format_tempo(248.0, o) == "124"


def test_tempo_ohne_bereich_bleibt_wie_es_ist():
    assert postprocess.format_tempo(62.0, opts(tempo_min=0, tempo_max=0)) == "62"


def test_kein_tempo_ergibt_leer():
    assert postprocess.format_tempo(0, opts()) == ""


@pytest.mark.parametrize("muster, erwartet", [
    ("", "vocals"),
    ("{name} - {key}", "vocals - 06A"),
    ("{name} - {key} - {tempo}", "vocals - 06A - 124bpm"),
    ("{key} - {name}", "06A - vocals"),
    ("{key} - {tempo} - {name}", "06A - 124bpm - vocals"),
])
def test_dateinamen_muster(muster, erwartet):
    assert postprocess.build_filename("vocals", BEISPIEL, opts(rename_pattern=muster)) == erwartet


def test_dateiname_ohne_tonart_hat_keine_doppelten_trenner():
    """Fehlt ein Feld, darf kein 'vocals -  - 124bpm' herauskommen."""
    ohne = {**BEISPIEL, "camelot": "–", "key_tonic": ""}
    name = postprocess.build_filename("vocals", ohne, opts(rename_pattern="{name} - {key} - {tempo}"))
    assert " -  - " not in name
    assert name == "vocals - 124bpm"


@pytest.mark.parametrize("roh", ["a/b", "a:b", "a*b", 'a"b', "a?b", "a<b>c", "a|b"])
def test_dateinamen_ohne_verbotene_zeichen(roh):
    """Ein Songtitel mit Schrägstrich darf keinen Unterordner erzeugen."""
    sauber = postprocess.sanitize_filename(roh)
    assert not set(sauber) & set('/\\:*?"<>|')


def test_vorlagen_sind_vollstaendig():
    """Jede Vorlage braucht Beschriftung, Erklärung und gültige Werte."""
    for key, preset in postprocess.PRESETS.items():
        assert preset["label"] and preset["hint"], key
        for feld, wert in preset["settings"].items():
            assert feld in postprocess.DEFAULT_SETTINGS, f"{key}: {feld}"
            if feld == "tag_pattern":
                assert wert in postprocess.TAG_PATTERNS
            if feld == "tag_target":
                assert wert in postprocess.TAG_TARGETS
            if feld == "key_notation":
                assert wert in postprocess.KEY_NOTATIONS


def test_einstellungen_fuellen_standardwerte_auf():
    s = postprocess.settings({"tagging": {"tag_pattern": "key"}})
    assert s["tag_pattern"] == "key"
    assert s["tempo_min"] == postprocess.DEFAULT_SETTINGS["tempo_min"]


def test_unbekannte_schluessel_werden_ignoriert():
    s = postprocess.settings({"tagging": {"gibtesnicht": 1}})
    assert "gibtesnicht" not in s
