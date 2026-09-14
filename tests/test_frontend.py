"""Prüfungen an der Oberfläche, die ohne Browser auskommen.

`static/index.html` ist eine einzelne Datei mit eingebettetem CSS und
JavaScript. Ein voller Browsertest wäre hier unverhältnismäßig; geprüft wird
deshalb, dass die Stellen vorhanden und konsistent sind, an denen schon
einmal etwas kaputtging.
"""

import re
from pathlib import Path

import pytest

INDEX = Path(__file__).resolve().parent.parent / "static" / "index.html"


@pytest.fixture(scope="module")
def html() -> str:
    return INDEX.read_text()


def test_camelot_hat_alle_zwoelf_positionen(html):
    block = re.search(r"const CAM_HUE = \{(.*?)\};", html, re.S)
    assert block, "CAM_HUE fehlt"
    hues = dict(re.findall(r"(\d+):(\d+)", block.group(1)))
    assert sorted(int(k) for k in hues) == list(range(1, 13))


def test_camelot_farbtoene_sind_gueltig(html):
    block = re.search(r"const CAM_HUE = \{(.*?)\};", html, re.S).group(1)
    for _pos, hue in re.findall(r"(\d+):(\d+)", block):
        assert 0 <= int(hue) < 360


def test_cover_url_ist_in_anfuehrungszeichen(html):
    """Ohne Anführungszeichen zerbricht die URL an Klammern im Ordnernamen.

    encodeURIComponent lässt Klammern stehen; ein Ordner wie
    "Titel (Remix)" schließt sonst das CSS-url(...) vorzeitig und das Cover
    erscheint nie.
    """
    match = re.search(r"art\.style\.backgroundImage\s*=\s*`([^`]+)`", html)
    assert match, "Zuweisung des Coverbilds nicht gefunden"
    assert match.group(1).startswith('url("')


def test_theme_wechsel_laeuft_ueber_css(html):
    """Die Chipfarben dürfen nicht per matchMedia in JavaScript entstehen.

    Sonst bleiben sie beim Umschalten zwischen Hell und Dunkel auf dem alten
    Stand, bis die Karte neu gezeichnet wird.
    """
    assert "light-dark(" in html
    assert "color-scheme:light dark" in html
    assert "matchMedia" not in html


def test_tag_felder_stimmen_mit_dem_server_ueberein(html):
    """Jedes Feld der Oberfläche muss der Server auch kennen."""
    import postprocess

    block = re.search(r"const TAG_LABELS = \[(.*?)\];", html, re.S)
    assert block, "TAG_LABELS fehlt"
    keys = {k for k, _label in re.findall(r"\['(\w+)','([^']+)'\]", block.group(1))}
    assert keys == set(postprocess.TAG_FIELDS)


def test_keine_doppelten_element_ids(html):
    ids = re.findall(r'\sid="([^"]+)"', html)
    assert len(ids) == len(set(ids))
