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


def test_camelot_faerbt_alle_zwoelf_positionen(html):
    """Jede Position des Rads braucht ihren eigenen Farbton.

    Geprüft wird die Eigenschaft, nicht die Umsetzung: ob die Farbe aus einer
    Tabelle oder einer Formel kommt, ist gleichgültig – sie muss nur für alle
    zwölf Positionen verschieden sein.
    """
    formel = re.search(r"camHue\s*=\s*\(n\)\s*=>\s*(.+?);", html)
    tabelle = re.search(r"CAM_HUE\s*=\s*\{(.*?)\}", html, re.S)
    assert formel or tabelle, "keine Camelot-Farblogik gefunden"

    if tabelle:
        hues = {int(v) for _k, v in re.findall(r"(\d+):(\d+)", tabelle.group(1))}
    else:
        # Die Formel im Test nachvollziehen, um alle zwölf Werte zu bekommen.
        hues = {((n % 12) * 30 + 196) % 360 for n in range(1, 13)}
    assert len(hues) == 12
    assert all(0 <= h < 360 for h in hues)


def test_camelot_unterscheidet_dur_und_moll(html):
    """A (Moll) muss blasser aussehen als B (Dur), sonst verliert das Rad seinen Sinn."""
    assert re.search(r"mode\s*===\s*'B'", html) or "[data-cam$=\"A\"]" in html


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
    """Das Farbschema muss allein aus CSS kommen.

    Hängt es an `matchMedia`, frieren die Farben beim Umschalten zwischen Hell
    und Dunkel ein, bis die Karte neu gezeichnet wird. Ob das über
    `light-dark()` oder über Variablen in einer Media-Query gelöst ist, ist
    gleichgültig – beides reagiert von selbst.
    """
    assert "light-dark(" in html or "prefers-color-scheme" in html
    # matchMedia für Touch-Erkennung ist in Ordnung, fürs Theme nicht.
    assert "matchMedia('(prefers-color-scheme" not in html
    assert 'matchMedia("(prefers-color-scheme' not in html


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


# --------------------------------------------------------------------------- #
# Einstellungs-Lightbox
# --------------------------------------------------------------------------- #

def test_lightbox_steht_vor_dem_skript(html):
    """Das Markup muss im DOM sein, bevor die Handler gebunden werden.

    Stand die Lightbox am Ende des body, lieferte `$('set-lb')` beim
    Ausführen des Skripts null – Schließen per Rand-Klick und Escape waren
    dadurch wirkungslos, ohne dass ein Fehler auffiel.
    """
    assert html.index('id="set-lb"') < html.rindex("<script>")


def test_lightbox_laesst_sich_schliessen(html):
    """Drei Wege hinaus: Knopf, Rand-Klick und Escape."""
    assert "$('set-close').onclick" in html
    assert "$('set-lb').onclick" in html
    assert "ev.key === 'Escape'" in html


def test_alle_bereiche_haben_einen_inhalt(html):
    """Jeder Eintrag der Seitenleiste muss auch etwas anzeigen."""
    block = re.search(r"function bereiche\(\) \{(.*?)\n\}", html, re.S)
    assert block, "bereiche() nicht gefunden"
    eintraege = set(re.findall(r"\['(\w+)',", block.group(1)))
    gezeichnet = set(re.findall(r"SET\.bereich === '(\w+)'", html))
    assert eintraege, "keine Bereiche gefunden"
    assert eintraege == gezeichnet, f"ohne Inhalt: {eintraege - gezeichnet}"


def test_vorlagen_kommen_vom_server(html):
    """Die Oberfläche darf keine eigene Vorlagenliste führen.

    Geprüft wird, dass über `m.presets` iteriert wird statt über ein
    eingebautes Objekt – erwähnt werden dürfen die Programme natürlich, etwa
    in der Erklärung zum Export.
    """
    skript = html.split("<script>")[-1]
    assert "m.presets" in skript
    assert not re.search(r"(const|let|var)\s+\w*PRESETS?\s*=", skript)


def test_einstellungen_sind_auf_dem_telefon_bedienbar(html):
    """Bereiche als Leiste statt Seitenspalte, und antippbare Ziele."""
    assert "@media (max-width:760px)" in html
    assert ".set-nav button{" in html and "min-height:44px" in html


def test_vorschau_in_jedem_bereich(html):
    """Ohne die Vorschau müsste man raten, was eine Einstellung bewirkt."""
    assert "set-preview" in html
    assert "/api/settings/preview" in html


# --------------------------------------------------------------------------- #
# Modus: trennen oder nur analysieren
# --------------------------------------------------------------------------- #

def test_modus_schalter_hat_beide_wege(html):
    assert 'id="mode"' in html
    assert 'data-v="separate"' in html and 'data-v="analyze"' in html


def test_modell_und_format_verschwinden_beim_analysieren(html):
    """Beides spielt ohne Trennung keine Rolle – stehen zu lassen wäre irreführend."""
    assert "$('model-picker').hidden = analyse" in html
    assert "$('format').hidden = analyse" in html


def test_upload_schickt_den_modus_mit(html):
    assert "fd.append('mode', state.mode)" in html


def test_analyse_ohne_modell_moeglich(html):
    """Ohne Trennung wird nie ein Modell geladen – es darf also fehlen."""
    assert "state.mode === 'separate' && !state.selected" in html


def test_karte_ohne_stems_wird_aufgebaut(html):
    """Eine reine Analyse hat keine Stems, aber eine Aufnahme.

    Ohne diese Ausnahme blieb die Karte leer – kein Player, keine Knöpfe,
    kein Weg zum nachträglichen Trennen.
    """
    assert "const hatInhalt = d.files.length ||" in html


def test_knoepfe_ohne_stems_werden_ausgeblendet(html):
    """Loops oder Mix ohne Stems führen ins Leere."""
    assert "brauchtStems" in html
    assert "'Jetzt trennen'" in html


# --------------------------------------------------------------------------- #
# Unsicheres Tempo
# --------------------------------------------------------------------------- #

def test_unsicheres_tempo_wird_sichtbar_gemacht(html):
    """Ein Tempo mit niedriger Konfidenz darf nicht wie eine Messung aussehen.

    Bei einem Track der Stichprobe vom 2026-09-20 stand bpm_confidence auf
    0,01, während die Karte 122,09 BPM ohne jeden Hinweis anzeigte – der
    Median der Beat-Abstände sagte 120,00. Wer das ungeprüft ins Set
    übernimmt, merkt es erst beim Mischen.

    Die Tonart macht es seit jeher richtig (`unsure` bei key_confidence
    < 0.25); das Tempo zog nach.
    """
    assert re.search(r"bpmUnsicher\s*=\s*a\.bpm_confidence\s*<", html), \
        "Die Karte muss eine niedrige Tempo-Konfidenz auswerten"


def test_tempo_schwelle_passt_zum_snapping(html):
    """0.5 ist dieselbe Grenze, unter der analysis._snap_bpm das Tempo nicht
    mehr auf glatte Werte zieht – unterhalb traut der Code der Schätzung
    ohnehin nicht."""
    treffer = re.search(r"bpmUnsicher\s*=\s*a\.bpm_confidence\s*<\s*([\d.]+)", html)
    assert treffer and float(treffer.group(1)) == 0.5


def test_unsicheres_tempo_zeigt_die_alternative(html):
    """Halbes oder doppeltes Tempo ist der häufigste Fehler – steht die
    Alternative daneben, sieht man sofort, worum es geht."""
    # Der Ausdruck geht über mehrere Zeilen bis zum abschließenden Semikolon.
    stelle = re.search(r"bpm\.innerHTML\s*=\s*(.+?);\s*$", html, re.S | re.M)
    assert stelle, "BPM-Anzeige nicht gefunden"
    assert "bpmUnsicher" in stelle.group(1)
    assert "bpm_alt" in stelle.group(1)


def test_unsicheres_tempo_auch_in_der_bibliothek(html):
    """Die Tabelle zeigt dieselben Werte wie die Karte und braucht denselben
    Hinweis – sonst ist der Wert dort wieder blank."""
    zeile = re.search(r'<td class="num[^"]*">\$\{a\.bpm.*?</td>', html)
    assert zeile, "BPM-Spalte der Bibliothek nicht gefunden"
    assert "bpmUnsicher" in zeile.group(0), "Die Zelle muss den Hinweis auswerten"

    # Und der Wert muss aus der Konfidenz kommen, nicht fest verdrahtet sein.
    quelle = re.search(r"const bpmUnsicher\s*=\s*a\.bpm\s*&&\s*a\.bpm_confidence\s*<\s*([\d.]+)", html)
    assert quelle and float(quelle.group(1)) == 0.5


def test_unsicheres_tempo_hat_eine_erklaerung(html):
    """Ein Zeichen ohne Erklärung verwirrt mehr, als es hilft."""
    assert re.search(r"title\s*=.*[Uu]nsicher|unsicher.*title\s*=", html) or \
        re.search(r'bpm\.title\s*=.*bpmUnsicher', html), \
        "Der Hinweis braucht einen Tooltip, der ihn erklärt"
