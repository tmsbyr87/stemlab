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
# Eine Auswahl statt eines Stapels
# --------------------------------------------------------------------------- #
#
# Wer sechs Dateien zur Analyse gibt, bekam sechs Karten untereinander – und
# nur die oberste hatte einen Player. Die Analysen gehören in die Liste, aus
# der man eine auswählt; die geöffnete Karte trägt dann Player und Optionen.

def test_fertige_analyse_bleibt_nicht_als_karte_stehen(html):
    """Nach dem Analysieren wandert der Track in die Liste, statt die
    Oberfläche mit einer weiteren Karte zu verlängern."""
    assert re.search(r"function raeumeAnalyseKarte", html), "Funktion fehlt"
    # Sie muss auch gerufen werden – eine tote Funktion räumt nichts.
    assert re.search(r"if \(d\.kind === 'analyze'\) raeumeAnalyseKarte", html), \
        "raeumeAnalyseKarte muss für fertige Analysen aufgerufen werden"


def test_analyse_karte_bekommt_denselben_aufbau_wie_die_bibliothek(html):
    """Eine frische Analyse und dieselbe aus der Liste geklickt dürfen sich
    nicht unterscheiden. Bisher landete die frische in buildActionResult
    (ohne Player), die geklickte in buildResult (mit Player)."""
    treffer = re.search(r"d\.kind === 'separate' \|\| d\.kind === 'library'([^\n]*)", html)
    assert treffer, "Verzweigung zwischen den beiden Aufbauten nicht gefunden"
    assert "analyze" in treffer.group(0), \
        "Auch 'analyze' muss den vollen Aufbau mit Player bekommen"


def test_liste_kennt_einen_filter_fuer_diese_sitzung(html):
    """„Gerade analysiert" beantwortet die häufigste Frage nach einem Lauf:
    Was habe ich eben hinzugefügt?"""
    assert re.search(r'id="ana-filter-neu"', html), "Der Knopf fehlt"
    assert re.search(r"state\.nurDieseSitzung = !state\.nurDieseSitzung", html), \
        "Der Knopf muss den Filter umschalten"
    # Und der Filter muss in der Auswahl der Zeilen ankommen.
    assert re.search(r"!state\.nurDieseSitzung \|\| state\.neueDieseSitzung\.has", html), \
        "visibleAnalyses muss den Filter auswerten"


def test_nur_eine_karte_ist_gleichzeitig_offen(html):
    """Zwei offene Player nebeneinander stiften Verwirrung – und spielen
    womöglich gleichzeitig."""
    assert re.search(r"function schliesseAndereKarten", html), "Funktion fehlt"
    # Entscheidend ist der Aufruf beim Öffnen aus der Liste.
    assert re.search(r"schliesseAndereKarten\(karte\)", html), \
        "Beim Öffnen einer Karte müssen die anderen weichen"


def test_tanzbarkeit_stimmt_mit_dem_analysemodul_ueberein(html):
    """Die Formel steht zweimal: in analysis.py und in der Oberfläche.

    Das ist Absicht – server.py importiert das Analysemodul nicht, weil es
    librosa und PyTorch in den Serverstart zöge. Ältere Analysen ohne das
    Feld rechnet die Oberfläche deshalb selbst nach. Laufen die beiden
    Fassungen auseinander, zeigt die Liste andere Werte als die Karte.
    """
    import analysis

    js = re.search(r"function tanzbarkeit\(.*?\n\}", html, re.S)
    assert js, "Die Oberfläche braucht die Nachberechnung"
    quelltext = js.group(0)

    # Die Eckwerte der Formel müssen auf beiden Seiten dieselben sein.
    for zahl in ("110", "135", "60", "50", "55", "0.35", "0.65", "0.4", "0.6"):
        assert zahl in quelltext, f"Eckwert {zahl} fehlt in der Oberfläche"

    # Und die Grenzfälle müssen dasselbe ergeben.
    assert analysis.danceability(0, 8, 0.9) == 1
    assert analysis.danceability(124, 8, 0.9) >= 7


# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #

def test_dunkle_palette_gilt_auch_ohne_systemvorgabe(html):
    """Beide Paletten waren schon da, aber nur hinter prefers-color-scheme.
    Wer dunkel will, obwohl das System hell steht, kam nicht heran."""
    assert re.search(r':root\[data-theme="dark"\]', html), \
        "Die dunkle Palette braucht einen Selektor ohne Media Query"
    assert re.search(r':root\[data-theme="light"\]', html), \
        "Und hell muss sich gegen ein dunkles System durchsetzen können"


def test_theme_wird_gemerkt(html):
    """Sonst steht nach jedem Start wieder die Systemvorgabe."""
    assert re.search(r"localStorage\.getItem\('stemlab\.theme'\)", html), "Lesen fehlt"
    assert re.search(r"localStorage\.setItem\('stemlab\.theme',", html), "Schreiben fehlt"
    # Und beim Start muss der gemerkte Wert tatsächlich angewandt werden.
    assert re.search(r"themeSetzen\(themeLesen\(\)\)", html), \
        "Der gemerkte Wert muss beim Start greifen"


def test_theme_kennt_die_systemvorgabe_als_eigene_wahl(html):
    """Drei Zustände, nicht zwei: Wer 'automatisch' wählt, folgt dem Mac."""
    liste = re.search(r"const THEMES = \[(.*?)\];", html, re.S)
    assert liste, "Die Zustandsliste fehlt"
    for wert in ("'auto'", "'light'", "'dark'"):
        assert wert in liste.group(1), f"Zustand {wert} fehlt"
    # 'auto' entfernt das Attribut, statt einen eigenen Wert zu setzen –
    # nur so greift wieder die Systemvorgabe.
    assert re.search(r"removeAttribute\('data-theme'\)", html)


def test_theme_umschalter_ist_bedienbar(html):
    """Ein Knopf, der die drei Zustände durchläuft, mit erkennbarem Ziel."""
    assert re.search(r'id="theme-btn"', html)
    assert re.search(r"\$\('theme-btn'\)\.onclick", html), "Der Knopf braucht einen Klick-Handler"
    # Er muss die Zustände der Reihe nach durchlaufen.
    assert re.search(r"THEMES\[\(i \+ 1\) % THEMES\.length\]", html), \
        "Der Knopf muss zum nächsten Zustand weiterschalten"


# --------------------------------------------------------------------------- #
# Dreispaltiges Grundgerüst
# --------------------------------------------------------------------------- #

def test_layout_hat_drei_spalten(html):
    """Links Navigation, Mitte Arbeitsfläche, rechts der gewählte Track.
    Auf einem breiten Bildschirm ist eine 1060-px-Spalte Verschwendung."""
    assert re.search(r"\.shell\{[^}]*display:grid", html), "Grundgerüst fehlt"
    assert re.search(r'id="seitenleiste"', html)
    assert re.search(r'id="mitte"', html)
    assert re.search(r'id="infospalte"', html)


def test_layout_faellt_auf_schmalen_geraeten_zusammen(html):
    """Drei Spalten auf dem Telefon wären drei unlesbare Streifen."""
    assert re.search(r"@media \(max-width:\s*11\d\dpx\)\{[^@]*\.shell\{", html, re.S), \
        "Es braucht einen Haltepunkt, an dem die Spalten zusammenfallen"


def test_seitenleiste_zeigt_nur_erreichbare_ziele(html):
    """Ein Menüpunkt ohne Ziel ist schlimmer als keiner. Jeder Eintrag
    muss auf einen Bereich zeigen, den es gibt."""
    # Die Einträge entstehen im Skript, nicht im Markup – geprüft wird
    # deshalb die Liste, aus der sie gebaut werden.
    liste = re.search(r"const BEREICHE = \[(.*?)\];", html, re.S)
    assert liste, "Die Bereichsliste fehlt"
    ziele = re.findall(r"ziel: '([a-z-]+)'", liste.group(1))
    assert ziele, "Keine Navigationsziele"
    for ziel in ziele:
        assert re.search(rf'id="{ziel}"', html), f"Ziel {ziel} existiert nicht im Markup"

    # Und die drei Bereiche, um die es geht, müssen erreichbar sein.
    for pflicht in ("bereich-neu", "bereich-analysen", "jobs-card"):
        assert pflicht in ziele, f"{pflicht} fehlt in der Seitenleiste"


def test_bestehende_bereiche_behalten_ihre_ids(html):
    """Der Umbau ordnet um, er benennt nicht um – sonst brechen die
    übrigen Prüfungen und der Zustand der Oberfläche."""
    for id_ in ("jobs", "ana-body", "ana-table", "wheel", "ana-search", "models"):
        assert re.search(rf'id="{id_}"', html), f"{id_} verschwunden"


def test_infospalte_beansprucht_leer_keinen_platz(html):
    """Bevor ein Track gewählt ist, hat die rechte Spalte nichts zu zeigen.
    Eine leere 300-px-Spalte würde die Arbeitsfläche grundlos schmaler
    machen."""
    assert re.search(r"\.shell:not\(\.hat-auswahl\)\{grid-template-columns:200px minmax\(0,1fr\)\}", html)
    assert re.search(r"\.shell:not\(\.hat-auswahl\) \.infospalte\{display:none\}", html)


# --------------------------------------------------------------------------- #
# Rechte Spalte: Analyse, Tags, Export
# --------------------------------------------------------------------------- #

def test_infospalte_hat_drei_reiter(html):
    """Alles zum gewählten Track an einer Stelle, statt in Aufklappern
    verteilt über die Karte."""
    liste = re.search(r"const REITER = \[(.*?)\];", html, re.S)
    assert liste, "Die Reiterliste fehlt"
    for schluessel in ("'analyse'", "'tags'", "'export'"):
        assert schluessel in liste.group(1), f"Reiter {schluessel} fehlt"


def test_infospalte_zeigt_die_analysewerte(html):
    """Tonart, Tempo, Takte, Energie und Tanzbarkeit – die Werte, die ein
    DJ vor dem Auflegen braucht."""
    fn = re.search(r"function reiterAnalyse\(.*?\n\}", html, re.S)
    assert fn, "Der Analyse-Reiter fehlt"
    # Und er muss auch gerufen werden – eine tote Funktion zeigt nichts.
    assert re.search(r"state\.infoReiter === 'analyse'\) reiterAnalyse\(", html), \
        "reiterAnalyse muss für den Analyse-Reiter aufgerufen werden"
    quelle = fn.group(0)
    for feld in ("camelot", "bpm", "key_de", "energy"):
        assert feld in quelle, f"{feld} wird nicht angezeigt"
    # Die Tanzbarkeit kommt aus dem Feld oder wird nachgerechnet.
    assert "danceability" in quelle and "tanzbarkeit(" in quelle


def test_infospalte_zeigt_energie_und_tanzbarkeit_als_balken(html):
    """Zwei Zahlen auf derselben Skala 1–10 lassen sich als Balken
    vergleichen, als Ziffern nicht."""
    assert re.search(r"\.balken\{", html), "Die Balken-Darstellung fehlt"


def test_infospalte_wird_nur_mit_auswahl_gezeigt(html):
    """Ohne gewählten Track hat die Spalte nichts zu sagen."""
    assert re.search(r"classList\.toggle\('hat-auswahl', !!d\)", html), \
        "Die Spalte darf nur erscheinen, wenn tatsächlich ein Track gewählt ist"


def test_tags_reiter_kennt_dieselben_felder_wie_der_server(html):
    """Wie schon für die Lightbox geprüft: Die Feldliste darf nicht
    auseinanderlaufen."""
    import postprocess

    liste = re.search(r"const TAG_LABELS = \[(.*?)\];", html, re.S)
    assert liste, "Die Feldliste fehlt"
    felder = set(re.findall(r"\['([a-z]+)',", liste.group(1)))
    assert felder == set(postprocess.TAG_FIELDS), \
        f"Oberfläche und Server sind uneins: {felder ^ set(postprocess.TAG_FIELDS)}"
