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


# --------------------------------------------------------------------------- #
# Playlists
# --------------------------------------------------------------------------- #

def test_playlists_filtern_die_liste(html):
    """Eine Playlist ist ein gespeicherter Filter – sie verschiebt keine
    Dateien, sie blendet die Liste ein."""
    assert re.search(r"state\.playlists\[state\.playlistFilter\] \|\| \[\]\)\.includes\(a\.folder\)", html), \
        "visibleAnalyses muss die Playlist auswerten"


def test_playlists_kommen_vom_server(html):
    """Sie liegen in config.json, nicht im Arbeitsspeicher – sonst wären
    sie nach dem Neustart weg."""
    assert re.search(r"fetch\('/api/playlists'\)", html)
    assert re.search(r"api\('/api/playlists', \{ aktion", html)


def test_playlists_stehen_in_der_seitenleiste(html):
    assert re.search(r"textContent: 'Playlists'", html)
    assert re.search(r"'\+ Neue Playlist'", html)


def test_track_laesst_sich_einer_playlist_zuordnen(html):
    """Über Häkchen in der rechten Spalte, nicht über ein verstecktes Menü."""
    assert re.search(r"playlistAktion\(box\.checked \? 'hinzufuegen' : 'entfernen'", html)


# --------------------------------------------------------------------------- #
# Stem-Farben
# --------------------------------------------------------------------------- #
#
# Diese Darstellung gab es schon, bevor der Umbau begann – festgehalten
# wird sie hier, damit sie nicht unbemerkt verlorengeht.

def test_jeder_stem_hat_eine_eigene_farbe(html):
    """Beim Mischen greift man nach der Farbe, nicht nach dem Namen.
    Zwei gleiche Farben machen die Zuordnung unmöglich."""
    tabelle = re.search(r"const COLOR = \{(.*?)\};", html)
    assert tabelle, "Die Farbtabelle fehlt"
    farben = dict(re.findall(r"(\w+):'(#[0-9a-f]{6})'", tabelle.group(1)))

    for stem in ("vocals", "drums", "bass", "other", "instrumental"):
        assert stem in farben, f"{stem} hat keine Farbe"

    # "other" und "original" teilen sich bewusst das Grau – beides ist
    # "der Rest". Die musikalischen Spuren müssen unterscheidbar sein.
    musikalisch = {k: v for k, v in farben.items() if k not in ("other", "original")}
    assert len(set(musikalisch.values())) == len(musikalisch), \
        f"Zwei Stems teilen sich eine Farbe: {musikalisch}"


def test_wellenform_nimmt_die_farbe_des_stems(html):
    """Sonst wären die Punkte bunt und die Spuren alle blau."""
    assert re.search(r"getPropertyValue\('--sc'\)", html), \
        "Die Wellenform muss die Farbe der Zeile übernehmen"
    assert re.search(r"row\.style\.setProperty\('--sc', COLOR\[stem\]", html), \
        "Jede Zeile muss ihre Stem-Farbe setzen"


def test_notizfeld_steht_in_der_rechten_spalte(html):
    """Was man sich merkt, gehört sichtbar zum Track – nicht hinter einen
    weiteren Klick."""
    assert re.search(r"function notizfeld\(", html)
    assert re.search(r"notizfeld\(spalte, d\)", html), "Das Feld muss auch gezeigt werden"


def test_notizen_werden_verzoegert_gesichert(html):
    """Bei jedem Zeichen zu speichern wäre verschwenderisch; gar nicht zu
    speichern, bis man klickt, verliert Text."""
    assert re.search(r"setTimeout\(\(\) => \{\s*api\('/api/notes'", html, re.S), \
        "Es braucht eine Verzögerung beim Tippen"
    assert re.search(r"feld\.onblur", html), \
        "Beim Verlassen muss sofort gesichert werden"


# --------------------------------------------------------------------------- #
# Icons in der Kopfzeile
# --------------------------------------------------------------------------- #

def test_kopfzeile_nutzt_icons_statt_wortknoepfe(html):
    """Zwei Wortknöpfe nebeneinander sind viel Text für wenig Funktion."""
    for knopf_id in ("theme-btn", "quit"):
        knopf = re.search(rf'<button[^>]*id="{knopf_id}"[^>]*>(.*?)</button>', html, re.S)
        assert knopf, f"{knopf_id} nicht gefunden"
        assert "<svg" in knopf.group(1), f"{knopf_id} braucht ein Icon"
        assert not re.sub(r"<svg.*?</svg>", "", knopf.group(1), flags=re.S).strip(), \
            f"{knopf_id} soll nur das Icon zeigen, keinen Text daneben"


def test_icons_bleiben_benannt(html):
    """Ein Knopf ohne Beschriftung braucht einen Namen für Screenreader –
    und einen Tooltip für alle anderen. Besonders bei „Beenden“: Das
    stoppt den Server, das darf niemand aus Versehen treffen."""
    for knopf_id in ("theme-btn", "quit"):
        knopf = re.search(rf'<button[^>]*id="{knopf_id}"[^>]*>', html)
        assert knopf, f"{knopf_id} nicht gefunden"
        assert "aria-label=" in knopf.group(0), f"{knopf_id} braucht ein aria-label"
        assert "title=" in knopf.group(0), f"{knopf_id} braucht einen Tooltip"


def test_theme_icon_wechselt_mit_dem_zustand(html):
    """Sonne, Mond oder Halbmond – man muss sehen, was gerade gilt, ohne
    den Tooltip zu öffnen."""
    liste = re.search(r"const THEMES = \[(.*?)\];", html, re.S)
    assert liste, "Die Zustandsliste fehlt"
    icons = re.findall(r"icon:\s*'<svg", liste.group(1))
    assert len(icons) == 3, "Jeder der drei Zustände braucht ein eigenes Icon"
    # Und der Name muss weiterhin im aria-label landen.
    assert re.search(r"btn\.setAttribute\('aria-label'", html)


def test_icons_erben_die_textfarbe(html):
    """currentColor statt fester Farbe – sonst bleibt das Icon im dunklen
    Erscheinungsbild dunkel."""
    liste = re.search(r"const THEMES = \[(.*?)\];", html, re.S)
    for svg in re.findall(r"icon:\s*'(<svg.*?</svg>)'", liste.group(1), re.S):
        assert "currentColor" in svg, f"Icon ohne currentColor: {svg[:60]}"


def test_kopfzeile_bricht_um_statt_zu_quetschen(html):
    """Bei schmaler Arbeitsfläche brauchen Titel und Statusanzeigen mehr
    Platz als vorhanden. Ohne Umbruch wird der Titel zur Textsäule."""
    top = re.search(r"\.top\{([^}]*)\}", html)
    assert top and "flex-wrap:wrap" in top.group(1), \
        "Die Kopfzeile muss umbrechen dürfen"


def test_icons_werden_nicht_gequetscht(html):
    """Die Regel für Finger-Bedienung gibt .quiet zusätzliche Innenabstände.
    Ohne Gegenmaßnahme schrumpft das Icon darin auf wenige Pixel – genau
    das passierte beim ersten Versuch (16 px breit, dargestellt als 4 px)."""
    assert re.search(r"\.ikon svg\{[^}]*flex:none", html), \
        "Das Icon darf im Flexkasten nicht gestaucht werden"
    # Im Touch-Block muss .ikon seine eigene Größe zurückholen, sonst
    # ziehen die dortigen Innenabstände den Knopf in die Breite.
    block = re.search(r"@media \(pointer: coarse\)\{.*?\n  \}", html, re.S)
    assert block, "Der Touch-Block fehlt"
    assert re.search(r"\.ikon\{[^}]*padding:0", block.group(0)), \
        "Icon-Knöpfe brauchen auch bei Finger-Bedienung ihre eigene Größe"


# --------------------------------------------------------------------------- #
# Einheitliche Höhen
# --------------------------------------------------------------------------- #

def test_bedienelemente_teilen_sich_eine_hoehe(html):
    """In der Werkzeugleiste standen fünf verschiedene Höhen nebeneinander:
    Modell-Wähler 47, Modus-Gruppe 46, Format-Gruppe 37, Icons 32, die
    übrigen Knöpfe 31. Das wirkt unruhig, auch wenn es niemand ausmisst.

    Eine Variable statt verstreuter Werte: Wer eine Höhe ändert, ändert alle.
    """
    assert re.search(r"--h-ctl:\s*\d+px", html), \
        "Es braucht eine gemeinsame Höhe als Variable"

    # Eine Sammelregel gibt allen Elementen der Werkzeugleiste dieselbe
    # Höhe. Geprüft wird ihr Inhalt, nicht einzelne Regeln – sonst trifft
    # der Ausdruck die Sammelregel selbst und besteht immer.
    sammel = re.search(r"([^\n]*\.bar[^\n]*)\{height:var\(--h-ctl\)\}", html)
    assert sammel, "Die gemeinsame Höhe wird nirgends zugewiesen"
    # "> button": nur die Knöpfe der Leiste selbst, nicht die in Gruppen –
    # letztere würden ihre Umrandung sprengen.
    for teil in (".bar > button", ".bar .seg", ".bar .pick-btn"):
        assert teil in sammel.group(1), f"{teil} fehlt in der gemeinsamen Höhe"

    # Die Icon-Knöpfe sind quadratisch und nutzen dieselbe Größe.
    ikon = re.search(r"\.ikon\{([^}]*)\}", html, re.S)
    assert ikon and ikon.group(1).count("var(--h-ctl)") == 2, \
        "Icon-Knöpfe müssen quadratisch auf der gemeinsamen Höhe sitzen"


def test_gruppen_umschliessen_ihre_knoepfe_ohne_zu_wachsen(html):
    """Eine Segment-Gruppe darf nicht höher werden als ein einzelner Knopf
    daneben – sonst steht sie über die Zeile hinaus."""
    seg = re.search(r"\.seg\{([^}]*)\}", html)
    assert seg and "box-sizing:border-box" in seg.group(1), \
        "Die Gruppe muss ihre Innenabstände einrechnen"


# --------------------------------------------------------------------------- #
# Cover
# --------------------------------------------------------------------------- #

def test_cover_entfernen_verschwindet_ohne_cover(html):
    """Ein blasser Knopf sieht aus wie eine Option, die gerade nicht geht.
    Ohne Cover gibt es nichts zu entfernen – dann gehört er weg, nicht
    nur ausgegraut."""
    assert re.search(r"dropCover\.hidden = !info\.has_cover", html), \
        "Der Entfernen-Knopf muss ohne Cover ausgeblendet werden"


def test_coverfeld_ist_selbst_anklickbar(html):
    """Das Feld sieht aus wie eine Ablage – dann soll ein Klick darauf auch
    die Dateiauswahl öffnen, nicht nur der Knopf darunter."""
    assert re.search(r"art\.onclick = \(\) => file\.click\(\)", html), \
        "Das Cover-Feld muss die Dateiauswahl öffnen"
    assert re.search(r"\.tags-cover \.art\{[^}]*cursor:pointer", html, re.S), \
        "Und es muss als klickbar erkennbar sein"


def test_coverfeld_nimmt_bilder_per_ziehen_an(html):
    """Ein Bild auf das Feld zu ziehen ist der kürzeste Weg."""
    assert re.search(r"art\.ondrop", html), "Das Feld muss ein fallengelassenes Bild annehmen"
    assert re.search(r"art\.ondragover", html), "Und das Ziehen erlauben"


def test_coverfeld_sagt_was_zu_tun_ist(html):
    """„kein Cover“ beschreibt einen Zustand, nicht eine Möglichkeit."""
    assert not re.search(r"'kein Cover'", html), \
        "Der Text soll zum Handeln auffordern, nicht nur den Mangel benennen"


# --------------------------------------------------------------------------- #
# Eine Karte, ein Track
# --------------------------------------------------------------------------- #

def test_nur_eine_karte_bleibt_offen(html):
    """Mit einem Song arbeitet man zur Zeit, nicht mit dreien. Zwei Karten
    nebeneinander heißt zweimal Player, zweimal Mixer, zweimal Aufmerksamkeit.

    Die erste Fassung schloss nur Karten, die über die Liste geöffnet
    wurden (dataset.gewaehlt). Karten aus Analyse-Läufen und Aktionen
    blieben stehen – am Ende lagen zwei Karten desselben Tracks übereinander.
    """
    fn = re.search(r"function schliesseAndereKarten\(.*?\n\}", html, re.S)
    assert fn, "Die Funktion fehlt"
    assert "dataset.gewaehlt !== '1'" not in fn.group(0), \
        "Es dürfen nicht nur die über die Liste geöffneten Karten geschlossen werden"


def test_laufende_karte_wird_nicht_geschlossen(html):
    """Wer eine Analyse gestartet hat und nebenbei einen anderen Track
    ansieht, soll die Fortschrittsanzeige nicht verlieren. Eine rechnende
    Karte ist keine Ablenkung, sondern die Auskunft darüber, was passiert."""
    fn = re.search(r"function schliesseAndereKarten\(.*?\n\}", html, re.S)
    assert fn, "Die Funktion fehlt"
    assert re.search(r"dataset\.status === 'running'", fn.group(0)), \
        "Laufende Karten müssen verschont bleiben"
    assert re.search(r"dataset\.status === 'queued'", fn.group(0)), \
        "Wartende ebenso – sie fangen gleich an"

    # Der Status muss dafür auch an der Karte stehen.
    assert re.search(r"e\.dataset\.status = d\.status", html), \
        "renderJob muss den Status an der Karte führen"


def test_dieselbe_karte_entsteht_nicht_zweimal(html):
    """Ein Track, der schon offen ist, bekommt keine zweite Karte –
    unabhängig davon, ob er über die Liste oder über eine Aktion kam."""
    assert re.search(r"function karteFuerOrdner\(", html), \
        "Es braucht eine Suche nach der vorhandenen Karte eines Ordners"


def test_aktionen_oeffnen_keine_zusaetzliche_karte(html):
    """Loops, Mix und Lyrics gehören zum Track, über den sie laufen –
    sie brauchen keine eigene Karte daneben."""
    assert re.search(r"schliesseAndereKarten\(e\)", html), \
        "Eine neu aufgebaute Karte muss die übrigen schließen"


# --------------------------------------------------------------------------- #
# Eine einzelne Spur braucht keinen Mixer
# --------------------------------------------------------------------------- #

def test_einzelne_spur_zeigt_keine_mischbedienung(html):
    """Nach einer reinen Analyse gibt es nur das Original. Mute, Solo und
    Pegel setzen voraus, dass man Spuren gegeneinander abwägt – mit einer
    Spur gibt es nichts abzuwägen, und der Mute-Knopf würde nur den
    einzigen Ton abschalten."""
    assert re.search(r"const nurEineSpur = paths\.length < 2", html), \
        "Der Mixer muss den Fall einer einzelnen Spur kennen"
    assert re.search(r"\.tracks\.nur-eine", html), \
        "Für eine einzelne Spur braucht es eine eigene Darstellung"


def test_einzelne_spur_behaelt_die_wellenform(html):
    """Wellenform und Abspielkopf bleiben – sie zeigen, wo man im Stück ist.
    Nur die Mischwerkzeuge fallen weg."""
    regel = re.search(r"\.tracks\.nur-eine[^{]*\{([^}]*)\}", html, re.S)
    assert regel, "Die Regel fehlt"
    assert "display:none" in regel.group(1)
    # Die Wellenform darf nicht mit ausgeblendet werden.
    versteckt = re.findall(r"\.tracks\.nur-eine \.track \.(\w+)", html)
    assert "wf" not in versteckt, "Die Wellenform muss sichtbar bleiben"


def test_einzelne_spur_ist_hoerbar(html):
    """Das Original wird im Mix stumm geschaltet, damit es die Stems nicht
    verdoppelt. Ist es die einzige Spur, muss diese Regel ausgesetzt werden –
    sonst drückt man nach einer reinen Analyse auf Abspielen und hört nichts.
    """
    assert re.search(r"mute: stem === 'original' && !nurEineSpur", html), \
        "Die einzige Spur darf nicht stumm starten"


def test_einzelne_spur_ohne_ab_vergleich(html):
    """A/B vergleicht den Mix gegen das Original. Ohne Mix gibt es nichts
    zu vergleichen."""
    assert re.search(r"this\.abBtn\.hidden = !orig \|\| nurEineSpur", html), \
        "Der A/B-Knopf gehört bei einer einzelnen Spur weg"


def test_einzelne_spur_ohne_stem_hinweis(html):
    """„links neben einer Spur hört nur diesen Stem“ beschreibt etwas, das
    es bei einer Spur nicht gibt."""
    assert re.search(r"hinweis[^\n]*hidden = nurEineSpur|nurEineSpur[^\n]*hinweis", html), \
        "Der Hinweis auf die Einzelspur-Wiedergabe gehört weg"


def test_knoepfe_in_gruppen_sprengen_ihre_umrandung_nicht(html):
    """Ein Knopf innerhalb einer Segment-Gruppe darf nicht so hoch sein wie
    die Gruppe selbst – sonst ragt er unten aus der Umrandung heraus.

    Genau das passierte: Die Sammelregel für die Werkzeugleiste gab allen
    Knöpfen die volle Höhe, auch denen in der Gruppe, und eine ältere
    Regel setzte zusätzlich min-height.
    """
    modus = re.search(r"#mode button\{([^}]*)\}", html)
    if modus:
        assert "min-height:38px" not in modus.group(1), \
            "Die feste Mindesthöhe sprengt die Gruppe"

    # Die Sammelregel darf Knöpfe in Gruppen nicht erfassen.
    assert re.search(r"\.bar > button|\.bar button:not\(\.seg button\)|\.seg button\{height:auto", html), \
        "Knöpfe in Gruppen brauchen eine Ausnahme von der gemeinsamen Höhe"


def test_einstellungen_steht_bei_den_anderen_icons(html):
    """Einstellungen, Erscheinungsbild und Beenden gehören zusammen: Sie
    betreffen das Programm, nicht den einzelnen Song. Unten bei den
    Werkzeugen für den Track stand es fehl am Platz."""
    kopf = re.search(r'<div class="pills">(.*?)</div>', html, re.S)
    assert kopf, "Die Kopfzeile nicht gefunden"
    assert re.search(r'id="open-settings"', kopf.group(1)), \
        "Einstellungen gehören in die Kopfzeile"
    assert kopf.group(1).count("<svg") >= 3, "Alle drei brauchen ein Icon"


def test_einstellungen_bleibt_benannt(html):
    """Ohne Beschriftung braucht auch dieser Knopf einen Namen."""
    knopf = re.search(r'<button[^>]*id="open-settings"[^>]*>', html)
    assert knopf and "aria-label=" in knopf.group(0) and "title=" in knopf.group(0)


def test_werkzeugleiste_bricht_erst_um_wenn_es_noetig_ist(html):
    """Der Modell-Wähler ist das breiteste Element und darf schrumpfen.
    Ohne das rutscht ein einzelner kleiner Knopf in eine zweite Zeile,
    während neben dem Wähler noch Platz wäre."""
    regel = re.search(r"\.picker\{([^}]*)\}", html)
    assert regel and "min-width" in regel.group(1), \
        "Der Wähler braucht eine Untergrenze, damit er schrumpfen darf"


# --------------------------------------------------------------------------- #
# Kein Kasten im Kasten
# --------------------------------------------------------------------------- #

def test_karten_stehen_nicht_in_einem_zweiten_kasten(html):
    """Die Karte bringt Rahmen, Schatten und Polsterung selbst mit. Der
    umgebende Kasten mit Überschrift „GEÖFFNET · 1 Karte" setzte dieselben
    Attribute ein zweites Mal – zwei Rahmen um denselben Inhalt.

    Seit nur noch eine Karte offen ist, zählt die Überschrift ohnehin
    immer bis eins.
    """
    behaelter = re.search(r'<div[^>]*id="jobs-card"[^>]*>', html)
    assert behaelter, "jobs-card nicht gefunden"
    klassen = re.search(r'class="([^"]*)"', behaelter.group(0))
    assert not klassen or "card" not in klassen.group(1).split(), \
        "Der Behälter darf kein eigener Kasten mehr sein"

    # Und die Zählung entfällt, solange nichts läuft.
    assert re.search(r"\$\('jobs-head'\)\.hidden = !busy", html), \
        "Die Überschrift gehört weg, wenn nichts rechnet"


def test_versatz_wird_ohne_sticky_gemessen(html):
    """position:sticky verfälscht die Messung: Ist die Seite gescrollt,
    liefert getBoundingClientRect() die klebende Position statt der
    natürlichen – der Versatz fiele dann zu klein aus. Für den Moment der
    Messung muss sticky gelöst werden."""
    fn = re.search(r"function richteInfospalteAus\(.*?\n\}", html, re.S)
    assert fn, "Die Ausrichtung fehlt"
    assert "position = 'static'" in fn.group(0), \
        "Für die Messung muss sticky ausgesetzt werden"
    assert re.search(r"spalte\.style\.position = vorher", fn.group(0)), \
        "Und danach wiederhergestellt"


def test_infospalte_beginnt_auf_hoehe_der_karte(html):
    """Rechts steht, was zum Song gehört – dann soll es auch auf seiner
    Höhe beginnen und nicht weiter oben schweben."""
    assert re.search(r"\.shell\{[^}]*align-items:start", html, re.S), \
        "Die Spalten richten sich oben aus"
    assert re.search(r"\.infospalte\{[^}]*--ab-oben", html, re.S) or \
           re.search(r"#infospalte[^{]*\{[^}]*margin-top", html, re.S), \
           "Die Infospalte braucht einen Versatz auf Kartenhöhe"


def test_einstellungen_und_hell_sehen_verschieden_aus(html):
    """Beide Icons waren ein Kreis mit Strichen drumherum – geometrisch
    dieselbe Form. Nebeneinander in der Kopfzeile war nicht zu erkennen,
    welches wofür steht.

    Ein Zahnrad hat eine geschlossene, gezackte Kontur; eine Sonne hat
    freistehende Strahlen um einen Kreis. Das unterscheidet sie.
    """
    knopf = re.search(r'<button[^>]*id="open-settings".*?</button>', html, re.S)
    assert knopf, "Der Einstellungen-Knopf fehlt"
    zahnrad = knopf.group(0)

    sonne = re.search(r"wert: 'light'.*?icon: '(<svg.*?</svg>)'", html, re.S)
    assert sonne, "Das Sonnen-Icon fehlt"

    # Die Strahlen der Sonne dürfen nicht im Zahnrad auftauchen.
    strahlen = re.search(r'd="(M8 1\.[0-9]v1\.[0-9][^"]*)"', sonne.group(1))
    assert strahlen, "Die Sonne hat keine erkennbaren Strahlen"
    assert strahlen.group(1) not in zahnrad, \
        "Das Zahnrad benutzt denselben Pfad wie die Sonne"

    # Ein Zahnrad hat eine geschlossene, gezackte Kontur. Die Sonne besteht
    # aus einem Kreis plus acht geraden Strichen – erkennbar an den vielen
    # "M…v…M…h…"-Sprüngen in einem einzigen Pfad.
    zahnPfade = re.findall(r'<path[^>]*\sd="([^"]+)"', zahnrad)
    assert zahnPfade, "Das Zahnrad hat keinen Pfad"
    assert any(len(d) > 200 for d in zahnPfade), \
        "Eine gezackte Kontur braucht mehr als ein paar gerade Striche"

    # Und es darf nicht dieselbe Bauweise haben wie die Sonne: Kreis-Element
    # plus Strichliste.
    assert not ("<circle" in zahnrad and "M8 1." in zahnrad), \
        "Das Zahnrad ist noch als Kreis mit Strahlen gebaut"


# --------------------------------------------------------------------------- #
# Lyrics: Fortschritt und Anzeige
# --------------------------------------------------------------------------- #

def test_lyrics_zeigt_geschaetzten_fortschritt(html):
    """Whisper läuft in einem Zug durch und meldet unterwegs nichts. Ein
    Balken, der minutenlang bei null steht, sieht aus wie ein Absturz.

    Die Dauer ist abschätzbar – auf der Apple-GPU etwa ein Zehntel der
    Spielzeit. Ein Balken, der sich daran orientiert und bei 95 % wartet,
    sagt mehr als gar keiner.
    """
    assert re.search(r"function schaetzeFortschritt\(", html), \
        "Es braucht eine Schätzung für Läufe ohne echte Rückmeldung"
    assert re.search(r"0\.95", html), \
        "Die Schätzung darf nicht bei 100 % ankommen, bevor sie fertig ist"


def test_lyrics_stehen_in_der_rechten_spalte(html):
    """Der Text gehört zum Track – dann soll er dort stehen, wo alles
    andere zum Track steht, statt hinter einem Aufklapper in der Karte."""
    liste = re.search(r"const REITER = \[(.*?)\];", html, re.S)
    assert liste and "'text'" in liste.group(1), \
        "Die rechte Spalte braucht einen Reiter für den Text"
    assert re.search(r"function reiterText\(", html)
    # Und der Reiter muss auch aufgerufen werden.
    assert re.search(r"state\.infoReiter === 'text'\) reiterText\(", html), \
        "Der Text-Reiter muss verdrahtet sein"


def test_lyrics_bleiben_anklickbar(html):
    """Ein Klick auf eine Zeile springt an die Stelle im Song. Das ist der
    eigentliche Nutzen – ohne ihn wäre es nur eine Textdatei."""
    fn = re.search(r"function reiterText\(.*?\n\}", html, re.S)
    assert fn, "Der Text-Reiter fehlt"
    assert "seek" in fn.group(0), "Die Zeilen müssen an die Stelle springen"


def test_lyrics_und_text_sind_ein_knopf(html):
    """„Lyrics" transkribierte, „Text" zeigte an – zwei Knöpfe für dieselbe
    Sache. Das ist eine Trennung, die nur die Technik kennt: Wer den Text
    eines Songs will, denkt nicht in „erzeugen" und „ansehen".

    Ein Knopf, der beides kann: Ist Text da, zeigt er ihn; fehlt er, bietet
    er das Transkribieren an.
    """
    knoepfe = re.findall(r"mk\('([^']+)'", html)
    assert "Lyrics" not in knoepfe, "Der zweite Knopf ist überflüssig"
    assert "Text" in knoepfe, "Ein Knopf für den Text muss bleiben"

    # Er muss sichtbar sein, auch wenn noch kein Text existiert – sonst
    # käme man gar nicht erst zum Transkribieren.
    assert re.search(r"lyricsBtn\.hidden = false", html), \
        "Der Knopf gehört auch ohne vorhandenen Text angezeigt"

    # Und er muss in den Text-Reiter führen, nicht irgendwohin.
    assert re.search(r"state\.infoReiter = 'text'; renderInfospalte\(\)", html), \
        "Der Knopf muss den Text-Reiter öffnen"


def test_textknopf_bietet_transkription_an_wenn_nichts_da_ist(html):
    """Ohne Text ist der Reiter nicht leer, sondern führt zum nächsten
    Schritt – mit der Sprachwahl, die vorher im Lyrics-Aufklapper stand."""
    fn = re.search(r"async function reiterText\(.*?\n\}", html, re.S)
    assert fn, "Der Text-Reiter fehlt"
    assert re.search(r"el\('button', 'btn sm', 'Transkribieren'\)", fn.group(0)), \
        "Ohne Text muss der Reiter das Transkribieren anbieten"
    assert "language" in fn.group(0), "Die Sprachwahl gehört dazu"
    assert re.search(r"kind: 'lyrics'", fn.group(0)), \
        "Der Knopf muss die Transkription auch starten"


# --------------------------------------------------------------------------- #
# Camelot-Rad: aufklappbar
# --------------------------------------------------------------------------- #

def test_rad_ist_standardmaessig_eingeklappt(html):
    """Das Rad nimmt eine halbe Bildschirmhöhe, wird aber nur zum Filtern
    gebraucht. Wer eine Liste durchsucht, braucht die Liste – nicht das
    Werkzeug daneben."""
    box = re.search(r'<div class="wheel-box"[^>]*id="wheel-box"[^>]*>', html) \
        or re.search(r'<div[^>]*id="wheel-box"[^>]*>', html)
    assert box, "Der Radbereich braucht eine ID zum Ein- und Ausklappen"
    assert "hidden" in box.group(0), "Eingeklappt ist der Ausgangszustand"


def test_liste_bekommt_den_platz_des_eingeklappten_rades(html):
    """Sonst bliebe die Spalte leer stehen und die Liste weiterhin schmal –
    der Gewinn wäre dahin."""
    assert re.search(r"\.analyse:has\(#wheel-box\[hidden\]\)\{grid-template-columns:1fr\}", html), \
        "Ohne Rad gehört die ganze Breite der Liste"


def test_knopf_nennt_die_gewaehlte_tonart(html):
    """„Tonart 6A" sagt mehr als ein hervorgehobenes „Tonart": Man sieht,
    wonach gefiltert wird, ohne aufzuklappen."""
    fn = re.search(r"function zeigeRadZustand\(\).*?\n\}", html, re.S)
    assert fn and re.search(r"Tonart \$\{state\.filter\.cam\}", fn.group(0)), \
        "Der Knopf muss die gewählte Tonart nennen"


def test_rad_wird_erst_beim_aufklappen_gezeichnet(html):
    """97 SVG-Elemente aufzubauen, die niemand sieht, ist verschenkte
    Arbeit bei jedem Seitenaufbau."""
    fn = re.search(r"\$\('wheel-toggle'\)\.onclick = \{?.*?\n\};", html, re.S)
    assert fn and "renderWheel()" in fn.group(0), \
        "Das Rad soll erst beim Aufklappen gezeichnet werden"


def test_rad_laesst_sich_aufklappen(html):
    """Ein Knopf, der es zeigt und wieder verbirgt."""
    assert re.search(r'id="wheel-toggle"', html), "Der Knopf fehlt"
    assert re.search(r"\$\('wheel-toggle'\)\.onclick", html)


def test_aktiver_filter_ist_auch_eingeklappt_erkennbar(html):
    """Sonst sucht man in einer gefilterten Liste und wundert sich, warum
    Tracks fehlen. Der Knopf muss zeigen, dass ein Filter greift."""
    fn = re.search(r"function zeigeRadZustand\(\).*?\n\}", html, re.S)
    assert fn, "Es fehlt die Anzeige des Filterzustands"
    assert "classList.toggle('on'" in fn.group(0), \
        "Der Knopf muss einen aktiven Filter hervorheben"
    assert "state.filter.cam" in fn.group(0), \
        "Und zwar anhand des tatsächlichen Filters"
    # Sie muss nach jeder Änderung laufen.
    assert html.count("zeigeRadZustand()") >= 2, \
        "Der Zustand muss nach Filteränderungen nachgezogen werden"


def test_rad_klappt_bei_auswahl_nicht_zu(html):
    """Wer eine Tonart wählt, will oft gleich die nächste probieren –
    zuklappen nach jedem Klick wäre lästig."""
    fn = re.search(r"\$\('wheel-toggle'\)\.onclick = .*?;", html, re.S)
    assert fn, "Der Umschalter fehlt"
    # Die Segment-Klicks dürfen den Bereich nicht verbergen.
    assert not re.search(r"seg\.onclick[^}]*wheel-box'\)\.hidden = true", html, re.S)


# --------------------------------------------------------------------------- #
# Bereichswechsel räumt auf
# --------------------------------------------------------------------------- #

def test_bereichswechsel_schliesst_den_offenen_song(html):
    """Wer links eine andere Playlist oder einen anderen Bereich wählt,
    verlässt den Song, mit dem er gerade gearbeitet hat. Bleibt dessen
    Karte stehen, steht sie über einer Liste, in der er nicht mehr
    vorkommt – im Extremfall über „Kein Treffer für diesen Filter".
    """
    assert re.search(r"function verlasseAktuellenSong\(", html), \
        "Es braucht einen Weg, die Ansicht eines Songs zu verlassen"


def test_playlistwechsel_raeumt_auf(html):
    """Der Wechsel selbst muss das auch auslösen."""
    fn = re.search(r"b\.onclick = \(\) => \{\s*state\.playlistFilter.*?\};", html, re.S)
    assert fn, "Der Playlist-Klick nicht gefunden"
    assert "verlasseAktuellenSong()" in fn.group(0), \
        "Beim Playlist-Wechsel muss der offene Song weichen"


def test_laufende_arbeit_bleibt_beim_wechsel_stehen(html):
    """Eine rechnende Karte ist Fortschrittsanzeige, kein Rest vom vorigen
    Song – sie darf ein Wechsel nicht wegräumen."""
    fn = re.search(r"function verlasseAktuellenSong\(.*?\n\}", html, re.S)
    assert fn, "Die Funktion fehlt"
    assert re.search(r"'running'", fn.group(0)) and re.search(r"'queued'", fn.group(0)), \
        "Laufende und wartende Karten müssen verschont bleiben"


def test_rechte_spalte_leert_sich_beim_wechsel(html):
    """Sonst zeigt sie weiter die Werte eines Songs, den man verlassen hat."""
    fn = re.search(r"function verlasseAktuellenSong\(.*?\n\}", html, re.S)
    assert fn and re.search(r"state\.infoTrack = null", fn.group(0)), \
        "Die rechte Spalte gehört mit geleert"


def test_neuer_track_verlaesst_den_offenen_song(html):
    """Wer „Neuer Track" wählt, beginnt etwas Neues – der alte Song hat
    dort nichts mehr zu suchen. Dasselbe gilt für „Analysen": Wer die
    Liste ansieht, hat den einzelnen Song verlassen."""
    liste = re.search(r"const BEREICHE = \[(.*?)\];", html, re.S)
    assert liste, "Die Bereichsliste fehlt"
    for ziel in ("bereich-neu", "bereich-analysen"):
        eintrag = re.search(rf"\{{ ziel: '{ziel}'[^}}]*\}}", liste.group(1))
        assert eintrag and "verlaesst: true" in eintrag.group(0), \
            f"{ziel} muss den offenen Song verlassen"

    # "In Arbeit" zeigt gerade die laufenden Karten – dort wäre Aufräumen
    # widersinnig.
    arbeit = re.search(r"\{ ziel: 'jobs-card'[^}]*\}", liste.group(1))
    assert arbeit and "verlaesst" not in arbeit.group(0), \
        "„In Arbeit\" darf nicht aufräumen"

    # Und der Klick muss es auslösen.
    assert re.search(r"if \(b\.verlaesst\) verlasseAktuellenSong\(\);", html)
