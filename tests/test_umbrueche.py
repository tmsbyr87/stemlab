"""Umbruchpunkte aus den Stems.

Der Vorgänger benannte Abschnitte nach dem Summenpegel eines gemasterten
Tracks. Gemessen schwankte der über einen ganzen Clubtrack um 12 % – der
Limiter drückt genau das weg, was die Messung suchte. Heraus kam ein
47-Takte-"Drop" und ein 52-Takte-"Aufbau", während zwei Abschnitte mit
denselben Kennzahlen verschiedene Namen trugen.

In den Stems steht dasselbe Arrangement unübersehbar: Bei einem echten
Track fielen die Drums an einer Stelle auf 2 % ihres Maximums. Deshalb
misst diese Funktion, *welche Stems wann spielen*, und benennt nichts.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
import soundfile as sf

import analysis
from conftest import kick_track, tone


SR = analysis.SAMPLE_RATE


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Server mit einem Zielordner im temporären Verzeichnis."""
    from fastapi.testclient import TestClient

    import server

    monkeypatch.setattr(server, "output_root", lambda: tmp_path)
    # Fremde Host-Header werden abgewiesen; ohne base_url gäbe es überall 403.
    return TestClient(server.app, base_url="http://127.0.0.1")


def _schreibe(ordner, name, y):
    sf.write(str(ordner / f"{name}.wav"), y.astype("float32"), SR)


def _takte(anzahl, bpm=120.0):
    """Downbeats für `anzahl` Takte zu 4 Schlägen."""
    pro_takt = 4 * 60.0 / bpm
    return [i * pro_takt for i in range(anzahl + 1)]


@pytest.fixture
def arrangement(tmp_path):
    """Ein Ordner mit vier Stems und einem gebauten Arrangement.

    32 Takte zu 2 s: Drums durchgehend außer Takt 17–24, Bass ab Takt 9,
    Vocals ab Takt 17. Die Umbrüche liegen damit bei 9, 17 und 25.
    """
    takte, pro_takt = 32, 2.0
    n = int(takte * pro_takt * SR)

    drums = kick_track(n)
    drums[int(16 * pro_takt * SR):int(24 * pro_takt * SR)] = 0.0

    bass = np.zeros(n, dtype="float32")
    ab = int(8 * pro_takt * SR)
    bass[ab:] = tone(60.0, (n - ab) / SR)[: n - ab] * 0.5

    vocals = np.zeros(n, dtype="float32")
    ab = int(16 * pro_takt * SR)
    vocals[ab:] = tone(440.0, (n - ab) / SR)[: n - ab] * 0.4

    other = tone(880.0, n / SR)[:n] * 0.2

    for name, y in (("drums", drums), ("bass", bass), ("vocals", vocals), ("other", other)):
        _schreibe(tmp_path, name, y)
    return tmp_path, _takte(takte, bpm=120.0)


# --------------------------------------------------------------------------- #
# Die Umbrüche selbst
# --------------------------------------------------------------------------- #

def test_findet_die_gebauten_umbrueche(arrangement):
    """Wo ein Stem ein- oder aussetzt, liegt ein Umbruch."""
    ordner, downbeats = arrangement
    u = analysis.umbrueche(ordner, downbeats)

    takte = [x["takt"] for x in u]
    assert takte[0] == 1, "Der Track beginnt bei Takt 1"
    for erwartet in (9, 17, 25):
        assert erwartet in takte, f"Bei Takt {erwartet} ändert sich die Besetzung"


def test_liefert_die_besetzung_statt_eines_namens(arrangement):
    """Kein "Drop", kein "Aufbau" – nur wer spielt.

    Die Namen waren die Deutung, die auf dem gemasterten Summenpegel nicht
    trug. Wer spielt, ist dagegen messbar.
    """
    ordner, downbeats = arrangement
    u = analysis.umbrueche(ordner, downbeats)

    for x in u:
        assert "art" not in x, "Die Funktion benennt nichts"
        assert isinstance(x["stems"], list)

    nach_takt = {x["takt"]: set(x["stems"]) for x in u}
    assert "bass" not in nach_takt[1], "Der Bass kommt erst bei Takt 9"
    assert "bass" in nach_takt[9]
    assert "drums" not in nach_takt[17], "Takt 17–24 ist ohne Drums gebaut"
    assert "drums" in nach_takt[25]


def test_umbrueche_liegen_auf_achtergruppen(arrangement):
    """Tanzmusik ist in Acht- und Sechzehnergruppen gebaut. Ein Umbruch bei
    Takt 43 wäre für die Produktion unbrauchbar."""
    ordner, downbeats = arrangement
    for x in analysis.umbrueche(ordner, downbeats):
        assert (x["takt"] - 1) % 8 == 0, f"Takt {x['takt']} liegt nicht auf einer Achtergruppe"


def test_jeder_umbruch_kennt_seine_laenge(arrangement):
    """Ohne Länge müsste die Oberfläche sie aus dem nächsten Umbruch
    errechnen – und der letzte hätte keine."""
    ordner, downbeats = arrangement
    u = analysis.umbrueche(ordner, downbeats)

    assert all(x["takte"] > 0 for x in u)
    assert sum(x["takte"] for x in u) == 32, "Zusammen ergeben sie den ganzen Track"


def test_gleichbleibende_besetzung_gibt_keinen_umbruch(tmp_path):
    """Läuft alles durch, gibt es genau einen Block – und nicht bei jeder
    Achtergruppe einen."""
    n = int(32 * 2.0 * SR)
    for name, y in (("drums", kick_track(n)), ("bass", tone(60.0, n / SR)[:n] * 0.5)):
        _schreibe(tmp_path, name, y)

    u = analysis.umbrueche(tmp_path, _takte(32))
    assert len(u) == 1
    assert u[0]["takt"] == 1 and u[0]["takte"] == 32


# --------------------------------------------------------------------------- #
# Grenzen
# --------------------------------------------------------------------------- #

def test_ohne_stems_keine_umbrueche(tmp_path):
    """Eine reine Analyse hat keine Stems – dann gibt es nichts zu messen,
    und erfundene Umbrüche wären schlimmer als gar keine."""
    assert analysis.umbrueche(tmp_path, _takte(32)) == []


def test_zu_wenige_takte(tmp_path):
    """Unter zwei Achtergruppen ist kein Arrangement erkennbar."""
    n = int(8 * 2.0 * SR)
    _schreibe(tmp_path, "drums", kick_track(n))
    assert analysis.umbrueche(tmp_path, _takte(8)) == []


def test_ohne_downbeats(tmp_path):
    """Ohne Taktraster gibt es keine Achtergruppen."""
    n = int(32 * 2.0 * SR)
    _schreibe(tmp_path, "drums", kick_track(n))
    assert analysis.umbrueche(tmp_path, []) == []


def test_original_zaehlt_nicht_als_stem(tmp_path):
    """original.wav enthält alles – als Stem gelesen wäre immer alles an."""
    n = int(32 * 2.0 * SR)
    _schreibe(tmp_path, "original", kick_track(n))
    assert analysis.umbrueche(tmp_path, _takte(32)) == []


# --------------------------------------------------------------------------- #
# Anschluss an die Analyse-Datei
# --------------------------------------------------------------------------- #

def test_nachtragen_schreibt_in_die_analyse(arrangement):
    """Die Umbrüche brauchen die fertigen Stems – die gibt es erst nach der
    Trennung, nicht während der Analyse. Deshalb ein eigener Schritt, der
    die vorhandene analysis.json ergänzt."""
    import json

    ordner, downbeats = arrangement
    (ordner / "analysis.json").write_text(json.dumps({"bpm": 120.0, "downbeats": downbeats}))

    analysis.trage_umbrueche_nach(ordner)

    daten = json.loads((ordner / "analysis.json").read_text())
    assert daten["umbrueche"], "Die Umbrüche gehören in die Analyse-Datei"
    assert daten["bpm"] == 120.0, "Der Rest der Analyse bleibt unangetastet"
    assert {x["takt"] for x in daten["umbrueche"]} >= {1, 9, 17, 25}


def test_nachtragen_ohne_analysedatei_faellt_nicht_um(tmp_path):
    """Ohne analysis.json gibt es nichts zu ergänzen – und die Trennung
    darf daran nicht scheitern."""
    analysis.trage_umbrueche_nach(tmp_path)          # kein Fehler
    assert not (tmp_path / "analysis.json").exists()


def test_nachtragen_ohne_stems_schreibt_nichts(tmp_path):
    """Eine reine Analyse behält ihre Datei unverändert."""
    import json

    (tmp_path / "analysis.json").write_text(json.dumps({"bpm": 128.0, "downbeats": _takte(32)}))
    vorher = (tmp_path / "analysis.json").read_text()

    analysis.trage_umbrueche_nach(tmp_path)

    assert (tmp_path / "analysis.json").read_text() == vorher


def test_separate_traegt_die_umbrueche_nach():
    """Ohne den Aufruf in der Trennung bleibt das Feld für jeden echten
    Track leer – die Funktion liefe dann nur in Tests."""
    import inspect

    import engine

    quelle = inspect.getsource(engine.separate)
    assert "trage_umbrueche_nach" in quelle, \
        "Nach der Trennung müssen die Umbrüche aus den Stems entstehen"


def test_api_reicht_die_umbrueche_durch():
    """folder_summary kürzt lange Listen auf ihre Anzahl – die Umbrüche
    müssen die Ausnahme sein, sonst kommt in der Oberfläche eine "9" an
    und die Leiste kann nichts zeichnen. Genau das ist bei den Abschnitten
    schon einmal passiert."""
    import inspect

    import server

    quelle = inspect.getsource(server.folder_summary)
    treffer = re.search(r"durchreichen = \{([^}]*)\}", quelle)
    assert treffer, "Die Ausnahmeliste fehlt"
    assert '"umbrueche"' in treffer.group(1), \
        "Die Umbrüche dürfen nicht zu ihrer Anzahl werden"


# --------------------------------------------------------------------------- #
# Eigene Beschriftungen
# --------------------------------------------------------------------------- #

def test_marken_speichern_und_lesen(client, tmp_path, monkeypatch):
    """StemLab schlägt die Taktgrenzen vor, benennen darf sie der Nutzer.

    Die Beschriftungen liegen neben den Stems – wer den Ordner kopiert,
    nimmt sie mit. Getrennt von notes.txt, weil das Freitext ist und
    diese hier an Takte gebunden sind.
    """
    import server

    ordner = tmp_path / "Song"
    ordner.mkdir()
    monkeypatch.setattr(server, "_allowed_result_path", lambda p: ordner)

    leer = client.get("/api/marken", params={"path": str(ordner)})
    assert leer.status_code == 200
    assert leer.json()["marken"] == {}

    client.post("/api/marken", json={"folder": str(ordner), "marken": {"9": "Drop", "97": "Breakdown"}})

    assert (ordner / "marken.json").is_file(), "Die Marken gehören in den Ordner"
    assert client.get("/api/marken", params={"path": str(ordner)}).json()["marken"] == \
        {"9": "Drop", "97": "Breakdown"}


def test_leere_marke_wird_entfernt(client, tmp_path, monkeypatch):
    """Eine gelöschte Beschriftung darf nicht als leerer Text zurückbleiben."""
    import server

    ordner = tmp_path / "Song"
    ordner.mkdir()
    monkeypatch.setattr(server, "_allowed_result_path", lambda p: ordner)

    client.post("/api/marken", json={"folder": str(ordner), "marken": {"9": "Drop", "17": "  "}})
    assert client.get("/api/marken", params={"path": str(ordner)}).json()["marken"] == {"9": "Drop"}

    client.post("/api/marken", json={"folder": str(ordner), "marken": {}})
    assert not (ordner / "marken.json").exists(), "Keine leere Datei hinterlassen"


# --------------------------------------------------------------------------- #
# Die Schwelle: was gilt als "spielt"
# --------------------------------------------------------------------------- #

def test_zurueckgenommener_stem_gilt_weiter_als_spielend(tmp_path):
    """Der Unterschied zwischen "leiser" und "aus" ist der ganze Punkt.

    An echten Tracks gemessen: Fallen die Drums in einem Breakdown aus,
    gehen sie auf 2 % ihres Maximums zurück. Ein Stem, der nur um die
    Hälfte zurückgenommen ist, spielt weiter – sonst meldet die Leiste
    einen Umbruch, wo nur ein Filter zugedreht wurde.
    """
    takte, pro_takt = 32, 2.0
    n = int(takte * pro_takt * SR)

    leiser = tone(440.0, n / SR)[:n] * 0.6
    leiser[int(16 * pro_takt * SR):] *= 0.5        # halb so laut, nicht aus
    _schreibe(tmp_path, "other", leiser)
    _schreibe(tmp_path, "drums", kick_track(n))

    u = analysis.umbrueche(tmp_path, _takte(takte))
    assert len(u) == 1, f"Halbe Lautstärke ist kein Umbruch, bekam aber {len(u)} Blöcke"
    assert set(u[0]["stems"]) == {"other", "drums"}


def test_ausgesetzter_stem_gilt_als_aus(tmp_path):
    """Der Gegenfall: 2 % des Maximums – so sieht ein echter Breakdown aus –
    muss als "spielt nicht" durchgehen."""
    takte, pro_takt = 32, 2.0
    n = int(takte * pro_takt * SR)

    drums = kick_track(n)
    drums[int(16 * pro_takt * SR):] *= 0.02
    _schreibe(tmp_path, "drums", drums)
    _schreibe(tmp_path, "other", tone(440.0, n / SR)[:n] * 0.4)

    u = analysis.umbrueche(tmp_path, _takte(takte))
    assert len(u) == 2, "Der Aussetzer bei Takt 17 ist ein Umbruch"
    assert "drums" in u[0]["stems"]
    assert "drums" not in u[1]["stems"], "Bei 2 % spielen die Drums nicht mehr"
    assert u[1]["takt"] == 17


def test_restakte_gehen_nicht_verloren(tmp_path):
    """Ein Track mit 35 Takten hat vier volle Achtergruppen und drei Takte
    Rest. Fallen die weg, endet die Leiste vor dem Track."""
    takte, pro_takt = 35, 2.0
    n = int(takte * pro_takt * SR)
    _schreibe(tmp_path, "drums", kick_track(n))

    u = analysis.umbrueche(tmp_path, _takte(takte))
    assert sum(x["takte"] for x in u) == 35, \
        "Die Blöcke müssen den ganzen Track abdecken, auch die angebrochene Gruppe"


def test_ordner_ohne_audiodateien(tmp_path):
    """Ein Ordner mit nur einer analysis.json darf nicht in die Messung
    laufen – dort gibt es keine Stems, die man befragen könnte."""
    (tmp_path / "analysis.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("hallo")
    assert analysis.umbrueche(tmp_path, _takte(32)) == []
