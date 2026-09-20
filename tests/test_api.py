"""Die HTTP-Schnittstelle, insbesondere ihre Schutzmechanismen.

Der Server hört nur lokal, liefert aber Dateien aus. Die Pfadprüfung ist
deshalb die wichtigste Stelle: ohne sie liesse sich über `?path=` jede Datei
im Benutzerordner abrufen.
"""

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
from conftest import SR, tiny_png  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Server mit einem Zielordner im temporären Verzeichnis."""
    monkeypatch.setattr(server, "output_root", lambda: tmp_path)
    # Der Server weist fremde Host-Header ab; der Testclient meldet sich sonst
    # als "testserver" und bekäme überall 403.
    return TestClient(server.app, base_url="http://127.0.0.1")


@pytest.fixture
def track(tmp_path):
    folder = tmp_path / "Testsong"
    folder.mkdir()
    path = folder / "original.wav"
    sf.write(path, np.zeros(SR), SR)
    return path


def test_ping_antwortet(client):
    assert client.get("/api/ping").status_code == 200


def test_tags_lesen(client, track):
    response = client.get("/api/tags", params={"path": str(track)})
    assert response.status_code == 200
    assert "title" in response.json()


def test_tags_schreiben(client, track):
    response = client.post("/api/tags", json={"path": str(track), "values": {"title": "Neu"}})
    assert response.status_code == 200
    assert response.json()["title"] == "Neu"


def test_existierende_datei_ausserhalb_wird_abgewiesen(client, tmp_path):
    """Der Kern der Pfadprüfung: 403, nicht 404.

    Ein Test auf nicht existierende Pfade wie /etc/passwd beweist nichts – der
    käme auch bei ausgebauter Prüfung als 404 zurück. Hier liegt eine echte,
    lesbare Audiodatei außerhalb des Zielordners; nur die Prüfung verhindert
    die Auslieferung.
    """
    import numpy as np
    import soundfile as sf

    fremd = tmp_path.parent / "ausserhalb.wav"
    sf.write(fremd, np.zeros(SR), SR)
    try:
        response = client.get("/api/tags", params={"path": str(fremd)})
        assert response.status_code == 403
    finally:
        fremd.unlink()


@pytest.mark.parametrize("pfad", ["/etc/passwd", "/etc/hosts"])
def test_systempfade_werden_abgewiesen(client, pfad):
    """Systemdateien dürfen nie heraus – egal ob sie existieren."""
    assert client.get("/api/tags", params={"path": pfad}).status_code in (403, 404)


def test_hochklettern_wird_abgewiesen(client, tmp_path):
    response = client.get("/api/tags", params={"path": str(tmp_path / ".." / ".." / "etc" / "passwd")})
    assert response.status_code in (403, 404)


def test_nicht_audio_wird_abgewiesen(client, tmp_path):
    other = tmp_path / "Testsong"
    other.mkdir(exist_ok=True)
    text = other / "notiz.txt"
    text.write_text("kein Audio")
    assert client.get("/api/tags", params={"path": str(text)}).status_code == 404


def test_cover_ohne_bild_meldet_404(client, track):
    assert client.get("/api/cover", params={"path": str(track)}).status_code == 404


def test_cover_hochladen_und_abrufen(client, track):
    png = tiny_png()
    upload = client.post("/api/cover",
                         data={"path": str(track)},
                         files={"file": ("cover.png", png, "image/png")})
    assert upload.status_code == 200
    assert upload.json()["has_cover"] is True

    download = client.get("/api/cover", params={"path": str(track)})
    assert download.status_code == 200
    assert download.content == png


def test_nur_bilder_als_cover(client, track):
    response = client.post("/api/cover",
                           data={"path": str(track)},
                           files={"file": ("schad.txt", b"kein Bild", "text/plain")})
    assert response.status_code == 400


def test_cover_loeschen(client, track):
    client.post("/api/cover", data={"path": str(track)},
                files={"file": ("c.png", tiny_png(), "image/png")})
    response = client.post("/api/cover/delete", json={"path": str(track)})
    assert response.status_code == 200
    assert response.json()["has_cover"] is False


def test_fremder_host_wird_abgewiesen(client):
    """Schutz gegen DNS-Rebinding: nur lokale Hostnamen sind erlaubt."""
    response = client.get("/api/ping", headers={"Host": "boese.example.com"})
    assert response.status_code == 403


def test_fremder_origin_wird_abgewiesen(client, track):
    """Schutz gegen CSRF: schreibende Zugriffe brauchen einen lokalen Origin."""
    response = client.post("/api/tags",
                           json={"path": str(track), "values": {"title": "X"}},
                           headers={"Origin": "http://boese.example.com"})
    assert response.status_code == 403


def test_eigener_origin_wird_akzeptiert(client, track):
    response = client.post("/api/tags",
                           json={"path": str(track), "values": {"title": "X"}},
                           headers={"Origin": "http://127.0.0.1:8765"})
    assert response.status_code == 200


def test_audio_nur_aus_dem_zielordner(client, track, tmp_path):
    assert client.get("/api/audio", params={"path": str(track)}).status_code == 200
    assert client.get("/api/audio", params={"path": "/etc/hosts"}).status_code in (403, 404)


def test_text_nur_aus_dem_zielordner(client, tmp_path):
    ordner = tmp_path / "Testsong"
    ordner.mkdir(exist_ok=True)
    (ordner / "chords.txt").write_text("| Gm |")
    assert client.get("/api/text", params={"path": str(ordner / "chords.txt")}).status_code == 200
    assert client.get("/api/text", params={"path": "/etc/hosts"}).status_code in (403, 404)


@pytest.mark.parametrize("header, erwartet", [
    ("127.0.0.1", "127.0.0.1"),
    ("127.0.0.1:8765", "127.0.0.1"),
    ("LOCALHOST:8080", "localhost"),
    ("[::1]:8765", "[::1]"),
])
def test_hostnamen_werden_normalisiert(header, erwartet):
    """Port abschneiden, Groß-/Kleinschreibung angleichen, IPv6-Klammern behalten."""
    assert server._host_only(header) == erwartet


# --------------------------------------------------------------------------- #
# Einstellungen
# --------------------------------------------------------------------------- #

def test_einstellungen_lesen(client):
    d = client.get("/api/settings").json()
    assert "values" in d and "presets" in d
    assert set(d["presets"]) == {"rekordbox", "traktor", "serato"}


def test_einstellungen_speichern(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "CONFIG_PATH", tmp_path / "config.json")
    r = client.post("/api/settings", json={"values": {"tag_pattern": "key_energy"}})
    assert r.status_code == 200
    assert r.json()["values"]["tag_pattern"] == "key_energy"


def test_unbekannter_wert_wird_abgewiesen(client):
    """Ein Tippfehler darf nicht stillschweigend in der Konfiguration landen."""
    r = client.post("/api/settings", json={"values": {"tag_pattern": "quatsch"}})
    assert r.status_code == 400


def test_verdrehter_tempobereich_wird_abgewiesen(client):
    r = client.post("/api/settings", json={"values": {"tempo_min": 200, "tempo_max": 100}})
    assert r.status_code == 400


def test_vorschau_zeigt_das_ergebnis(client):
    """Die Vorschau ist der Kern der Bedienbarkeit – sie muss stimmen."""
    r = client.post("/api/settings/preview", json={"values": {
        "tag_pattern": "key_tempo_energy", "key_notation": "camelot",
        "key_leading_zero": True, "tempo_decimals": 0,
        "rename_pattern": "{name} - {key}"}})
    d = r.json()
    assert d["tag"] == "06A - 124 - 5"
    assert d["filename"] == "vocals - 06A.wav"


# --------------------------------------------------------------------------- #
# Nur analysieren statt trennen
# --------------------------------------------------------------------------- #

def test_analyse_braucht_kein_modell(client, tmp_path, monkeypatch):
    """Beim reinen Analysieren wird nie ein Modell geladen.

    Es darf also fehlen – sonst könnte man nicht analysieren, solange die
    Modelliste noch lädt.
    """
    aufrufe = {}

    def falsches_analyze(**kwargs):
        aufrufe.update(kwargs)
        class Ergebnis:
            folder = tmp_path / "Song"
            files: list = []
            extras: list = []
            seconds = 1.0
            analysis = {"bpm": 124.0}
        (tmp_path / "Song").mkdir(exist_ok=True)
        return Ergebnis()

    monkeypatch.setattr(server.engine, "analyze_only", falsches_analyze)
    r = client.post("/api/jobs",
                    data={"mode": "analyze", "model": ""},
                    files={"file": ("probe.wav", b"RIFF0000WAVE" + b"\0" * 100, "audio/wav")})
    assert r.status_code == 200


def test_unbekannter_modus_wird_abgewiesen(client):
    r = client.post("/api/jobs",
                    data={"mode": "quatsch", "model": ""},
                    files={"file": ("probe.wav", b"RIFF0000WAVE", "audio/wav")})
    assert r.status_code == 400


def test_nachtraegliches_trennen_braucht_eine_aufnahme(client, tmp_path):
    ordner = tmp_path / "Leer"
    ordner.mkdir()
    r = client.post("/api/separate-folder", json={"folder": str(ordner), "model": "demucs4"})
    assert r.status_code == 400
    assert "Aufnahme" in r.json()["detail"]


def test_nachtraegliches_trennen_lehnt_fertige_ordner_ab(client, tmp_path):
    """Ein Ordner mit Stems wurde schon getrennt – kein zweites Mal."""
    import numpy as np
    import soundfile as sf

    ordner = tmp_path / "Fertig"
    ordner.mkdir()
    sf.write(ordner / "original.wav", np.zeros(SR), SR)
    sf.write(ordner / "vocals.wav", np.zeros(SR), SR)
    r = client.post("/api/separate-folder", json={"folder": str(ordner), "model": "demucs4"})
    assert r.status_code == 400
    assert "bereits" in r.json()["detail"]


def test_quelle_im_ergebnisordner_wird_nicht_geloescht(tmp_path, monkeypatch):
    """original.wav muss das nachträgliche Trennen überleben.

    Der Worker räumt nach jedem Job die hochgeladene Zwischendatei weg. Beim
    nachträglichen Trennen ist die Quelle aber die original.wav im
    Ergebnisordner – ohne die Ausnahme löschte er sie mit und der
    A/B-Vergleich war hinüber.
    """
    import numpy as np
    import soundfile as sf

    quelle = tmp_path / "original.wav"
    sf.write(quelle, np.zeros(SR), SR)

    def trennung(**kwargs):
        class Ergebnis:
            folder = tmp_path
            files: list = []
            extras: list = []
            seconds = 1.0
            model_file = ""
            device = "cpu"
            analysis = None
        return Ergebnis()

    monkeypatch.setattr(server.engine, "separate", trennung)
    job = server.Job(id="t1", kind="separate", display_name="Song", source=quelle,
                     model_key="demucs4", options={"reuse": str(tmp_path)})
    server.run_job(job)
    assert quelle.exists(), "original.wav wurde gelöscht"


def test_hochgeladene_datei_wird_aufgeraeumt(tmp_path, monkeypatch):
    """Ohne reuse ist die Quelle eine Zwischendatei – die muss weg."""
    import numpy as np
    import soundfile as sf

    quelle = tmp_path / "upload.wav"
    sf.write(quelle, np.zeros(SR), SR)

    def trennung(**kwargs):
        class Ergebnis:
            folder = tmp_path
            files: list = []
            extras: list = []
            seconds = 1.0
            model_file = ""
            device = "cpu"
            analysis = None
        return Ergebnis()

    monkeypatch.setattr(server.engine, "separate", trennung)
    job = server.Job(id="t2", kind="separate", display_name="Song", source=quelle,
                     model_key="demucs4", options={})
    server.run_job(job)
    assert not quelle.exists()


# --------------------------------------------------------------------------- #
# Playlists
# --------------------------------------------------------------------------- #
#
# Playlists sind gespeicherte Filter, keine zweite Dateiverwaltung: Sie
# merken sich Ordnerpfade, die Dateien bleiben, wo sie sind. Deshalb reicht
# config.json – eine Datenbank wäre für eine Handvoll Listen zu viel.

def test_playlists_sind_anfangs_leer(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "CONFIG_PATH", tmp_path / "config.json")
    antwort = client.get("/api/playlists")
    assert antwort.status_code == 200
    assert antwort.json()["playlists"] == {}


def test_playlist_anlegen_und_wiederfinden(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "CONFIG_PATH", tmp_path / "config.json")
    client.post("/api/playlists", json={"aktion": "anlegen", "name": "Warmup"})

    assert "Warmup" in client.get("/api/playlists").json()["playlists"]


def test_playlist_nimmt_einen_track_auf(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "CONFIG_PATH", tmp_path / "config.json")
    ordner = tmp_path / "Ein Song"
    ordner.mkdir()
    client.post("/api/playlists", json={"aktion": "anlegen", "name": "Warmup"})

    client.post("/api/playlists", json={"aktion": "hinzufuegen", "name": "Warmup",
                                        "folder": str(ordner)})

    assert client.get("/api/playlists").json()["playlists"]["Warmup"] == [str(ordner)]


def test_playlist_nimmt_denselben_track_nicht_zweimal(client, tmp_path, monkeypatch):
    """Sonst steht ein Song doppelt im Set."""
    monkeypatch.setattr(server, "CONFIG_PATH", tmp_path / "config.json")
    ordner = tmp_path / "Ein Song"; ordner.mkdir()
    client.post("/api/playlists", json={"aktion": "anlegen", "name": "W"})
    for _ in range(2):
        client.post("/api/playlists", json={"aktion": "hinzufuegen", "name": "W",
                                            "folder": str(ordner)})

    assert len(client.get("/api/playlists").json()["playlists"]["W"]) == 1


def test_playlist_track_entfernen(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "CONFIG_PATH", tmp_path / "config.json")
    ordner = tmp_path / "Ein Song"; ordner.mkdir()
    client.post("/api/playlists", json={"aktion": "anlegen", "name": "W"})
    client.post("/api/playlists", json={"aktion": "hinzufuegen", "name": "W", "folder": str(ordner)})

    client.post("/api/playlists", json={"aktion": "entfernen", "name": "W", "folder": str(ordner)})

    assert client.get("/api/playlists").json()["playlists"]["W"] == []


def test_playlist_loeschen(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "CONFIG_PATH", tmp_path / "config.json")
    client.post("/api/playlists", json={"aktion": "anlegen", "name": "W"})

    client.post("/api/playlists", json={"aktion": "loeschen", "name": "W"})

    assert client.get("/api/playlists").json()["playlists"] == {}


def test_playlist_name_kollidiert_nicht(client, tmp_path, monkeypatch):
    """Zweimal derselbe Name darf die vorhandene Liste nicht leeren."""
    monkeypatch.setattr(server, "CONFIG_PATH", tmp_path / "config.json")
    ordner = tmp_path / "S"; ordner.mkdir()
    client.post("/api/playlists", json={"aktion": "anlegen", "name": "W"})
    client.post("/api/playlists", json={"aktion": "hinzufuegen", "name": "W", "folder": str(ordner)})

    client.post("/api/playlists", json={"aktion": "anlegen", "name": "W"})

    assert client.get("/api/playlists").json()["playlists"]["W"] == [str(ordner)]


def test_playlist_verweist_nicht_auf_geloeschte_ordner(client, tmp_path, monkeypatch):
    """Wer einen Ergebnisordner im Finder löscht, soll ihn nicht weiter
    in seinen Listen finden."""
    monkeypatch.setattr(server, "CONFIG_PATH", tmp_path / "config.json")
    bleibt = tmp_path / "Bleibt"; bleibt.mkdir()
    weg = tmp_path / "Weg"; weg.mkdir()
    client.post("/api/playlists", json={"aktion": "anlegen", "name": "W"})
    for o in (bleibt, weg):
        client.post("/api/playlists", json={"aktion": "hinzufuegen", "name": "W", "folder": str(o)})

    weg.rmdir()

    assert client.get("/api/playlists").json()["playlists"]["W"] == [str(bleibt)]


def test_playlist_ohne_namen_wird_abgelehnt(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "CONFIG_PATH", tmp_path / "config.json")
    antwort = client.post("/api/playlists", json={"aktion": "anlegen", "name": "  "})
    assert antwort.status_code == 400


def test_playlist_uebersteht_einen_neustart(client, tmp_path, monkeypatch):
    """Die Zuordnung liegt in config.json, nicht im Arbeitsspeicher."""
    pfad = tmp_path / "config.json"
    monkeypatch.setattr(server, "CONFIG_PATH", pfad)
    ordner = tmp_path / "S"; ordner.mkdir()
    client.post("/api/playlists", json={"aktion": "anlegen", "name": "W"})
    client.post("/api/playlists", json={"aktion": "hinzufuegen", "name": "W", "folder": str(ordner)})

    import json
    assert json.loads(pfad.read_text())["playlists"]["W"] == [str(ordner)]


# --------------------------------------------------------------------------- #
# Notizen
# --------------------------------------------------------------------------- #
#
# Was man sich zu einem Track merkt, gehört zum Track: eine notes.txt im
# Ergebnisordner, nicht in einer zentralen Datei. Wer den Ordner kopiert,
# nimmt die Notiz mit; wer ihn löscht, wird sie los.

def test_notiz_ist_anfangs_leer(client, tmp_path):
    ordner = tmp_path / "Ein Song"; ordner.mkdir()
    antwort = client.get(f"/api/notes?path={ordner}")
    assert antwort.status_code == 200
    assert antwort.json()["text"] == ""


def test_notiz_schreiben_und_lesen(client, tmp_path):
    ordner = tmp_path / "Ein Song"; ordner.mkdir()
    client.post("/api/notes", json={"folder": str(ordner), "text": "Intro ab 1:04 kürzen"})

    assert client.get(f"/api/notes?path={ordner}").json()["text"] == "Intro ab 1:04 kürzen"


def test_notiz_liegt_im_ergebnisordner(client, tmp_path):
    ordner = tmp_path / "Ein Song"; ordner.mkdir()
    client.post("/api/notes", json={"folder": str(ordner), "text": "Hallo"})

    assert (ordner / "notes.txt").read_text(encoding="utf-8") == "Hallo"


def test_notiz_haelt_umlaute_und_zeilenumbrueche(client, tmp_path):
    """Eine Notiz ist Fließtext, kein Formularfeld."""
    ordner = tmp_path / "Ein Song"; ordner.mkdir()
    text = "Erste Zeile: Größe prüfen\nZweite Zeile – mit Gedankenstrich\n\nDritte"
    client.post("/api/notes", json={"folder": str(ordner), "text": text})

    assert client.get(f"/api/notes?path={ordner}").json()["text"] == text


def test_notiz_bleibt_unveraendert_erhalten(client, tmp_path):
    """Eingerückte Zeilen und Leerzeichen am Rand sind Teil der Notiz.
    Wer eine Liste einrückt, will sie eingerückt wiederfinden."""
    ordner = tmp_path / "Ein Song"; ordner.mkdir()
    text = "  eingerückt\n    tiefer\nnormal  "
    client.post("/api/notes", json={"folder": str(ordner), "text": text})

    assert client.get(f"/api/notes?path={ordner}").json()["text"] == text


def test_notiz_ueberschreibt_statt_anzuhaengen(client, tmp_path):
    ordner = tmp_path / "Ein Song"; ordner.mkdir()
    client.post("/api/notes", json={"folder": str(ordner), "text": "alt"})

    client.post("/api/notes", json={"folder": str(ordner), "text": "neu"})

    assert client.get(f"/api/notes?path={ordner}").json()["text"] == "neu"


def test_leere_notiz_entfernt_die_datei(client, tmp_path):
    """Ein leerer Text soll keine leere Datei hinterlassen."""
    ordner = tmp_path / "Ein Song"; ordner.mkdir()
    client.post("/api/notes", json={"folder": str(ordner), "text": "etwas"})

    client.post("/api/notes", json={"folder": str(ordner), "text": "   "})

    assert not (ordner / "notes.txt").exists()
    assert client.get(f"/api/notes?path={ordner}").json()["text"] == ""


def test_notiz_ausserhalb_des_zielordners_wird_abgelehnt(client, tmp_path):
    """Derselbe Schutz wie für alle Pfade: nichts außerhalb des Zielordners."""
    antwort = client.post("/api/notes", json={"folder": "/etc", "text": "nein"})
    assert antwort.status_code >= 400


# --------------------------------------------------------------------------- #
# Sammelanalyse eines Ordners
# --------------------------------------------------------------------------- #
#
# Wer ein Genre-Profil bauen will, braucht 30 und mehr Tracks. Die einzeln
# hineinzuziehen ist keine Arbeitsweise – der Ordner wandert als Ganzes in
# die Warteschlange.

def _wav(pfad, sekunden=1.0):
    import numpy as np, soundfile as sf
    sf.write(str(pfad), np.zeros(int(44100 * sekunden), dtype="float32"), 44100)
    return pfad


def test_sammelanalyse_reiht_jede_audiodatei_ein(client, tmp_path, monkeypatch):
    quelle = tmp_path / "Sammlung"; quelle.mkdir()
    for name in ("a.mp3", "b.wav", "c.flac"):
        _wav(quelle / name)
    monkeypatch.setattr(server.WORK, "put", lambda _id: None)

    antwort = client.post("/api/analyze-folder", json={"folder": str(quelle)})

    assert antwort.status_code == 200
    assert antwort.json()["eingereiht"] == 3


def test_sammelanalyse_ignoriert_fremde_dateien(client, tmp_path, monkeypatch):
    """Cover-Bilder, Playlisten und Notizen liegen in Musikordnern herum."""
    quelle = tmp_path / "Sammlung"; quelle.mkdir()
    _wav(quelle / "song.mp3")
    (quelle / "cover.jpg").write_bytes(b"x")
    (quelle / "liste.m3u8").write_text("x")
    (quelle / "notiz.txt").write_text("x")
    monkeypatch.setattr(server.WORK, "put", lambda _id: None)

    assert client.post("/api/analyze-folder", json={"folder": str(quelle)}).json()["eingereiht"] == 1


def test_sammelanalyse_ueberspringt_bereits_analysierte(client, tmp_path, monkeypatch):
    """Ein zweiter Lauf über denselben Ordner soll nicht alles neu rechnen –
    bei 50 Tracks wären das 20 Minuten für nichts."""
    quelle = tmp_path / "Sammlung"; quelle.mkdir()
    _wav(quelle / "schon da.mp3"); _wav(quelle / "neu.mp3")
    fertig = tmp_path / "schon da"; fertig.mkdir()
    (fertig / "analysis.json").write_text('{"bpm": 124}')
    monkeypatch.setattr(server, "output_root", lambda: tmp_path)
    monkeypatch.setattr(server.WORK, "put", lambda _id: None)

    antwort = client.post("/api/analyze-folder", json={"folder": str(quelle)})

    assert antwort.json()["eingereiht"] == 1
    assert antwort.json()["uebersprungen"] == 1


def test_sammelanalyse_kann_alles_neu_rechnen(client, tmp_path, monkeypatch):
    """Wer die Analyse verbessert hat, will sie auf alles anwenden."""
    quelle = tmp_path / "Sammlung"; quelle.mkdir()
    _wav(quelle / "schon da.mp3")
    fertig = tmp_path / "schon da"; fertig.mkdir()
    (fertig / "analysis.json").write_text("{}")
    monkeypatch.setattr(server, "output_root", lambda: tmp_path)
    monkeypatch.setattr(server.WORK, "put", lambda _id: None)

    antwort = client.post("/api/analyze-folder", json={"folder": str(quelle), "neu": True})

    assert antwort.json()["eingereiht"] == 1


def test_sammelanalyse_geht_in_unterordner(client, tmp_path, monkeypatch):
    """Sammlungen sind nach Genre sortiert – eine Ebene tiefer reicht."""
    quelle = tmp_path / "Sammlung"; (quelle / "Techno").mkdir(parents=True)
    _wav(quelle / "oben.mp3"); _wav(quelle / "Techno" / "unten.mp3")
    monkeypatch.setattr(server.WORK, "put", lambda _id: None)

    assert client.post("/api/analyze-folder", json={"folder": str(quelle)}).json()["eingereiht"] == 2


def test_sammelanalyse_leerer_ordner(client, tmp_path, monkeypatch):
    leer = tmp_path / "Leer"; leer.mkdir()
    monkeypatch.setattr(server.WORK, "put", lambda _id: None)

    antwort = client.post("/api/analyze-folder", json={"folder": str(leer)})

    assert antwort.status_code == 400
    assert "keine" in antwort.json()["detail"].lower()


def test_sammelanalyse_trennt_keine_stems(client, tmp_path, monkeypatch):
    """Analysieren dauert Sekunden, Trennen Minuten. Für ein Profil braucht
    es nur die Analyse."""
    quelle = tmp_path / "S"; quelle.mkdir(); _wav(quelle / "a.mp3")
    eingereiht = []
    monkeypatch.setattr(server.WORK, "put", lambda job_id: eingereiht.append(job_id))

    client.post("/api/analyze-folder", json={"folder": str(quelle)})

    with server.JOBS_LOCK:
        arten = {server.JOBS[i].kind for i in eingereiht}
    assert arten == {"analyze"}


def test_sammelanalyse_loescht_die_quelldateien_nicht(client, tmp_path, monkeypatch):
    """Der wichtigste Test dieser Funktion.

    Bei hochgeladenen Dateien räumt der Worker die Zwischendatei weg. Beim
    Sammellauf zeigt die Quelle aber auf die Musiksammlung des Nutzers –
    dort etwas zu löschen wäre unverzeihlich.
    """
    quelle = tmp_path / "Sammlung"; quelle.mkdir()
    datei = _wav(quelle / "kostbar.mp3")
    monkeypatch.setattr(server, "output_root", lambda: tmp_path / "ziel")
    (tmp_path / "ziel").mkdir()

    def falsche_analyse(source, output_root, **kwargs):
        ordner = output_root / "kostbar"
        ordner.mkdir(exist_ok=True)
        return server.engine.JobResult(folder=ordner, seconds=0.1, device="cpu")

    monkeypatch.setattr(server.engine, "analyze_only", falsche_analyse)
    # Die Warteschlange kann von anderen Tests gefüllt sein – deshalb den
    # eigenen Job über seine ID holen, statt blind zu entnehmen.
    antwort = client.post("/api/analyze-folder", json={"folder": str(quelle)})
    job_id = antwort.json()["ids"][0]
    with server.JOBS_LOCK:
        job = server.JOBS[job_id]

    server.run_job(job)

    assert datei.exists(), "Die Quelldatei des Nutzers wurde gelöscht"


# --------------------------------------------------------------------------- #
# Profil einer Playlist
# --------------------------------------------------------------------------- #

def _analyse_ordner(wurzel, name, **werte):
    import json
    ordner = wurzel / name
    ordner.mkdir(parents=True, exist_ok=True)
    daten = {"bpm": 124.0, "energy": 7, "danceability": 8, "camelot": "8A",
             "key_mode": "minor", "seconds_analyzed": 380.0, "downbeats": 190}
    daten.update(werte)
    (ordner / "analysis.json").write_text(json.dumps(daten))
    return ordner


def test_profil_einer_playlist(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "output_root", lambda: tmp_path)
    monkeypatch.setattr(server, "CONFIG_PATH", tmp_path / "config.json")
    a = _analyse_ordner(tmp_path, "Song A", bpm=124)
    b = _analyse_ordner(tmp_path, "Song B", bpm=126)
    _analyse_ordner(tmp_path, "Song C", bpm=90)      # nicht in der Playlist
    client.post("/api/playlists", json={"aktion": "anlegen", "name": "Set"})
    for o in (a, b):
        client.post("/api/playlists", json={"aktion": "hinzufuegen", "name": "Set", "folder": str(o)})

    antwort = client.get("/api/profil?playlist=Set")

    assert antwort.status_code == 200
    p = antwort.json()["profil"]
    assert p["anzahl"] == 2
    assert p["tempo"]["median"] == 125


def test_profil_ueber_alle_analysen(client, tmp_path, monkeypatch):
    """Ohne Playlist: das Profil des ganzen Zielordners."""
    monkeypatch.setattr(server, "output_root", lambda: tmp_path)
    for i, bpm in enumerate((120, 124, 128)):
        _analyse_ordner(tmp_path, f"Song {i}", bpm=bpm)

    p = client.get("/api/profil").json()["profil"]

    assert p["anzahl"] == 3
    assert p["tempo"]["median"] == 124


def test_profil_kennzeichnet_kleine_gruppen(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "output_root", lambda: tmp_path)
    _analyse_ordner(tmp_path, "Einziger")

    assert client.get("/api/profil").json()["profil"]["belastbar"] is False


def test_profil_unbekannte_playlist(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "output_root", lambda: tmp_path)
    monkeypatch.setattr(server, "CONFIG_PATH", tmp_path / "config.json")

    assert client.get("/api/profil?playlist=Gibtsnicht").status_code == 404


def test_profil_ueberspringt_kaputte_analysen(client, tmp_path, monkeypatch):
    """Eine unlesbare Datei darf das Profil nicht verhindern."""
    monkeypatch.setattr(server, "output_root", lambda: tmp_path)
    _analyse_ordner(tmp_path, "Gut")
    kaputt = tmp_path / "Kaputt"; kaputt.mkdir()
    (kaputt / "analysis.json").write_text("{kein json")

    assert client.get("/api/profil").json()["profil"]["anzahl"] == 1


def test_folder_reicht_die_abschnitte_durch(client, tmp_path, monkeypatch):
    """Listen werden in ihre Länge umgewandelt – bei beats und downbeats
    ist das richtig, denn niemand braucht 800 Zeitpunkte in der Oberfläche.

    Für die Abschnitte gilt das nicht: Ohne sie kann die Arrangement-
    Ansicht nichts zeichnen, und "7" sagt nichts über den Aufbau.
    """
    import json
    ordner = tmp_path / "Song"; ordner.mkdir()
    (ordner / "analysis.json").write_text(json.dumps({
        "bpm": 124, "beats": [0.0, 0.5, 1.0], "downbeats": [0.0, 2.0],
        "segments": [{"art": "Intro", "takte": 16, "takt_von": 1},
                     {"art": "Drop", "takte": 32, "takt_von": 17}],
    }))
    (ordner / "original.wav").write_bytes(b"")
    monkeypatch.setattr(server, "output_root", lambda: tmp_path)

    antwort = client.get(f"/api/folder?path={ordner}")
    a = antwort.json()["analysis"]

    assert isinstance(a["segments"], list), "Die Abschnitte müssen Abschnitte bleiben"
    assert a["segments"][1]["art"] == "Drop"
    # Beats und Downbeats dagegen weiterhin als Anzahl.
    assert a["beats"] == 3
    assert a["downbeats"] == 2
