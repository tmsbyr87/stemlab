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
