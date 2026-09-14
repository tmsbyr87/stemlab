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
from conftest import SR  # noqa: E402


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


@pytest.mark.parametrize("pfad", [
    "/etc/passwd",
    "/etc/hosts",
    "~/.ssh/id_rsa",
])
def test_pfade_ausserhalb_werden_abgewiesen(client, pfad):
    """Nur Dateien im Zielordner dürfen heraus."""
    response = client.get("/api/tags", params={"path": pfad})
    assert response.status_code in (403, 404)


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
    from test_tags import tiny_png

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
    from test_tags import tiny_png

    client.post("/api/cover", data={"path": str(track)},
                files={"file": ("c.png", tiny_png(), "image/png")})
    response = client.post("/api/cover/delete", json={"path": str(track)})
    assert response.status_code == 200
    assert response.json()["has_cover"] is False


def test_fremder_host_wird_abgewiesen(client):
    """Schutz gegen DNS-Rebinding: nur lokale Hostnamen sind erlaubt."""
    response = client.get("/api/ping", headers={"Host": "boese.example.com"})
    assert response.status_code == 403
