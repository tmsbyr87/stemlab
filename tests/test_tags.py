"""Tags und Coverbilder in WAV, FLAC und MP3.

Die drei Formate speichern Metadaten unterschiedlich: WAV und MP3 als ID3,
FLAC als Vorbis-Kommentare mit eigenem Bildblock. Der Tag-Editor muss in allen
dreien lesen, schreiben, leeren und Cover setzen können.
"""

import struct
import zlib

import numpy as np
import pytest
import soundfile as sf

import postprocess
from conftest import SR

FIELDS = {
    "title": "We Are (Remix)", "artist": "AM I RIGHT", "album": "Testalbum",
    "label": "Testlabel", "remixer": "Remixer", "composer": "Komponist",
    "grouping": "Gruppe", "genre": "Techno", "year": "2026",
    "key": "Gm", "bpm": "124", "comment": "06A - 5",
}


def tiny_png() -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00")) + chunk(b"IEND", b""))


@pytest.fixture(params=["wav", "flac"])
def audio(request, tmp_path):
    """Eine kurze Datei je Format. MP3 bleibt außen vor – das Schreiben
    bräuchte einen Encoder, den nicht jedes System mitbringt."""
    path = tmp_path / f"probe.{request.param}"
    sf.write(path, np.zeros(SR), SR)
    return path


def test_tags_ueberleben_das_schreiben(audio):
    postprocess.write_tags(audio, FIELDS)
    read = postprocess.read_tags(audio)
    for key, value in FIELDS.items():
        assert read[key] == value, f"{key} in {audio.suffix}"


def test_leeres_feld_loescht_den_tag(audio):
    postprocess.write_tags(audio, FIELDS)
    postprocess.write_tags(audio, {"label": ""})
    assert postprocess.read_tags(audio)["label"] == ""


def test_andere_felder_bleiben_beim_teilupdate(audio):
    postprocess.write_tags(audio, FIELDS)
    postprocess.write_tags(audio, {"title": "Neuer Titel"})
    read = postprocess.read_tags(audio)
    assert read["title"] == "Neuer Titel"
    assert read["artist"] == FIELDS["artist"]


def test_datei_ohne_tags_liefert_leere_felder(audio):
    read = postprocess.read_tags(audio)
    assert read["title"] == ""
    assert read["has_cover"] is False


def test_cover_schreiben_und_lesen(audio):
    png = tiny_png()
    postprocess.write_cover(audio, png, "image/png")
    read = postprocess.read_tags(audio)
    assert read["has_cover"] is True
    assert read["cover_mime"] == "image/png"
    data, mime = postprocess.read_cover(audio)
    assert data == png


def test_cover_entfernen(audio):
    postprocess.write_cover(audio, tiny_png(), "image/png")
    postprocess.remove_cover(audio)
    assert postprocess.read_tags(audio)["has_cover"] is False
    assert postprocess.read_cover(audio) is None


def test_cover_ersetzt_das_alte(audio):
    """Zwei Cover hintereinander dürfen nicht beide in der Datei landen."""
    postprocess.write_cover(audio, tiny_png(), "image/png")
    postprocess.write_cover(audio, tiny_png(), "image/png")
    postprocess.remove_cover(audio)
    assert postprocess.read_cover(audio) is None


def test_fremde_dateitypen_werden_abgelehnt(audio):
    with pytest.raises(ValueError):
        postprocess.write_cover(audio, b"GIF89a", "image/gif")


def test_zu_grosses_cover_wird_abgelehnt(audio):
    with pytest.raises(ValueError):
        postprocess.write_cover(audio, b"x" * (postprocess.MAX_COVER_BYTES + 1), "image/jpeg")


def test_unbekannte_felder_werden_ignoriert(audio):
    """Ein Feld, das es nicht gibt, darf nicht zum Absturz führen."""
    postprocess.write_tags(audio, {"gibtesnicht": "wert", "title": "Titel"})
    assert postprocess.read_tags(audio)["title"] == "Titel"


def test_umlaute_und_sonderzeichen(audio):
    postprocess.write_tags(audio, {"title": "Grüße aus Köln – ößä", "artist": "Æther"})
    read = postprocess.read_tags(audio)
    assert read["title"] == "Grüße aus Köln – ößä"
    assert read["artist"] == "Æther"
