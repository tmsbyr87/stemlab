"""Tags und Coverbilder in WAV, FLAC und MP3.

Die drei Formate speichern Metadaten unterschiedlich: WAV und MP3 als ID3,
FLAC als Vorbis-Kommentare mit eigenem Bildblock. Der Tag-Editor muss in allen
dreien lesen, schreiben, leeren und Cover setzen können.
"""

import numpy as np
import pytest
import soundfile as sf

import postprocess
from conftest import SR, tiny_png

FIELDS = {
    "title": "We Are (Remix)", "artist": "AM I RIGHT", "album": "Testalbum",
    "label": "Testlabel", "remixer": "Remixer", "composer": "Komponist",
    "grouping": "Gruppe", "genre": "Techno", "year": "2026",
    "key": "Gm", "bpm": "124", "comment": "06A - 5",
}


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


# --------------------------------------------------------------------------- #
# tag_folder: der Weg, den jede fertige Trennung nimmt
# --------------------------------------------------------------------------- #

ANALYSE = {"bpm": 124.0, "key_id3": "Gm", "camelot": "6A"}


@pytest.fixture
def ordner(tmp_path):
    """Ergebnisordner mit zwei Stems und dem Mainmix."""
    for name in ("vocals.wav", "drums.wav", "original.wav"):
        sf.write(tmp_path / name, np.zeros(SR), SR)
    return tmp_path


def test_tag_folder_schreibt_in_alle_stems(ordner):
    dateien = postprocess.tag_folder(ordner, ANALYSE, "Testsong", False, lambda _m: None)
    assert len(dateien) == 2                       # original.wav ist kein Stem
    for path in dateien:
        tags = postprocess.read_tags(path)
        assert tags["bpm"] == "124"
        assert tags["key"] == "Gm"


def test_tag_folder_laesst_den_mainmix_in_ruhe(ordner):
    """`original.wav` bekommt seine Tags über den Editor, nicht automatisch."""
    postprocess.tag_folder(ordner, ANALYSE, "Testsong", False, lambda _m: None)
    assert (ordner / "original.wav").exists()
    assert postprocess.read_tags(ordner / "original.wav")["bpm"] == ""


def test_umbenennen_mit_tempo_und_camelot(ordner):
    dateien = postprocess.tag_folder(ordner, ANALYSE, "Testsong", True, lambda _m: None)
    namen = sorted(p.name for p in dateien)
    assert namen == ["drums - 124bpm - 6A.wav", "vocals - 124bpm - 6A.wav"]


def test_zweites_umbenennen_haengt_nichts_an(ordner):
    """Ein zweiter Lauf darf nicht 'vocals - 124bpm - 6A - 124bpm - 6A.wav' erzeugen."""
    postprocess.tag_folder(ordner, ANALYSE, "Testsong", True, lambda _m: None)
    dateien = postprocess.tag_folder(ordner, ANALYSE, "Testsong", True, lambda _m: None)
    assert all(p.name.count("bpm") == 1 for p in dateien)


def test_ohne_camelot_nur_das_tempo(ordner):
    dateien = postprocess.tag_folder(ordner, {"bpm": 124.0, "key_id3": "Gm", "camelot": "–"},
                                     "Testsong", True, lambda _m: None)
    assert sorted(p.name for p in dateien) == ["drums - 124bpm.wav", "vocals - 124bpm.wav"]


def test_ohne_tempo_wird_nicht_umbenannt(ordner):
    """Ohne erkanntes Tempo ergäbe 'vocals - 0bpm.wav' keinen Sinn."""
    dateien = postprocess.tag_folder(ordner, {"bpm": 0, "key_id3": "", "camelot": ""},
                                     "Testsong", True, lambda _m: None)
    assert sorted(p.name for p in dateien) == ["drums.wav", "vocals.wav"]


def test_leerer_ordner_meldet_nichts(tmp_path):
    gesagt = []
    assert postprocess.tag_folder(tmp_path, ANALYSE, "Leer", False, gesagt.append) == []
    assert gesagt == []


# --------------------------------------------------------------------------- #
# tag_file je Format
# --------------------------------------------------------------------------- #

def _tonspur(pfad, sekunden=0.3, sr=44100):
    sf.write(str(pfad), np.zeros(int(sekunden * sr), dtype="float32"), sr)
    return pfad


@pytest.mark.parametrize("endung", [".wav", ".flac"])
def test_tag_file_schreibt_in_beide_formate(tmp_path, endung):
    """WAV geht über ID3, FLAC über Vorbis-Kommentare – zwei ganz
    verschiedene Wege durch dieselbe Funktion."""
    pfad = _tonspur(tmp_path / f"vocals{endung}")

    postprocess.tag_file(pfad, 128.0, "8A", "8A", "Mein Song", "Vocals")

    tags = postprocess.read_tags(pfad)
    assert tags["title"] == "Mein Song – Vocals"
    assert tags["album"] == "Mein Song"
    assert tags["bpm"] == "128"
    assert tags["key"] == "8A"


@pytest.mark.parametrize("endung", [".wav", ".flac"])
def test_tag_file_rundet_das_tempo(tmp_path, endung):
    pfad = _tonspur(tmp_path / f"vocals{endung}")

    postprocess.tag_file(pfad, 127.6, "8A", "8A", "Song", "Vocals")

    assert postprocess.read_tags(pfad)["bpm"] == "128"


@pytest.mark.parametrize("endung", [".wav", ".flac"])
def test_tag_file_ohne_tempo_schreibt_kein_bpm(tmp_path, endung):
    pfad = _tonspur(tmp_path / f"vocals{endung}")

    postprocess.tag_file(pfad, 0, "8A", "8A", "Song", "Vocals")

    assert not postprocess.read_tags(pfad).get("bpm")


@pytest.mark.parametrize("endung", [".wav", ".flac"])
def test_tag_file_ohne_tonart_schreibt_keinen_key(tmp_path, endung):
    pfad = _tonspur(tmp_path / f"vocals{endung}")

    postprocess.tag_file(pfad, 128.0, "", "", "Song", "Vocals")

    assert not postprocess.read_tags(pfad).get("key")


@pytest.mark.parametrize("endung", [".wav", ".flac"])
def test_tag_file_schreibt_camelot_in_den_kommentar(tmp_path, endung):
    pfad = _tonspur(tmp_path / f"vocals{endung}")

    postprocess.tag_file(pfad, 128.0, "Am", "8A", "Song", "Vocals")

    assert "Camelot 8A" in postprocess.read_tags(pfad)["comment"]


@pytest.mark.parametrize("endung", [".wav", ".flac"])
def test_tag_file_ohne_camelot_kein_kommentar(tmp_path, endung):
    """"–" ist der Platzhalter für "unbekannt" und gehört nicht in die Datei."""
    pfad = _tonspur(tmp_path / f"vocals{endung}")

    postprocess.tag_file(pfad, 128.0, "Am", "–", "Song", "Vocals")

    assert not postprocess.read_tags(pfad).get("comment")


def test_tag_file_auf_bereits_getaggter_datei(tmp_path):
    """Ein zweiter Lauf überschreibt, statt die Tags zu verdoppeln."""
    pfad = _tonspur(tmp_path / "vocals.wav")
    postprocess.tag_file(pfad, 100.0, "1A", "1A", "Alt", "Vocals")

    postprocess.tag_file(pfad, 128.0, "8A", "8A", "Neu", "Vocals")

    tags = postprocess.read_tags(pfad)
    assert tags["bpm"] == "128"
    assert tags["title"] == "Neu – Vocals"
