"""Namen, Endungen und Modellauflösung in engine.py.

Diese Helfer sind rein: keine Modelle, kein ffmpeg, kein Netz. Trotzdem
entscheiden sie, wie Ordner heißen, welche Datei als Video gilt und welches
Trennmodell am Ende läuft. Ein Fehler hier fällt erst auf, wenn eine Datei
unter falschem Namen im Ausgabeordner liegt oder die Trennung das falsche
Modell zieht.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import engine


# --------------------------------------------------------------------------- #
# safe_song_name
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("eingabe, erwartet", [
    # Bekannte Endungen fallen weg, unbekannte bleiben Teil des Namens.
    ("Song.mp3", "Song"),
    ("Song.MP3", "Song"),
    ("Clip.mov", "Clip"),
    ("Song.txt", "Song.txt"),
    ("Ohne Endung", "Ohne Endung"),
    # Nur die letzte Endung geht, der Rest ist Name.
    ("Mix.v2.wav", "Mix.v2"),
])
def test_safe_song_name_endungen(eingabe, erwartet):
    assert engine.safe_song_name(eingabe) == erwartet


@pytest.mark.parametrize("eingabe, erwartet", [
    # Ein Slash ist bereits ein Pfadtrenner: Path zerlegt den Namen, bevor
    # die Ersetzung greift. Übrig bleibt das letzte Segment.
    ("AC/DC - Thunder.mp3", "DC - Thunder"),
    ("Was?.mp3", "Was"),
    ('Titel: "Schön".flac', "Titel- -Schön"),
    ("a\\b|c*d.wav", "a-b-c-d"),
    # Mehrere verbotene Zeichen am Stück werden zu einem Bindestrich.
    ("a<<>>b.wav", "a-b"),
])
def test_safe_song_name_ersetzt_verbotene_zeichen(eingabe, erwartet):
    """Zeichen, an denen das Dateisystem oder der Finder hängenbleibt."""
    assert engine.safe_song_name(eingabe) == erwartet


def test_safe_song_name_entfernt_steuerzeichen():
    assert engine.safe_song_name("Lied\x00\x1fName.wav") == "Lied-Name"


@pytest.mark.parametrize("eingabe", ["", ".", "..", "   ", "...", "- . -.wav"])
def test_safe_song_name_faellt_auf_output_zurueck(eingabe):
    """Ein leerer Ordnername wäre unbrauchbar, also gibt es einen Rückfall."""
    assert engine.safe_song_name(eingabe) == "output"


def test_safe_song_name_kuerzt_auf_150_zeichen():
    """Sonst reißt der Pfad das Limit des Dateisystems."""
    name = engine.safe_song_name("x" * 400 + ".wav")
    assert len(name) == 150
    assert set(name) == {"x"}


def test_safe_song_name_behaelt_pfadlose_basis():
    """Ein durchgereichter Pfad darf keine Ordner in den Namen schleppen."""
    assert "/" not in engine.safe_song_name("/tmp/Ordner/Song.mp3")


# --------------------------------------------------------------------------- #
# stem_label
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("dateiname, erwartet", [
    ("Song_(Vocals)_model.wav", "Vocals"),
    ("Song_(Instrumental)_mel_band.flac", "Instrumental"),
    # Die Modellnamen stecken selbst in Klammern – die letzte gewinnt nicht
    # immer das, was man erwartet, deshalb ist die Regel hier festgenagelt.
    ("(Alt)_Song_(Drums).wav", "Drums"),
    ("Song_( Bass ).wav", "Bass"),
])
def test_stem_label_nimmt_letzte_klammer(dateiname, erwartet):
    assert engine.stem_label(dateiname) == erwartet


@pytest.mark.parametrize("dateiname", ["Song.wav", "Song_Vocals.wav", "", "Song_(unvollstaendig.wav"])
def test_stem_label_ohne_klammer_ist_none(dateiname):
    assert engine.stem_label(dateiname) is None


def test_stem_label_leere_klammer():
    """`()` ist kein Label – der Inhalt muss nichtleer sein."""
    assert engine.stem_label("Song_().wav") is None


# --------------------------------------------------------------------------- #
# is_video
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ["a.mp4", "a.MOV", "a.mkv", "a.webm", "a.Avi", "a.3gp"])
def test_is_video_erkennt_videoendungen(name):
    assert engine.is_video(Path(name)) is True


@pytest.mark.parametrize("name", ["a.wav", "a.mp3", "a.flac", "a.txt", "a", "a.mp4.wav"])
def test_is_video_lehnt_alles_andere_ab(name):
    assert engine.is_video(Path(name)) is False


def test_video_und_audio_endungen_ueberschneiden_sich_nicht():
    """Sonst wäre unklar, ob eine Datei die Tonspur-Extraktion braucht."""
    assert not set(engine.AUDIO_SUFFIXES) & set(engine.VIDEO_SUFFIXES)


# --------------------------------------------------------------------------- #
# first_available
# --------------------------------------------------------------------------- #

@pytest.fixture
def modellliste(monkeypatch):
    """Setzt die bekannte Modellliste und stellt sie danach wieder her."""
    def setzen(namen):
        monkeypatch.setattr(engine, "_available_models", set(namen))
    return setzen


def test_first_available_nimmt_ersten_bekannten(modellliste):
    modellliste({"b.ckpt", "c.ckpt"})
    assert engine.first_available(["a.ckpt", "b.ckpt", "c.ckpt"]) == "b.ckpt"


def test_first_available_haelt_die_reihenfolge_der_kandidaten(modellliste):
    """Die Kandidatenliste ist nach Qualität sortiert, nicht die Modellliste."""
    modellliste({"c.ckpt", "b.ckpt"})
    assert engine.first_available(["c.ckpt", "b.ckpt"]) == "c.ckpt"


def test_first_available_faellt_auf_ersten_kandidaten_zurueck(modellliste):
    """Kein Kandidat bekannt: lieber der erste als gar keiner."""
    modellliste({"x.ckpt"})
    assert engine.first_available(["a.ckpt", "b.ckpt"]) == "a.ckpt"


def test_first_available_ohne_modellliste(modellliste):
    """Vor dem ersten Katalogabgleich ist die Liste leer – ohne Netz der Normalfall."""
    modellliste(set())
    assert engine.first_available(["a.ckpt", "b.ckpt"]) == "a.ckpt"


# --------------------------------------------------------------------------- #
# _pretty_stem und _holds_stems
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("dateiname, erwartet", [
    ("Song_(Vocals).wav", "vocals"),
    ("Song_(Vocal).wav", "vocals"),
    ("Song_(No Vocals).wav", "instrumental"),
    ("Song_(Instrumental).wav", "instrumental"),
    ("Song_(Drums).wav", "drums"),
    # Unbekanntes Label: kleingeschrieben und auf Dateinamen-taugliche
    # Zeichen reduziert, statt es wegzuwerfen.
    ("Song_(Lead Guitar 2).wav", "lead_guitar_2"),
    ("Song_(Strings & Brass).wav", "strings_brass"),
])
def test_pretty_stem(dateiname, erwartet):
    assert engine._pretty_stem(dateiname) == erwartet


@pytest.mark.parametrize("dateiname", ["Song.wav", "Song_().wav", "Song_(---).wav"])
def test_pretty_stem_ohne_brauchbares_label_ist_none(dateiname):
    assert engine._pretty_stem(dateiname) is None


def test_holds_stems_erkennt_audiodateien(tmp_path):
    (tmp_path / "Song_(Vocals).wav").write_bytes(b"")
    assert engine._holds_stems(tmp_path) is True


@pytest.mark.parametrize("name", ["egal.mp3", "egal.flac"])
def test_holds_stems_kennt_die_ausgabeformate(tmp_path, name):
    (tmp_path / name).write_bytes(b"")
    assert engine._holds_stems(tmp_path) is True


def test_holds_stems_ohne_audio_ist_false(tmp_path):
    (tmp_path / "analysis.json").write_bytes(b"{}")
    (tmp_path / "notiz.txt").write_bytes(b"")
    assert engine._holds_stems(tmp_path) is False


def test_holds_stems_ignoriert_unterordner(tmp_path):
    """Ein Ordner namens "x.wav" ist keine Stemdatei."""
    (tmp_path / "unterordner.wav").mkdir()
    assert engine._holds_stems(tmp_path) is False


def test_holds_stems_auf_nicht_vorhandenem_pfad(tmp_path):
    assert engine._holds_stems(tmp_path / "gibtsnicht") is False
    assert engine._holds_stems(tmp_path / "datei.wav") is False
