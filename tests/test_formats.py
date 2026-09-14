"""Zeitstempel, Dateinamen und Sidecar-Dateien.

Kleine Helfer, die still falsch sein können: ein verrutschter Zeitstempel
fällt erst auf, wenn jemand die LRC-Datei im Player öffnet.
"""

import json

import numpy as np
import pytest
import soundfile as sf

import analysis
import postprocess
from conftest import SR, beat_grid


@pytest.mark.parametrize("sekunden, erwartet", [
    (0.0, "[00:00.00]"),
    (76.12, "[01:16.12]"),
    (59.99, "[00:59.99]"),
    (600.0, "[10:00.00]"),
])
def test_lrc_zeitstempel(sekunden, erwartet):
    """Karaoke-Format: [mm:ss.hh]."""
    assert postprocess._fmt_lrc(sekunden) == erwartet


@pytest.mark.parametrize("sekunden, erwartet", [
    (0.0, "00:00:00,000"),
    (76.12, "00:01:16,120"),
    (3661.5, "01:01:01,500"),
])
def test_srt_zeitstempel(sekunden, erwartet):
    """Untertitel-Format: hh:mm:ss,mmm mit Komma vor den Millisekunden."""
    assert postprocess._fmt_srt(sekunden) == erwartet


@pytest.mark.parametrize("dateiname, erwartet", [
    ("vocals.wav", "vocals"),
    ("drums.flac", "drums"),
    ("vocals - 124bpm - 6A.wav", "vocals"),
    ("bass_final.mp3", "bass"),
    ("other (2).wav", "other"),
])
def test_stemname_aus_dateiname(dateiname, erwartet):
    """Der Stemname muss auch nach dem Umbenennen erkannt werden.

    Mit aktivierter Umbenennung heißt die Datei `vocals - 124bpm - 6A.wav`;
    der Mixer und der Tag-Editor brauchen daraus weiterhin `vocals`.
    """
    from pathlib import Path

    assert postprocess.stem_name(Path(dateiname)) == erwartet


def test_stems_ohne_original(tmp_path):
    """`original.wav` ist der Mainmix, kein Stem – er darf nicht mitgezählt werden."""
    for name in ("vocals.wav", "drums.wav", "original.wav"):
        sf.write(tmp_path / name, np.zeros(100), SR)
    namen = [p.name for p in postprocess.stem_files(tmp_path)]
    assert "original.wav" not in namen
    assert set(namen) == {"vocals.wav", "drums.wav"}


def test_stems_in_musikalischer_reihenfolge(tmp_path):
    """Vocals zuerst, dann Drums und Bass – nicht alphabetisch."""
    for name in ("other.wav", "bass.wav", "vocals.wav", "drums.wav"):
        sf.write(tmp_path / name, np.zeros(100), SR)
    namen = [postprocess.stem_name(p) for p in postprocess.stem_files(tmp_path)]
    assert namen == ["vocals", "drums", "bass", "other"]


def test_akkorde_pro_takt():
    """Je zwei Downbeats ein Akkord."""
    chroma = np.zeros((12, 200))
    chroma[[7, 10, 2], :] = 1.0          # G, A#, D – ein Gm-Dreiklang
    downbeats = [0.0, 2.0, 4.0, 6.0]
    chords = analysis._chords(chroma, 2048, downbeats, [], 8.0)
    assert len(chords) >= len(downbeats) - 1
    assert all(c["chord"] == "Gm" for c in chords)
    assert [c["bar"] for c in chords][:3] == [1, 2, 3]


def test_stille_takte_werden_als_nc_markiert():
    """Wo nichts klingt, steht N.C. statt eines geratenen Akkords."""
    chroma = np.zeros((12, 200))
    chords = analysis._chords(chroma, 2048, [0.0, 2.0, 4.0], [], 6.0)
    assert all(c["chord"] == "N.C." for c in chords)


def test_akkordblatt_bricht_um():
    chords = [{"chord": name} for name in ["Gm", "C", "F", "D#", "Gm", "C"]]
    zeilen = analysis._chord_sheet(chords, per_line=4).splitlines()
    assert len(zeilen) == 2
    assert zeilen[0].count("|") == 5


def test_leeres_akkordblatt():
    assert analysis._chord_sheet([]) == ""


def test_klickspur_ist_gueltiges_midi(tmp_path):
    """Header und Track-Chunk müssen stimmen, sonst öffnet es keine DAW."""
    path = tmp_path / "click.mid"
    beats = list(beat_grid(124.0, seconds=20.0))
    analysis.write_click_midi(path, beats, beats[::4], 124.0)
    data = path.read_bytes()
    assert data[:4] == b"MThd"
    assert b"MTrk" in data
    assert data.endswith(b"\xff\x2f\x00")


def test_sidecars_werden_geschrieben(tmp_path):
    """analysis.json, beats.json und click.mid landen neben den Stems."""
    result = analysis.Analysis(
        bpm=124.0, key="G minor", key_de="g-Moll", camelot="6A",
        beats=list(beat_grid(124.0, seconds=20.0)),
        downbeats=list(beat_grid(124.0, seconds=20.0))[::4],
        chords=[{"bar": 1, "start": 0.0, "end": 2.0, "chord": "Gm", "confidence": 0.5}],
        chord_sheet="| Gm |",
    )
    analysis.write_sidecars(tmp_path, result)

    gespeichert = json.loads((tmp_path / "analysis.json").read_text())
    assert gespeichert["bpm"] == 124.0
    assert gespeichert["camelot"] == "6A"

    beats = json.loads((tmp_path / "beats.json").read_text())
    assert beats["bpm"] == 124.0
    assert len(beats["beats"]) == len(result.beats)

    assert (tmp_path / "click.mid").exists()
    assert "g-Moll" in (tmp_path / "chords.txt").read_text()


def test_zusammenfassung_ohne_listen():
    """Für Ereignisse und Tags zählen nur die Längen, nicht die Rohdaten."""
    result = analysis.Analysis(bpm=124.0, beats=[0.0, 0.5, 1.0], downbeats=[0.0], chords=[{"bar": 1}])
    summary = result.summary()
    assert summary["beats"] == 3
    assert summary["downbeats"] == 1
    assert summary["chords"] == 1
    assert summary["bpm"] == 124.0
