"""Die ffmpeg-Grenze in engine.py.

Geprüft wird, was StemLab selbst tut: welche Argumente es baut und wie es
die Rückmeldung von ffmpeg in eine Fehlermeldung übersetzt. Ob ffmpeg die
Argumente akzeptiert, kann hier niemand feststellen – die CI installiert
ffmpeg nicht (siehe Kommentar in .github/workflows/tests.yml), und ein
falsches Flag fiele weiterhin erst zur Laufzeit auf.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import engine


@pytest.fixture
def ffmpeg_aufruf(monkeypatch):
    """Fängt den subprocess-Aufruf ab und liefert die Argumentliste zurück."""
    aufrufe: list[list[str]] = []

    def einrichten(returncode: int = 0, stderr: str = "", exe: str | None = "/usr/bin/ffmpeg"):
        monkeypatch.setattr(engine.shutil, "which", lambda _name: exe)

        def fake_run(args, **_kwargs):
            aufrufe.append(list(args))
            return subprocess.CompletedProcess(args, returncode, stdout="", stderr=stderr)

        monkeypatch.setattr(engine.subprocess, "run", fake_run)
        return aufrufe

    einrichten.aufrufe = aufrufe  # type: ignore[attr-defined]
    return einrichten


# --------------------------------------------------------------------------- #
# Fehlerpfade
# --------------------------------------------------------------------------- #

def test_fehlendes_ffmpeg_nennt_den_brew_befehl(ffmpeg_aufruf):
    """Die Meldung landet unverändert in der Oberfläche – sie muss sagen, was zu tun ist."""
    ffmpeg_aufruf(exe=None)

    with pytest.raises(RuntimeError, match="brew install ffmpeg"):
        engine._run_ffmpeg(["-i", "a.wav", "b.wav"])


def test_ffmpeg_path_meldet_none_wenn_nichts_gefunden(monkeypatch):
    monkeypatch.setattr(engine.shutil, "which", lambda _name: None)
    assert engine.ffmpeg_path() is None


def test_fehler_hebt_die_letzte_stderr_zeile_heraus(ffmpeg_aufruf):
    """ffmpeg schreibt viel; die letzte Zeile nennt üblicherweise den Grund."""
    ffmpeg_aufruf(returncode=1, stderr="Zeile eins\nZeile zwei\nInvalid data found\n")

    with pytest.raises(RuntimeError, match=r"^ffmpeg: Invalid data found$"):
        engine._run_ffmpeg(["-i", "a.wav", "b.wav"])


def test_fehler_ohne_stderr_nennt_den_exit_code(ffmpeg_aufruf):
    ffmpeg_aufruf(returncode=69, stderr="")

    with pytest.raises(RuntimeError, match="Exit-Code 69"):
        engine._run_ffmpeg(["-i", "a.wav", "b.wav"])


def test_fehler_mit_nur_leerzeilen_nennt_den_exit_code(ffmpeg_aufruf):
    ffmpeg_aufruf(returncode=1, stderr="   \n\n  ")

    with pytest.raises(RuntimeError, match="Exit-Code 1"):
        engine._run_ffmpeg(["-i", "a.wav", "b.wav"])


def test_erfolg_wirft_nicht(ffmpeg_aufruf):
    ffmpeg_aufruf(returncode=0, stderr="nur eine Warnung")
    engine._run_ffmpeg(["-i", "a.wav", "b.wav"])


# --------------------------------------------------------------------------- #
# Argumentbildung
# --------------------------------------------------------------------------- #

def test_immer_ueberschreiben_und_ohne_rueckfragen(ffmpeg_aufruf):
    """-y schreibt ohne Nachfrage, -nostdin verhindert, dass ffmpeg auf
    eine Eingabe wartet, die in einem Serverprozess nie kommt."""
    aufrufe = ffmpeg_aufruf()
    engine._run_ffmpeg(["-i", "a.wav", "b.wav"])

    argv = aufrufe[0]
    assert argv[0] == "/usr/bin/ffmpeg"
    assert argv[1:5] == ["-y", "-nostdin", "-loglevel", "error"]
    assert argv[5:] == ["-i", "a.wav", "b.wav"]


def test_to_wav_zieht_die_erste_tonspur_als_stereo(ffmpeg_aufruf, tmp_path):
    """Videos bringen oft mehrere Tonspuren mit; die Analyse braucht genau eine."""
    aufrufe = ffmpeg_aufruf()
    engine._to_wav(tmp_path / "clip.mov", tmp_path / "original.wav")

    argv = aufrufe[0]
    assert "-vn" in argv, "Videospur muss raus"
    assert argv[argv.index("-map") + 1] == "0:a:0"
    assert argv[argv.index("-ar") + 1] == "44100"
    assert argv[argv.index("-ac") + 1] == "2"
    assert argv[argv.index("-codec:a") + 1] == "pcm_s16le"
    assert argv[-1] == str(tmp_path / "original.wav")


@pytest.mark.parametrize("fmt, erwartet", [
    ("mp3", ["-codec:a", "libmp3lame", "-b:a", "320k"]),
    ("flac", ["-codec:a", "flac"]),
    # wav ist das Ausgangsformat der Trennung – dann kopiert ffmpeg nur.
    ("wav", []),
])
def test_convert_waehlt_den_codec(ffmpeg_aufruf, tmp_path, fmt, erwartet):
    aufrufe = ffmpeg_aufruf()
    engine._convert(tmp_path / "stem.wav", tmp_path / f"stem.{fmt}", fmt)

    argv = aufrufe[0]
    quelle = argv.index("-i")
    assert argv[quelle:quelle + 2] == ["-i", str(tmp_path / "stem.wav")]
    assert argv[quelle + 2:-1] == erwartet
    assert argv[-1] == str(tmp_path / f"stem.{fmt}")


def test_convert_reicht_die_bitrate_durch(ffmpeg_aufruf, tmp_path):
    aufrufe = ffmpeg_aufruf()
    engine._convert(tmp_path / "a.wav", tmp_path / "a.mp3", "mp3", bitrate="192k")

    assert "192k" in aufrufe[0]
    assert "320k" not in aufrufe[0]


def test_convert_kennt_unbekannte_formate_nicht(ffmpeg_aufruf, tmp_path):
    """Ein unbekanntes Format setzt keinen Codec – ffmpeg rät dann selbst."""
    aufrufe = ffmpeg_aufruf()
    engine._convert(tmp_path / "a.wav", tmp_path / "a.ogg", "ogg")

    assert "-codec:a" not in aufrufe[0]


def test_pfade_werden_als_text_uebergeben(ffmpeg_aufruf, tmp_path):
    """subprocess bekommt keine Path-Objekte untergeschoben."""
    aufrufe = ffmpeg_aufruf()
    engine._to_wav(tmp_path / "a.mov", tmp_path / "b.wav")

    assert all(isinstance(teil, str) for teil in aufrufe[0])


def test_umlaute_und_leerzeichen_bleiben_unveraendert(ffmpeg_aufruf, tmp_path):
    """Ohne Shell gibt es nichts zu escapen – das muss so bleiben."""
    quelle = tmp_path / "Grüße aus Köln & Co.wav"
    aufrufe = ffmpeg_aufruf()
    engine._convert(quelle, tmp_path / "ziel.mp3", "mp3")

    assert str(quelle) in aufrufe[0]
