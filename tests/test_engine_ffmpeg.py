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
