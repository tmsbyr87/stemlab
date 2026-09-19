"""Mixdown und Tonhöhen-/Tempoänderung.

Beide Wege gehen über ffmpeg, das in der CI fehlt. Geprüft wird daher,
was StemLab selbst baut: die Filterkette und die Dateinamen. Eine
Ausnahme ist der Phasenvocoder-Rückfall in shift_file – der läuft über
librosa, das in der CI liegt, und wird deshalb echt gerechnet.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import postprocess
from conftest import SR


@pytest.fixture
def ffmpeg_argv(monkeypatch):
    """Fängt jeden ffmpeg-Aufruf ab und liefert die Argumentlisten."""
    aufrufe: list[list[str]] = []

    def einrichten(stdout: str = "", returncode: int = 0, stderr: str = ""):
        monkeypatch.setattr(postprocess.shutil, "which", lambda _n: "/usr/bin/ffmpeg")

        def fake_run(args, **_kwargs):
            aufrufe.append(list(args))
            return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr=stderr)

        monkeypatch.setattr(postprocess.subprocess, "run", fake_run)
        return aufrufe

    einrichten.aufrufe = aufrufe  # type: ignore[attr-defined]
    return einrichten


def _stem(ordner, name, sekunden=2.0, pegel=0.5, sr=SR):
    sf.write(str(ordner / f"{name}.wav"),
             np.full(int(sekunden * sr), pegel, dtype="float32"), sr, subtype="PCM_24")
    return ordner / f"{name}.wav"


@pytest.fixture
def stemordner(tmp_path):
    _stem(tmp_path, "vocals")
    _stem(tmp_path, "drums")
    _stem(tmp_path, "bass")
    return tmp_path


def _kette(argv: list[str]) -> str:
    return argv[argv.index("-filter_complex") + 1]


# --------------------------------------------------------------------------- #
# export_mix
# --------------------------------------------------------------------------- #

def test_mix_setzt_je_stem_einen_pegel(stemordner, ffmpeg_argv, stille):
    aufrufe = ffmpeg_argv()
    postprocess.export_mix(stemordner, {"vocals": -3.0, "drums": 0.0, "bass": 2.5},
                           "Mein Mix", stille)

    kette = _kette(aufrufe[0])
    assert "volume=-3.00dB" in kette
    assert "volume=0.00dB" in kette
    assert "volume=2.50dB" in kette


def test_mix_mischt_ohne_normalisierung(stemordner, ffmpeg_argv, stille):
    """normalize=0: sonst zöge ffmpeg die Pegel wieder zusammen und die
    eingestellten dB-Werte wären wirkungslos."""
    aufrufe = ffmpeg_argv()
    postprocess.export_mix(stemordner, {}, "Mix", stille)

    kette = _kette(aufrufe[0])
    assert "amix=inputs=3" in kette
    assert "normalize=0" in kette


def test_mix_laesst_ausgeblendete_stems_weg(stemordner, ffmpeg_argv, stille):
    """Unter -60 dB ist ein Stem stumm – der gehört nicht in die Kette."""
    aufrufe = ffmpeg_argv()
    postprocess.export_mix(stemordner, {"vocals": -80.0, "drums": 0.0, "bass": 0.0},
                           "Ohne Vocals", stille)

    argv = aufrufe[0]
    assert "amix=inputs=2" in _kette(argv)
    assert not any("vocals.wav" in a for a in argv)


def test_mix_ohne_hoerbaren_stem_wirft(stemordner, ffmpeg_argv, stille):
    ffmpeg_argv()
    with pytest.raises(RuntimeError, match="Kein Stem im Mix"):
        postprocess.export_mix(stemordner, {"vocals": -99, "drums": -99, "bass": -99},
                               "Leer", stille)


def test_mix_im_leeren_ordner_wirft(tmp_path, ffmpeg_argv, stille):
    ffmpeg_argv()
    with pytest.raises(RuntimeError, match="Kein Stem im Mix"):
        postprocess.export_mix(tmp_path, {}, "Leer", stille)


def test_mix_saeubert_den_dateinamen(stemordner, ffmpeg_argv, stille):
    aufrufe = ffmpeg_argv()
    postprocess.export_mix(stemordner, {}, "Mix: Version 2 / final?", stille)

    assert Path(aufrufe[0][-1]).name == "Mix Version 2  final.wav"


def test_mix_ohne_brauchbares_label_heisst_mix(stemordner, ffmpeg_argv, stille):
    aufrufe = ffmpeg_argv()
    postprocess.export_mix(stemordner, {}, "???", stille)

    assert Path(aufrufe[0][-1]).name == "mix.wav"


def test_mix_landet_im_unterordner(stemordner, ffmpeg_argv, stille):
    aufrufe = ffmpeg_argv()
    ziel = postprocess.export_mix(stemordner, {}, "Mix", stille)

    assert ziel.parent == stemordner / "mixes"
    assert ziel.parent.is_dir()
    assert aufrufe[0][-1] == str(ziel)


def test_mix_schreibt_24_bit(stemordner, ffmpeg_argv, stille):
    aufrufe = ffmpeg_argv()
    postprocess.export_mix(stemordner, {}, "Mix", stille)

    assert "pcm_s24le" in aufrufe[0]


def test_mix_meldet_den_dateinamen(stemordner, ffmpeg_argv, stille):
    ffmpeg_argv()
    postprocess.export_mix(stemordner, {}, "Mein Mix", stille)

    assert "Mein Mix.wav" in stille.meldungen[-1]


# --------------------------------------------------------------------------- #
# shift_file – mit Rubber Band
# --------------------------------------------------------------------------- #

def test_shift_nutzt_rubberband_wenn_vorhanden(tmp_path, ffmpeg_argv, stille):
    quelle = _stem(tmp_path, "vocals")
    aufrufe = ffmpeg_argv(stdout="... rubberband ...")

    postprocess.shift_file(quelle, tmp_path / "ziel.wav", 2, 1.0, stille)

    # Erster Aufruf ist die Filterabfrage, zweiter die Umrechnung.
    filter_arg = aufrufe[-1][aufrufe[-1].index("-af") + 1]
    assert filter_arg.startswith("rubberband=")
    # Zwei Halbtöne sind der Faktor 2^(2/12) = 1,122462.
    assert "pitch=1.122462" in filter_arg
    assert "tempo=1.000000" in filter_arg


def test_shift_rechnet_halbtoene_in_ein_verhaeltnis_um(tmp_path, ffmpeg_argv, stille):
    """Eine Oktave nach oben ist exakt der Faktor 2."""
    quelle = _stem(tmp_path, "vocals")
    aufrufe = ffmpeg_argv(stdout="rubberband")

    postprocess.shift_file(quelle, tmp_path / "ziel.wav", 12, 1.0, stille)

    assert "pitch=2.000000" in aufrufe[-1][aufrufe[-1].index("-af") + 1]


def test_shift_reicht_das_tempo_durch(tmp_path, ffmpeg_argv, stille):
    quelle = _stem(tmp_path, "vocals")
    aufrufe = ffmpeg_argv(stdout="rubberband")

    postprocess.shift_file(quelle, tmp_path / "ziel.wav", 0, 1.25, stille)

    filter_arg = aufrufe[-1][aufrufe[-1].index("-af") + 1]
    assert "tempo=1.250000" in filter_arg
    assert "pitch=1.000000" in filter_arg


# --------------------------------------------------------------------------- #
# shift_file – Phasenvocoder-Rückfall
# --------------------------------------------------------------------------- #

def test_shift_ohne_rubberband_rechnet_selbst(tmp_path, monkeypatch, stille):
    """Fehlt der Filter in ffmpeg, springt librosa ein – hier echt gerechnet."""
    monkeypatch.setattr(postprocess, "_has_rubberband", lambda: False)
    quelle = tmp_path / "ton.wav"
    dauer = 1.0
    t = np.linspace(0, dauer, int(SR * dauer), endpoint=False)
    sf.write(str(quelle), (0.5 * np.sin(2 * np.pi * 220 * t)).astype("float32"), SR)

    ziel = postprocess.shift_file(quelle, tmp_path / "hoch.wav", 12, 1.0, stille)

    assert ziel.exists()
    daten, sr = sf.read(str(ziel))
    # Eine Oktave höher: der stärkste Anteil liegt bei rund 440 Hz.
    spektrum = np.abs(np.fft.rfft(daten if daten.ndim == 1 else daten[:, 0]))
    spitze = np.fft.rfftfreq(len(daten), 1 / sr)[np.argmax(spektrum)]
    assert 420 < spitze < 460, f"erwartet ~440 Hz, gemessen {spitze:.0f} Hz"


def test_shift_ohne_rubberband_meldet_den_rueckfall(tmp_path, monkeypatch, stille):
    monkeypatch.setattr(postprocess, "_has_rubberband", lambda: False)
    quelle = _stem(tmp_path, "vocals", sekunden=0.5)

    postprocess.shift_file(quelle, tmp_path / "ziel.wav", 1, 1.0, stille)

    assert any("Rubber Band" in m for m in stille.meldungen)


def test_shift_ohne_rubberband_dehnt_die_zeit(tmp_path, monkeypatch, stille):
    monkeypatch.setattr(postprocess, "_has_rubberband", lambda: False)
    quelle = _stem(tmp_path, "vocals", sekunden=2.0)

    ziel = postprocess.shift_file(quelle, tmp_path / "schnell.wav", 0, 2.0, stille)

    with sf.SoundFile(str(ziel)) as f:
        assert abs(len(f) / f.samplerate - 1.0) < 0.1, "doppeltes Tempo halbiert die Länge"


# --------------------------------------------------------------------------- #
# _has_rubberband
# --------------------------------------------------------------------------- #

def test_has_rubberband_erkennt_den_filter(ffmpeg_argv):
    ffmpeg_argv(stdout="Filters:\n  rubberband  Apply time-stretching\n")
    assert postprocess._has_rubberband() is True


def test_has_rubberband_ohne_filter(ffmpeg_argv):
    ffmpeg_argv(stdout="Filters:\n  volume  Change volume\n")
    assert postprocess._has_rubberband() is False


def test_has_rubberband_ohne_ffmpeg(monkeypatch):
    """Kein ffmpeg heißt kein Rubber Band – und keine Ausnahme nach außen."""
    monkeypatch.setattr(postprocess.shutil, "which", lambda _n: None)
    assert postprocess._has_rubberband() is False


# --------------------------------------------------------------------------- #
# shift_folder
# --------------------------------------------------------------------------- #

@pytest.fixture
def gefaelschtes_shift(monkeypatch):
    """Ersetzt shift_file und protokolliert die Zielpfade."""
    aufrufe = []

    def fake(src, dst, semitones, tempo_ratio, say):
        aufrufe.append(dict(src=src, dst=dst, semitones=semitones, tempo=tempo_ratio))
        dst.write_bytes(b"")
        return dst

    monkeypatch.setattr(postprocess, "shift_file", fake)
    return aufrufe


def test_shift_folder_nimmt_alle_stems(stemordner, gefaelschtes_shift, stille):
    postprocess.shift_folder(stemordner, [], 2, 1.0, stille)

    assert {a["src"].stem for a in gefaelschtes_shift} == {"vocals", "drums", "bass"}


def test_shift_folder_beschraenkt_auf_die_auswahl(stemordner, gefaelschtes_shift, stille):
    postprocess.shift_folder(stemordner, ["vocals", "bass"], 2, 1.0, stille)

    assert {a["src"].stem for a in gefaelschtes_shift} == {"vocals", "bass"}


def test_shift_folder_benennt_nach_halbtoenen(stemordner, gefaelschtes_shift, stille):
    """Das Vorzeichen muss mit: +2st und -2st sind verschiedene Dateien."""
    postprocess.shift_folder(stemordner, ["vocals"], 2, 1.0, stille)
    postprocess.shift_folder(stemordner, ["vocals"], -2, 1.0, stille)

    namen = [a["dst"].name for a in gefaelschtes_shift]
    assert namen == ["vocals_+2st.wav", "vocals_-2st.wav"]


def test_shift_folder_benennt_nach_tempo(stemordner, gefaelschtes_shift, stille):
    postprocess.shift_folder(stemordner, ["vocals"], 0, 1.25, stille)

    assert gefaelschtes_shift[0]["dst"].name == "vocals_x1.25.wav"


def test_shift_folder_kombiniert_beides(stemordner, gefaelschtes_shift, stille):
    postprocess.shift_folder(stemordner, ["vocals"], 3, 0.5, stille)

    assert gefaelschtes_shift[0]["dst"].name == "vocals_+3st_x0.5.wav"


def test_shift_folder_ohne_aenderung_heisst_copy(stemordner, gefaelschtes_shift, stille):
    postprocess.shift_folder(stemordner, ["vocals"], 0, 1.0, stille)

    assert gefaelschtes_shift[0]["dst"].name == "vocals_copy.wav"


def test_shift_folder_landet_im_unterordner(stemordner, gefaelschtes_shift, stille):
    pfade = postprocess.shift_folder(stemordner, ["vocals"], 2, 1.0, stille)

    assert pfade[0].parent == stemordner / "shifted"
