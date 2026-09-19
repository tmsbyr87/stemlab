"""Vocal-Veredelung und die Steuerung des DJ-Exports.

Beides orchestriert: refine_vocals hängt Trennmodelle hintereinander,
export_dj entscheidet, welche Exporter laufen. Die Arbeit selbst machen
andere – hier zählt, dass die Reihenfolge und die Auswahl stimmen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import engine
import postprocess
from conftest import SR


def _stem(ordner, name, sekunden=1.0):
    pfad = ordner / f"{name}.wav"
    sf.write(str(pfad), np.zeros(int(sekunden * SR), dtype="float32"), SR, subtype="PCM_24")
    return pfad


@pytest.fixture
def vocalordner(tmp_path):
    _stem(tmp_path, "vocals")
    _stem(tmp_path, "drums")
    return tmp_path


@pytest.fixture
def gefaelschtes_run_model(monkeypatch, tmp_path):
    """Ersetzt engine.run_model und legt die angeforderten Ausgabedateien an.

    Die Kette wird über die Labels gesteuert: Jeder Aufruf liefert die
    Dateien, die der nächste Schritt erwartet.
    """
    aufrufe = []
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)

    def einrichten(*ausgabefolgen):
        folgen = list(ausgabefolgen)

        def fake(model, src, say, on_progress=None, preset=None):
            aufrufe.append(dict(model=model, src=Path(src)))
            labels = folgen.pop(0) if folgen else []
            erzeugt = []
            for i, label in enumerate(labels):
                p = scratch / f"lauf{len(aufrufe)}_{i}_({label})_modell.wav"
                # Erkennbare Länge je Position: so lässt sich später
                # feststellen, WELCHE Ausgabe behalten wurde.
                sf.write(str(p), np.zeros(SR // 10 * (i + 1), dtype="float32"), SR)
                erzeugt.append(p)
            return erzeugt

        monkeypatch.setattr(engine, "run_model", fake)
        monkeypatch.setattr(engine, "first_available", lambda kandidaten: kandidaten[0])
        return aufrufe

    return einrichten


# --------------------------------------------------------------------------- #
# refine_vocals
# --------------------------------------------------------------------------- #

def test_refine_ohne_vocalstem_wirft(tmp_path, stille):
    _stem(tmp_path, "drums")
    with pytest.raises(RuntimeError, match="Kein Vocal-Stem"):
        postprocess.refine_vocals(tmp_path, ["dereverb"], stille)


def test_refine_einzelner_schritt(vocalordner, gefaelschtes_run_model, stille):
    gefaelschtes_run_model(["Dry"])

    pfade = postprocess.refine_vocals(vocalordner, ["dereverb"], stille)

    assert [p.name for p in pfade] == ["vocals_dry.wav"]
    assert pfade[0].parent == vocalordner / "refined"
    assert pfade[0].exists()


def test_refine_kette_baut_die_suffixe_auf(vocalordner, gefaelschtes_run_model, stille):
    """Jeder Schritt hängt sein Kürzel an – der Dateiname erzählt die Kette."""
    gefaelschtes_run_model(["Dry"], ["Dry"])

    pfade = postprocess.refine_vocals(vocalordner, ["dereverb", "denoise"], stille)

    assert [p.name for p in pfade] == ["vocals_dry.wav", "vocals_dry_clean.wav"]


def test_refine_arbeitet_auf_dem_vorigen_ergebnis(vocalordner, gefaelschtes_run_model, stille):
    """Der zweite Schritt darf nicht wieder vom Original ausgehen."""
    aufrufe = gefaelschtes_run_model(["Dry"], ["Dry"])

    postprocess.refine_vocals(vocalordner, ["dereverb", "denoise"], stille)

    assert aufrufe[0]["src"].name == "vocals.wav"
    assert aufrufe[1]["src"].name == "vocals_dry.wav"


def test_refine_waehlt_das_modell_ueber_first_available(vocalordner, gefaelschtes_run_model, stille):
    aufrufe = gefaelschtes_run_model(["Dry"])

    postprocess.refine_vocals(vocalordner, ["dereverb"], stille)

    assert aufrufe[0]["model"] == postprocess.REFINE_STEPS["dereverb"]["candidates"][0]


@pytest.mark.parametrize("label", ["Dry", "No Reverb", "No Echo"])
def test_refine_erkennt_jedes_erlaubte_label(vocalordner, gefaelschtes_run_model, stille, label):
    """Die Modelle benennen ihre Ausgabe unterschiedlich – alle müssen zählen."""
    gefaelschtes_run_model([label])

    assert postprocess.refine_vocals(vocalordner, ["dereverb"], stille)


def test_refine_nimmt_die_erste_passende_ausgabe(vocalordner, gefaelschtes_run_model, stille):
    """Liefert ein Modell mehrere passende Spuren, zählt die erste.

    Die Kandidatenlabels in REFINE_STEPS sind nach Güte sortiert; die
    zweite Spur ist eine Variante, keine Verbesserung.
    """
    gefaelschtes_run_model(["Dry", "No Reverb"])

    pfade = postprocess.refine_vocals(vocalordner, ["dereverb"], stille)

    # Genau eine Datei – die zweite passende Spur wird nicht zusätzlich
    # behalten und überschreibt die erste auch nicht.
    assert [p.name for p in pfade] == ["vocals_dry.wav"]
    # Die erste Ausgabe ist an ihrer Länge erkennbar (SR//10), die zweite
    # wäre doppelt so lang. Ohne diese Prüfung bliebe offen, welche der
    # beiden passenden Spuren tatsächlich behalten wurde.
    with sf.SoundFile(str(pfade[0])) as f:
        assert len(f) == SR // 10


def test_refine_bei_unerwarteter_ausgabe_wirft(vocalordner, gefaelschtes_run_model, stille):
    """Liefert das Modell nichts Brauchbares, ist Abbruch besser als eine
    stillschweigend falsche Datei."""
    gefaelschtes_run_model(["Instrumental"])

    with pytest.raises(RuntimeError, match="unerwartete Ausgabe"):
        postprocess.refine_vocals(vocalordner, ["dereverb"], stille)


def test_refine_haelt_das_backing_fest(vocalordner, gefaelschtes_run_model, stille):
    """Beim Lead/Backing-Schritt ist auch die zweite Spur wertvoll."""
    gefaelschtes_run_model(["Vocals", "Instrumental"])

    pfade = postprocess.refine_vocals(vocalordner, ["leadback"], stille)
    namen = sorted(p.name for p in pfade)

    assert namen == ["vocals_backing.wav", "vocals_lead.wav"]
    assert all(p.exists() for p in pfade)


def test_refine_raeumt_den_scratch_auf(vocalordner, gefaelschtes_run_model, stille):
    """Nicht verwendete Ausgaben bleiben nicht liegen."""
    gefaelschtes_run_model(["Dry", "Other"])

    postprocess.refine_vocals(vocalordner, ["dereverb"], stille)

    uebrig = list((vocalordner.parent / "scratch").glob("*.wav")) \
        if (vocalordner.parent / "scratch").is_dir() else []
    assert uebrig == []


def test_refine_meldet_die_anzahl(vocalordner, gefaelschtes_run_model, stille):
    gefaelschtes_run_model(["Dry"], ["Dry"])

    postprocess.refine_vocals(vocalordner, ["dereverb", "denoise"], stille)

    assert "2 Dateien" in stille.meldungen[-1]


def test_refine_ohne_schritte_tut_nichts(vocalordner, gefaelschtes_run_model, stille):
    aufrufe = gefaelschtes_run_model()

    assert postprocess.refine_vocals(vocalordner, [], stille) == []
    assert aufrufe == []


# --------------------------------------------------------------------------- #
# export_dj
# --------------------------------------------------------------------------- #

@pytest.fixture
def exporter(monkeypatch):
    """Ersetzt die beiden Exporter und die Cue-Suche."""
    protokoll = {"rekordbox": 0, "traktor": 0, "cues": 0}

    def fake_rb(folder, analysis, cues, song, opts):
        protokoll["rekordbox"] += 1
        protokoll["cues_an_rb"] = len(cues)
        (folder / "rekordbox.xml").write_text("<x/>")
        return folder / "rekordbox.xml"

    def fake_tk(folder, analysis, cues, song, opts):
        protokoll["traktor"] += 1
        (folder / "traktor.nml").write_text("<x/>")
        return folder / "traktor.nml"

    def fake_cues(audio, downbeats, total):
        protokoll["cues"] += 1
        return [{"name": "Intro", "time": 0.0}, {"name": "Drop", "time": 32.0}]

    monkeypatch.setattr(postprocess, "export_rekordbox", fake_rb)
    monkeypatch.setattr(postprocess, "export_traktor", fake_tk)
    monkeypatch.setattr(postprocess, "detect_cues", fake_cues)
    return protokoll


@pytest.fixture
def analyseordner(tmp_path):
    _stem(tmp_path, "original")
    return tmp_path


def _cfg(**werte):
    """Einstellungen liegen unter "tagging" – flach übergeben wirkt nichts."""
    return {"tagging": werte}


def test_export_dj_schreibt_ohne_einstellung_nichts(analyseordner, exporter, stille):
    """Beide Exporter sind standardmäßig AUS. Wer nichts einstellt, bekommt
    keine Fremdformate in seinen Ordner geschrieben."""
    pfade = postprocess.export_dj(analyseordner, {}, "Song", {}, stille)

    assert pfade == []
    assert exporter["rekordbox"] == 0 and exporter["traktor"] == 0


def test_export_dj_schreibt_beide_wenn_eingeschaltet(analyseordner, exporter, stille):
    pfade = postprocess.export_dj(analyseordner, {}, "Song",
                                  _cfg(export_rekordbox=True, export_traktor=True), stille)

    assert exporter["rekordbox"] == 1 and exporter["traktor"] == 1
    assert {p.name for p in pfade} == {"rekordbox.xml", "traktor.nml"}


def test_export_dj_nur_rekordbox(analyseordner, exporter, stille):
    pfade = postprocess.export_dj(analyseordner, {}, "Song",
                                  _cfg(export_rekordbox=True), stille)

    assert exporter["traktor"] == 0
    assert [p.name for p in pfade] == ["rekordbox.xml"]


def test_export_dj_nur_traktor(analyseordner, exporter, stille):
    pfade = postprocess.export_dj(analyseordner, {}, "Song",
                                  _cfg(export_traktor=True), stille)

    assert exporter["rekordbox"] == 0
    assert [p.name for p in pfade] == ["traktor.nml"]


def test_export_dj_ohne_exporter_meldet_nichts(analyseordner, exporter, stille):
    postprocess.export_dj(analyseordner, {}, "Song", {}, stille)

    assert not any("Export geschrieben" in m for m in stille.meldungen)


def test_export_dj_sucht_cues_und_reicht_sie_weiter(analyseordner, exporter, stille):
    """Die Cue-Suche läuft auch ohne eingeschalteten Exporter – sie ist
    standardmäßig an und kostet nur Rechenzeit, keine fremden Dateien."""
    postprocess.export_dj(analyseordner, {}, "Song",
                          _cfg(export_rekordbox=True), stille)

    assert exporter["cues"] == 1
    assert exporter["cues_an_rb"] == 2


def test_export_dj_kann_cues_abschalten(analyseordner, exporter, stille):
    postprocess.export_dj(analyseordner, {}, "Song",
                          _cfg(export_rekordbox=True, export_cues=False), stille)

    assert exporter["cues"] == 0
    assert exporter["cues_an_rb"] == 0


def test_export_dj_ohne_original_sucht_keine_cues(tmp_path, exporter, stille):
    """Ohne original.wav gibt es nichts zu analysieren – der Export selbst
    soll aber trotzdem entstehen."""
    pfade = postprocess.export_dj(tmp_path, {}, "Song",
                                  _cfg(export_rekordbox=True, export_traktor=True), stille)

    assert exporter["cues"] == 0
    assert len(pfade) == 2


def test_export_dj_meldet_die_cue_zahl(analyseordner, exporter, stille):
    postprocess.export_dj(analyseordner, {}, "Song",
                          _cfg(export_rekordbox=True), stille)

    assert "2 Cue-Points" in stille.meldungen[-1]
    assert "rekordbox.xml" in stille.meldungen[-1]
