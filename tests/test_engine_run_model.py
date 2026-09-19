"""run_model: Pfadauflösung und MPS-Rückfall.

Was nach der Trennung aus den Ausgaben der Bibliothek wird, und
was passiert, wenn die Apple-GPU mitten im Lauf aussteigt. Der
Rückfallzweig läuft auf einem funktionierenden Rechner nie.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

import engine


# --------------------------------------------------------------------------- #
# run_model – Pfadauflösung
# --------------------------------------------------------------------------- #

@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """Legt SCRATCH in einen Testordner, damit nichts in Application Support landet."""
    ordner = tmp_path / "scratch"
    ordner.mkdir()
    monkeypatch.setattr(engine, "SCRATCH", ordner)
    return ordner


def test_run_model_loest_relative_pfade_gegen_scratch_auf(separator_fabrik, scratch, stille):
    """audio_separator liefert blanke Dateinamen – die liegen im Ausgabeordner."""
    (scratch / "song_(Vocals).wav").write_bytes(b"")
    separator_fabrik(ausgaben=["song_(Vocals).wav"])

    pfade = engine.run_model("modell.ckpt", Path("quelle.wav"), stille)

    assert pfade == [scratch / "song_(Vocals).wav"]


def test_run_model_laesst_absolute_pfade_stehen(separator_fabrik, scratch, tmp_path, stille):
    woanders = tmp_path / "woanders.wav"
    woanders.write_bytes(b"")
    separator_fabrik(ausgaben=[str(woanders)])

    assert engine.run_model("modell.ckpt", Path("quelle.wav"), stille) == [woanders]


def test_run_model_laesst_nicht_existierende_dateien_weg(separator_fabrik, scratch, stille):
    """Die Bibliothek nennt gelegentlich Dateien, die sie nicht geschrieben hat."""
    (scratch / "da.wav").write_bytes(b"")
    separator_fabrik(ausgaben=["da.wav", "fehlt.wav"])

    assert engine.run_model("modell.ckpt", Path("quelle.wav"), stille) == [scratch / "da.wav"]


def test_run_model_haelt_die_reihenfolge_der_stems(separator_fabrik, scratch, stille):
    for name in ("a.wav", "b.wav", "c.wav"):
        (scratch / name).write_bytes(b"")
    separator_fabrik(ausgaben=["c.wav", "a.wav", "b.wav"])

    pfade = engine.run_model("modell.ckpt", Path("quelle.wav"), stille)
    assert [p.name for p in pfade] == ["c.wav", "a.wav", "b.wav"]


def test_run_model_reicht_die_quelle_als_text_durch(separator_fabrik, scratch, stille):
    fabrik = separator_fabrik(ausgaben=[])
    engine.run_model("modell.ckpt", Path("/pfad/quelle.wav"), stille)

    assert fabrik.letzter.separate_aufrufe == ["/pfad/quelle.wav"]


def test_run_model_merkt_sich_das_geraet(separator_fabrik, scratch, stille):
    """Die Oberfläche zeigt danach an, worauf die Trennung lief."""
    separator_fabrik(ausgaben=[], torch_device="mps:0")
    engine.run_model("modell.ckpt", Path("quelle.wav"), stille)
    assert engine.run_model.last_device == "mps"

    separator_fabrik(ausgaben=[], torch_device="cpu")
    engine.run_model("anderes.ckpt", Path("quelle.wav"), stille)
    assert engine.run_model.last_device == "cpu"


def test_run_model_meldet_fortschritt(separator_fabrik, scratch, stille):
    """Der ProgressTap hängt während separate() an stderr und füttert die Oberfläche."""
    meldungen = []
    fabrik = separator_fabrik(ausgaben=[], fortschritt=["30%|", "60%|"])

    engine.run_model("modell.ckpt", Path("quelle.wav"), stille,
                     on_progress=lambda p, d: meldungen.append((p, d)))

    assert meldungen == [(30, 1), (60, 1)]
    assert fabrik.ladevorgaenge == 1


def test_run_model_stellt_stderr_wieder_her(separator_fabrik, scratch, stille):
    separator_fabrik(ausgaben=[])
    vorher = sys.stderr
    engine.run_model("modell.ckpt", Path("quelle.wav"), stille)
    assert sys.stderr is vorher


def test_run_model_haengt_den_logtap_wieder_ab(separator_fabrik, scratch, stille):
    """Sonst sammeln sich mit jedem Job weitere Handler am Bibliotheks-Logger."""
    separator_fabrik(ausgaben=[])
    logger = logging.getLogger("audio_separator")
    vorher = len(logger.handlers)

    engine.run_model("modell.ckpt", Path("quelle.wav"), stille)

    assert len(logger.handlers) == vorher


def test_run_model_reicht_bibliotheksmeldungen_durch(separator_fabrik, scratch, stille, monkeypatch):
    """audio_separator loggt seinen Fortschritt – der gehört in die Oberfläche.

    Im Betrieb setzt _make_separator log_level=INFO, und die Bibliothek legt
    das auf ihrem Logger ab. Der Fake tut das nicht, also hier von Hand –
    sonst verwirft logging die Meldung auf WARNING-Niveau, bevor der Tap
    sie überhaupt sieht.
    """
    logger = logging.getLogger("audio_separator")
    monkeypatch.setattr(logger, "level", logging.INFO)
    separator_fabrik(ausgaben=[], logmeldungen=["Lade Gewichte"])

    engine.run_model("modell.ckpt", Path("quelle.wav"), stille)

    assert "Lade Gewichte" in stille.meldungen

# --------------------------------------------------------------------------- #
# run_model – MPS-Rückfall
# --------------------------------------------------------------------------- #
#
# Dieser Zweig läuft auf einem funktionierenden Rechner nie: Er greift erst,
# wenn PyTorch mitten in der Trennung eine Operation auf der Apple-GPU nicht
# unterstützt. Ohne Test bliebe er bis zu dem Tag ungeprüft, an dem ein
# Nutzer ihn braucht – und dann ist es zu spät, ihn zu bemerken.

def test_mps_fehler_loest_genau_einen_cpu_versuch_aus(separator_fabrik, scratch, stille):
    (scratch / "ergebnis.wav").write_bytes(b"")
    fabrik = separator_fabrik(
        ausgaben=["ergebnis.wav"],
        fehler=[RuntimeError("MPS backend out of memory"), None],
    )

    pfade = engine.run_model("modell.ckpt", Path("quelle.wav"), stille)

    assert pfade == [scratch / "ergebnis.wav"]
    assert fabrik.ladevorgaenge == 2, "Modell muss für die CPU neu geladen werden"
    assert engine._force_cpu is True
    assert any("CPU" in m for m in stille.meldungen)
    # Beide Separatoren wurden gerufen – und das Ergebnis stammt vom zweiten.
    # Ohne diese Prüfung bliebe offen, ob run_model den neu geladenen
    # Separator überhaupt verwendet oder still am gescheiterten festhält.
    assert fabrik.erzeugte[0].separate_aufrufe == ["quelle.wav"]
    assert fabrik.erzeugte[1].separate_aufrufe == ["quelle.wav"]
    assert fabrik.letzter is fabrik.erzeugte[1]


@pytest.mark.parametrize("text", ["MPS backend out of memory", "not implemented for mps"])
def test_mps_erkennung_ist_gross_und_kleinschreibung(separator_fabrik, scratch, stille, text):
    """PyTorch meldet mal "MPS", mal "mps" – beides muss greifen."""
    separator_fabrik(ausgaben=[], fehler=[RuntimeError(text), None])

    engine.run_model("modell.ckpt", Path("quelle.wav"), stille)

    assert engine._force_cpu is True


def test_anderer_fehler_wird_durchgereicht(separator_fabrik, scratch, stille):
    """Nur MPS-Fehler rechtfertigen einen zweiten Versuch. Alles andere fliegt."""
    fabrik = separator_fabrik(ausgaben=[], fehler=[ValueError("Datei kaputt")])

    with pytest.raises(ValueError, match="Datei kaputt"):
        engine.run_model("modell.ckpt", Path("quelle.wav"), stille)

    assert fabrik.ladevorgaenge == 1, "kein zweiter Versuch"
    assert engine._force_cpu is False


def test_kein_zweiter_versuch_wenn_schon_auf_cpu(separator_fabrik, scratch, stille):
    """Läuft es bereits auf der CPU, ist ein MPS-Fehler nicht mehr erklärbar –
    dann endlos zu wiederholen würde den Fehler nur verschleiern."""
    engine._force_cpu = True
    fabrik = separator_fabrik(ausgaben=[], fehler=[RuntimeError("MPS kaputt")])

    with pytest.raises(RuntimeError, match="MPS kaputt"):
        engine.run_model("modell.ckpt", Path("quelle.wav"), stille)

    assert fabrik.ladevorgaenge == 1


def test_zweiter_mps_fehler_fliegt(separator_fabrik, scratch, stille):
    """Wenn auch der CPU-Versuch scheitert, gibt es kein drittes Mal."""
    fabrik = separator_fabrik(
        ausgaben=[],
        fehler=[RuntimeError("MPS kaputt"), RuntimeError("MPS immer noch kaputt")],
    )

    with pytest.raises(RuntimeError, match="immer noch"):
        engine.run_model("modell.ckpt", Path("quelle.wav"), stille)

    assert fabrik.ladevorgaenge == 2


def test_mps_rueckfall_stellt_stderr_wieder_her(separator_fabrik, scratch, stille):
    """Der Rückfall tauscht stderr zweimal – am Ende muss das Original stehen."""
    separator_fabrik(ausgaben=[], fehler=[RuntimeError("MPS kaputt"), None])
    vorher = sys.stderr

    engine.run_model("modell.ckpt", Path("quelle.wav"), stille)

    assert sys.stderr is vorher


def test_mps_rueckfall_stellt_stderr_auch_bei_fehler_wieder_her(separator_fabrik, scratch, stille):
    separator_fabrik(ausgaben=[], fehler=[ValueError("kaputt")])
    vorher = sys.stderr

    with pytest.raises(ValueError):
        engine.run_model("modell.ckpt", Path("quelle.wav"), stille)

    assert sys.stderr is vorher


def test_mps_rueckfall_meldet_weiter_fortschritt(separator_fabrik, scratch, stille):
    """Nach dem Tausch muss der Tap wieder hängen, sonst friert die Anzeige ein."""
    meldungen = []
    separator_fabrik(ausgaben=[], fehler=[RuntimeError("MPS kaputt"), None],
                     fortschritt=["40%|"])

    engine.run_model("modell.ckpt", Path("quelle.wav"), stille,
                     on_progress=lambda p, d: meldungen.append((p, d)))

    assert (40, 1) in meldungen
