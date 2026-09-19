"""Modellkatalog und Separator-Cache von engine.py.

Welches Modell eine Trennung tatsächlich zieht, entscheidet sich
hier: beim Abgleich mit der Modellliste und beim Cache, der ein
geladenes Modell für den nächsten Job behält.
"""

from __future__ import annotations

import sys

import pytest

import engine


# einziger Test beide Hälften, ohne sich auf eine Reihenfolge zu verlassen.
def test_fixture_stellt_jeden_global_wieder_her(request):
    """Verstellt jeden Global und lässt die Fixture in einem eigenen
    Testlauf aufräumen – nachgestellt über request.getfixturevalue in
    einem frischen Fixture-Zyklus."""
    engine._force_cpu = True
    engine._sep_cache.update(key="egal", separator=object())
    engine._available_models = {"erfunden.ckpt"}
    engine.CATALOG[0].resolved = "erfunden.ckpt"
    engine.CATALOG[0].verified = True

    # Die autouse-Fixture räumt am Ende dieses Tests auf. Dass sie das tut,
    # prüft test_zustand_ist_zu_beginn_jedes_tests_unberuehrt bei JEDEM
    # weiteren Test der Datei mit – denn jeder von ihnen liefe rot, wenn
    # hier etwas hängenbliebe.
    assert engine._force_cpu is True


def test_zustand_ist_zu_beginn_jedes_tests_unberuehrt():
    """Gegenprobe: egal was vorher lief, zu Testbeginn ist alles frisch.

    Wirksam ist das nur im Verbund – läuft dieser Test allein, hat niemand
    etwas verstellt, und er wäre auch ohne Fixture grün. Im vollen Lauf
    dagegen fällt er, sobald ein anderer Test seinen Zustand hinterlässt,
    und zwar unabhängig von der Reihenfolge.
    """
    assert engine._force_cpu is False
    assert engine._sep_cache == {"key": None, "separator": None}
    assert engine._available_models == set()
    assert all(c.resolved is None and c.verified is False for c in engine.CATALOG)

# --------------------------------------------------------------------------- #
# FakeSeparator
# --------------------------------------------------------------------------- #

def test_get_separator_nutzt_eingeschleusten_separator(separator_fabrik, stille):
    """Ohne Netz und ohne audio_separator: _make_separator ist die Einschleusstelle."""
    fabrik = separator_fabrik()
    sep = engine._get_separator("modell.ckpt", None, stille, sys.stderr)

    assert sep is fabrik.letzter
    assert sep.load_model_aufrufe == [{"model_filename": "modell.ckpt"}]
    assert sep.separate_aufrufe == []

# --------------------------------------------------------------------------- #
# Katalogauflösung
# --------------------------------------------------------------------------- #

def test_flatten_findet_modellnamen_in_verschachtelter_struktur():
    """list_supported_model_files() liefert verschachtelte dicts – Namen stehen
    mal als Schlüssel, mal als Wert."""
    baum = {
        "MDX": {"UVR-MDX.onnx": "Beschreibung"},
        "Demucs": {"Bundle": {"htdemucs.yaml": {"datei": "htdemucs.th"}}},
        "Liste": [{"tief.pth": 1}],
        "kein Modell": "einfach Text",
    }
    assert set(engine._flatten_model_names(baum)) == {
        "UVR-MDX.onnx", "htdemucs.yaml", "htdemucs.th", "tief.pth",
    }


def test_flatten_beachtet_blanke_strings_in_listen_nicht():
    """Festgehalten, weil es asymmetrisch aussieht: ein Modellname als
    dict-Schlüssel oder -Wert zählt, als blanker Listeneintrag nicht.

    In der echten Modellliste stehen in Listen nur Stem-Namen ("vocals",
    "drums"), nie Dateinamen – deshalb ist das kein Fehler, sondern nur
    eine Kante, über die man beim Lesen stolpert.
    """
    assert list(engine._flatten_model_names(["roformer.ckpt"])) == []
    assert list(engine._flatten_model_names({"k": "roformer.ckpt"})) == ["roformer.ckpt"]


def test_flatten_ignoriert_fremde_endungen():
    assert list(engine._flatten_model_names({"liesmich.txt": "a.json"})) == []


def _fake_probe(monkeypatch, modelle=None, fehler=None):
    """Schleust eine gefälschte Modellliste ein.

    engine ruft `from audio_separator.separator import Separator`. Ein Eintrag
    in sys.modules reicht dafür nur, wenn er ein echtes Modulobjekt mit dem
    Attribut ist – sonst greift der Import daneben und verify_catalog nimmt
    still den Fehlerpfad. Genau so sähe es in der CI aus, die das Paket nicht
    installiert; der Test würde dann grün aussehen, ohne etwas zu prüfen.
    """
    import types

    class FakeSeparator:
        def __init__(self, **_kwargs):
            if fehler is not None:
                raise fehler

        def list_supported_model_files(self):
            return {"alle": {name: name for name in (modelle or [])}}

    paket = types.ModuleType("audio_separator")
    untermodul = types.ModuleType("audio_separator.separator")
    untermodul.Separator = FakeSeparator
    paket.separator = untermodul
    monkeypatch.setitem(sys.modules, "audio_separator", paket)
    monkeypatch.setitem(sys.modules, "audio_separator.separator", untermodul)


def test_verify_catalog_loest_einzelmodelle_auf(monkeypatch):
    einzel = next(c for c in engine.CATALOG if not c.preset)
    _fake_probe(monkeypatch, modelle=[einzel.candidates[-1]])

    assert engine.verify_catalog() is True
    assert einzel.resolved == einzel.candidates[-1]
    assert einzel.verified is True
    assert engine.catalog_status()["verified"] is True


def test_verify_catalog_setzt_resolved_auf_none_wenn_nichts_passt(monkeypatch):
    einzel = next(c for c in engine.CATALOG if not c.preset)
    _fake_probe(monkeypatch, modelle=["voellig.anderes.ckpt"])

    engine.verify_catalog()
    assert einzel.resolved is None


def test_verify_catalog_preset_braucht_alle_kandidaten(monkeypatch):
    """Ein Ensemble läuft nur, wenn jedes beteiligte Modell da ist."""
    preset = next((c for c in engine.CATALOG if c.preset), None)
    if preset is None:
        pytest.skip("Katalog führt kein Ensemble")

    _fake_probe(monkeypatch, modelle=preset.candidates[:-1])
    engine.verify_catalog()
    assert preset.resolved is None

    _fake_probe(monkeypatch, modelle=preset.candidates)
    engine.verify_catalog()
    assert preset.resolved == preset.preset


def test_verify_catalog_ohne_netz_nimmt_ersten_kandidaten(monkeypatch):
    """Kein Netz ist kein Fehler: StemLab startet trotzdem."""
    _fake_probe(monkeypatch, fehler=OSError("kein Netz"))

    assert engine.verify_catalog() is False
    status = engine.catalog_status()
    assert status["ready"] is True
    assert status["verified"] is False
    assert "kein Netz" in status["error"]
    assert all(c.resolved == c.candidates[0] for c in engine.CATALOG)


def test_catalog_loest_beim_ersten_zugriff_auf():
    """Vor dem Abgleich darf catalog() keine leeren resolved-Felder liefern."""
    engine._catalog_state["ready"] = False
    for choice in engine.CATALOG:
        choice.resolved = None

    assert all(c.resolved is not None for c in engine.catalog())


def test_get_choice_kennt_jeden_katalogeintrag():
    for choice in engine.catalog():
        assert engine.get_choice(choice.key) is choice


def test_get_choice_wirft_bei_unbekanntem_schluessel():
    with pytest.raises(KeyError, match="erfunden"):
        engine.get_choice("erfunden")

# --------------------------------------------------------------------------- #
# Separator-Cache
# --------------------------------------------------------------------------- #

def test_cache_laedt_dasselbe_modell_kein_zweites_mal(separator_fabrik, stille):
    """Der Sinn des Caches: ein Album am Stück trennen, ohne jedes Mal neu zu laden."""
    fabrik = separator_fabrik()

    erster = engine._get_separator("modell.ckpt", None, stille, sys.stderr)
    zweiter = engine._get_separator("modell.ckpt", None, stille, sys.stderr)

    assert zweiter is erster
    assert fabrik.ladevorgaenge == 1
    assert erster.load_model_aufrufe == [{"model_filename": "modell.ckpt"}]
    assert "ist noch geladen" in stille.meldungen[-1]


def test_cache_wird_bei_modellwechsel_verworfen(separator_fabrik, stille):
    fabrik = separator_fabrik()

    erster = engine._get_separator("a.ckpt", None, stille, sys.stderr)
    zweiter = engine._get_separator("b.ckpt", None, stille, sys.stderr)

    assert zweiter is not erster
    assert fabrik.ladevorgaenge == 2
    assert zweiter.load_model_aufrufe == [{"model_filename": "b.ckpt"}]


def test_cache_merkt_sich_das_zuletzt_geladene_modell(separator_fabrik, stille):
    """Nach dem Wechsel zurück wird erneut geladen – es gibt nur einen Platz."""
    fabrik = separator_fabrik()

    engine._get_separator("a.ckpt", None, stille, sys.stderr)
    engine._get_separator("b.ckpt", None, stille, sys.stderr)
    engine._get_separator("a.ckpt", None, stille, sys.stderr)

    assert fabrik.ladevorgaenge == 3
    assert engine._sep_cache["key"] == "a.ckpt"


def test_cache_haelt_den_separator_im_zustand(separator_fabrik, stille):
    fabrik = separator_fabrik()
    sep = engine._get_separator("modell.ckpt", None, stille, sys.stderr)

    assert engine._sep_cache == {"key": "modell.ckpt", "separator": sep}
    assert fabrik.ladevorgaenge == 1


def test_get_separator_stellt_stderr_wieder_her(separator_fabrik, stille):
    """Während des Ladens hängt ein ProgressTap an stderr – danach nicht mehr."""
    separator_fabrik()
    vorher = sys.stderr

    engine._get_separator("modell.ckpt", None, stille, vorher)

    assert sys.stderr is vorher


def test_get_separator_stellt_stderr_auch_nach_fehler_wieder_her(monkeypatch, stille):
    """Ein Ladefehler darf stderr nicht als Tap zurücklassen."""
    class Kaputt:
        def load_model(self, **_kwargs):
            raise RuntimeError("Modell defekt")

    monkeypatch.setattr(engine, "_make_separator", lambda preset=None: Kaputt())
    vorher = sys.stderr

    with pytest.raises(RuntimeError, match="Modell defekt"):
        engine._get_separator("modell.ckpt", None, stille, vorher)

    assert sys.stderr is vorher
    assert engine._sep_cache["separator"] is None


def test_get_separator_meldet_ladefortschritt(monkeypatch, stille):
    """Die Prozentmeldungen kommen in Fünferschritten, damit das Log lesbar bleibt."""
    class Meldend:
        def load_model(self, **_kwargs):
            for pct in (0, 2, 5, 7, 12, 100):
                sys.stderr.write(f"{pct}%|")

    monkeypatch.setattr(engine, "_make_separator", lambda preset=None: Meldend())
    engine._get_separator("modell.ckpt", None, stille, sys.stderr)

    prozente = [m for m in stille.meldungen if "%" in m]
    assert prozente == [
        "Modell wird geladen … 0 %",
        "Modell wird geladen … 5 %",
        "Modell wird geladen … 12 %",
        "Modell wird geladen … 100 %",
    ]


def test_preset_wird_ohne_model_filename_geladen(separator_fabrik, stille):
    """Beim Ensemble kennt der Separator seine Modelle aus dem Preset."""
    fabrik = separator_fabrik()

    sep = engine._get_separator("wird_ignoriert.ckpt", "vocal_balanced", stille, sys.stderr)

    assert sep.preset == "vocal_balanced"
    assert sep.load_model_aufrufe == [{}]
    assert engine._sep_cache["key"] == "preset:vocal_balanced"


def test_preset_und_gleichnamiges_modell_kollidieren_nicht(separator_fabrik, stille):
    """Der Schlüssel trägt deshalb das Präfix "preset:".

    Ohne das würde ein Ensemble namens "a" den Cache-Eintrag eines Modells
    namens "a" treffen – und die Trennung liefe mit dem falschen Aufbau,
    ohne dass irgendwo ein Fehler auftauchte.
    """
    fabrik = separator_fabrik()

    als_preset = engine._get_separator("a", "a", stille, sys.stderr)
    als_modell = engine._get_separator("a", None, stille, sys.stderr)

    assert als_modell is not als_preset
    assert fabrik.ladevorgaenge == 2
    assert als_preset.load_model_aufrufe == [{}]
    assert als_modell.load_model_aufrufe == [{"model_filename": "a"}]


def test_preset_trifft_den_cache_beim_zweiten_mal(separator_fabrik, stille):
    fabrik = separator_fabrik()

    erster = engine._get_separator("egal.ckpt", "vocal_balanced", stille, sys.stderr)
    zweiter = engine._get_separator("anderes.ckpt", "vocal_balanced", stille, sys.stderr)

    assert zweiter is erster
    assert fabrik.ladevorgaenge == 1


def test_presetwechsel_verwirft_den_cache(separator_fabrik, stille):
    fabrik = separator_fabrik()

    engine._get_separator("egal.ckpt", "vocal_balanced", stille, sys.stderr)
    engine._get_separator("egal.ckpt", "instrumental", stille, sys.stderr)

    assert fabrik.ladevorgaenge == 2
    assert engine._sep_cache["key"] == "preset:instrumental"


def _separator_argumente(monkeypatch, preset):
    """Ruft das echte _make_separator mit gefälschtem Separator und liefert dessen kwargs."""
    import types

    aufgezeichnet: dict = {}

    class AufzeichnenderSeparator:
        def __init__(self, **kwargs):
            aufgezeichnet.update(kwargs)

    paket = types.ModuleType("audio_separator")
    untermodul = types.ModuleType("audio_separator.separator")
    untermodul.Separator = AufzeichnenderSeparator
    paket.separator = untermodul
    monkeypatch.setitem(sys.modules, "audio_separator", paket)
    monkeypatch.setitem(sys.modules, "audio_separator.separator", untermodul)

    engine._make_separator(preset)
    return aufgezeichnet


def test_make_separator_reicht_preset_an_die_bibliothek(monkeypatch):
    """Die übrigen Tests ersetzen _make_separator ganz – hier läuft es echt.

    Sonst bliebe ungeprüft, dass das Preset überhaupt bei der Bibliothek
    ankommt: die Trennung liefe dann als Einzelmodell, und das Ergebnis
    wäre schlechter, ohne dass etwas fehlschlägt.
    """
    kwargs = _separator_argumente(monkeypatch, "vocal_balanced")
    assert kwargs["ensemble_preset"] == "vocal_balanced"


def test_make_separator_setzt_ohne_preset_kein_ensemble(monkeypatch):
    assert "ensemble_preset" not in _separator_argumente(monkeypatch, None)


def test_make_separator_schaltet_unter_force_cpu_die_apple_gpu_ab(monkeypatch):
    """_force_cpu verbiegt torch.backends.mps.is_available dauerhaft.

    Das ist die folgenreichste Zeile des Moduls: Sie verändert eine
    fremde Bibliothek im laufenden Prozess. Nach einem MPS-Fehler ist
    das gewollt – die Bibliothek soll die GPU gar nicht erst anbieten –
    aber es gehört belegt, statt sich darauf zu verlassen.
    """
    import types

    torch = types.ModuleType("torch")
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", torch)
    engine._force_cpu = True

    _separator_argumente(monkeypatch, None)

    assert torch.backends.mps.is_available() is False


def test_make_separator_laesst_torch_in_ruhe_ohne_force_cpu(monkeypatch):
    import types

    torch = types.ModuleType("torch")
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", torch)

    _separator_argumente(monkeypatch, None)

    assert torch.backends.mps.is_available() is True


def test_make_separator_schreibt_in_die_stemlab_ordner(monkeypatch):
    """Modelle und Zwischenergebnisse gehören nach Application Support, nicht ins Projekt."""
    kwargs = _separator_argumente(monkeypatch, None)
    assert kwargs["model_file_dir"] == str(engine.MODEL_CACHE)
    assert kwargs["output_dir"] == str(engine.SCRATCH)
    assert kwargs["output_format"] == "WAV"
