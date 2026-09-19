"""Zustandsbehafteter Teil von engine.py: Cache, Gerätewahl, Taps.

engine hält drei Modul-Globals, die Tests aneinander koppeln würden:
_sep_cache (das geladene Modell), _force_cpu (einmal gesetzt, bleibt es für
den Prozess) und _available_models. Die Fixture unten setzt sie zurück,
damit kein Test den nächsten beeinflusst.
"""

from __future__ import annotations

import logging
import sys

import pytest

import engine


@pytest.fixture(autouse=True)
def engine_zustand():
    """Sichert die Modul-Globals von engine und stellt sie danach wieder her.

    _sep_cache wird flach kopiert, nicht per Referenz gehalten: engine ruft
    darauf .update(), was das Original sonst mitverändern würde.
    """
    cache = dict(engine._sep_cache)
    force_cpu = engine._force_cpu
    modelle = set(engine._available_models)
    katalog = {c.key: (c.resolved, c.verified) for c in engine.CATALOG}
    try:
        yield
    finally:
        engine._sep_cache.clear()
        engine._sep_cache.update(cache)
        engine._force_cpu = force_cpu
        engine._available_models = modelle
        for choice in engine.CATALOG:
            choice.resolved, choice.verified = katalog[choice.key]


class FakeSeparator:
    """Doppelgänger für audio_separator.Separator.

    Hält fest, wie er gerufen wurde, und kann auf Wunsch beim Trennen eine
    Ausnahme werfen – damit lässt sich der MPS-Rückfall in run_model prüfen,
    ohne je ein Modell zu laden.
    """

    def __init__(self, preset=None, ausgaben=None, fehler=None,
                 torch_device="cpu", onnx_execution_provider=None):
        self.preset = preset
        self._ausgaben = list(ausgaben or [])
        self._fehler = list(fehler or [])
        self.torch_device = torch_device
        self.onnx_execution_provider = onnx_execution_provider
        self.load_model_aufrufe: list[dict] = []
        self.separate_aufrufe: list[str] = []

    def load_model(self, **kwargs):
        self.load_model_aufrufe.append(kwargs)

    def separate(self, pfad):
        self.separate_aufrufe.append(pfad)
        if self._fehler:
            fehler = self._fehler.pop(0)
            if fehler is not None:
                raise fehler
        return list(self._ausgaben)


class SeparatorFabrik:
    """Merkt sich jeden erzeugten Fake – _get_separator liefert ihn nur zurück."""

    def __init__(self, **vorgaben):
        self.vorgaben = vorgaben
        self.erzeugte: list[FakeSeparator] = []

    def __call__(self, preset=None):
        sep = FakeSeparator(preset=preset, **self.vorgaben)
        self.erzeugte.append(sep)
        return sep

    @property
    def letzter(self) -> FakeSeparator:
        return self.erzeugte[-1]

    @property
    def ladevorgaenge(self) -> int:
        return len(self.erzeugte)


@pytest.fixture
def separator_fabrik(monkeypatch):
    """Schleust FakeSeparator an der Stelle ein, an der engine das echte Paket importiert."""
    def einrichten(**vorgaben) -> SeparatorFabrik:
        fabrik = SeparatorFabrik(**vorgaben)
        monkeypatch.setattr(engine, "_make_separator", fabrik)
        return fabrik
    return einrichten


@pytest.fixture
def stille():
    """on_log-Rückruf, der die Meldungen sammelt statt sie zu drucken."""
    meldungen: list[str] = []

    def sammeln(text: str) -> None:
        meldungen.append(text)

    sammeln.meldungen = meldungen  # type: ignore[attr-defined]
    return sammeln


# Die Fixture selbst braucht einen Beleg: Diese beiden Tests laufen in der
# Reihenfolge ihrer Definition. Der erste verstellt jeden Global, der zweite
# erwartet überall den Ausgangswert. Ohne Fixture fällt der zweite.
def test_zustand_verstellen():
    engine._force_cpu = True
    engine._sep_cache.update(key="egal", separator=object())
    engine._available_models = {"erfunden.ckpt"}
    engine.CATALOG[0].resolved = "erfunden.ckpt"
    engine.CATALOG[0].verified = True


def test_zustand_ist_wieder_am_ausgangspunkt():
    assert engine._force_cpu is False
    assert engine._sep_cache == {"key": None, "separator": None}
    assert engine._available_models == set()
    assert engine.CATALOG[0].resolved is None
    assert engine.CATALOG[0].verified is False


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
# _ProgressTap
# --------------------------------------------------------------------------- #

class FakeStream:
    """Nimmt entgegen, was der Tap durchreicht."""

    def __init__(self, tty: bool = False):
        self.geschrieben: list[str] = []
        self.geleert = 0
        self._tty = tty
        self.encoding = "utf-8"

    def write(self, text: str) -> int:
        self.geschrieben.append(text)
        return len(text)

    def flush(self) -> None:
        self.geleert += 1

    def isatty(self) -> bool:
        return self._tty


@pytest.fixture
def tap():
    """Liefert (Tap, Strom, Meldungen) – Meldungen sind (Prozent, Durchgang)."""
    def bauen(tty: bool = False):
        strom = FakeStream(tty)
        meldungen: list[tuple[int, int]] = []
        return engine._ProgressTap(strom, lambda p, d: meldungen.append((p, d))), strom, meldungen
    return bauen


def test_progresstap_liest_prozente_aus_tqdm(tap):
    tapper, _strom, meldungen = tap()
    tapper.write(" 42%|####      | 42/100")
    assert meldungen == [(42, 1)]


def test_progresstap_reicht_text_unveraendert_weiter(tap):
    """Der Tap spiegelt stderr, er verschluckt es nicht."""
    tapper, strom, _ = tap()
    tapper.write("irgendwas ohne Prozent")
    assert strom.geschrieben == ["irgendwas ohne Prozent"]


def test_progresstap_mehrere_prozente_in_einem_block(tap):
    tapper, _strom, meldungen = tap()
    tapper.write("10%| ... 20%| ... 30%|")
    assert meldungen == [(10, 1), (20, 1), (30, 1)]


def test_progresstap_zaehlt_durchgang_hoch_wenn_prozent_zurueckfaellt(tap):
    """Ein neuer tqdm-Balken beginnt wieder bei null – das ist Durchgang 2."""
    tapper, _strom, meldungen = tap()
    tapper.write("50%|")
    tapper.write("10%|")
    assert meldungen == [(50, 1), (10, 2)]


@pytest.mark.parametrize("zweiter, erwarteter_durchgang", [
    # Regel: pct + 5 < last. Kleine Rückschritte sind tqdm-Rauschen,
    # erst ein echter Sprung nach unten ist ein neuer Durchgang.
    (46, 1),   # 46+5=51, nicht < 50
    (45, 1),   # 45+5=50, nicht < 50
    (44, 2),   # 44+5=49  < 50
    (0, 2),
])
def test_progresstap_grenze_der_durchgangserkennung(tap, zweiter, erwarteter_durchgang):
    tapper, _strom, meldungen = tap()
    tapper.write("50%|")
    tapper.write(f"{zweiter}%|")
    assert meldungen[-1] == (zweiter, erwarteter_durchgang)


def test_progresstap_werfender_rueckruf_bricht_nicht_ab(tap):
    """Ein Fehler in der Oberfläche darf die laufende Trennung nicht killen."""
    strom = FakeStream()
    tapper = engine._ProgressTap(strom, lambda _p, _d: 1 / 0)
    assert tapper.write("50%|") == len("50%|")
    assert strom.geschrieben == ["50%|"]


def test_progresstap_ueberlebt_geschlossenen_strom():
    """Beim Beenden kann stderr schon zu sein."""
    class Kaputt:
        encoding = "utf-8"

        def write(self, _text):
            raise ValueError("closed")

        def flush(self):
            raise ValueError("closed")

    tapper = engine._ProgressTap(Kaputt(), lambda _p, _d: None)
    assert tapper.write("50%|") == len("50%|")
    tapper.flush()


def test_progresstap_meldet_writable_und_isatty(tap):
    """audio_separator prüft beides, bevor es tqdm anwirft."""
    tapper, _strom, _ = tap(tty=True)
    assert tapper.writable() is True
    assert tapper.isatty() is True
    assert tap(tty=False)[0].isatty() is False


# --------------------------------------------------------------------------- #
# _LogTap
# --------------------------------------------------------------------------- #

def _record(text: str, level: int = logging.INFO, args=None) -> logging.LogRecord:
    return logging.LogRecord("audio_separator", level, __file__, 1, text, args, None)


def test_logtap_reicht_meldung_weiter(stille):
    engine._LogTap(stille).emit(_record("Lade Modell"))
    assert stille.meldungen == ["Lade Modell"]


def test_logtap_setzt_platzhalter_ein(stille):
    """getMessage() statt record.msg – sonst stünde "%s %%" in der Oberfläche."""
    engine._LogTap(stille).emit(_record("Modell %s bei %d %%", args=("mdx", 50)))
    assert stille.meldungen == ["Modell mdx bei 50 %"]


def test_logtap_werfender_rueckruf_bricht_nicht_ab():
    """Logging darf die Trennung nie zum Absturz bringen."""
    engine._LogTap(lambda _m: 1 / 0).emit(_record("egal"))


def test_logtap_haengt_unter_info_nichts_durch(stille):
    """Handler-Level INFO: DEBUG-Rauschen der Bibliothek bleibt draußen."""
    logger = logging.getLogger("test-stemlab-logtap")
    logger.setLevel(logging.DEBUG)
    tap = engine._LogTap(stille)
    logger.addHandler(tap)
    try:
        logger.debug("unsichtbar")
        logger.info("sichtbar")
    finally:
        logger.removeHandler(tap)
    assert stille.meldungen == ["sichtbar"]


# --------------------------------------------------------------------------- #
# describe_device
# --------------------------------------------------------------------------- #

class FakeDevice:
    def __init__(self, torch_device="cpu", onnx_execution_provider=None):
        self.torch_device = torch_device
        self.onnx_execution_provider = onnx_execution_provider


@pytest.mark.parametrize("sep, erwartet", [
    (FakeDevice("mps:0"), "mps"),
    (FakeDevice("cpu"), "cpu"),
    (FakeDevice("cuda:0"), "cpu"),
    # CoreML sticht das Torch-Device: die Trennung läuft dann über ONNX.
    (FakeDevice("cpu", ["CoreMLExecutionProvider", "CPUExecutionProvider"]), "coreml"),
    (FakeDevice("mps:0", "CoreMLExecutionProvider"), "coreml"),
    (FakeDevice("cpu", ["CPUExecutionProvider"]), "cpu"),
    (FakeDevice("cpu", []), "cpu"),
])
def test_describe_device(sep, erwartet):
    assert engine.describe_device(sep) == erwartet


def test_describe_device_ohne_attribute():
    """Ein Separator ohne torch_device darf nicht durchschlagen."""
    assert engine.describe_device(object()) == "cpu"


def test_detect_device_ohne_torch_ist_cpu(monkeypatch):
    """Ohne PyTorch – der Normalfall in der CI – bleibt es bei der CPU."""
    monkeypatch.setitem(sys.modules, "torch", None)
    assert engine.detect_device() == "cpu"


def test_detect_device_respektiert_force_cpu(monkeypatch):
    """Nach einem MPS-Fehler meldet die Oberfläche nicht weiter "mps"."""
    engine._force_cpu = True
    assert engine.detect_device() == "cpu"


def test_describe_device_nimmt_nur_den_ersten_provider():
    """onnxruntime probiert die Provider der Reihe nach: der erste gewinnt.

    Ohne das Entpacken auf provider[0] würde str(liste) geprüft, und ein
    CoreML weiter hinten in der Liste würde fälschlich als aktiv gemeldet.
    """
    sep = FakeDevice("cpu", ["CPUExecutionProvider", "CoreMLExecutionProvider"])
    assert engine.describe_device(sep) == "cpu"


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
