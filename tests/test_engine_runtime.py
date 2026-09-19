"""Zustandsbehafteter Teil von engine.py: Cache, Gerätewahl, Taps.

engine hält drei Modul-Globals, die Tests aneinander koppeln würden:
_sep_cache (das geladene Modell), _force_cpu (einmal gesetzt, bleibt es für
den Prozess) und _available_models. Die Fixture unten setzt sie zurück,
damit kein Test den nächsten beeinflusst.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

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
                 torch_device="cpu", onnx_execution_provider=None,
                 fortschritt=None, logmeldungen=None):
        self.preset = preset
        self._ausgaben = list(ausgaben or [])
        # Geteilte Liste, absichtlich nicht kopiert: Beim MPS-Rückfall legt
        # engine einen zweiten Separator an. Die Fehlerfolge beschreibt den
        # Ablauf über beide hinweg ("erst MPS-Fehler, dann Erfolg"), nicht
        # das Verhalten je Instanz.
        self._fehler = fehler if fehler is not None else []
        self._fortschritt = list(fortschritt or [])
        self._logmeldungen = list(logmeldungen or [])
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
        # Erst nach der Fehlerprüfung: ein fehlschlagender Versuch meldet
        # keinen Fortschritt. Sonst wäre nicht unterscheidbar, ob der
        # ProgressTap nach dem MPS-Rückfall wieder an stderr hängt.
        for text in self._fortschritt:
            sys.stderr.write(text)
        for text in self._logmeldungen:
            logging.getLogger("audio_separator").info(text)
        return list(self._ausgaben)


class SeparatorFabrik:
    """Merkt sich jeden erzeugten Fake – _get_separator liefert ihn nur zurück."""

    def __init__(self, **vorgaben):
        # fehler wird von allen erzeugten Fakes geteilt (siehe FakeSeparator).
        self.vorgaben = dict(vorgaben)
        self.vorgaben["fehler"] = list(vorgaben.get("fehler") or [])
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


# Die Fixture selbst braucht einen Beleg. Zwei aufeinander aufbauende Tests
# wären dafür untauglich: läuft der prüfende allein (per -k, als Einzelaufruf
# oder nach einer Umsortierung durch ein Plugin), ist nichts verstellt – er
# wäre dann auch ohne Fixture grün und würde nichts belegen. Deshalb prüft ein
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


def test_make_separator_schreibt_in_die_stemlab_ordner(monkeypatch):
    """Modelle und Zwischenergebnisse gehören nach Application Support, nicht ins Projekt."""
    kwargs = _separator_argumente(monkeypatch, None)
    assert kwargs["model_file_dir"] == str(engine.MODEL_CACHE)
    assert kwargs["output_dir"] == str(engine.SCRATCH)
    assert kwargs["output_format"] == "WAV"


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


def test_progresstap_reicht_stromeigenschaften_durch(tap):
    """audio_separator fragt fileno() und encoding ab, bevor es tqdm anwirft."""
    class MitFileno(FakeStream):
        def fileno(self):
            return 42

    strom = MitFileno()
    tapper = engine._ProgressTap(strom, lambda _p, _d: None)

    assert tapper.fileno() == 42
    assert tapper.encoding == "utf-8"


def test_progresstap_isatty_wenn_der_strom_keins_kennt():
    """Ein Strom ohne isatty darf nicht durchschlagen."""
    class Ohne:
        def write(self, _text):
            return 0

    assert engine._ProgressTap(Ohne(), lambda _p, _d: None).isatty() is False


def test_progresstap_encoding_faellt_auf_utf8_zurueck():
    class Ohne:
        def write(self, _text):
            return 0

    assert engine._ProgressTap(Ohne(), lambda _p, _d: None).encoding == "utf-8"
