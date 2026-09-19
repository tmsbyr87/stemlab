"""Fortschritt, Log und Gerätemeldung von engine.py.

Die Teile, die zwischen audio_separator und der Oberfläche
vermitteln: tqdm-Prozente aus stderr, Logmeldungen der Bibliothek
und die Frage, worauf die Trennung gerade läuft.
"""

from __future__ import annotations

import logging
import sys

import pytest

import engine


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


def _torch_mit_mps(monkeypatch, verfuegbar=True, prozessor="arm"):
    """Stellt ein torch mit verfügbarer Apple-GPU nach.

    Ohne das prüft ein Test auf "cpu" gar nichts: Fehlt torch – der
    Normalfall in der CI –, liefert detect_device ohnehin "cpu", ganz
    gleich was _force_cpu sagt. Der Test wäre dort grün und leer.
    """
    import types

    torch = types.ModuleType("torch")
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: verfuegbar))
    monkeypatch.setitem(sys.modules, "torch", torch)

    plattform = types.ModuleType("platform")
    plattform.uname = lambda: types.SimpleNamespace(processor=prozessor)
    monkeypatch.setitem(sys.modules, "platform", plattform)


def test_detect_device_meldet_mps_auf_apple_silicon(monkeypatch):
    """Gegenstück zu den cpu-Fällen: sonst wäre nie belegt, dass mps
    überhaupt jemals herauskommt."""
    _torch_mit_mps(monkeypatch)
    assert engine.detect_device() == "mps"


def test_detect_device_respektiert_force_cpu(monkeypatch):
    """Nach einem MPS-Fehler meldet die Oberfläche nicht weiter "mps".

    Die Apple-GPU ist hier ausdrücklich verfügbar – nur _force_cpu
    verhindert sie. Ohne dieses Nachstellen prüfte der Test nichts.
    """
    _torch_mit_mps(monkeypatch)
    engine._force_cpu = True
    assert engine.detect_device() == "cpu"


def test_detect_device_ohne_apple_silicon_ist_cpu(monkeypatch):
    """Intel-Mac: mps meldet sich verfügbar, taugt aber nicht."""
    _torch_mit_mps(monkeypatch, prozessor="i386")
    assert engine.detect_device() == "cpu"


def test_detect_device_ohne_verfuegbares_mps_ist_cpu(monkeypatch):
    _torch_mit_mps(monkeypatch, verfuegbar=False)
    assert engine.detect_device() == "cpu"


def test_describe_device_nimmt_nur_den_ersten_provider():
    """onnxruntime probiert die Provider der Reihe nach: der erste gewinnt.

    Ohne das Entpacken auf provider[0] würde str(liste) geprüft, und ein
    CoreML weiter hinten in der Liste würde fälschlich als aktiv gemeldet.
    """
    sep = FakeDevice("cpu", ["CPUExecutionProvider", "CoreMLExecutionProvider"])
    assert engine.describe_device(sep) == "cpu"
