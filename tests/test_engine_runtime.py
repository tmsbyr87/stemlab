"""Zustandsbehafteter Teil von engine.py: Cache, Gerätewahl, Taps.

engine hält drei Modul-Globals, die Tests aneinander koppeln würden:
_sep_cache (das geladene Modell), _force_cpu (einmal gesetzt, bleibt es für
den Prozess) und _available_models. Die Fixture unten setzt sie zurück,
damit kein Test den nächsten beeinflusst.
"""

from __future__ import annotations

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
