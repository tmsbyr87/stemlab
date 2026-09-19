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
