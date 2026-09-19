"""Zustandsbehafteter Teil von engine.py: Cache, Gerätewahl, Taps.

engine hält drei Modul-Globals, die Tests aneinander koppeln würden:
_sep_cache (das geladene Modell), _force_cpu (einmal gesetzt, bleibt es für
den Prozess) und _available_models. Die Fixture unten setzt sie zurück,
damit kein Test den nächsten beeinflusst.
"""

from __future__ import annotations

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
