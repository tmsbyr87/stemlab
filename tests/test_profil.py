"""Genre-Profil: Verteilungen über eine Menge von Tracks.

Ein Profil beschreibt, was eine Sammlung gemeinsam hat – Tempo, Tonart,
Energie, Aufbau. Nicht als Mittelwert, sondern als Spannweite: "80 % liegen
zwischen 122 und 126" sagt mehr als "Mittelwert 124", und ein Ausreißer
verschiebt es nicht.
"""

from __future__ import annotations

import pytest

import profil


def _a(**werte):
    """Eine Analyse mit Vorgabewerten, die sich einzeln überschreiben lassen.

    Camelot und Tonart gehören zusammen: A ist Moll, B ist Dur. Wer nur
    das eine setzt, bekommt das andere passend dazu – sonst entstehen
    widersprüchliche Testdaten, die etwas anderes prüfen als gedacht.
    """
    basis = {"bpm": 124.0, "energy": 7, "danceability": 8, "camelot": "8A",
             "key_mode": "minor", "seconds_analyzed": 380.0, "downbeats": 190}
    basis.update(werte)
    if "camelot" in werte and "key_mode" not in werte:
        basis["key_mode"] = "major" if str(werte["camelot"]).endswith("B") else "minor"
    return basis


# --------------------------------------------------------------------------- #
# Verteilungen
# --------------------------------------------------------------------------- #

def test_profil_nennt_median_und_spannweite():
    """Der Median hält stand, wo ein Mittelwert kippt."""
    p = profil.erstelle([_a(bpm=b) for b in (120, 122, 124, 126, 128)])

    assert p["tempo"]["median"] == 124
    assert p["tempo"]["min"] == 120
    assert p["tempo"]["max"] == 128


def test_ausreisser_verschiebt_den_median_nicht():
    """Ein halb erkannter Track mit 62 BPM darf das Profil nicht kippen."""
    ohne = profil.erstelle([_a(bpm=b) for b in (124, 125, 126)])
    mit = profil.erstelle([_a(bpm=b) for b in (124, 125, 126, 62)])

    assert abs(mit["tempo"]["median"] - ohne["tempo"]["median"]) <= 1


def test_profil_nennt_den_kernbereich():
    """Die Spannweite zeigt die Ränder, das Quartil den Kern: Wo liegt die
    Mehrheit, wenn man die Ausreißer wegdenkt?"""
    p = profil.erstelle([_a(bpm=b) for b in (118, 122, 123, 124, 125, 126, 132)])

    assert p["tempo"]["q1"] >= 120
    assert p["tempo"]["q3"] <= 130


def test_fehlende_werte_zaehlen_nicht_als_null():
    """Ältere Analysen kennen `danceability` noch nicht. Sie als 0 zu werten
    zöge den Median nach unten – in der vorhandenen Sammlung fehlt das Feld
    in 9 von 24 Dateien."""
    mit_luecken = [_a(), _a(), {k: v for k, v in _a().items() if k != "energy"}]

    p = profil.erstelle(mit_luecken)

    assert p["energie"]["median"] == 7
    assert p["energie"]["anzahl"] == 2, "Nur die Tracks mit Wert zählen"


# --------------------------------------------------------------------------- #
# Tonart
# --------------------------------------------------------------------------- #

def test_profil_zaehlt_die_haeufigsten_tonarten():
    """Der deutlichste Unterschied zwischen zwei Genres in der Vorprobe."""
    p = profil.erstelle([_a(camelot=c) for c in ("8A", "8A", "8A", "5A", "12B")])

    assert p["tonart"]["haeufigste"][0] == ("8A", 3)
    assert p["tonart"]["moll_anteil"] == pytest.approx(0.8, abs=0.01)


def test_moll_anteil_bei_reinem_dur():
    p = profil.erstelle([_a(key_mode="major", camelot="8B") for _ in range(4)])
    assert p["tonart"]["moll_anteil"] == 0.0


# --------------------------------------------------------------------------- #
# Umfang und Belastbarkeit
# --------------------------------------------------------------------------- #

def test_profil_nennt_seine_groesse():
    p = profil.erstelle([_a() for _ in range(17)])
    assert p["anzahl"] == 17


def test_kleine_gruppen_werden_als_unsicher_gekennzeichnet():
    """Unter etwa 15 Tracks sind Verteilungen Zufall. Das gehört angezeigt,
    nicht verschwiegen – sonst liest jemand aus fünf Tracks eine Regel."""
    assert profil.erstelle([_a() for _ in range(5)])["belastbar"] is False
    assert profil.erstelle([_a() for _ in range(20)])["belastbar"] is True


def test_leeres_profil_wirft_nicht():
    p = profil.erstelle([])
    assert p["anzahl"] == 0 and p["belastbar"] is False


# --------------------------------------------------------------------------- #
# Aufbau
# --------------------------------------------------------------------------- #

def test_profil_rechnet_laenge_in_takten():
    """Produziert wird in Takten, nicht in Minuten."""
    p = profil.erstelle([_a(downbeats=n) for n in (180, 190, 200)])

    assert p["takte"]["median"] == 190


def test_profil_haelt_auch_die_spielzeit_fest():
    p = profil.erstelle([_a(seconds_analyzed=s) for s in (360, 380, 400)])
    assert p["laenge"]["median"] == pytest.approx(380, abs=1)


def test_takte_auch_aus_der_rohen_analyse():
    """In analysis.json steht unter "downbeats" die Liste der Zeitpunkte,
    in der API-Fassung deren Anzahl. Ein Profil bekommt beides zu sehen –
    an den echten Dateien fiel genau das auf."""
    roh = [_a(downbeats=[0.0, 1.9, 3.8, 5.7]) for _ in range(3)]
    assert profil.erstelle(roh)["takte"]["median"] == 4

    gemischt = [_a(downbeats=[0.0] * 190), _a(downbeats=190)]
    assert profil.erstelle(gemischt)["takte"]["median"] == 190


def test_fehlende_felder_verschieben_den_median_nicht():
    """Die schärfere Fassung: Nicht nur die Anzahl muss stimmen, auch der
    Median darf sich nicht bewegen, wenn Felder fehlen.

    Würden fehlende Werte als 0 gezählt, sänke der Median bei drei
    Tracks mit Energie 8 und drei ohne Feld von 8 auf 4.
    """
    mit = [_a(energy=8) for _ in range(3)]
    ohne = [{k: v for k, v in _a().items() if k != "energy"} for _ in range(3)]

    p = profil.erstelle(mit + ohne)

    assert p["energie"]["median"] == 8
    assert p["energie"]["anzahl"] == 3


def test_null_und_negativ_zaehlen_nicht_als_messwert():
    """Eine Energie von 0 heißt "nicht gemessen", nicht "sehr leise" –
    und ein Tempo von 0 ist keine Angabe."""
    p = profil.erstelle([_a(bpm=124), _a(bpm=126), _a(bpm=0)])

    assert p["tempo"]["anzahl"] == 2
    assert p["tempo"]["min"] == 124
