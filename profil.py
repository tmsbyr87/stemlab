"""Genre-Profil: Was eine Sammlung von Tracks gemeinsam hat.

Ein Profil beschreibt eine Menge von Analysen als Verteilung, nicht als
Mittelwert. "80 % liegen zwischen 122 und 126 BPM" ist eine Aussage, mit
der man arbeiten kann; "Mittelwert 124,3" täuscht eine Genauigkeit vor,
die nicht da ist, und kippt beim ersten halb erkannten Tempo.

Was ein Profil NICHT ist: eine Erfolgsformel. Es beschreibt, was die
ausgewählten Tracks gemeinsam haben – nicht, was sie gut macht. Dieselben
Merkmale hat mittelmäßige Musik desselben Genres auch. Wer daraus eine
Regel ableiten will, braucht eine Vergleichsgruppe.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

# Unter dieser Zahl sind Verteilungen Zufall. Bei acht Tracks liegt der
# Median irgendwo, und zwei Ausreißer bestimmen die Spannweite.
MINDESTGROESSE = 15


def _verteilung(werte: list[float]) -> dict:
    """Median, Quartile und Ränder – oder ein leeres Feld, wenn nichts da ist.

    Median und Quartile statt Mittelwert und Streuung: Ein Track, dessen
    Tempo halbiert erkannt wurde, verschiebt den Mittelwert um zehn BPM,
    den Median um keinen.
    """
    sauber = [float(w) for w in werte if w is not None and float(w) > 0]
    if not sauber:
        return {"anzahl": 0}
    reihe = np.array(sorted(sauber))
    return {
        "anzahl": len(reihe),
        "median": round(float(np.median(reihe)), 1),
        "q1": round(float(np.percentile(reihe, 25)), 1),
        "q3": round(float(np.percentile(reihe, 75)), 1),
        "min": round(float(reihe[0]), 1),
        "max": round(float(reihe[-1]), 1),
    }


def _anzahl(wert):
    """Listen zählen, Zahlen durchreichen.

    In analysis.json steht unter "downbeats" die Liste der Zeitpunkte, in
    der API-Fassung dagegen deren Anzahl. Ein Profil bekommt beides zu
    sehen, je nachdem woher die Analyse stammt.
    """
    if isinstance(wert, (list, tuple)):
        return len(wert)
    return wert


def _feld(analysen: list[dict], name: str) -> list:
    """Nur die Tracks, die das Feld kennen.

    Ältere Analysen kennen `danceability` nicht – in einer vorhandenen
    Sammlung fehlt es in 9 von 24 Dateien. Ein fehlendes Feld als 0 zu
    werten zöge den Median nach unten und behauptete etwas Falsches.

    Die Filterung in _verteilung() würde solche Nullen ohnehin verwerfen;
    diese Prüfung hier ist die zweite Absicherung derselben Sache. Sie
    bleibt, weil sie die Absicht an der Stelle ausdrückt, an der das Feld
    gelesen wird – eine Mutationsprobe kann sie deshalb nicht
    unterscheiden.
    """
    return [_anzahl(a[name]) for a in analysen if a.get(name) is not None]


def erstelle(analysen: list[dict]) -> dict:
    """Ein Profil aus einer Menge von Analysen."""
    n = len(analysen)
    if not n:
        return {"anzahl": 0, "belastbar": False}

    camelots = [a.get("camelot") for a in analysen if a.get("camelot") and a["camelot"] != "–"]
    moll = sum(1 for a in analysen if a.get("key_mode") == "minor")
    mit_tonart = sum(1 for a in analysen if a.get("key_mode"))

    return {
        "anzahl": n,
        # Ehrlich über die eigene Aussagekraft: Wer aus fünf Tracks eine
        # Regel liest, irrt sich – das gehört angezeigt.
        "belastbar": n >= MINDESTGROESSE,
        "mindestgroesse": MINDESTGROESSE,
        "tempo": _verteilung(_feld(analysen, "bpm")),
        "energie": _verteilung(_feld(analysen, "energy")),
        "tanzbarkeit": _verteilung(_feld(analysen, "danceability")),
        "laenge": _verteilung(_feld(analysen, "seconds_analyzed")),
        "takte": _verteilung(_feld(analysen, "downbeats")),
        "tonart": {
            "haeufigste": Counter(camelots).most_common(5),
            "moll_anteil": round(moll / mit_tonart, 2) if mit_tonart else 0.0,
            "anzahl": mit_tonart,
        },
    }
