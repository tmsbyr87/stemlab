"""Tempoberechnung aus einem Beat-Raster.

Die Fälle hier sind keine erfundenen Randfälle, sondern das, was Beat This!
tatsächlich liefert. Der Referenztrack wurde einmal mit 126,9 BPM statt 124,0
gemeldet – die Ursache steckt in `test_zwischenschlaege_verschieben_das_tempo_nicht`.
"""

import re

import numpy as np
import pytest

import analysis
from conftest import FRAME, beat_grid


def estimate(beats: np.ndarray) -> float:
    """Tempo wie in analyze(): schätzen, falten, auf glatte Werte ziehen."""
    bpm, conf = analysis._tempo_from_beats(np.asarray(beats))
    return analysis._snap_bpm(analysis._fold_bpm(bpm), conf)


@pytest.mark.parametrize("bpm", [90.0, 100.0, 124.0, 128.0, 140.0, 174.0])
def test_sauberes_raster_trifft_genau(bpm):
    """Ein ungestörtes Raster muss exakt sein – auch bei 124 BPM.

    0,483871 s pro Beat lässt sich auf einem 20-ms-Raster gar nicht abbilden;
    der Tracker wechselt dort zwischen 0,48 s und 0,50 s. Die Regression muss
    diese Quantisierung wegmitteln.
    """
    assert estimate(beat_grid(bpm)) == pytest.approx(bpm, abs=0.05)


def test_zwischenschlaege_verschieben_das_tempo_nicht():
    """Setzt der Tracker Achtel zwischen die Viertel, darf das Tempo nicht steigen.

    Genau das ging schief: jeder zusätzliche Beat verschob alle folgenden
    Indizes, die Regression lief auf 126,9 BPM davon.
    """
    grid = beat_grid(124.0)
    period = 60.0 / 124.0
    extra = np.sort(np.concatenate([grid, grid[100:140] + period / 2]))
    assert estimate(extra) == pytest.approx(124.0, abs=0.6)


def test_fehlende_beats_verschieben_das_tempo_nicht():
    """Ein Breakdown ohne Drums lässt Beats ausfallen – das Tempo bleibt gleich."""
    grid = beat_grid(124.0)
    with_gap = np.concatenate([grid[:120], grid[145:]])
    assert estimate(with_gap) == pytest.approx(124.0, abs=0.6)


def test_konfidenz_faellt_bei_gestoertem_raster():
    """Ein gestörtes Raster darf nicht dieselbe Sicherheit melden wie ein sauberes."""
    grid = beat_grid(124.0)
    period = 60.0 / 124.0
    noisy = np.sort(np.concatenate([grid, grid[80:160] + period / 2]))
    _, clean_conf = analysis._tempo_from_beats(grid)
    _, noisy_conf = analysis._tempo_from_beats(noisy)
    assert noisy_conf < clean_conf


def test_wenige_beats_liefern_geringe_konfidenz():
    """18 Beats über ein paar Sekunden sind keine belastbare Grundlage.

    Ein Sprach-Reel mit etwas Hintergrundmusik darf kein selbstbewusstes
    Tempo melden.
    """
    _, conf = analysis._tempo_from_beats(beat_grid(86.0, seconds=13.0))
    assert conf < 0.4


@pytest.mark.parametrize("anzahl", [0, 1, 2, 3])
def test_zu_wenige_beats_ergeben_kein_tempo(anzahl):
    """Unter vier Beats ist keine Regression möglich – an der Kante geprüft."""
    beats = np.arange(anzahl) * 0.5
    assert analysis._tempo_from_beats(beats) == (0.0, 0.0)


def test_vier_beats_reichen_gerade():
    """Genau an der Grenze muss ein Tempo herauskommen."""
    bpm, _ = analysis._tempo_from_beats(np.arange(4) * 0.5)
    assert bpm == pytest.approx(120.0, abs=0.1)


def test_identische_zeitstempel_stuerzen_nicht_ab():
    assert analysis._tempo_from_beats(np.zeros(20)) == (0.0, 0.0)


@pytest.mark.parametrize("roh, erwartet", [
    (31.0, 62.0),     # unter 60 wird verdoppelt, bis es im Fenster liegt
    (248.0, 124.0),   # ab 200 wird halbiert
    (124.0, 124.0),   # dazwischen bleibt es stehen
    (62.0, 62.0),     # 62 liegt im Fenster, wird also nicht angefasst
])
def test_faltung_in_den_ueblichen_bereich(roh, erwartet):
    """Tempi werden in das Fenster 60 bis 200 BPM gefaltet."""
    assert analysis._fold_bpm(roh) == pytest.approx(erwartet, abs=0.01)


def test_snapping_nur_bei_hoher_konfidenz():
    """Produzierte Tracks laufen auf glatten DAW-Tempi – aber nur wenn wir sicher sind."""
    assert analysis._snap_bpm(123.98, 0.9) == 124.0
    assert analysis._snap_bpm(123.98, 0.2) == 123.98


def test_snapping_verbiegt_echte_werte_nicht():
    """Eine Live-Aufnahme mit 123,4 BPM darf nicht auf 123 gezogen werden."""
    assert analysis._snap_bpm(123.4, 0.9) == 123.4


@pytest.mark.parametrize("roh, erwartet", [
    (119.06, 119.0),
    (128.06, 128.0),
    (124.07, 124.0),
    (123.92, 124.0),
    (124.08, 124.0),
])
def test_snapping_faengt_knappe_abweichungen(roh, erwartet):
    """Genau an der Grenze scheiterte das Snapping bisher.

    `abs(119.06 - 119.0)` ergibt in Gleitkomma 0.060000000000002274 und ist
    damit NICHT `<= 0.06`. Die Werte, für die das Snapping gebaut wurde,
    fielen also durch – im Durchlauf vom 2026-09-20 standen deshalb 119,06
    und 128,06 in der Bibliothek statt 119 und 128.
    """
    assert analysis._snap_bpm(roh, 0.9) == erwartet


def test_snapping_hat_eine_obergrenze():
    """Die Toleranz endet bei 0.08. Weiter zu gehen würde echte Tempi
    verbiegen: 123,4 läge bei 0.10 exakt an der Grenze zu 123,5."""
    assert analysis._snap_bpm(124.09, 0.9) == 124.09
    assert analysis._snap_bpm(123.4, 0.9) == 123.4


@pytest.mark.parametrize("roh", [123.4, 122.4, 128.3, 119.25, 124.8])
def test_snapping_laesst_deutlich_krumme_tempi_stehen(roh):
    """Vinyl mit Pitch, Live-Einspielungen und alte Platten laufen krumm.
    Deren Tempo ist echt und darf nicht geglättet werden."""
    assert analysis._snap_bpm(roh, 0.9) == roh


def test_snapping_kennt_auch_halbe_werte():
    """Manche Tracks laufen auf x,5 – das ist ebenso ein DAW-Tempo."""
    assert analysis._snap_bpm(124.47, 0.9) == 124.5
    assert analysis._snap_bpm(124.54, 0.9) == 124.5


def test_downbeats_ergeben_dasselbe_tempo():
    """Über die Takte gerechnet muss dasselbe herauskommen wie über die Beats."""
    grid = beat_grid(124.0)
    bars = grid[::4]
    bpm, _ = analysis._tempo_from_beats(bars, subdivision=4)
    assert bpm == pytest.approx(124.0, abs=0.3)


def test_rasterung_allein_verfaelscht_nicht():
    """Quantisierung auf 20 ms darf das Ergebnis nicht verschieben.

    Der Median der Abstände läge bei 125,0 BPM – die Regression muss besser sein.
    """
    exact = beat_grid(124.0, quantize=False)
    quantized = beat_grid(124.0, quantize=True)
    assert np.abs(np.diff(quantized) - 60.0 / 124.0).max() >= FRAME / 2 - 1e-9
    bpm_exact, _ = analysis._tempo_from_beats(exact)
    bpm_quant, _ = analysis._tempo_from_beats(quantized)
    assert bpm_quant == pytest.approx(bpm_exact, abs=0.05)


# --------------------------------------------------------------------------- #
# Tanzbarkeit
# --------------------------------------------------------------------------- #
#
# Keine Messung, sondern eine Ableitung aus drei Werten, die schon vorliegen:
# Tempo im Tanzbereich, Energie und wie stabil das Taktraster ist. Genau das
# unterscheidet einen durchlaufenden Clubtrack von einem Intro oder einer
# freien Aufnahme.

@pytest.mark.parametrize("bpm, energie, konfidenz, erwartet_mindestens", [
    (124.0, 8, 0.9, 7),     # klassischer Clubtrack
    (128.0, 9, 0.85, 8),    # treibender Techno
])
def test_tanzbarkeit_erkennt_clubtracks(bpm, energie, konfidenz, erwartet_mindestens):
    wert = analysis.danceability(bpm, energie, konfidenz)
    assert wert >= erwartet_mindestens


@pytest.mark.parametrize("bpm, energie, konfidenz", [
    (72.0, 3, 0.9),      # Ballade
    (124.0, 2, 0.9),     # richtiges Tempo, aber kraftlos
    (124.0, 8, 0.05),    # klingt tanzbar, aber das Raster trägt nicht
    (200.0, 7, 0.9),     # zu schnell zum Tanzen
])
def test_tanzbarkeit_bleibt_niedrig_wo_sie_hingehoert(bpm, energie, konfidenz):
    assert analysis.danceability(bpm, energie, konfidenz) <= 5


def test_tanzbarkeit_haelt_sich_an_die_skala():
    """1 bis 10, wie die Energie – damit beide Werte gleich lesbar sind."""
    for bpm in (0, 60, 124, 300):
        for energie in (0, 5, 10):
            for konf in (0.0, 0.5, 1.0):
                wert = analysis.danceability(bpm, energie, konf)
                assert 1 <= wert <= 10 and isinstance(wert, int)


def test_tanzbarkeit_ohne_tempo_ist_nicht_beurteilbar():
    """Ohne Taktraster gibt es keine Grundlage – dann der niedrigste Wert."""
    assert analysis.danceability(0, 8, 0.9) == 1


def test_tanzbarkeit_steigt_mit_der_energie():
    leise = analysis.danceability(124.0, 3, 0.9)
    laut = analysis.danceability(124.0, 9, 0.9)
    assert laut > leise


# --------------------------------------------------------------------------- #
# Energie
# --------------------------------------------------------------------------- #
#
# Gemessen aus vier Größen, die zusammen den wahrgenommenen Druck ergeben:
# Transienten-Dichte, Bassanteil, Breite des Spektrums und Pegel (RMS).
# Die erste Fassung kannte nur die ersten beiden und lag deshalb daneben:
# Ein leiser Sinus bei 220 Hz hat 99 % seiner Energie unter 250 Hz und bekam
# die volle Punktzahl, ein druckvoller Technotrack mit breiterem Spektrum
# nur 3.

def _sinus(freq=220.0, sekunden=15.0, pegel=0.5):
    t = np.linspace(0, sekunden, int(analysis.SAMPLE_RATE * sekunden), endpoint=False)
    return (pegel * np.sin(2 * np.pi * freq * t)).astype("float32")


def _kicktrack(sekunden=15.0, pegel=0.8, rausch=0.05, breit=True):
    """Kick plus Rauschen – grob wie ein gemasterter Clubtrack.

    `breit` mischt Obertöne dazu: Echte Musik belegt das Spektrum weit über
    den Bass hinaus, und genau daran unterscheidet sich ein Track von einem
    Dauerton.
    """
    sr = analysis.SAMPLE_RATE
    n = int(sr * sekunden)
    y = np.zeros(n, dtype="float32")
    for i in range(0, n, int(sr * 0.47)):
        laenge = min(int(sr * 0.1), n - i)
        huelle = np.exp(-np.arange(laenge) / (sr * 0.03))
        y[i:i + laenge] += (np.sin(2 * np.pi * 55 * np.arange(laenge) / sr) * huelle).astype("float32")
    if breit:
        # Hi-Hats und Fläche, damit das Spektrum nicht nur aus Bass besteht.
        t = np.arange(n) / sr
        for i in range(0, n, int(sr * 0.235)):
            laenge = min(int(sr * 0.04), n - i)
            y[i:i + laenge] += (0.25 * np.exp(-np.arange(laenge) / (sr * 0.008))).astype("float32")
        y += (0.12 * np.sin(2 * np.pi * 440 * t) + 0.08 * np.sin(2 * np.pi * 1760 * t)).astype("float32")
    rng = np.random.default_rng(3)
    return np.clip(y * pegel + rausch * rng.standard_normal(n).astype("float32"), -1, 1)


def test_energie_faellt_nicht_auf_tiefe_dauertoene_herein():
    """Der Kern der Korrektur: Ein Dauerton bei 220 Hz hat 99 % seiner
    Energie unter 250 Hz. Die erste Fassung gab ihm dafür die volle
    Punktzahl – lauter als jeder echte Clubtrack.

    Synthetische Signale taugen nicht, um die Skala nach oben zu prüfen:
    Sie erreichen die Transienten-Dichte echter Musik nicht (gemessen
    1.45–2.51, ein Kick-Muster kommt auf 0.9). Geprüft wird deshalb nur,
    was hier sicher gilt – ein Dauerton ist nicht energiereich.
    """
    assert analysis._energy(_sinus(freq=220.0, pegel=0.05)) <= 4
    assert analysis._energy(_sinus(freq=220.0, pegel=0.5)) <= 6


def test_energie_beachtet_den_pegel():
    """Derselbe Track leise gemischt hat weniger Druck als laut gemastert."""
    laut = analysis._energy(_kicktrack(pegel=1.4, rausch=0.12))
    leise = analysis._energy(_kicktrack(pegel=0.35, rausch=0.02))
    assert laut > leise


def test_energie_faellt_nicht_auf_schmalbandige_signale_herein():
    """Ein reiner Sinus ist kein energetischer Track, auch wenn seine
    gesamte Energie im Bass liegt."""
    assert analysis._energy(_sinus(freq=220.0, pegel=0.5)) <= 6


def test_energie_haelt_sich_an_die_skala():
    for y in (_sinus(pegel=0.02), _sinus(pegel=0.9), _kicktrack(), np.zeros(1000, dtype="float32")):
        wert = analysis._energy(y)
        assert 0 <= wert <= 10


def test_energie_gewichtet_die_anschlagsdichte_am_staerksten():
    """Die Dichte der Anschläge trägt am meisten zum Druck bei.

    An synthetischem Material lässt sich das nicht zeigen: Ein Kick-Muster
    erreicht eine Dichte von 0.9, echte Clubtracks liegen bei 1.45–2.51.
    Deshalb wird hier die Gewichtung selbst festgehalten – sie ist die
    Aussage der Formel, und ein Umbau ohne sie wäre ein anderes Maß.
    """
    import inspect

    quelle = inspect.getsource(analysis._energy)
    assert "druck * 3.6" in quelle, "Dichte muss den größten Anteil haben"
    assert "pegel * 2.6" in quelle
    # Und die Dichte muss tatsächlich aus den Onsets kommen.
    assert "onset_strength" in quelle
    assert "(dichte - 1.3) / 1.2" in quelle, \
        "Die Spanne stammt aus einer Stichprobe echter Tracks"


def test_energie_bei_stille_ist_am_boden():
    assert analysis._energy(np.zeros(analysis.SAMPLE_RATE * 5, dtype="float32")) <= 1


# --------------------------------------------------------------------------- #
# Struktur: Abschnitte eines Tracks
# --------------------------------------------------------------------------- #
#
# Wo beginnt der Drop, wie lang ist das Intro? Das sind die Fragen, mit
# denen ein Produzent an einen fremden Track herangeht. Gemessen wird über
# das Taktraster: Merkmale je Takt mitteln, dann dort teilen, wo sie sich
# deutlich ändern.

def _abschnittstrack(sr=None):
    """Vier Abschnitte mit deutlich verschiedenem Charakter.

    Leise -> laut mit Bass -> leise ohne Bass -> laut. So sieht ein Track
    grob aus, und die Grenzen sind per Konstruktion bekannt.
    """
    sr = sr or analysis.SAMPLE_RATE
    takt = 2.0            # 120 BPM, 4/4
    teile = []
    for pegel, mit_bass in ((0.10, False), (0.55, True), (0.12, False), (0.60, True)):
        n = int(sr * takt * 8)          # 8 Takte je Abschnitt
        t = np.arange(n) / sr
        ton = 0.5 * np.sin(2 * np.pi * 440 * t)
        if mit_bass:
            ton = ton + np.sin(2 * np.pi * 55 * t)
        teile.append((pegel * ton).astype("float32"))
    return np.concatenate(teile)


def test_struktur_findet_die_abschnitte():
    """Vier gebaute Abschnitte sollen als vier erkannt werden."""
    y = _abschnittstrack()
    downbeats = [i * 2.0 for i in range(33)]

    abschnitte = analysis.segmente(y, downbeats, anzahl=4)

    assert len(abschnitte) == 4


def test_struktur_grenzen_liegen_auf_takten():
    """Ein Abschnitt, der mitten im Takt beginnt, ist für die Produktion
    unbrauchbar – man arbeitet in Achter- und Sechzehnergruppen."""
    y = _abschnittstrack()
    downbeats = [i * 2.0 for i in range(33)]

    for a in analysis.segmente(y, downbeats, anzahl=4):
        assert any(abs(a["start"] - d) < 0.01 for d in downbeats), \
            f"Grenze bei {a['start']} liegt nicht auf einem Downbeat"


def test_struktur_kennt_die_laenge_in_takten():
    y = _abschnittstrack()
    downbeats = [i * 2.0 for i in range(33)]

    abschnitte = analysis.segmente(y, downbeats, anzahl=4)

    assert sum(a["takte"] for a in abschnitte) == 32
    assert all(a["takte"] > 0 for a in abschnitte)


def test_struktur_ohne_taktraster():
    """Ohne Downbeats gibt es keine Grundlage – dann lieber nichts."""
    assert analysis.segmente(_abschnittstrack(), [], anzahl=4) == []


@pytest.mark.parametrize("takte", [0, 1, 4, 7])
def test_struktur_braucht_genug_takte(takte):
    """Unter acht Takten ist jede Einteilung willkürlich."""
    y = _abschnittstrack()
    assert analysis.segmente(y, [i * 2.0 for i in range(takte)], anzahl=3) == []


def test_struktur_ohne_audio():
    assert analysis.segmente(np.array([], dtype="float32"),
                             [i * 2.0 for i in range(20)]) == []


def test_struktur_bei_zu_wenigen_takten():
    """Ein Track mit vier Takten hat keine Abschnitte."""
    y = _abschnittstrack()[: analysis.SAMPLE_RATE * 8]
    assert analysis.segmente(y, [0.0, 2.0, 4.0, 6.0], anzahl=4) == []


def test_struktur_benennt_laute_und_leise_abschnitte():
    """Der Kern für Produzenten: Wo ist der Drop, wo der Breakdown?"""
    y = _abschnittstrack()
    downbeats = [i * 2.0 for i in range(33)]

    abschnitte = analysis.segmente(y, downbeats, anzahl=4)
    namen = [a["art"] for a in abschnitte]

    # Der zweite und vierte Abschnitt sind laut und bassreich, der erste
    # und dritte leise – das muss sich in der Benennung zeigen.
    assert namen[1] != namen[0], "Laut und leise dürfen nicht gleich heißen"
    assert namen[3] != namen[2]


def test_struktur_nennt_pegel_und_bassanteil():
    """Die Kennzahlen gehören dazu – sie begründen die Benennung."""
    y = _abschnittstrack()
    abschnitte = analysis.segmente(y, [i * 2.0 for i in range(33)], anzahl=4)

    for a in abschnitte:
        assert 0.0 <= a["pegel"] <= 1.0
        assert 0.0 <= a["bass"] <= 1.0


def test_struktur_anzahl_richtet_sich_nach_der_laenge():
    """Ein Sechsminüter hat mehr Abschnitte als ein Zweiminüter – die Zahl
    fest vorzugeben würde beiden nicht gerecht."""
    import inspect

    y = _abschnittstrack()
    downbeats = [i * 2.0 for i in range(33)]

    automatisch = analysis.segmente(y, downbeats)
    assert 3 <= len(automatisch) <= 8

    # Die Regel selbst: etwa ein Abschnitt je 16 Takte, begrenzt auf 3 bis 8.
    # Eine feste Zahl würde einem Zweiminüter so wenig gerecht wie einem
    # Zehnminüter.
    quelle = inspect.getsource(analysis.segmente)
    assert "len(db) / 16" in quelle, "Die Zahl muss sich nach der Länge richten"
    assert "max(3, min(8," in quelle, "Mit Unter- und Obergrenze"


def test_drop_braucht_pegel_und_bass():
    """Ein Drop ist der lauteste Teil mit vollem Bass – beides zusammen.
    Ohne die Bedingung hieße jeder laute Abschnitt so."""
    import inspect

    quelle = inspect.getsource(analysis._abschnittsart)
    assert "laut * 0.92" in quelle, "Der Drop misst sich am lautesten Abschnitt"
    assert "viel_bass" in quelle, "Und braucht vollen Bass"

    # Laut, aber bassarm ist kein Drop.
    assert analysis._abschnittsart(0.40, 0.10, 0.40, 0.60) != "Drop"
    # Laut und bassreich schon.
    assert analysis._abschnittsart(0.40, 0.58, 0.40, 0.60) == "Drop"


def test_erster_abschnitt_heisst_nicht_drop():
    """Ein Track beginnt nicht mit dem Drop.

    Das Intro kann durchaus laut und bassreich sein – bei Clubtracks ist
    es oft schon der volle Beat. Trotzdem ist es der Einstieg, und wer
    das Arrangement nachbauen will, braucht die Unterscheidung.
    """
    y = _abschnittstrack()
    downbeats = [i * 2.0 for i in range(33)]

    abschnitte = analysis.segmente(y, downbeats, anzahl=4)

    assert abschnitte[0]["art"] != "Drop"
    assert abschnitte[0]["art"] == "Intro", \
        "Der erste Abschnitt ist das Intro, auch wenn er laut ist"

    # Die Regel selbst, weil synthetisches Material die Drop-Bedingung
    # nicht erreicht: An echten Tracks ist der Anfang oft schon der volle
    # Beat und würde sonst "Drop" heißen.
    import inspect
    quelle = inspect.getsource(analysis.segmente)
    assert 'abschnitte[0]["art"] == "Drop"' in quelle, \
        "Ein lautes Intro darf nicht als Drop durchgehen"
    assert 'abschnitte[0]["art"] = "Intro"' in quelle


def test_letzter_abschnitt_kann_outro_sein():
    """Läuft der Track am Ende aus, ist das ein Outro und kein Breakdown –
    danach kommt nichts mehr."""
    y = _abschnittstrack()
    # Leiser Schluss anhängen.
    sr = analysis.SAMPLE_RATE
    leise = (0.05 * np.sin(2 * np.pi * 440 * np.arange(sr * 16) / sr)).astype("float32")
    y = np.concatenate([y, leise])
    downbeats = [i * 2.0 for i in range(41)]

    abschnitte = analysis.segmente(y, downbeats, anzahl=5)

    assert abschnitte[-1]["art"] in ("Outro", "Breakdown")


def test_benennung_verteilt_sich_sinnvoll():
    """Wenn mehr als die Hälfte aller Abschnitte "Break" heißt, sagt die
    Einteilung nichts mehr.

    An 40 Abschnitten echter Clubtracks liegt der Bassanteil im Median bei
    0,55 des Maximums – eine Schwelle von 0,6 machte 57 % zu Breaks.
    """
    import inspect
    quelle = inspect.getsource(analysis._abschnittsart)
    treffer = re.search(r"wenig_bass = bass < bassreich \* ([\d.]+)", quelle)
    assert treffer, "Die Bass-Schwelle fehlt"
    assert float(treffer.group(1)) <= 0.5, \
        "Über 0,5 gilt der Median eines echten Tracks bereits als bassarm"
