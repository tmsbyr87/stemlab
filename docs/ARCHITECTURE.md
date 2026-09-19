# Architecture Overview

Generated: 2026-09-19

StemLab ist eine lokale macOS-Anwendung: ein FastAPI-Server mit einer
Single-File-Weboberfläche, der Musikdateien in Stems zerlegt, sie analysiert
(Tempo, Tonart, Energie, Akkorde, Lyrics) und taggt oder für DJ-Software
exportiert. Kein Framework-Scaffolding, kein Paketmanager-Workspace – fünf
Quelldateien auf der obersten Ebene.

## Tech Stack

| Kategorie | Verwendet |
|---|---|
| Sprache | Python 3.10–3.12 (14 .py), HTML/JS/CSS (1 Datei) |
| Web-Framework | FastAPI + uvicorn, StaticFiles, SSE über StreamingResponse |
| Frontend | `static/index.html` – 1609 Zeilen, kein Build, kein npm |
| Audio/ML | audio-separator (Roformer/MDX/Demucs), beat_this, librosa, numpy, scipy |
| Lyrics | mlx-whisper (Apple GPU), faster-whisper (CPU-Rückfall) |
| Tags | mutagen (WAV/FLAC/MP3) |
| Externe Tools | ffmpeg, optional rubberband |
| Tests | pytest 8, httpx/TestClient; `pytest.ini`, `testpaths = tests` |
| CI | GitHub Actions: pytest auf macos-14 × Py 3.10/3.11/3.12; shellcheck |
| Paketierung | venv + `requirements.txt`; Installation über Shell-Skripte |
| Datenbank | keine – Zustand liegt in JSON-Sidecars neben den Audiodateien |

## File Structure

```
StemLab/
  server.py             # FastAPI-App, Job-Queue, HTTP-API      (849 Z.)
  engine.py             # Stem-Trennung, Modellkatalog, ffmpeg  (683 Z.)
  postprocess.py        # Tags, Cues, DJ-Export, Lyrics, Mixe  (1100 Z.)
  analysis.py           # Tempo, Tonart, Energie, Akkorde       (499 Z.)
  static/index.html     # gesamte Oberfläche, ohne Build       (1609 Z.)
  tests/                # 9 Testmodule + conftest-Fixtures
  .github/workflows/    # pytest-Matrix und shellcheck
  app/, docs/           # nur Icon bzw. Logo (je 1 PNG)
  setup.sh install.sh start.sh "StemLab installieren.command"
  _to_delete/           # untracked Arbeitsreste (ignoriert)
  venv/                 # untracked
```

## Module Map

| Modul | Verantwortung | Hängt ab von |
|---|---|---|
| `server.py` | HTTP-API, Job-Queue mit Worker-Thread, SSE-Events, Konfiguration, Pfad-Whitelisting | engine, postprocess |
| `engine.py` | Modellkatalog und -auflösung, Separator-Aufrufe, Fortschritts-/Log-Taps, ffmpeg-Konvertierung, Gerätewahl | analysis (indirekt über Job-Fluss) |
| `postprocess.py` | Alles nach der Trennung: Tags, Cover, Dateinamen, Cue-Erkennung, Rekordbox-/Traktor-XML, Loops, Pitch-Shift, Mixdown, Wellenformen, Transkription | analysis-Ergebnisse (als dict), ffmpeg |
| `analysis.py` | Reine Signalanalyse: Beats/Downbeats, BPM-Faltung, Chroma/Tonart, Energie, Akkorde, Click-MIDI, Sidecar-JSONs | numpy, librosa, beat_this, scipy |
| `static/index.html` | Gesamte Oberfläche: Upload, Job-Ansicht, Player, Tag-Editor, Einstellungen-Lightbox | `/api/*` |

`postprocess.py` mit 1100 Zeilen und ~40 öffentlichen Funktionen deckt
mehrere Themen ab (Metadaten, DJ-Export, Audiobearbeitung, Lyrics). Das ist
der plausibelste Kandidat für eine Aufteilung, falls die Datei weiter wächst.

## Entry Points

| Typ | Datei | Zweck |
|---|---|---|
| App | `server.py` (`_startup`, uvicorn) | Startet Server, wärmt Modellkatalog vor |
| Start | `start.sh` | Sucht freien Port 8765–8799, öffnet Browser |
| Installation | `setup.sh`, `install.sh`, `StemLab installieren.command` | venv, Abhängigkeiten, App-Bundle |
| API | `/api/jobs`, `/api/actions`, `/api/events` (SSE), `/api/settings`, `/api/tags`, `/api/cover`, `/api/export`, `/api/separate-folder`, `/api/library`, `/api/analyses`, `/api/audio`, `/api/quit` | 30+ Routen in `server.py` |
| Tests | `tests/test_*.py` | pytest über `pytest.ini` |

## Dependencies

```
server.py       → engine, postprocess
engine.py       → (audio-separator, ffmpeg)
postprocess.py  → numpy, mutagen, ffmpeg, whisper
analysis.py     → numpy, librosa, scipy, beat_this
static/index.html → server.py (HTTP)
```

Keine zyklischen Importe. Die Richtung ist durchgehend sauber: `server` oben,
`analysis` unten, keine Rückwärtskanten.

## Test Coverage

```
[TESTED]   analysis.py      → test_tempo, test_key, test_formats
[TESTED]   postprocess.py   → test_tags, test_export, test_settings, test_lyrics, test_formats
[TESTED]   server.py        → test_api (TestClient)
[TESTED]   static/index.html→ test_frontend (statische Prüfungen)
[UNTESTED] engine.py        → kein Test importiert engine
```

Geschätzte Abdeckung: 4 von 5 Modulen haben zugehörige Tests (~80 %).
Die Tests erzeugen ihr Audiomaterial synthetisch (`conftest.py`: `kick_track`,
`beat_grid`, `progression`, `tiny_png`) und brauchen weder Modelle noch GPU –
deshalb läuft die CI ohne PyTorch.

## Concerns

```
[UNTESTED]  engine.py – 683 Zeilen, 0 Tests. Die reinen Teile
            (safe_song_name, stem_label, _pretty_stem, _fold/first_available,
            is_video, catalog-Auflösung) wären ohne Modelle testbar.
[LARGE]     postprocess.py 1100 Z. · static/index.html 1609 Z. ·
            server.py 849 Z. – jeweils deutlich über 300 Zeilen.
[MIXED]     postprocess.py bündelt Tags, DJ-Export, Audiobearbeitung
            und Lyrics in einem Modul.
[CRUFT]     _to_delete/ enthält alte Server- und HTML-Stände
            (server.v3.py, index.v3/v4/new.html). Untracked und
            in .gitignore, aber im Arbeitsverzeichnis.
```

Keine zyklischen Abhängigkeiten, keine Orphan-Dateien, kein veralteter Code –
alle Quelldateien wurden in den letzten Tagen angefasst (14./15.09.2026).
