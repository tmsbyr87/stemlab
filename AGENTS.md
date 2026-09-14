# StemLab installieren – Anleitung für Coding-Agenten

Diese Datei richtet sich an einen Agenten (Claude Code, Codex, Cursor), der
StemLab auf einem Mac einrichten soll. Sie beschreibt den Ablauf, die Stellen,
an denen es erfahrungsgemäß klemmt, und wie du das behebst.

Wenn ein Mensch dieses Dokument liest: die normale Installation steht in der
[README](README.md) und braucht keinen Agenten.

## Auftrag

Richte StemLab auf diesem Mac ein, sodass am Ende `~/Applications/StemLab.app`
existiert und startet.

Melde dem Nutzer am Ende kurz, was du getan hast. Nenne dabei: die App liegt in
`~/Applications`, fertige Stems landen in `~/Music/StemLab`, das Protokoll in
`~/Library/Logs/StemLab.log`.

## Vorprüfung

Brich ab und sag es dem Nutzer, wenn eine dieser Bedingungen nicht stimmt:

| Prüfung | Befehl | Erwartet |
|---|---|---|
| Betriebssystem | `uname -s` | `Darwin`. Alles andere ist nicht unterstützt. |
| Architektur | `uname -m` | `arm64`. Bei `x86_64` läuft es ohne GPU und **ohne `mlx-whisper`** – die Lyrics laufen dann über das langsamere `faster-whisper` auf der CPU. Weise darauf hin und frag, ob fortgefahren werden soll. |
| macOS-Version | `sw_vers -productVersion` | 13 oder höher. |
| Plattenplatz | `df -g ~ \| awk 'NR==2{print $4}'` | Mindestens 3 GB für die Einrichtung (Python-Umgebung rund 1,5 GB). Für die Modelle kommen später rund 2 GB dazu, die lädt das Setup aber nicht. |
| Netz | `curl -sI -o /dev/null -w '%{http_code}' https://codeload.github.com` | Eine 2xx- oder 3xx-Antwort. Geprüft wird `codeload` (dort liegt das ZIP), nicht `github.com`. |
| Homebrew | `command -v brew` | Siehe unten – fehlt es, klär das **vor** dem Einzeiler. |

**Homebrew vorab klären.** Fehlt Homebrew, will `setup.sh` es mitten im Lauf
installieren und fragt dabei nach dem Passwort des Nutzers. In einer
Agenten-Shell kann er dort nichts eintippen – die Installation bliebe hängen.
Fehlt `brew` also, bitte den Nutzer, es vorher selbst in einem eigenen
Terminal zu installieren ([brew.sh](https://brew.sh)), und fahre erst danach
fort.

**Vor dem Ausführen fragen.** Der Einzeiler unten führt ein Skript aus dem Netz
mit den Rechten des Nutzers aus und installiert über Homebrew systemweit
Software. Hol dir dafür einmal ausdrücklich das Einverständnis, bevor du ihn
startest.

## Ablauf

**Zuerst prüfen, ob das Projekt schon lokal liegt.** Häufig hat der Nutzer das
ZIP bereits entpackt oder das Repo geklont – dann darfst du keine zweite Kopie
herunterladen, sonst zeigt `StemLab.app` am Ende auf die heruntergeladene
statt auf die, die er vor sich hat. Liegt neben dir eine `setup.sh` (oder
nennt der Nutzer einen Ordner), dann richte **diesen** ein:

```bash
cd <Projektordner> && bash setup.sh
```

Nur wenn nichts lokal vorliegt, hol es aus dem Netz:

```bash
curl -fsSL https://raw.githubusercontent.com/tmsbyr87/stemlab/main/install.sh | bash
```

Der Installer lädt das Projekt als ZIP nach
`~/Library/Application Support/StemLab/app`, ruft `setup.sh` auf und startet
StemLab am Ende.

Beide Wege sind **wiederholbar**: `setup.sh` mehrfach auszuführen schadet
nicht, und eine vorhandene `venv` bleibt stehen. Nach einem Abbruch darfst du
also einfach neu starten, statt aufzuräumen.

**Zeitbudget:** Der `pip install`-Schritt lädt unter anderem PyTorch und
braucht je nach Leitung 5 bis 20 Minuten. Setz das Timeout deines
Tool-Aufrufs entsprechend hoch (mindestens 20 Minuten), sonst killst du die
Installation mittendrin und hältst sie für gescheitert.

`setup.sh` erledigt dabei der Reihe nach:

1. Homebrew prüfen, bei Bedarf installieren – das sollte die Vorprüfung oben
   schon abgefangen haben
2. `ffmpeg` prüfen, bei Bedarf per Homebrew installieren
3. Ein passendes Python suchen (**3.10 bis 3.12**), sonst `python@3.11`
   installieren
4. Virtuelle Umgebung in `venv/` anlegen und `requirements.txt` installieren
   (dauert mehrere Minuten, lädt unter anderem PyTorch)
5. `~/Applications/StemLab.app` bauen, inklusive Icon

Die Modelle lädt StemLab **nicht** beim Setup, sondern erst beim ersten
Gebrauch. Das ist Absicht: niemand soll 2 GB für Funktionen laden, die er nie
benutzt. Lade sie nicht vorab, außer der Nutzer bittet ausdrücklich darum.

## Wenn es klemmt

**Die Installation fragt nach einem Passwort.**
Dann fehlte Homebrew und die Vorprüfung oben wurde übersprungen. Brich den
Lauf ab, lass den Nutzer Homebrew in einem eigenen Terminal installieren und
starte danach neu – wiederholen schadet nicht. Gib niemals ein Passwort ein
und frag auch nicht danach.

**Meldung „Der Projektpfad enthält Anführungszeichen, `$`, `` ` `` oder `\`".**
`setup.sh` bricht bei diesen Zeichen bewusst ab, weil der Launcher Pfade in ein
Shell-Skript einsetzt. Benenne den übergeordneten Ordner um oder installiere an
einen anderen Ort:
```bash
curl -fsSL https://raw.githubusercontent.com/tmsbyr87/stemlab/main/install.sh | STEMLAB_DIR="$HOME/stemlab" bash
```

Die Zuweisung muss vor `bash` am Ende der Pipe stehen, nicht vor `curl`.

**Kein passendes Python gefunden.**
Das Fenster ist eng: **3.10 bis einschließlich 3.12**. Ein systemweites 3.13
oder neuer wird ignoriert, weil nicht alle Abhängigkeiten dafür Räder
bereitstellen. `setup.sh` installiert dann selbst `python@3.11` – lass es das
tun, statt die Grenze im Skript aufzuweichen.

**`pip install` bricht bei einem Paket ab.**
Meistens PyTorch oder `mlx-whisper`. Prüfe zuerst die Python-Version
(`venv/bin/python --version`) – liegt sie außerhalb 3.10–3.12, ist das die
Ursache. Danach die Fehlermeldung des Pakets selbst lesen, nicht blind
wiederholen.

**`PyTorch … Apple-GPU (MPS): nein – CPU-Modus`.**
Läuft, ist aber deutlich langsamer. Ursache ist fast immer ein PyTorch-Rad für
die falsche Architektur:
```bash
venv/bin/pip install --force-reinstall torch torchaudio
```

Sag dem Nutzer vorher Bescheid: das lädt PyTorch noch einmal komplett und
dauert einige Minuten. Es betrifft nur die `venv`, nicht sein System.

**Die App startet nicht nach dem Doppelklick.**
Das Protokoll steht in `~/Library/Logs/StemLab.log`. Häufigste Ursache: alle
Ports zwischen 8765 und 8799 belegt. Prüfen mit
`lsof -nP -iTCP:8765-8799 -sTCP:LISTEN`.

**Gatekeeper blockiert den Installer.**
Nur beim Doppelklick-Weg relevant, nicht beim Einzeiler. Der Nutzer muss
einmalig Rechtsklick → Öffnen wählen; diesen Schritt kannst du ihm nicht
abnehmen.

`StemLab installieren.command` entfernt danach selbst die Quarantäne vom
entpackten Ordner – das ist in Ordnung, weil der Nutzer die Datei mit seinem
Klick bereits freigegeben hat. Du selbst solltest `xattr -dr` nicht auf
Ordner anwenden, die der Nutzer nicht bewusst geladen und freigegeben hat.

**Wenn du aufräumen musst.**
Ein Neustart ist meist besser als Aufräumen, weil beide Wege wiederholbar sind.
Musst du doch zurückbauen, darfst du `~/Applications/StemLab.app` und den
Projektordner samt `venv` löschen. `~/Library/Application Support/StemLab/models`
darfst du behalten, das erspart erneutes Laden. **Nicht** anfassen:
`~/Music/StemLab` – dort liegen die Ergebnisse des Nutzers.

## Prüfen, ob es geklappt hat

```bash
test -x ~/Applications/StemLab.app/Contents/MacOS/StemLab && echo "App gebaut"
```

Prüfe auf den **Launcher**, nicht auf das Verzeichnis: `setup.sh` legt
`StemLab.app` schon an, bevor die Pakete installiert sind. Ein `test -d` wäre
auch bei einer auf halbem Weg abgebrochenen Installation wahr.

Die Python-Umgebung liegt immer **neben** `setup.sh`, also im Projektordner –
beim Einzeiler unter `~/Library/Application Support/StemLab/app`, beim
ZIP-Weg im entpackten Ordner. Der Launcher in der App kennt den Pfad, du
findest ihn so:

```bash
PROJEKT="$(grep -m1 '^PY=' ~/Applications/StemLab.app/Contents/MacOS/StemLab | sed 's|.*="||;s|/venv/bin/python"||')"
"$PROJEKT/venv/bin/python" -c "import server, engine, analysis, postprocess; print('Module laden')"
```

Geprüft werden die Module des Projekts selbst, nicht einzelne Pakete:
`librosa` und `torch` stehen nicht in `requirements.txt`, sie kommen über
`audio-separator` mit. Ein Import-Check auf sie würde bei einer intakten
Installation fehlschlagen, sobald sich deren Abhängigkeiten ändern.

Für einen echten Test brauchst du eine Audiodatei. Frag den Nutzer danach,
lade keine aus dem Netz. Der erste Lauf dauert länger, weil dann die Modelle
geladen werden.

## Was du wissen solltest, bevor du am Code etwas änderst

Falls der Nutzer nach der Installation Änderungen möchte:

| Datei | Zweck |
|---|---|
| `server.py` | Lokaler Webserver, Warteschlange, API, SSE |
| `engine.py` | Modellkatalog und Trennung, kapselt `audio-separator` |
| `analysis.py` | Tempo, Tonart, Akkorde, MIDI-Klick |
| `postprocess.py` | Tags und Cover, Loops, Pitch/Tempo, Mix, Lyrics |
| `static/index.html` | Komplette Oberfläche in einer Datei |

Konventionen: Kommentare und Oberfläche auf Deutsch, Code englisch benannt.
Kommentare erklären das **Warum**, nicht das Was.

### Tests

```bash
venv/bin/python -m pytest
```

212 Tests, rund anderthalb Sekunden, **keine Audiodateien nötig** – das
Material wird synthetisch erzeugt, seine Wahrheit steht dadurch per
Konstruktion fest. Ein Raster aus exakt 124 BPM muss 124 BPM ergeben, eine
g-Moll-Kadenz muss g-Moll ergeben.

| Datei | Deckt ab |
|---|---|
| `tests/test_tempo.py` | Tempo aus dem Beat-Raster, Störungen, Faltung, Snapping |
| `tests/test_key.py` | Tonart, Bandbegrenzung, Camelot-Tabelle |
| `tests/test_lyrics.py` | Filterung erfundener Zeilen, Floskelerkennung |
| `tests/test_tags.py` | Tags und Cover in WAV und FLAC |
| `tests/test_api.py` | HTTP-Schnittstelle, Pfad- und Host-Schutz |
| `tests/test_frontend.py` | Camelot-Farben, Cover-URL, Feldabgleich mit dem Server |
| `tests/test_export.py` | Rekordbox- und Traktor-Format, Cue-Erkennung |
| `tests/test_formats.py` | Zeitstempel, Dateinamen, Akkorde, MIDI-Klick, Sidecars |

Die Tests sind **Regressionstests für real aufgetretene Fehler**. Jeder
prüfbare Fall stand einmal falsch im Code. Wenn einer rot wird, hast du sehr
wahrscheinlich einen dieser Fehler wieder eingebaut – lies den Docstring, dort
steht, worum es ging.

Bei jedem Push laufen sie über GitHub Actions auf macOS gegen Python 3.10,
3.11 und 3.12, dazu `shellcheck` über die Installationsskripte.

Änderst du etwas an Analyse, Tags oder API, **schreib den Test zuerst** und
sieh ihn scheitern. Sonst weisst du nicht, ob er den Fehler überhaupt fangen
würde.

### Offene Punkte – hier ist Vorsicht angebracht

Was die Tests **nicht** abdecken, damit du dich nicht in falscher Sicherheit
wiegst:

- **Kein echtes Audio.** Getestet wird gegen synthetische Signale. Dass die
  Tonart an echter Musik stimmt, ist an zwei Tracks gegen Mixed In Key
  belegt – nicht an einem Testsatz.
- **Die Schwellen der Lyrics-Filterung sind an einem einzigen Track
  kalibriert** (`SILENCE_DB = -40`, `FILLER_DB = -20` in `postprocess.py`).
  Die Tests prüfen, dass die Logik greift, nicht dass die Werte für jedes
  Material passen. Bei sehr leise abgemischtem oder geflüstertem Gesang
  könnten echte Zeilen wegfallen. Wenn du daran drehst, miss die Pegel echter
  und erfundener Segmente, statt zu raten.
- **Das Tempo geht von einem durchgehenden DAW-Raster aus.** Bei Musik mit
  echten Tempowechseln sollte die Downbeat-Gegenprobe greifen und die
  Konfidenz senken – dieser Fall ist nicht getestet.
- **Die Trennmodelle selbst sind nicht getestet.** Sie brauchen GPU und
  Gigabyte an Gewichten; geprüft wird nur, was ohne sie läuft.
- **Die Analyse läuft vor der Trennung** auf dem Mix, nicht auf den Stems. Für
  die Tonart wäre der `other`-Stem sauberer; die Bandbegrenzung auf
  100–1000 Hz war die günstigere Lösung und reicht bislang.

### Grenzen

Halte dich an diese, auch wenn der Nutzer es lockerer sieht:

- Installiere nichts systemweit außer über die Wege, die `setup.sh` ohnehin
  geht (Homebrew, `ffmpeg`, `python@3.11`).
- Lösche keine vorhandenen Ergebnisordner unter `~/Music/StemLab`. Dort liegt
  Arbeit des Nutzers.
- Lade keine urheberrechtlich geschützte Musik zum Testen herunter.
- Schreibe keine Passwörter irgendwohin und frag nicht danach.
