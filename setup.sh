#!/bin/bash
# ---------------------------------------------------------------------------
# StemLab – Einrichtung auf macOS (Apple Silicon)
# Legt eine isolierte Python-Umgebung an, installiert alles Nötige und baut
# eine StemLab.app in ~/Applications. Mehrfaches Ausführen ist unschädlich.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/venv"
APP="$HOME/Applications/StemLab.app"
LOGFILE="$HOME/Library/Logs/StemLab.log"

say()  { printf "\n\033[1m▸ %s\033[0m\n" "$1"; }
warn() { printf "\033[33m  ! %s\033[0m\n" "$1"; }
die()  { printf "\033[31m  ✗ %s\033[0m\n" "$1"; exit 1; }

[[ "$(uname -s)" == "Darwin" ]] || die "Dieses Skript ist für macOS."
[[ "$(uname -m)" == "arm64" ]] || warn "Kein Apple Silicon erkannt – läuft, aber ohne GPU-Beschleunigung."
case "$HERE" in *[\"\$\`\\]*) die "Der Projektpfad enthält Anführungszeichen, \$, \` oder \\ – bitte den Ordner umbenennen." ;; esac

# --- Werkzeuge ------------------------------------------------------------
say "Prüfe Werkzeuge"
if ! command -v brew >/dev/null 2>&1; then
  for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do [[ -x "$b" ]] && eval "$("$b" shellenv)" && break; done
fi
if ! command -v brew >/dev/null 2>&1; then
  warn "Homebrew fehlt. Installation startet (fragt nach deinem Passwort)."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do [[ -x "$b" ]] && eval "$("$b" shellenv)" && break; done
fi
command -v ffmpeg >/dev/null 2>&1 || { echo "  ffmpeg wird installiert …"; brew install ffmpeg; }
FFMPEG_DIR="$(dirname "$(command -v ffmpeg)")"

PY=""
for cand in python3.12 python3.11 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    v="$("$cand" -c 'import sys;print("%d%02d"%sys.version_info[:2])')"
    if [[ "$v" -ge 310 && "$v" -lt 313 ]]; then PY="$(command -v "$cand")"; break; fi
  fi
done
if [[ -z "$PY" ]]; then
  echo "  Python 3.11 wird installiert …"
  brew install python@3.11
  PY="$(brew --prefix)/opt/python@3.11/bin/python3.11"
fi
echo "  Python: $("$PY" --version) · ffmpeg: $(ffmpeg -version | head -1 | cut -d' ' -f3) in $FFMPEG_DIR"

chmod +x "$HERE/start.sh" "$HERE/setup.sh" 2>/dev/null || true

# --- Umgebung -------------------------------------------------------------
say "Richte Python-Umgebung ein (beim ersten Mal ein paar Minuten)"
[[ -d "$VENV" ]] || "$PY" -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip wheel
"$VENV/bin/pip" install --quiet --upgrade -r "$HERE/requirements.txt"
echo "  Pakete installiert."

"$VENV/bin/python" - <<'PYCHECK'
import platform, torch
mps = torch.backends.mps.is_available() and platform.uname().processor == "arm"
print(f"  PyTorch {torch.__version__} · Apple-GPU (MPS): {'ja' if mps else 'nein – CPU-Modus'}")
PYCHECK

# --- App-Bundle -----------------------------------------------------------
say "Baue StemLab.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" "$(dirname "$LOGFILE")"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>StemLab</string>
  <key>CFBundleDisplayName</key><string>StemLab</string>
  <key>CFBundleIdentifier</key><string>local.stemlab</string>
  <key>CFBundleVersion</key><string>2.0</string>
  <key>CFBundleShortVersionString</key><string>2.0</string>
  <key>CFBundleExecutable</key><string>StemLab</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleIconFile</key><string>StemLab</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSUIElement</key><true/>
</dict></plist>
PLIST

# Der Launcher: Pfade werden JETZT eingesetzt, alles mit \$ erst zur Laufzeit.
cat > "$APP/Contents/MacOS/StemLab" <<LAUNCH
#!/bin/bash
# StemLab-Launcher – startet den lokalen Server und öffnet die Oberfläche.
export PATH="$FFMPEG_DIR:/opt/homebrew/bin:/usr/local/bin:\$PATH"
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$HERE" || exit 1
LOG="$LOGFILE"
PY="$VENV/bin/python"
echo "--- \$(date) Start ---" >> "\$LOG"

# Läuft schon eine Instanz? Dann nur das Fenster öffnen.
for p in \$(seq 8765 8799); do
  if curl -s --max-time 1 "http://127.0.0.1:\$p/api/ping" 2>/dev/null | grep -q stemlab; then
    open "http://127.0.0.1:\$p/"; exit 0
  fi
done

PORT="\$("\$PY" -c '
import socket
for p in range(8765, 8800):
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", p)) != 0:
            print(p); break
')"
if [[ -z "\$PORT" ]]; then
  osascript -e 'display alert "StemLab" message "Kein freier Port zwischen 8765 und 8799." as critical'
  exit 1
fi

export STEMLAB_PORT="\$PORT"
"\$PY" "$HERE/server.py" >> "\$LOG" 2>&1 &
SERVER=\$!
trap 'kill \$SERVER 2>/dev/null' EXIT INT TERM

for _ in \$(seq 1 120); do
  kill -0 "\$SERVER" 2>/dev/null || break
  curl -s --max-time 2 "http://127.0.0.1:\$PORT/api/ping" >/dev/null 2>&1 && { open "http://127.0.0.1:\$PORT/"; break; }
  sleep 0.5
done
if ! kill -0 "\$SERVER" 2>/dev/null; then
  osascript -e 'display alert "StemLab konnte nicht starten" message "Details stehen in ~/Library/Logs/StemLab.log" as critical'
  exit 1
fi
wait "\$SERVER"
LAUNCH
chmod +x "$APP/Contents/MacOS/StemLab"

if [[ -f "$HERE/app/icon.png" ]]; then
  TMPICON="$(mktemp -d)"; ICONSET="$TMPICON/StemLab.iconset"; mkdir -p "$ICONSET"
  for sz in 16 32 64 128 256 512; do
    sips -z $sz $sz "$HERE/app/icon.png" --out "$ICONSET/icon_${sz}x${sz}.png" >/dev/null
    sips -z $((sz*2)) $((sz*2)) "$HERE/app/icon.png" --out "$ICONSET/icon_${sz}x${sz}@2x.png" >/dev/null
  done
  iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/StemLab.icns" 2>/dev/null || warn "Icon konnte nicht gebaut werden (kosmetisch)."
  rm -rf "$TMPICON"
fi
touch "$APP"

say "Fertig"
cat <<DONE
  StemLab liegt in ~/Applications – per Doppelklick starten.
  Beim ersten Gebrauch eines Modells lädt es dessen Gewichte (65 MB bis 650 MB),
  Beat This! (77 MB) und Whisper (~1,5 GB) ebenso – danach läuft alles offline.

  Alternativ im Terminal:   $HERE/start.sh
  Stems landen in:          ~/Music/StemLab
  Protokoll:                $LOGFILE
DONE
