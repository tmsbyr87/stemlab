#!/bin/bash
# StemLab ohne App-Bundle starten (Strg-C beendet).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
export PYTORCH_ENABLE_MPS_FALLBACK=1
[[ -x "$HERE/venv/bin/python" ]] || { echo "Bitte zuerst ./setup.sh ausführen."; exit 1; }

for p in $(seq 8765 8799); do
  if curl -s --max-time 1 "http://127.0.0.1:$p/api/ping" 2>/dev/null | grep -q stemlab; then
    echo "StemLab läuft bereits auf Port $p – öffne das Fenster."; open "http://127.0.0.1:$p/"; exit 0
  fi
done
PORT="$("$HERE/venv/bin/python" -c '
import socket
for p in range(8765, 8800):
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", p)) != 0:
            print(p); break
')"
[[ -n "$PORT" ]] || { echo "Kein freier Port zwischen 8765 und 8799."; exit 1; }
export STEMLAB_PORT="$PORT"
( for _ in $(seq 1 120); do curl -s --max-time 2 "http://127.0.0.1:$PORT/api/ping" >/dev/null 2>&1 && { open "http://127.0.0.1:$PORT/"; break; }; sleep 0.5; done ) &
exec "$HERE/venv/bin/python" "$HERE/server.py"
