#!/bin/bash
# ---------------------------------------------------------------------------
# StemLab – Doppelklick-Installation.
# Diese Datei liegt im entpackten Ordner. Ein Doppelklick öffnet Terminal.app
# und richtet alles ein; danach liegt StemLab in ~/Applications.
#
# Warum ein eigener Starter und nicht setup.sh direkt: Dateien aus einem ZIP
# kommen ohne Ausführbar-Bit an. setup.sh liesse sich also gar nicht
# doppelklicken. Hier wird es deshalb ausdrücklich mit bash aufgerufen.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

printf '\033[1m\n  StemLab wird eingerichtet\033[0m\n'
printf '  Das dauert beim ersten Mal ein paar Minuten.\n'
printf '  Du kannst dieses Fenster offen lassen.\n\n'

if [[ ! -f "$HERE/setup.sh" ]]; then
  printf '\033[31m  Die Datei setup.sh fehlt neben diesem Installer.\033[0m\n'
  printf '  Bitte das ZIP noch einmal entpacken und den ganzen Ordner behalten.\n\n'
  read -r -p "  Mit der Eingabetaste schliessen. " _
  exit 1
fi

# Quarantäne-Markierung entfernen, damit macOS den entpackten Ordner nicht
# bei jedem Schritt erneut prüft.
xattr -dr com.apple.quarantine "$HERE" 2>/dev/null || true

if bash "$HERE/setup.sh"; then
  printf '\n\033[32m  Fertig.\033[0m StemLab liegt jetzt in deinem Programme-Ordner.\n'
  printf '  Es öffnet sich gleich von selbst.\n\n'
  open "$HOME/Applications/StemLab.app" 2>/dev/null || true
  printf '  Dieses Fenster kannst du schliessen.\n\n'
else
  printf '\n\033[31m  Die Einrichtung ist fehlgeschlagen.\033[0m\n'
  printf '  Details stehen oben und in ~/Library/Logs/StemLab.log\n\n'
  read -r -p "  Mit der Eingabetaste schliessen. " _
  exit 1
fi
