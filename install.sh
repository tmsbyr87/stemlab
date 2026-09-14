#!/bin/bash
# ---------------------------------------------------------------------------
# StemLab – Installation mit einem Befehl.
#
#   curl -fsSL https://raw.githubusercontent.com/tmsbyr87/stemlab/main/install.sh | bash
#
# Lädt das Projekt als ZIP nach ~/Library/Application Support/StemLab/app
# und richtet es ein.
# Bewusst ohne git: auf einem frischen Mac würde `git clone` erst die Xcode
# Command Line Tools nachinstallieren wollen. curl und unzip sind immer da.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="tmsbyr87/stemlab"
BRANCH="main"
# Neben Modelle und Konfiguration, nicht in den Programme-Ordner: dort liegt
# nur StemLab.app, damit im Finder kein zweiter, verwirrender Eintrag steht.
DEST="${STEMLAB_DIR:-$HOME/Library/Application Support/StemLab/app}"

say()  { printf "\n\033[1m▸ %s\033[0m\n" "$1"; }
die()  { printf "\033[31m  ✗ %s\033[0m\n" "$1"; exit 1; }

[[ "$(uname -s)" == "Darwin" ]] || die "StemLab läuft nur auf macOS."

say "Lade StemLab"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
curl -fsSL "https://codeload.github.com/$REPO/zip/refs/heads/$BRANCH" -o "$TMP/stemlab.zip" \
  || die "Download fehlgeschlagen – ist die Internetverbindung da?"
unzip -q "$TMP/stemlab.zip" -d "$TMP" || die "Das Archiv liess sich nicht entpacken."
SRC="$TMP/stemlab-$BRANCH"
[[ -f "$SRC/setup.sh" ]] || die "Das Archiv sieht unvollständig aus."

# Eine vorhandene Installation behält ihre venv, damit nicht alles neu lädt.
say "Lege alles in $DEST ab"
mkdir -p "$DEST"
for item in "$SRC"/*; do
  name="$(basename "$item")"
  [[ "$name" == "venv" ]] && continue
  # ${DEST:?} bricht ab, statt bei leerer Variable "/$name" zu löschen.
  rm -rf "${DEST:?}/$name"
  mv "$item" "$DEST/$name"
done

say "Richte ein"
bash "$DEST/setup.sh"

open "$HOME/Applications/StemLab.app" 2>/dev/null || true
