#!/usr/bin/env bash
# Lakh MIDI Dataset -- Clean MIDI subset (~17k files, artist/title in the path).
#
# Used only to validate the chroma chord estimator against symbolic ground
# truth (src/chords.py::validate_against_lmd). It is not a training input, so a
# failure here degrades one analysis rather than blocking the pipeline.
#   bash scripts/download_lmd.sh [target_dir]
set -euo pipefail

TARGET="${1:-data/raw/lmd_clean}"
URL="http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz"

mkdir -p "$(dirname "$TARGET")"

if [ -d "$TARGET" ] && [ -n "$(ls -A "$TARGET" 2>/dev/null)" ]; then
  echo "[skip] $TARGET already populated"
else
  ARCHIVE="$(dirname "$TARGET")/clean_midi.tar.gz"
  if [ -s "$ARCHIVE" ]; then
    echo "[skip] archive already downloaded"
  else
    echo "[get ] clean_midi.tar.gz (~200 MB)"
    curl -fL --retry 3 -o "$ARCHIVE" "$URL"
  fi

  if command -v md5sum >/dev/null 2>&1; then
    echo "[info] md5 = $(md5sum "$ARCHIVE" | cut -d' ' -f1)"
    echo "       compare against the checksum listed on https://colinraffel.com/projects/lmd/"
  fi

  echo "[untar] extracting"
  mkdir -p "$TARGET"
  tar -xzf "$ARCHIVE" -C "$TARGET" --strip-components=1
  rm -f "$ARCHIVE"
fi

N_FILES="$(find "$TARGET" -name '*.mid' | wc -l)"
N_DIRS="$(find "$TARGET" -mindepth 1 -maxdepth 1 -type d | wc -l)"
echo "[info] $N_FILES MIDI files in $N_DIRS artist folders"
echo "[done] validate the chord estimator with:"
echo "       python -m src.chords --lmd-dir $TARGET --n-samples 200"
