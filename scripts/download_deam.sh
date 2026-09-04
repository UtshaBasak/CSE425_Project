#!/usr/bin/env bash
# DEAM / MediaEval Emotion in Music: 1,802 excerpts with valence and arousal.
#
# DEAM is distributed through Zenodo record 1188976, which does not publish
# stable per-file checksums in the download URL. This script therefore verifies
# by expected file count and prints the md5 it observed so you can record it.
#   bash scripts/download_deam.sh [target_dir]
set -euo pipefail

TARGET="${1:-data/raw/deam}"
BASE="https://zenodo.org/records/1188976/files"

mkdir -p "$TARGET"
cd "$TARGET"

fetch() {
  archive="$1"
  dest="$2"
  if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
    echo "[skip] $dest/ already present"
    return 0
  fi
  if [ -s "$archive" ]; then
    echo "[skip] $archive already downloaded"
  else
    echo "[get ] $archive"
    curl -fL --retry 3 -o "$archive" "$BASE/$archive?download=1"
  fi
  if command -v md5sum >/dev/null 2>&1; then
    echo "[info] md5($archive) = $(md5sum "$archive" | cut -d' ' -f1)"
  fi
  echo "[unzip] $archive -> $dest"
  mkdir -p "$dest"
  unzip -q -o "$archive" -d "$dest"
  rm -f "$archive"
}

fetch "DEAM_audio.zip" "audio_raw"
fetch "DEAM_Annotations.zip" "annotations"
fetch "metadata.zip" "metadata"

# The audio archive nests everything under MEMD_audio/; flatten it to audio/ so
# config.datasets.deam.audio needs no special case.
if [ -d "audio_raw/MEMD_audio" ] && [ ! -d "audio" ]; then
  mv "audio_raw/MEMD_audio" "audio"
  rmdir "audio_raw" 2>/dev/null || true
fi

COUNT="$(find audio -name '*.mp3' 2>/dev/null | wc -l)"
echo "[info] $COUNT audio excerpts (expected 1802)"
if [ "$COUNT" -ne 1802 ]; then
  echo "[warn] unexpected file count -- check the extraction"
fi
echo "[done] now run: python scripts/verify_datasets.py"
