#!/usr/bin/env bash
# MagnaTagATune: 25,863 x 29 s clips + 188-tag annotations.
#
# The audio ships as three split archives that must be concatenated before
# unzipping -- unzipping mp3.zip.001 on its own fails with a confusing
# "cannot find zipfile directory" and is the single most common setup mistake.
#
# Idempotent: every step is skipped if its output already exists.
#   bash scripts/download_mtat.sh [target_dir]
set -euo pipefail

TARGET="${1:-data/raw/mtat}"
BASE="https://mirg.city.ac.uk/datasets/magnatagatune"
AUDIO_DIR="$TARGET/audio"

mkdir -p "$TARGET" "$AUDIO_DIR"
cd "$TARGET"

# ---- annotations ------------------------------------------------------------
for f in annotations_final.csv clip_info_final.csv; do
  if [ -s "$f" ]; then
    echo "[skip] $f already present"
  else
    echo "[get ] $f"
    curl -fL --retry 3 -o "$f" "$BASE/$f"
  fi
done

# ---- audio ------------------------------------------------------------------
if [ -n "$(ls -A "$AUDIO_DIR" 2>/dev/null)" ]; then
  echo "[skip] audio/ is not empty; refusing to re-extract"
else
  for part in 001 002 003; do
    if [ -s "mp3.zip.$part" ]; then
      echo "[skip] mp3.zip.$part already downloaded"
    else
      echo "[get ] mp3.zip.$part"
      curl -fL --retry 3 -o "mp3.zip.$part" "$BASE/mp3.zip.$part"
    fi
  done

  echo "[join] concatenating the three parts into mp3_all.zip"
  cat mp3.zip.001 mp3.zip.002 mp3.zip.003 > mp3_all.zip

  # If curl truncated a part, this is where you find out -- not three hours
  # into feature extraction. Compare against the checksum published alongside
  # the archives on the mirror above.
  if command -v md5sum >/dev/null 2>&1; then
    echo "[info] md5(mp3_all.zip) = $(md5sum mp3_all.zip | cut -d' ' -f1)"
    echo "       verify this against the published checksum on $BASE"
  else
    echo "[warn] md5sum unavailable; skipping checksum verification"
  fi

  echo "[unzip] extracting into audio/ (this takes a while)"
  unzip -q -o mp3_all.zip -d "audio"
  rm -f mp3_all.zip
fi

# MTAT ships a handful of zero-byte mp3s; report them rather than silently
# letting them become tracks of digital silence.
EMPTY="$(find "$AUDIO_DIR" -name '*.mp3' -size 0 | wc -l)"
echo "[info] $(find "$AUDIO_DIR" -name '*.mp3' | wc -l) mp3 files, $EMPTY empty"
echo "[done] now run: python scripts/verify_datasets.py"
