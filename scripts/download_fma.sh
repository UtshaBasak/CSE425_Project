#!/usr/bin/env bash
# FMA-small: 8,000 x 30 s clips across 8 balanced genres, plus the metadata.
#
# fma_small.zip is ~7.2 GB and fma_metadata.zip ~342 MB. The sha1 checksums
# below are the ones published in the FMA repository README; a mismatch means a
# truncated download, and is fatal here rather than a warning, because an
# unzip failure three gigabytes later is much harder to diagnose.
#   bash scripts/download_fma.sh [target_dir]
set -euo pipefail

TARGET="${1:-data/raw/fma}"
BASE="https://os.unil.cloud.switch.ch/fma"

mkdir -p "$TARGET"
cd "$TARGET"

sha1_for() {
  case "$1" in
    fma_metadata.zip) echo "f0df49ffe5f2a6008d7dc83c6915b31835dfe733" ;;
    fma_small.zip)    echo "ade154f733639d52e35e32f5593efe5be76c6d70" ;;
    *)                echo "" ;;
  esac
}

for archive in fma_metadata.zip fma_small.zip; do
  dir="${archive%.zip}"
  if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
    echo "[skip] $dir/ already extracted"
    continue
  fi

  if [ -s "$archive" ]; then
    echo "[skip] $archive already downloaded"
  else
    echo "[get ] $archive"
    curl -fL --retry 3 -o "$archive" "$BASE/$archive"
  fi

  EXPECTED="$(sha1_for "$archive")"
  if command -v sha1sum >/dev/null 2>&1 && [ -n "$EXPECTED" ]; then
    ACTUAL="$(sha1sum "$archive" | cut -d' ' -f1)"
    if [ "$ACTUAL" != "$EXPECTED" ]; then
      echo "[FAIL] sha1 mismatch for $archive"
      echo "       got      $ACTUAL"
      echo "       expected $EXPECTED"
      echo "       Delete the file and re-run; a truncated archive will fail to unzip."
      exit 1
    fi
    echo "[ ok ] sha1 verified for $archive"
  else
    echo "[warn] sha1sum unavailable; skipping checksum verification"
  fi

  echo "[unzip] $archive"
  unzip -q -o "$archive"
  rm -f "$archive"
done

echo "[info] $(find fma_small -name '*.mp3' 2>/dev/null | wc -l) mp3 files"
echo "[done] now run: python scripts/verify_datasets.py"
