#!/usr/bin/env python
"""Locate the payload on Kaggle and stage it into /kaggle/working.

Run this as the FIRST notebook cell:

    !python /kaggle/input/<anything>/scripts/kaggle_setup.py   # or paste inline

...though the usual way is to paste its body into a cell, because the file lives
inside the payload you are trying to find.

Why this exists instead of a plain `tar xzf`:

* **Kaggle decompresses archives on upload.** Upload
  `kaggle_payload_task1.tar.gz` and the dataset you get back contains the
  extracted tree, not the tarball -- so `tar xzf .../kaggle_payload_task1.tar.gz`
  fails with "No such file or directory".
* **The mount path varies.** Depending on how the dataset was added it can be
  `/kaggle/input/<slug>/`, `/kaggle/input/datasets/<owner>/<slug>/`, or nested
  one level deeper again. Hardcoding it is how you get a path error two minutes
  into a 45-minute run.
* **`/kaggle/input` is read-only.** Training writes `results/` and checkpoints,
  so the tree has to be copied into `/kaggle/working` first.

Both layouts are handled: an extracted tree is copied, a surviving tarball is
unpacked.
"""
from __future__ import annotations

import os
import shutil
import sys
import tarfile
from pathlib import Path

INPUT_ROOT = Path(os.environ.get("GBMC_INPUT_ROOT", "/kaggle/input"))
WORK_ROOT = Path(os.environ.get("GBMC_WORK_ROOT", "/kaggle/working"))


def stage(input_root: Path = INPUT_ROOT, work_root: Path = WORK_ROOT) -> Path:
    """Put a runnable copy of the project in ``work_root`` and return that path."""
    work_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        raise SystemExit(
            f"{input_root} does not exist. Attach the dataset with 'Add Input' "
            "in the notebook sidebar."
        )

    # Case 1: the archive survived upload (rare -- Kaggle usually extracts it).
    tarball = next(iter(sorted(input_root.rglob("kaggle_payload_task1.tar.gz"))), None)
    if tarball is not None:
        print(f"found archive: {tarball}")
        with tarfile.open(tarball) as tf:
            tf.extractall(work_root)
        print(f"extracted -> {work_root}")
        return work_root

    # Case 2: Kaggle already extracted it. Find the tree by a file we know is in
    # it, rather than by guessing the mount path.
    marker = next(iter(sorted(input_root.rglob("src/train.py"))), None)
    if marker is None:
        listing = sorted(p for p in input_root.rglob("*") if p.is_dir())[:20]
        raise SystemExit(
            "could not find the payload under "
            f"{input_root}.\nDirectories seen:\n  "
            + "\n  ".join(str(p) for p in listing)
            + "\n\nIs the dataset attached, and did the upload finish?"
        )

    payload = marker.parent.parent
    print(f"found extracted payload: {payload}")

    copied = 0
    for item in sorted(payload.iterdir()):
        target = work_root / item.name
        if target.exists():
            print(f"  skip {item.name} (already in working dir)")
            continue
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
        copied += 1
    print(f"copied {copied} top-level entries -> {work_root}")
    return work_root


def main(argv=None) -> int:
    work = stage()
    os.chdir(work)
    sys.path.insert(0, str(work))

    print(f"\ncwd: {os.getcwd()}")
    print("contents:", ", ".join(sorted(p.name for p in work.iterdir())))

    missing = [p for p in ("config.yaml", "src/train.py",
                           "scripts/kaggle_task1.py",
                           "data/splits/musiccaps_manifest.csv",
                           "data/splits/musiccaps_tag_vocab.json")
               if not (work / p).exists()]
    if missing:
        print("\nMISSING (the payload looks incomplete):")
        for p in missing:
            print(f"  {p}")
        print("Rebuild it locally with: make kaggle-payload")
        return 1

    print("\npayload complete -- ready to run:")
    print("  !pip -q install torch-geometric")
    print("  !python scripts/kaggle_task1.py --model bert-base-uncased --epochs 8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
