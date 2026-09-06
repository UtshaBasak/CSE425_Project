# C4 on Kaggle — the seven-mode fusion ablation

**What this is.** Twenty-one Task 3 runs: seven fusion modes × seeds 42 / 1337 /
2024, all on MTAT, all with `distilbert-base-uncased` and an epoch budget held
identical across every mode.

**Why it is not local.** `bert-base` measures **68 ms/row** on the GTX 1650, so
these twenty-one runs would take roughly **26 hours** here. That would have
forced cutting rows out of the ablation table under Phase C decision rule 3.6.
Two T4s running two shards concurrently avoid the cut entirely *and* leave the
local GPU free for C5–C7, which are running in parallel while you do this.

---

## Step 1 — upload the payload

| | |
|---|---|
| **File** | `gnn-bert-music-context/kaggle_payload_ablation.tar.gz` |
| **Size** | **105.0 MB** (237.6 MB uncompressed, 71 files) |
| **Rebuild** | `make kaggle-payload-ablation` |

It contains the four `features_*.h5` caches, `data/splits/`, the train-only
norm stats, `config.yaml`, `src/` and `scripts/`. **No mel caches** (5.8 GB,
and B2 runs locally in minutes), **no checkpoints**, **no audio**.

> The feature caches are the point. Task 3 builds its graphs on the fly from
> them — `DataBundle` sets `graph_dir=None` for real data — so a payload
> carrying the exported `.pt` graphs instead would be larger *and* unusable.

Kaggle → **Datasets → New Dataset** → drag the `.tar.gz` in → title it
`gbmc-ablation-payload` → **Private** → Create.

## Step 2 — notebook settings

| Setting | Value |
|---|---|
| Accelerator | **GPU T4 ×2** — not P100 (sm_60; this PyTorch has no kernels for it) |
| Internet | **On** — `distilbert-base-uncased` is fetched at runtime |
| Add Data | your `gbmc-ablation-payload` dataset |

## Step 3 — paste this one cell

```python
import os, shutil, subprocess, tarfile, time
from pathlib import Path

WORK, INPUT = Path("/kaggle/working"), Path("/kaggle/input")

# Kaggle sometimes decompresses archives on upload and sometimes does not, and
# the mount path varies, so find the payload rather than assuming either.
tarball = next(iter(sorted(INPUT.rglob("kaggle_payload_ablation.tar.gz"))), None)
if tarball:
    print("archive:", tarball)
    with tarfile.open(tarball) as tf:
        tf.extractall(WORK)
else:
    marker = next(iter(sorted(INPUT.rglob("src/train.py"))), None)
    if marker is None:
        raise SystemExit("payload not found -- is the dataset attached?")
    payload = marker.parent.parent
    print("payload:", payload)
    for item in sorted(payload.iterdir()):
        dst = WORK / item.name
        if not dst.exists():
            shutil.copytree(item, dst) if item.is_dir() else shutil.copy2(item, dst)

os.chdir(WORK)
print("contents:", sorted(p.name for p in WORK.iterdir()))

# Required: src.train imports the graph modules at module load, so the import
# fails without it even though this task needs no PyG data loaders. ~11 s.
subprocess.run(["pip", "-q", "install", "torch-geometric"], check=True)

# One process per GPU. CUDA_VISIBLE_DEVICES=i makes each see exactly one card,
# so both address it as cuda:0 and neither can contend for the other's memory.
procs = []
for shard in (0, 1):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard))
    log = open(f"/kaggle/working/shard{shard}.log", "w")
    procs.append((shard, log, subprocess.Popen(
        ["python", "scripts/kaggle_ablation.py",
         "--shard", str(shard), "--shards", "2",
         "--model", "distilbert-base-uncased",
         "--ablation-epochs", "6"],
        env=env, stdout=log, stderr=subprocess.STDOUT)))
    print(f"launched shard {shard} on GPU {shard}")

started = time.time()
for shard, log, proc in procs:
    code = proc.wait()
    log.close()
    print(f"shard {shard} exited {code} after {(time.time()-started)/60:.1f} min")

for shard in (0, 1):
    print(f"\n===== shard {shard} tail =====")
    print("".join(open(f"/kaggle/working/shard{shard}.log").readlines()[-25:]))
```

## Step 4 — Save & Run All, not Quick Save

**Save Version → Save & Run All (Commit).** Quick Save stores the notebook
without re-running it, so its version carries **no output files** — that cost
this project a full sweep once already. Only Save & Run All captures
`/kaggle/working`.

## Step 5 — bring it back

Output tab → **Download all**, then from the repo:

```bash
python scripts/import_kaggle_results.py ~/Downloads/archive.zip
```

The importer runs three guards on every incoming file and reports what it
accepted and refused:

1. **provenance** — anything synthetic is refused;
2. **threshold discipline** — anything not tuned on `val` is refused;
3. **vocabulary hash** — the ordered tag vocabulary is hashed into every result,
   and a file whose hash matches no current vocabulary is refused. A7.3 changed
   7 of the 50 MusicCaps tags, so a pre-fix result is numerically fine and
   semantically incompatible, and nothing about the file would reveal that.

Exit code 2 means something was rejected; the reasons are printed.

Then:

```bash
python scripts/fusion_ablation.py --corpus mtat --summary-only
```

which applies the noise-floor rule: mean ± sd over seeds, a Δ-vs-best column,
and `best_mode = None` when the top rows sit within 0.0288 macro-F1 of each
other.

---

## Notes

**Resumable.** Each shard skips any run whose result JSON already exists with
matching provenance and mode, so if the 12-hour limit cuts the session, re-run
the same cell and it continues from where it stopped.

**Sharding is exact.** Shard *i* takes every *N*th run from offset *i* from a
fixed-order list. Verified locally for 1–4 shards: every run appears in exactly
one shard, with no overlap and no coordination between the two processes.
`python scripts/kaggle_ablation.py --shards 2 --dry-run` prints the split.

**Wall-clock.** Each run prints its seconds/epoch and ms/row, and each shard
prints its throughput against the local 68 ms/row figure. That is what turns the
26-hour local estimate into a measured number rather than an extrapolation.

**Expected duration.** Unknown until the first runs report — that is the point
of printing ms/row. If a T4 is ~4× the GTX 1650 and DistilBERT is ~2× BERT-base,
the estimate is roughly 3 h per shard, comfortably inside the 12-hour limit.
