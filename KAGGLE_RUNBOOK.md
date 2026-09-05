# Kaggle runbook — Task 1 (A6.4)

Task 1 is BERT predicting tags from text. It is the only part of the project that
wants more GPU-hours than one laptop card gives comfortably, so it runs on
Kaggle. Everything else fits locally (see `state/vram_report.md`).

**Budget:** ~15 minutes of your attention, ~45 minutes of unattended GPU.

---

## 0. Two corrections to the earlier notes

| Claim | Verdict |
|---|---|
| "Internet must be On, needs phone verification" | **Correct, and critical.** `bert-base-uncased` is fetched from HuggingFace at runtime. Without internet the run dies at model load. |
| "Drop the `pip install torch-geometric` line" | **Wrong — keep it.** Task 1 needs no graphs, but `src.train` imports the graph modules at module load, so the import fails without it. Verified by blocking the import: `src.train` raises `ImportError`. It is a pure-Python wheel and takes **11 seconds**, not several minutes. (The slow PyG installs are `pyg-lib`/`torch-scatter`, which this project never installs.) |
| "6–10 hours of quota" | That is a three-seed figure. **One seed is ~45 min** — 2,095 MusicCaps rows (65 steps/epoch at batch 32) and 16,881 MTAT rows (527 steps/epoch). |

A bug was also fixed before this runbook was usable: Task 1 used to load rows
through the graph dataset, which opens the HDF5 feature caches. Those are
deliberately excluded from the payload, so the run died ~30 s in on
`FileNotFoundError: features.h5`. Task 1 now reads text straight from the
manifests (`TextTagDataset`), verified end to end from the 3.4 MB payload in a
clean directory.

---

## 1. Phone-verify your Kaggle account (do this first)

`kaggle.com` → your avatar → **Settings** → **Phone Verification**.

This gates the "Internet" toggle on notebooks. It can take a few minutes to come
through, so start it before anything else.

## 2. Upload the payload as a private dataset

The payload is already built at the repo root:

```
gnn-bert-music-context/kaggle_payload_task1.tar.gz     (3.4 MB)
```

Rebuild it any time with `make kaggle-payload` or:

```bash
python scripts/make_kaggle_payload.py --no-graphs --out kaggle_payload_task1.tar.gz
```

On Kaggle: **Datasets → New Dataset** → drag the `.tar.gz` in → title it
something like `gbmc-task1-payload` → **Private** → Create.

You do **not** need the slug — the staging cell in step 4 finds the payload
wherever Kaggle mounts it.

> It contains manifests, text variants, both tag vocabularies, the train-only
> normalisation stats, `config.yaml`, `src/` and `scripts/`. No audio, no feature
> caches, no checkpoints — the builder refuses to package those.

**Kaggle will decompress the `.tar.gz` on upload.** That is expected; the staging
cell handles both the extracted tree and a surviving archive.

## 3. Create the notebook

**Code → New Notebook**, then in the right-hand sidebar:

> **Do not pick P100.** It is compute capability sm_60 (Pascal), and the PyTorch
> on current Kaggle images ships kernels for sm_70 and newer only. Every CUDA
> call then fails with `no kernel image is available for execution on the
> device`, once per run, after the model has already downloaded. T4 is sm_75 and
> works. `get_device()` now checks this at startup and says so explicitly rather
> than letting the whole sweep die cryptically.

| Setting | Value |
|---|---|
| Accelerator | **GPU T4 x2** — *not* P100 (see below; the code uses one GPU) |
| Internet | **On** ← the step everyone misses |
| Persistence | Files only (optional) |
| Add Data | your `gbmc-task1-payload` dataset |

## 4. Paste and run

**Do not use `tar xzf`.** Kaggle decompresses archives when it creates a
dataset, so the tarball no longer exists inside `/kaggle/input` — and the mount
path varies between `/kaggle/input/<slug>/` and
`/kaggle/input/datasets/<owner>/<slug>/`. Hardcoding either is how you lose two
minutes to a path error.

Also: `/kaggle/input` is read-only, and training writes `results/`, so the tree
has to be copied into `/kaggle/working` first.

**Cell 1 — stage the payload (no slug needed, finds it wherever it landed):**

```python
import os, shutil, tarfile
from pathlib import Path

WORK, INPUT = Path("/kaggle/working"), Path("/kaggle/input")

tarball = next(iter(sorted(INPUT.rglob("kaggle_payload_task1.tar.gz"))), None)
if tarball:                                   # archive survived upload
    print("archive:", tarball)
    with tarfile.open(tarball) as tf:
        tf.extractall(WORK)
else:                                         # Kaggle already extracted it
    marker = next(iter(sorted(INPUT.rglob("src/train.py"))), None)
    if marker is None:
        raise SystemExit("payload not found under /kaggle/input -- is the dataset attached?")
    payload = marker.parent.parent
    print("payload:", payload)
    for item in sorted(payload.iterdir()):
        dst = WORK / item.name
        if dst.exists():
            continue
        shutil.copytree(item, dst) if item.is_dir() else shutil.copy2(item, dst)

os.chdir(WORK)
print("cwd:", os.getcwd())
print("contents:", sorted(p.name for p in WORK.iterdir()))
```

Expect `contents: ['config.yaml', 'data', 'requirements.txt', 'scripts', 'src']`.
If `scripts` is missing, your dataset predates the payload fix — rebuild with
`make kaggle-payload` and upload a new version.

**Cell 2 — run:**

```python
!pip -q install torch-geometric
!python scripts/kaggle_task1.py --model bert-base-uncased --epochs 8
```

Sanity-check the first minute:

- `tag vocabulary: 50 tags from musiccaps_tag_vocab.json` — right vocabulary
- `text_source=caption_masked applied to N rows` — labels are not in the input
- `leakage check passed` — splits are artist-disjoint
- `task 1 epoch 1/8 | loss ... | val macro_f1=...` — it is training

## 5. Save & Run All — not Quick Save

**Save Version → Save & Run All (Commit)**. This re-executes the notebook
detached, survives closing the browser, and gets the 12-hour limit instead of the
interactive idle timeout.

> **This is the step that decides whether you get your results back.**
> `/kaggle/working` is discarded when an interactive session ends, and **Quick
> Save stores the notebook without re-running it**, so its version carries no
> output files. Download the Output of a Quick-Saved version and you get an empty
> tree — typically just `kaggle/working/.virtual_documents`. Only
> **Save & Run All (Commit)** captures the files.
>
> If it already happened: the console log has everything. Save it to a file and
> run `python scripts/recover_results_from_log.py run.log`, which rebuilds the
> result JSONs (including per-epoch history) and stamps them
> `recovered_from_log: true`. Only the per-tag thresholds are unrecoverable.

### What it runs

| # | Corpus | `text_source` | Freeze mode | Purpose |
|---|---|---|---|---|
| 1 | MusicCaps | `caption_masked` | `frozen_probe` | cheap floor |
| 2 | MusicCaps | `caption_masked` | `top_n` | the usual recipe |
| 3 | MusicCaps | `caption_masked` | `full_ft` | **the headline Task 1 number** |
| 4 | MusicCaps | `caption_raw` | `full_ft` | leakage demo — labels visible in the caption |
| 5 | MTAT | `metadata` | `full_ft` | non-circular secondary, weak text |

Runs 3 and 4 are identical except for masking, so their difference is a clean
measurement of how much the raw caption leaks its own labels. That gap is a
result worth reporting in its own right.

Results are written after **every** run to `results/task1_seed42_<tag>.json` plus
a rolling `results/task1_sweep.json`, so a session that dies mid-sweep still
leaves you everything finished so far.

---

## 6. Bring the results back

When the commit finishes: notebook → **Output** tab → **Download all** (a zip of
`/kaggle/working`).

Then, from the repo:

```bash
python scripts/import_kaggle_results.py ~/Downloads/archive.zip
```

It copies the result JSONs into `results/`, prints the comparison table, and
prints the leakage gap. It **refuses** anything whose provenance is synthetic or
whose thresholds did not come from the validation split, and it will not silently
overwrite an existing local result (`--force` if you mean it).

Then regenerate the tables and figures:

```bash
python -m src.evaluate --device cuda
```

Finally put the headline number into `report/final_report.md` §6.1, which
currently reads `[TBD — Kaggle]` for the B3/T1 row.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Dies at model load, `OSError`/connection error | Internet is Off | Sidebar → Internet On (needs phone verification) |
| `CUDA error: no kernel image is available for execution on the device` | Accelerator is **P100** (sm_60); current torch needs sm_70+ | Sidebar → Accelerator → **GPU T4 x2**, then Run All again. No re-upload needed. |
| `ModuleNotFoundError: torch_geometric` | dropped the pip line | Put it back; it costs 11 s |
| `FileNotFoundError: features.h5` | payload predates the Task 1 fix | Rebuild with `make kaggle-payload` |
| `/bin/bash: line 1: slug: No such file or directory` | the literal `<slug>` placeholder was left in; bash read `<` as redirection | Use the Cell 1 above — it needs no slug |
| `can't open file '.../scripts/kaggle_task1.py'` | either Cell 1 did not run, or the dataset predates the payload fix that added `scripts/` | Re-upload a freshly built payload |
| `tar: ...tar.gz: Cannot open` | Kaggle already extracted the archive | Use Cell 1; do not call `tar` |
| `LEAKAGE: ... artist_id(s) span multiple splits` | edited manifests by hand | Rebuild: `python -m src.splits --validate-audio --prune-to-cache` |
| Run stops early at epoch 4-5 | early stopping, `patience=3` | Working as intended; `best_epoch` is in the JSON |
| Output tab is empty / zip has only `.virtual_documents` | used **Quick Save** instead of **Save & Run All**, or the interactive session ended | Re-run with Save & Run All, or recover from the log with `scripts/recover_results_from_log.py` |
| Out of GPU quota | 30 h/week | The sweep is ~45 min; `--only full_ft` cuts it to ~25 min |

## Quota note

Kaggle gives ~30 GPU-hours/week. One seed of this sweep is roughly 45 minutes.
Three seeds (42, 1337, 2024) — which the report wants for the headline rows only
— is about 2.5 hours. That is comfortable; the 6–10 hour figure in the earlier
notes was an over-estimate.
