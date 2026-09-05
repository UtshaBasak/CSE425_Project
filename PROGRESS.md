# PROGRESS

Last session ended: (in progress)
Currently resuming at: A3.2-A3.5 (extraction running)

Status markers: `[ ]` todo, `[~]` in progress, `[x]` done, `[!]` blocked, `[-]` skipped.

## Phase A0 — Fix Day 1-2 gaps
- [x] A0.1 Quarantine synthetic artifacts
- [x] A0.2 Verify real BERT checkpoints load under installed transformers
- [x] A0.3 Document chord-validation methodology honestly
- [x] A0.4 Real-encoder VRAM probe on GTX 1650
- [x] A0.5 Confirm item-level resumability of extraction + graph building
- [x] A0.6 Lock the Xtext decision into config

## Phase A0 GATE: PASSED — pytest 148/148, reports written, artifacts quarantined

## Phase A1 — Data verification
- [x] A1.1 verify_datasets.py on real data, output archived
- [x] A1.2 MusicCaps retry pass COMPLETE — 82.4% of retries succeeded
- [x] A1.3 FMA errata cross-check

## Phase A2 — Splits
- [x] A2.1 All 5 manifests built
- [x] A2.2 Leakage assertions pass on all (incl. the combined manifest)
- [x] A2.3 Both artist-disjoint variants recorded for MTAT

## Phase A2 GATE: PASSED — all leakage assertions green, both MTAT variants persisted

## Phase A3 — Feature extraction (LONG)
- [x] A3.1 Train-only norm stats plan verified
- [~] A3.2 MTAT extracted (background job running)
- [~] A3.3 FMA-small extracted (background job running)
- [~] A3.4 MusicCaps extracted (background job running)
- [~] A3.5 DEAM extracted (background job running)
- [ ] A3.6 Norm stats computed + persisted

## Phase A4 — Graphs
- [ ] A4.1 Segment graphs, all datasets
- [ ] A4.2 Chord graphs
- [ ] A4.3 Hetero graphs
- [ ] A4.4 GATE: visual inspection of 5 real graphs
- [ ] A4.5 Export 20 REAL sample graphs

## Phase A5 — EDA
- [ ] A5.1 eda.ipynb executed on real data

## Phase A6 — First real runs
- [ ] A6.1 Task 2 on FMA-small (local GPU)
- [ ] A6.2 CNN baseline B2 (local GPU)
- [ ] A6.3 Kaggle payload built
- [ ] A6.4 Task 1 launched on Kaggle

## Notes / blockers

- **Session 2 start (2026-09-05).** Cold start: no `PROGRESS.md`, no `state/`, no git
  repository existed. Created all three from the Section 1 template and began at A0.1.
- Inherited from Day 1-2: 1.6 GB `results/checkpoints/`, 20 plots and 20 sample
  graphs, **all produced from synthetic data**. A0.1 quarantined them into
  `results/_synthetic_smoke/` (git-ignored, with its own README).
- **A0.1 added a `provenance` field to the graph contract.** `dataset` cannot
  distinguish fake from real, because the synthetic generator deliberately reuses
  the real corpus names so the `-1`/`nan` sentinel pattern matches. Provenance is
  stamped on graphs, manifest rows, checkpoints and result JSONs, and
  `evaluate.py` / `export_sample_graphs.py` refuse synthetic input without
  `--synthetic`.
- **OPEN, for A2 — cross-corpus artist leakage.** Running `evaluate` on the
  concatenated real manifests raises `LEAKAGE: 39 artist_id(s) span multiple
  splits`. Each of the four manifests is individually leak-free, but splits were
  assigned per corpus independently, so one artist can land in FMA-train and
  DEAM-test. Task 3 trains on MTAT+DEAM jointly, so this is a live correctness
  problem, not a formality. Fix in A2 with a global reconciliation pass.

### A0 findings worth carrying forward

- **A0.3 the chord number was not what it looked like.** `validate_against_lmd`
  compares MIDI-derived chroma against labels read from the same MIDI, so it
  never touched audio. Re-measured at n=200: **0.774** frame agreement over
  471,576 frames. It bounds template matching in isolation and is now impossible
  to quote without its caveat.
- **A0.4 the VRAM expectation in the spec was wrong, for a structural reason.**
  Fusion costs only ~14 MB more than the text encoder alone; peak memory is
  dominated by AdamW state, not activations. Nothing needs Kaggle for *memory* —
  only for wall-clock. Also fixed a real probe bug: HF silently skips gradient
  checkpointing in eval mode, which had made the ckpt rows identical.
- **A0.5 atomic writes were missing entirely** despite being a protocol rule.
  Added, plus a key sidecar so a corrupt HDF5 cannot erase the record of what
  completed.
- **A0.6 MTAT `Xtext` was the degenerate case the spec warns about** — the
  manifest literally put the tag string into `text`. Now metadata-only, with a
  regression test.

### A1/A2 findings

- **ffmpeg was never installed.** `yt-dlp` cannot trim to the 10 s window without
  it, so A1.2 would have failed every clip. Installed `Gyan.FFmpeg 9.0.1` via
  winget. Feature extraction does *not* need it — libsndfile decodes the mp3s we
  already have — so Phase A3 was never blocked.
- **A1.3 FMA errata: exactly 3 bad tracks**, and two independent methods agree.
  `not_found.pickle` covers the medium/large subsets and has zero overlap with
  fma_small, so it would have caught nothing on its own; a size sweep of the disk
  found 99134, 108925 and 133297 at 1–2 KB each, all of which fail to decode.
  Manifest is now 7,997.
- **A2 fixed the cross-corpus leak from A0.1.** 39 artists spanned corpora
  (e.g. present in both FMA-train and DEAM-test) because each corpus assigned its
  splits independently. Reconciling moved 52 rows — deam 46, fma 6 — and the
  leakage assertion now runs on the *concatenation*, which is the object Task 3
  and the evaluation actually load.
- **MTAT val is thin after repair**: 977 clips from just 14 artists. That is the
  honest cost of artist-disjointness on a 229-artist corpus, and it means val
  macro-F1 will be noisy. Worth a sentence in the report.
- **MusicCaps is growing while we work.** The A1.2 recovery pass is succeeding on
  ~61% of retries. Pre-recovery gallery was **1,481**; it has already passed
  1,597. Final counts must be re-recorded at the end of Phase A, with both
  numbers reported.

### A1.2 result — the MusicCaps recovery worked, and it changes the data story

The retry pass re-attempted all 2,740 previously-failed ids and **succeeded on
82.4%** of them. Most original failures were transient (rate limiting, timeouts),
not deleted videos.

| | before recovery | after recovery |
|---|---|---|
| usable rows | 2,781 / 5,521 (50.4%) | **5,043 / 5,521 (91.3%)** |
| train survivors | 1,300 / 2,663 (48.8%) | **2,409 / 2,663 (90.5%)** |
| eval survivors = **Task 4 gallery** | **1,481** / 2,858 (51.8%) | **2,634** / 2,858 (92.2%) |

Both numbers are now in the README and the report, because R@10 out of 1,481 and
out of 2,634 are different claims. Attrition is still not random — it correlates
with video age and region — so the surviving subset remains a biased sample.

The pass crashed *after* all downloads completed, on a column collision in the
log merge (`is_audioset_eval_x` / `_y`). Fixed, and the log is now regenerated
from the filesystem, which is authoritative anyway.

### A3 in flight

Extraction chain (mtat → fma → musiccaps → deam, 36,203 tracks, 6 workers) is
running in the background at ~190 tracks/min; ETA ~3 h from 07:06. Item-level
resumable, so an interrupted session just restarts it. Loaders were updated to
read the per-corpus caches.
