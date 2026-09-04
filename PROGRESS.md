# PROGRESS

Last session ended: (in progress)
Currently resuming at: A1.1

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
- [~] A1.1 verify_datasets.py on real data, output archived
- [ ] A1.2 MusicCaps retry pass launched (background)
- [ ] A1.3 FMA errata cross-check

## Phase A2 — Splits
- [ ] A2.1 All 5 manifests built
- [ ] A2.2 Leakage assertions pass on all
- [ ] A2.3 Both artist-disjoint variants recorded for MTAT

## Phase A3 — Feature extraction (LONG)
- [ ] A3.1 Train-only norm stats plan verified
- [ ] A3.2 MTAT extracted
- [ ] A3.3 FMA-small extracted
- [ ] A3.4 MusicCaps extracted
- [ ] A3.5 DEAM extracted
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
