# PROGRESS

Last session ended: 2026-09-05 13:10
Currently resuming at: A6.4 (operator action), then Phase B

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
- [x] A3.2 MTAT extracted — 21,358/21,361 (3 unreadable mp3s)
- [x] A3.3 FMA-small extracted — 7,994/7,994
- [x] A3.4 MusicCaps extracted — 4,830 after decode verification
- [x] A3.5 DEAM extracted — 1,802/1,802, 0 failures
- [x] A3.6 Norm stats computed + persisted (per corpus, train-only, tested)

## Phase A3 GATE: PASSED — 4 caches, 100% coverage, train-only norm stats

## Phase A4 — Graphs
- [x] A4.1 Segment graphs, all datasets — 35,984
- [x] A4.2 Chord graphs — 35,984
- [x] A4.3 Hetero graphs — 35,984
- [x] A4.4 GATE: PASSED on real MTAT — long-range 0.858, repeat recall 0.900
- [x] A4.5 Export 20 REAL sample graphs — all 4 corpora

## Phase A4 GATE: PASSED — A4.4 sanity gate green, 20 real sample graphs committed

## Phase A5 — EDA
- [x] A5.1 eda.ipynb executed on real data — 0 errors

## Phase A6 — First real runs
- [x] A6.1 Task 2 on real MTAT (local GPU) — macro-F1 0.3692
- [x] A6.2 CNN baseline B2 (local GPU) — macro-F1 0.1654, param-matched 96.2%
- [x] A6.3 Kaggle payload built — 3.4 MB (Task 1 is text-only)
- [!] A6.4 Task 1 on Kaggle — BLOCKED on operator action (see below)

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

---

## Phase B handoff (draft — finalise at the Phase A exit gate)

### What Phase B has to produce

| Item | Depends on | Notes |
|---|---|---|
| Task 3 fusion + multi-task DEAM | A3 caches, A4 graphs | alternating MTAT/DEAM batches; the masked loss is already written and tested |
| Task 4 contrastive retrieval | MusicCaps graphs | gallery is **2,634**; use `--precompute-text-embeddings` |
| Ablation: 7 fusion modes + rewired control | Task 3 | `src.evaluate --ablation-epochs N` already drives this |
| t-SNE by genre and mood + k-NN probe + silhouette | Task 3 embeddings | the probe numbers are the evidence, not the picture |
| 3 case studies, >= 1 a failure | Task 3, GATv2 | `attention_viz.generate_case_studies` picks the worst example by measured F1 |
| 10 retrieval examples, >= 2 failures | Task 4 | `evaluate.export_retrieval_examples` |
| Human eval: 5 listeners | Task 4 | **has a hard human dependency — line the raters up now** |
| 3-seed repeats on the headline rows | everything | seeds 42 / 1337 / 2024; cut non-headline seeds first if time runs short |

### Decisions already locked, do not relitigate

- **`Xtext`**: MusicCaps headline uses `caption_masked`; the raw caption is the
  leakage demo only; MTAT uses metadata. `data.text_source` selects it.
- **Splits**: artist-disjoint within *and* across corpora. The unrepaired MTAT
  variant exists solely for a cited comparison.
- **Thresholds**: tuned on val, frozen, applied once to test.
- **Norm stats**: per corpus, train split only, provenance asserted by test.
- **Never report accuracy** for multi-label tagging.

### Routing, from the measured VRAM table

Everything fits on the 1650. The binding constraint is wall-clock, not memory:
three freeze modes across three seeds is what needs Kaggle, not any single run.
bert-base above batch 32 (or seq 256) needs `bert.gradient_checkpointing: true`.

### Known caveats to carry into the report

1. MusicCaps attrition is **not random**; survivors are a biased sample even at
   91% recovery.
2. The chord agreement figure (0.774) is measured on MIDI-derived chroma and
   bounds template matching in isolation — it says nothing about audio.
3. MTAT val is thin after the artist-disjoint repair: 977 clips, 14 artists.
   Expect noisy validation curves and say so.
4. The mel cache is time-pooled to 256 frames; B2 sees the whole track at ~0.12 s
   resolution, which is finer than the GNN's ~1 s segments.

### A3 findings

- **A3.4 caught the failure mode the spec names.** 206 MusicCaps files are
  ~352-byte truncated stubs left by the *original* download. They pass
  `Path.exists()` and a non-zero size check, and only fail when something tries
  to decode them. Decode verification existed in `build_musiccaps_manifest` but
  had never actually run on real data, because `verify_datasets` passed
  `verify_decode=False` by default. Rebuilt with `--validate-audio`:

  | | rows | Task 4 gallery |
  |---|---|---|
  | one download pass | 2,781 | 1,481 |
  | after `--retry-failed` | 5,043 | 2,634 |
  | **after decode verification** | **4,830** | **2,503** |

  The last row is the one to report. The middle row was never real.

- **Windows spawn + torch = WinError 1455.** MusicCaps and DEAM extraction died
  because every `ProcessPoolExecutor` worker re-imported the parent module, which
  pulled in torch via `src.utils`, and six copies of torch exhausted the commit
  limit. Extraction workers need numpy and librosa only, so torch is now imported
  lazily inside the functions that use it. This also makes every CLI start faster.

- **Manifests are pruned to cached keys.** Three MTAT and three FMA rows exist on
  disk but do not decode; they are now absent from the manifest rather than
  raising a KeyError inside a dataloader worker.

- Extraction totals: MTAT 21,358 · FMA 7,994 · MusicCaps 4,830 · DEAM 1,802 =
  **35,984 tracks**, ~2h20 wall clock at ~196 tracks/min on 6 workers.


---

## Phase A exit criteria — audit

| # | Criterion | Status |
|---|---|---|
| 1 | Every A0–A6 item `[x]` or explicitly `[-]`/`[!]` | **A6.4 is `[!]`**, everything else done |
| 2 | No synthetic artifact reachable by `evaluate.py` without `--synthetic` | PASS — provenance field + hard guard + 12 tests |
| 3 | Four feature caches complete; norm stats provenance-tested train-only | PASS — 35,984 tracks, 100% manifest/cache coverage |
| 4 | Graph visual sanity gate passed, renders saved | PASS — MTAT long-range 0.858, repeat recall 0.900 |
| 5 | 20 **real** sample graphs committed | PASS — 5 from each of 4 corpora |
| 6 | Leakage assertions pass on all five manifests; both MTAT variants exist | PASS — including the concatenation |
| 7 | `state/vram_report.md` records real-encoder peaks and routing | PASS |
| 8 | `state/env_report.md` records BERT loadability and chord methodology | PASS |
| 9 | Task 2 + B2 have real numbers; Task 1 running on Kaggle | Task 2 + B2 **done**; Task 1 **blocked on operator** |
| 10 | `pytest` green | PASS |
| 11 | Everything committed with phase-ID messages | PASS |

### A6.4 — what is blocked, and exactly what to do

Launching a Kaggle run needs Kaggle credentials and a browser session, neither of
which exists in this environment. Everything that *can* be prepared is prepared,
and the local dry run confirms the path works end to end (frozen probe, 1 epoch,
distilbert on real MusicCaps: macro-F1 0.187 on 2,503 test clips).

1. Upload `kaggle_payload_task1.tar.gz` (3.4 MB) as a Kaggle dataset.
2. New notebook, GPU accelerator on, attach that dataset, then:

   ```
   !tar xzf /kaggle/input/<dataset>/kaggle_payload_task1.tar.gz -C /kaggle/working
   %cd /kaggle/working
   !pip -q install torch-geometric
   !python scripts/kaggle_task1.py --model bert-base-uncased --epochs 8
   ```
3. **Save & Run All** so it runs in the background against the 12-hour limit.

The sweep covers the three freeze modes plus the masked/raw caption pair; that
pair *is* the leakage result, and the script prints the gap explicitly.

### Why Task 1 needs no graphs

Task 1 is text-only, so the payload is 3.4 MB rather than the ~1.4 GB a
graph-carrying archive would be. Phase B will need the graph payload
(`make kaggle-payload`) for Tasks 3 and 4.
