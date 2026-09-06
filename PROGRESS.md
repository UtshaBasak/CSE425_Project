# PROGRESS

Last session ended: 2026-09-06 (phase C, running unattended)
Currently resuming at: whatever `state/queue_state.json` says is unfinished

A7's code is complete and pushed; its final runs (Task 1 sweep runs 4-5, then B2
on both domains) were still executing when Phase B began, so B0 started with the
work that does not contend for the GPU.

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

**Full step-by-step: see `KAGGLE_RUNBOOK.md`.** Summary below.

A blocking bug was found and fixed on 2026-09-05 by simulating Kaggle (extracting
the payload into a clean directory and running there): Task 1 loaded rows through
the graph dataset, which opens the HDF5 caches the payload deliberately omits, so
the run died ~30 s in on `FileNotFoundError: features.h5`. Task 1 now reads text
straight from the manifests. Also verified: `src.train` genuinely requires
torch-geometric to import even for Task 1, so the `pip install` line must stay
(11 s, pure-Python wheel).

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


## Phase A7 — Pre-Phase-B corrections

Five defects found after the first real results came in. Three of them changed
numbers that were already written down, which is the reason this phase exists
rather than being folded into Phase B.

- [x] **A7.0 Real Kaggle output imported.** The operator re-ran with *Save & Run
  All* and the output files exist this time. Every test metric matches the values
  previously recovered from the console log exactly, and the runs now carry their
  real per-tag thresholds and per-epoch history. No result in `results/` is
  second-hand any more. `output/` (863 MB, mostly checkpoints) added to the parent
  `.gitignore`.

- [x] **A7.1 FMA-small genre is now the Task 2 headline.** The deliverable asks
  for genre classification on FMA-small; what existed was MTAT multi-label
  tagging. Eight-way single-label classification is wired end to end
  (`masked_genre_loss`, `M.multiclass_metrics`, `GNNClassifier(n_tags=0)`), and
  MTAT multi-label is kept as a reported second domain because it is the only
  corpus whose label space Task 3 also uses. Accuracy is reported for the genre
  runs and only for them: one label per clip, eight near-balanced classes
  (train 797-801, test 100-101), chance 12.5%.

- [x] **A7.2 B2 rebuilt as a real mel-CNN.** Parameter-matching it to the GNN's
  432,690 parameters was a specification error and the likely cause of its
  0.1654. Three things were wrong and all three were ours, not the CNN's:

  | Was | Now |
  |---|---|
  | width searched to match 432,690 params | short-chunk CNN, 3.43M params (tags) / 2.89M (genre) |
  | mel mean-pooled to 256 columns (8.8 fps) | native resolution, 43.07 fps |
  | one global descriptor per 29 s clip | random 3 s excerpts; 9 chunks averaged at inference |

  `target_params` now **raises** rather than being quietly accepted, so the
  constraint cannot come back by accident. Compute budget is what is equalised
  (same GPU, same 30-epoch cap, same early-stopping rule) and both parameter
  counts and wall-clock times are reported. Full-resolution cache built by
  `scripts/build_mel_cache.py`: MTAT 4.03 GB, FMA 1.74 GB, 0 failures, float16 +
  gzip-4 + shuffle (1.63x measured, 2.2 ms/clip to read back).

- [x] **A7.3 Tag vocabularies were leaking, and one of them mattered.**
  `build_musiccaps_tag_vocab` ranked aspect frequency over all 5,521 MusicCaps
  clips, so the test split helped choose the label space before training began.
  Counting only the 2,095 train clips **swaps 7 of the 50 tags**, so every
  MusicCaps Task 1 number was invalidated and re-run. The MTAT vocabulary was
  audited the same way and turned out to be **unaffected in content** — its
  top-50 set is identical either way, only the frequency ordering moves — so
  MTAT-scored results did not need re-running for this reason. Fixed anyway;
  both vocabulary files now record `split_used` and `n_clips_counted`, and six
  tests cover it, including one asserting that train-only and all-split
  selection still *disagree* (without it the other five would pass whether or
  not the fix were present).

- [x] **A7.4 Threshold stability quantified rather than assumed.** MTAT's
  validation split is 977 clips from 14 artists, so the per-tag thresholds fitted
  on it carry a variance term that a point estimate hides. `_fit` now dumps the
  val/test score matrices of the selected model; `scripts/threshold_bootstrap.py`
  resamples validation 100x, re-tunes all 50 thresholds per replicate and applies
  each to the fixed test split. **Decision rule fixed in advance:** spread above
  0.02 test macro-F1 means tuned numbers may not stand alone and every table
  carries the fixed-0.5 number beside them. The verdict is written into the JSON.

- [x] **A7.5** pytest re-run, PROGRESS.md updated, committed with phase-ID
  messages, pushed.

### Also in A7 (not requested, but load-bearing)

- `--run-tag` on `src.train`. Two configurations of the same task were writing to
  the same result file; the genre smoke run overwrote the MTAT Task 2 result and
  it had to be restored from git.
- `report/final_report.tex` replaces the markdown report as the living document,
  in IEEEtran two-column form. `report/fill_report.py` injects every number from
  `results/*.json` into an AUTOGEN macro block, so the prose contains no literal
  figures and a stale number cannot survive a re-run. Missing results render as
  `\textit{pending}` and the script names them.
- The Task 1 sweep now runs **locally**. This GPU is sm_75, so the work that went
  to Kaggle for wall-clock reasons is reproducible here — roughly 2 min/epoch
  against Kaggle's 12 s, which is slow but unattended.

### Left as-is, deliberately

The 0.774 chord-validation figure keeps its caveat unchanged: it is measured on
MIDI-derived chroma, validates the template-matching step in isolation, and is
not an audio-domain chord accuracy.


---

# PHASE B

## The noise floor governs every ablation from here on

A7.4 measured a **tuned test macro-F1 spread of 0.0288** across 100 validation
resamples on MTAT. That is not a footnote on the tagging tables; it is a
measurement floor for the project. **Any difference below ~0.029 macro-F1 on
MTAT is indistinguishable from validation-split noise.**

Consequences, applied without exception to every ablation table:

1. tuned **and** fixed-0.5 columns;
2. 3 seeds, mean +- sd, on **every** row, not just headline rows;
3. a "delta vs best" column and the 0.0288 figure stated in the caption;
4. no claimed ordering between rows whose intervals overlap -- "within noise of
   each other", said plainly.


---

# PHASE C — execution

## The plan changed: C4 moved to Kaggle

`bert-base` measures **68 ms/row** on this GTX 1650. The seven-mode x three-seed
fusion ablation is 21 runs, which at that rate is **~26 h locally** — and Phase C
decision rule 3.6 would then have forced cutting rows out of the ablation table.

Moving C4 to a two-T4 Kaggle session avoids the cut *and* frees the local GPU, so
C5–C7 run in parallel with the sweep instead of queueing behind it. The revised
local order is **C1 → C2 → C3 → C5 → C6 → C7**, with C4 merged in on import.

| | |
|---|---|
| Payload | `kaggle_payload_ablation.tar.gz`, **105.0 MB** (237.6 MB raw, 71 files) |
| Build | `make kaggle-payload-ablation` |
| Instructions | `state/kaggle_c4_instructions.md` — one paste-able cell |
| Runner | `scripts/kaggle_ablation.py --shard I --shards N` |
| Merge | `scripts/import_kaggle_results.py <archive.zip>` |

The payload carries the four `features_*.h5` caches rather than the exported
`.pt` graphs, because Task 3 builds graphs on the fly from those — `DataBundle`
sets `graph_dir=None` for real data — so a graph payload would be both larger
(450 MB) and unusable. No mel caches, no checkpoints, no audio.

**Sharding verified before upload.** Shard *i* takes every *N*th run from a
fixed-order list; checked for 1–4 shards that every run appears in exactly one
shard with no overlap, which matters because the two shards run as separate
processes that cannot see each other. `--dry-run` prints the split.

**Import is guarded three ways**: synthetic provenance refused, thresholds not
from `val` refused, and — new — the ordered tag vocabulary is hashed into every
result and a file whose hash matches no current vocabulary is refused. A7.3
changed 7 of the 50 MusicCaps tags, so a pre-fix result is numerically fine and
semantically incompatible, and nothing about the file would show it.

## How the local queue runs now

One process owns it: `scripts/run_queue.py`. The bash chains it replaces waited
on log markers and twice produced **two waiters on one job, and so two
concurrent training runs** — once corrupting a result file. A single sequential
owner cannot do that.

- **Watchdog, not halt.** A failed step is logged with its traceback, marked
  `[!]`, and the queue moves on; failures are retried once at the end. One
  failure must never idle the GPU for hours.
- **Resume.** Each step declares the artifact that proves it finished — and
  *freshness* is checked, not just existence: `baselines_seed42.json` exists
  from before the A7.2 rework holding the old parameter-matched `B2_mel_cnn`,
  and `structural_controls.json` exists holding a dry run. Skipping on existence
  would have silently dropped two Task 2 deliverables.
- **Commit and push after every step**, not every gate, so a dead session still
  leaves results on GitHub.
- **Estimates from measurement.** Each step records its real wall-clock and the
  remaining estimate is rescaled by how the predictions are actually tracking.

Live status, rewritten by the runner after every step, is in the QUEUE block
below. `python scripts/run_queue.py --plan` shows it without running anything.

## Estimates to watch

- **C2** is the widest band (~200 min prior, extrapolated from 68 ms/row plus
  GNN and HDF5 overhead). It gets tightened the moment the step completes.
- **B2** was never timed; the 1.5–2.5 h band is a prior. Whether it held is
  recorded in the queue table.

<!-- QUEUE:BEGIN (rewritten by scripts/run_queue.py) -->

## Local run queue -- live status

Updated 2026-09-06 09:57. **6 done, 0 failed, 15 remaining.**

Estimated **1.8 h** of local GPU left, from 5 measured step(s) (estimates running 0.44x of prediction).

C4 is not in this queue: the seven-mode ablation runs on Kaggle so it does not sit in front of C5-C7 for a day. See `state/kaggle_c4_instructions.md`; merge with `scripts/import_kaggle_results.py`.

| Step | Phase | State | Wall-clock |
|---|---|---|---|
| C1a baselines B1/B2/B4 | C1 | [x] | 46.7 min |
| C1b structural controls | C1 | [x] | 20.8 min |
| C2 Task 3 headline (MTAT, bert-base) | C2 | [x] | 68.9 min |
| C3 Task 4 headline (MusicCaps dual encoder) | C3 | [x] | 4.5 min |
| C3 retrieval export | C3 | [x] | 38.7 min |
| C3 listening study | C3 | [x] | 0.0 min |
| C5 Task 3 MusicCaps bert_only | C5 | [~] | running |
| C5 Task 3 MusicCaps gnn_only | C5 | [ ] | ~11 min est. |
| C5 Task 3 MusicCaps cross_attention | C5 | [ ] | ~11 min est. |
| C6 zero-shot vs supervised | C6 | [ ] | ~7 min est. |
| C6 full evaluation (t-SNE, S_graph, case studies) | C6 | [ ] | ~13 min est. |
| C6 threshold bootstrap | C6 | [ ] | ~2 min est. |
| C6 compact figures | C6 | [ ] | ~1 min est. |
| C6 genre confusion figure | C6 | [ ] | ~1 min est. |
| C7 Task 2 genre seed 1337 | C7 | [ ] | ~2 min est. |
| C7 Task 2 genre seed 2024 | C7 | [ ] | ~2 min est. |
| C7 Task 2 tags seed 1337 | C7 | [ ] | ~4 min est. |
| C7 Task 2 tags seed 2024 | C7 | [ ] | ~4 min est. |
| C7 Task 4 seed 1337 | C7 | [ ] | ~20 min est. |
| C7 Task 4 seed 2024 | C7 | [ ] | ~20 min est. |
| report fill + structural check | C7 | [ ] | ~0 min est. |

<!-- QUEUE:END -->

## Phase B0 - gaps to close before Task 3

- [x] **B0.1 structural controls wired.** `graph.rewire` applies
  degree-preserving double-edge swaps at load time, seeded per track id with
  `crc32` (not `hash`, which Python randomises per process -- that would have
  made the control silently irreproducible between runs). The edge-type ablation
  needs no new code: `graph.temporal_edges` / `graph.similarity_edges` are
  already honoured at build time and `MusicGraphDataset` builds on the fly, so
  `--override` is enough. Four tests, including one that asserts the rewiring is
  identical across dataset instances, changes the topology, and preserves every
  degree. **Runs queued behind the A7 pipeline** -- both are GPU jobs.

- [x] **B0.2 the 0.3692 -> 0.3737 change is NOT the vocabulary fix.** Checked
  rather than asserted:

  | | epochs | best epoch | macro-F1 | micro-F1 | AUC-PR | params |
  |---|---|---|---|---|---|---|
  | old | 10 (cap) | **8** | 0.3692 | 0.4124 | 0.3915 | 432,690 |
  | new | 30 (cap) | 11, stopped 14 | 0.3737 | 0.4102 | 0.3939 | 432,690 |

  The old run's best epoch was 8 of a 10-epoch cap, i.e. it was **still
  improving when the budget ran out**. The new run reached epoch 11 before
  early stopping. Same architecture, same parameter count. The A7.3 finding
  stands: MTAT's top-50 *set* is identical train-only, only the frequency
  ordering moved, and macro-F1, micro-F1 and AUC-PR are all invariant to a
  permutation of the tag columns.

  The honest reading is stronger than "three more epochs helped": **+0.0045 is
  six times below the 0.0288 noise floor, and micro-F1 moved the other way
  (-0.0022).** The two runs are indistinguishable. Reported as such.

- [ ] **B0.3 B2-vs-GNN framing prepared.** B2 is now a 3.43M-parameter
  short-chunk CNN on a full-resolution cache and may beat the GNN's 42.3% on
  FMA-small genre. If it does it is the headline finding and stays that way --
  the GNN will not be re-tuned until it wins. The comparison is then about
  **representation granularity** (96-dim pooled segment statistics at ~1 s
  against full-resolution spectrograms), not about graphs. **B4 is the control
  that isolates structure**, because it uses the same 96-dim features without
  it, so the Task 2 discussion is built on the B4 -> GNN delta with B2 as an
  upper reference. Awaiting the B2 runs.

- [ ] **B0.4 Task 1 attention examples** must come from a corrected-vocabulary
  run. Blocked on the sweep finishing.

- [x] **B0.5 mood label sets defined in `config.yaml`**, not at plot time.
  Worth recording: of 15 proposed affect words, **7 are not in MTAT's top-50 at
  all** (`sad`, `happy`, `mellow`, `calm`, `dark`, `upbeat`, `eerie`). The
  surviving MTAT set -- soft, hard, ambient, quiet, loud, slow, fast, weird --
  is really texture and dynamics rather than affect, so the DEAM
  valence/arousal quadrants carry the actual mood story and the MTAT panel is
  labelled for what it is. Clips matching no mood tag are drawn in grey and
  **excluded** from the k-NN probe rather than pooled into an "other" class,
  which would inflate it.

- [x] **B0.6 page guard active.** `check_tex.py` now fails (non-zero exit) above
  10 pages, counts anything after `\appendix` separately, and writes the split
  into the header block. Three tests: fires at 12 pages, passes at 7, and does
  not count appendix material against the limit.



## Phase B1 - Task 3 fusion (22 marks)

Code complete and tested; the runs are queued behind the A7 pipeline and the
B0.1 controls, because all three want the same GPU.

- [x] **B1.2 DEAM imbalance handled explicitly**, three separate problems:

  | problem | what it does if ignored | fix |
  |---|---|---|
  | 1:14 size ratio | emotion head sees each DEAM track ~14x per MTAT epoch and overfits | `multitask.batch_ratio: [4, 1]`, recorded in every result |
  | raw 1-9 targets | MSE of 4-10 against per-tag BCE of 0.2; emotion takes the gradient | standardise with train-split stats |
  | one blended loss | a collapsing head hides inside a falling total | per-term losses printed every epoch |

  Train-split statistics are valence 4.901 +- 1.206 and arousal 4.861 +- 1.273
  over 1,277 tracks, cached to `data/splits/emotion_stats.json`;
  `compute_emotion_stats` refuses any split but train.

  **The companion change matters as much as the standardisation.** Predictions
  are inverted before MAE and RMSE, so the report stays on DEAM's original 1-9
  scale. R2 is invariant to the affine transform; MAE is not, and an "MAE of
  0.8" in standard deviations would mean nothing to a reader. A test asserts the
  inversion happens and that skipping it makes a perfect model look wrong.

  Early stopping follows the **tagging** metric with emotion auxiliary, per the
  PDF's `L_aux` framing: a blended criterion lets a collapsing tag head hide
  behind a good regression fit.

- [x] **B1.3 ablation harness** (`scripts/fusion_ablation.py`) with **two
  documented budgets**: distilbert at a reduced epoch count held *identical*
  across all seven modes for the sweep, then bert-base at full budget for the
  headline rows only. The budget is stamped into every result file so the two
  tables cannot be merged by accident.

  The noise-floor rule is enforced **in code, not in prose**: mean +- sd across
  seeds on every row, a delta-vs-best column, every row inside 0.0288 flagged,
  and `summarise()` returns `best_mode = None` when the top rows overlap. Three
  tests cover it, including one asserting no winner is named among
  indistinguishable rows.

- [ ] **B1.1 both domains** - MTAT metadata primary, MusicCaps `caption_masked`
  secondary. Queued. The contrast is the finding: fusion gain should scale with
  text informativeness, so a small or absent gain on MTAT metadata is the
  *expected* result and is explained by the text source rather than being a
  failure.



## Phase B2 - Task 4 retrieval (18 marks)

- [x] **B2.2 chance reference.** `random_retrieval_reference` is folded into
  every `retrieval_metrics` payload, so no table can report R@K without it.
  Against the 2,503-clip gallery, chance R@10 is 0.4%: an R@10 of 0.05 reads as
  failure in isolation and as 12x chance beside the reference. The multiple is
  reported too.
- [x] **B2.3 zero-shot** (`scripts/zero_shot_eval.py`): four prompt templates,
  per-template macro-F1, the spread, and a template ensemble, all with
  thresholds tuned on val. The MusicCaps corpus and vocabulary are **pinned** -
  an `--override` that would change them is refused, because the supervised
  reference is the Task 3 MusicCaps run and comparing against an MTAT-vocabulary
  model would be two different problems. Runs whose recorded `tag_vocab` does
  not match are skipped with a warning.
- [x] **B2.4 ten examples with >= 2 failures** already existed from Phase A
  (`export_retrieval_examples`); it selects the two worst-ranked queries
  deliberately.
- [ ] **B2.1 dual encoder run** - queued.

## Phase B3 - analysis

- [x] Mood colouring now comes from the B0.5 config rather than literals. The
  DEAM midpoint was hardcoded at 5, correct for a 1-9 scale and silently wrong
  for any other range. Quadrant names are stored **in the order of the index the
  code assigns**, not in circumplex order - the latter would have mislabelled
  the legend, which a scatter plot never reveals.
- [x] A third t-SNE panel for MTAT mood tags, reported separately because it is
  a different construct (texture and dynamics, not affect). Clips matching no
  mood tag are excluded from the k-NN probe, not pooled into an "other" class
  that would be trivially separable and inflate it.
- [x] t-SNE already carries k-NN probe and silhouette; `S_graph` real-vs-rewired
  plotting already exists and now has real rewired runs to consume.
- [ ] Three case studies on the **MusicCaps** Task 3 model - queued.

## Phase B4 - human evaluation

- [x] `scripts/make_listening_page.py` writes `sheet_key.json`, a self-contained
  HTML page with the clips embedded as base64, and the form question list in
  page order. Self-contained because a page referencing local paths breaks the
  moment it is emailed, and hosting the clips would publish copyrighted audio.
  No rating widgets - ratings go to a form so responses land in one CSV.
  Controls use **real audio** from an unrelated clip; a silent control is
  identifiable without listening and would measure attention to silence.
- [x] `scripts/analyse_human_eval.py` adapts the wide Google Forms export to
  long format, keyed on the **clip number parsed from the question text**, not
  column order. An off-by-one there would swap real pairs with controls and
  invert the headline finding, so it refuses a mismatched export rather than
  guessing. It states plainly when control discrimination is too small for the
  study to be informative.
- [ ] Awaiting `data/human_eval/raw_responses.csv` - a hard human dependency.

## Phase B6 - submission artifacts

- [x] **Demo notebook: 16.7 s on CPU**, zero errors, against a 2-minute budget -
  and measured while a training job was competing for the same cores, so the
  clean figure is lower.
- [x] **Fresh-clone test run, and it found three real defects**, none visible
  from the working tree: `data/raw/` and `results/retrieval_examples/` are
  absent from a checkout, so the prescribed-tree test failed on a clone (for a
  grader that is indistinguishable from a broken repo); and the two MusicCaps
  vocabulary tests failed rather than skipped without the gitignored raw CSV,
  which reports "the code is broken" when the truth is "the corpus is absent".
  Fixed; the clone now runs **198 passed, 5 skipped**.
- [x] All 20 committed sample graphs verified in the clone: they load, carry
  provenance `real`, and match the frozen contract.

## Phase B7 - report

- [x] A named subsection on the measurement floor and what it forbids, stating
  explicitly what refusing to rank costs and why it is still the stronger claim.
- [x] A section collecting the four protocol findings - non-disjoint canonical
  split, vocabulary chosen using test annotations, files that existed but did
  not decode, unstable operating point - as instances of one failure mode rather
  than as scattered footnotes.
- [x] Limitations expanded to the full list, led by the floor and the B2/GNN
  input-granularity asymmetry.
- [x] Reproducibility section: hardware, seeds, train-only provenance rules,
  threshold discipline, exact commands.
- Report is **7.4 pages** of a 6-10 limit; `check_tex.py` clean apart from
  macros pending on runs in flight.


### Why Task 1 needs no graphs

Task 1 is text-only, so the payload is 3.4 MB rather than the ~1.4 GB a
graph-carrying archive would be. Phase B will need the graph payload
(`make kaggle-payload`) for Tasks 3 and 4.
