# SESSION LOG

Append-only. One line per operation. Write a `START` line before any operation
expected to exceed two minutes, and a matching `DONE` / `FAILED` / `BLOCKED` line
when it resolves. The last entry is what a resuming session reads first.

Format:

```
[YYYY-MM-DD HH:MM] START  <step id> <what> — <expected duration, resumability>
[YYYY-MM-DD HH:MM] DONE   <step id> <what> — <result counts>
```

---

[2026-09-05 05:15] INFO   Session 2 begins. Cold start: no PROGRESS.md, no state/, no git repo.
[2026-09-05 05:15] DONE   Bootstrap — created PROGRESS.md, state/run_state.json, state/SESSION_LOG.md, logs/.
[2026-09-05 05:16] START  A0.1 Quarantine synthetic artifacts — fast, no long job.
[2026-09-05 05:19] DONE   A0.1 Quarantine synthetic artifacts — 1.6 GB moved to results/_synthetic_smoke/ (checkpoints, 20 plots, 20 sample graphs, 10 retrieval
                        examples, 8 result JSONs). Added `provenance` contract field, utils.guard_against_synthetic, guards in evaluate.py (manifest + checkpoints)
                        and export_sample_graphs.py, 12 tests. .gitignore updated.
[2026-09-05 05:19] FINDING A0.1 side-effect: running `evaluate` on the concatenated REAL manifests raises LEAKAGE on 39 artist_ids. Each corpus is individually leak-free, but
                        splits were assigned per-corpus independently, so one artist can sit in FMA-train and DEAM-test. Real cross-corpus leak; deferred to A2 to fix properly.
[2026-09-05 05:25] START  A0.4 VRAM probe on GTX 1650 — 34 configs, expect ~10 min
[2026-09-05 05:46] START  A0.6 rebuild manifests (MTAT metadata Xtext + MusicCaps text variants)
[2026-09-05 05:50] DONE   A0.2 BERT loadability - distilbert + bert-base both load, fast tokenizers, eager attentions OK. scripts/probe_env.py written.
[2026-09-05 05:50] DONE   A0.3 Chord methodology - route is MIDI-DERIVED CHROMA, no audio synthesized. Re-ran at n=200: 0.774 frame / 0.785 root over 471,576 frames.
                        Caveat embedded in docstring, every result JSON, state/env_report.md and report/final_report.md. Warns on every run.
[2026-09-05 05:50] DONE   A0.4 VRAM probe - 29 configs on the real GTX 1650. Found+fixed a probe bug (HF skips grad-checkpointing in eval mode).
                        bert-base b=64 needs checkpointing (4254 -> 2234 MB reserved); everything else local. state/vram_report.md.
[2026-09-05 05:50] DONE   A0.5 Resumability - extraction skips cached keys (kill/restart test). Added atomic writes (were MISSING), per-item HDF5 flush,
                        atomic key sidecar, corrupt-cache refusal, scripts/build_graphs.py with skip-existing. 14 tests.
[2026-09-05 05:50] DONE   A0.6 Xtext locked - MTAT text was the tag string (degenerate); now metadata-only. MusicCaps stripping fires on 2779/2781
                        captions, retains 72% of words. data.text_source key + sidecars. Manifests rebuilt. 14 tests.
[2026-09-05 05:50] GATE   A0 PASSED - pytest 148/148 green.
[2026-09-05 05:53] START  A1.2 musiccaps --retry-failed (2740 ids, 4 workers) — expect 1-3h, resumable, writes musiccaps_download_log.csv
[2026-09-05 05:57] START  A2 rebuild all manifests + cross-corpus reconciliation + LMD inventory
[2026-09-05 06:03] DONE   A1.1 verify_datasets on real data - archived to state/verify_datasets_20260905.txt. All 5 corpora present.
[2026-09-05 06:03] INFO   A1.2 ffmpeg was NOT installed (yt-dlp cannot trim without it). Installed Gyan.FFmpeg 9.0.1 via winget; also fixed
                        binary discovery so yt-dlp is found beside the venv interpreter. Retry pass then launched.
[2026-09-05 06:03] DONE   A1.3 FMA errata - 3 truncated tracks excluded (99134, 108925, 133297). not_found.pickle audio/clips lists do not
                        overlap fma_small; a size sweep of the disk independently found exactly the same 3. Manifest now 7,997.
[2026-09-05 06:03] DONE   A2   All 5 manifests + both MTAT variants. Fixed the cross-corpus artist leak found in A0.1: 39 artists spanned
                        corpora, 52 rows moved (deam 46, fma 6). Combined leakage assertion now passes across all 4 training corpora.
[2026-09-05 06:03] GATE   A2 PASSED - pytest 148/148 green.
[2026-09-05 07:05] DONE   A1.2 musiccaps retry — 2740 attempted, ~2263 recovered (82.4% hit rate). Crashed on a log-merge column collision AFTER
                        all downloads completed; bug fixed, log regenerated from the filesystem (authoritative). Files on disk: 2783 -> 5046.
[2026-09-05 07:06] START  A3.2-A3.5 extract_dataset mtat->fma->musiccaps->deam (36,203 tracks, 6 workers) — expect ~5h, item-level resumable
[2026-09-05 07:30] DONE   A3.1 train-only norm stats verified on a 40-track DEAM trial: 589 train segments, 96 dims, split=train recorded in the file.
[2026-09-05 07:30] INFO   A3 design: one HDF5 per corpus (features_<ds>.h5 + mels_<ds>.h5), written from a SINGLE decode pass. Mel patch is the
                        whole track pooled to 128x256 - full resolution would cost ~8 GB for MTAT to serve a model that pools anyway,
                        and 256 frames over 30 s is finer than the 32 GNN segments, so B2 is not disadvantaged.
[2026-09-05 07:30] DONE   A4.4 gate tooling written and trialled on real DEAM audio: long-range fraction 0.781, repeat recall 0.821. PASS.
                        Formal gate re-runs on MTAT once its extraction finishes.
[2026-09-05 08:58] DONE   A3.2 MTAT extracted - 21,358/21,361 written, 3 failed (unreadable mp3s, known MTAT zero-byte clips).
                        Train norm stats: 303,858 segments, 96 dims, split=train. Wall clock ~1h46 at ~196 tracks/min on 6 workers.
[2026-09-05 08:58] GATE   A4.4 PASSED on real MTAT - long-range fraction 0.858, repeat recall 0.900, mean similarity lag 5.76 segments,
                        0 tracks with isolated nodes. Similarity edges genuinely connect repeated sections; the graph is not a chain.
[2026-09-05 09:18] FIX    A3.4/A3.5 failed with WinError 1455 (paging file): every spawned worker re-imported torch via src.utils.
                        Made torch a lazy import in utils; extraction workers now load only numpy+librosa. Re-running musiccaps+deam.
[2026-09-05 09:30] START  A2 rebuild with --validate-audio (206 MusicCaps files are 352-byte stubs that pass an existence check)
[2026-09-05 09:32] START  A4.1-A4.3 build_graphs (segment+chord+hetero, 35,984 tracks) — resumable, skips existing .pt
[2026-09-05 09:34] DONE   A3.3 FMA extracted - 7,994/7,997 (3 known-truncated already excluded upstream). Train norm stats: 117,808 segments.
[2026-09-05 09:34] DONE   A3.4 MusicCaps extracted - 4,837 written, 206 FAILED: those files are ~352-byte truncated stubs from the ORIGINAL
                        download that pass an existence check but do not decode. Exactly the case the spec warns about.
[2026-09-05 09:34] DONE   A3.5 DEAM extracted - 1,802/1,802, 0 failures. Train norm stats: 24,263 segments.
[2026-09-05 09:34] FIX    Decode verification had never actually run on real data (verify_decode defaulted to False through verify_datasets).
                        Rebuilt manifests with --validate-audio: MusicCaps usable 5,043 -> 4,830, Task 4 gallery 2,634 -> 2,503.
[2026-09-05 09:34] DONE   A3.6 Per-corpus train-only norm stats persisted for all four corpora; provenance asserted by test.
[2026-09-05 09:34] DONE   Manifests pruned to cached keys (mtat -3, fma -3) so every listed row is loadable.
[2026-09-05 09:34] GATE   A3 PASSED - four caches complete, 100% manifest/cache coverage, norm stats train-only.
[2026-09-05 10:03] START  A5.1 execute eda.ipynb on real data
[2026-09-05 10:06] START  A6.1 Task 2 GNN on real MTAT, GPU, 10 epochs, batch 64
[2026-09-05 10:15] START  A6.2 baselines B1/B2/B4 on real MTAT, same budget as Task 2 (batch 64, 10 epochs)
[2026-09-05 10:52] START  A6.2b B2 re-run with train-only mel standardisation + 25 epochs (0.165 was below B4, i.e. undertrained)
[2026-09-05 11:08] DONE   A4.1-A4.3 graphs built - segment/chord/hetero for all 4 corpora, 35,984 x 3 = 107,952 .pt files, 0 failures.
[2026-09-05 11:08] DONE   A4.5 exported 20 REAL sample graphs spanning all 4 corpora; all 96-dim, all provenance=real, no isolated nodes.
[2026-09-05 11:08] FINDING MTAT top-50 vocabulary does not fit MusicCaps: 0.52 of 10.7 aspects match, 62% of clips get NO positive label.
                        Built a MusicCaps-native top-50 aspect vocabulary (2.83 labels/clip, 9.5% empty) + tags.vocab_source config key.
[2026-09-05 11:08] DONE   A5.1 eda.ipynb executed on real data, 0 errors. Shows the three-stage MusicCaps story and the A4.4 verdict.
[2026-09-05 11:08] DONE   A6.1 Task 2 GNN on real MTAT - test macro-F1 0.3692, micro-F1 0.4124, AUC-PR 0.3915 (3,500 clips, 432,690 params).
                        VRAM peak 37 MB, matching the A0.4 prediction of ~56 MB reserved.
[2026-09-05 11:08] DONE   A6.2 baselines - B1 random 0.0634, B1 majority 0.0000 at 93.6% element accuracy, B4 PCA+MLP 0.3239.
[2026-09-05 11:08] FIX    A6.2 B2 first came out at 0.1648, BELOW B4 on the same audio - undertrained, not an architectural finding.
                        Cause: B2 was the only model fed unstandardised input (raw dB, ~[-80,0]). Added train-only mel
                        standardisation and raised its budget to 25 epochs. Re-running; both numbers will be reported.
[2026-09-05 13:10] DONE   A6.2 B2 re-run with standardised input + 25 epochs: macro-F1 0.1654 vs 0.1648 before. NO improvement.
                        That rules out undertraining and bad conditioning. The remaining suspect is the input itself: the mel
                        cache is time-pooled to 256 frames for disk economy, which removes the fine spectro-temporal texture a
                        CNN depends on. Reported as a limitation of OUR cache, not as a CNN-vs-GNN result.
[2026-09-05 13:10] NOTE   Two run_baselines processes overlapped; the stale one overwrote the result once. Confirmed the final file is
                        from the fixed run (mel_normalised=true) before recording anything.
[2026-09-05 13:10] DONE   A6.3 Kaggle payload built: 3.4 MB for Task 1 (text-only, no graphs needed). scripts/kaggle_task1.py drives the
                        3 freeze modes + the masked/raw leakage pair. Validated the MusicCaps path locally: frozen probe,
                        1 epoch, distilbert -> macro-F1 0.187 on 2,503 test clips.
[2026-09-05 13:10] BLOCKED A6.4 cannot launch on Kaggle from here - no Kaggle credentials or browser in this environment. Payload and
                        runner are ready; this is a one-step manual action for the operator. See PROGRESS.md.
[2026-09-05 13:10] GATE   PHASE A COMPLETE except A6.4 (blocked on Kaggle credentials, which this environment does not have).
                        10 of 11 exit criteria pass; the 11th is the operator action documented in PROGRESS.md.

[2026-09-06 01:25] DONE   A7.0 Real Kaggle output imported (Save & Run All this time, not Quick Save). All five test metrics are
                        identical to the values recovered from the console log, and the runs now carry their real per-tag
                        thresholds and history. The recovered stubs are gone; nothing in results/ is second-hand any more.
[2026-09-06 01:26] DONE   A7.0 Added output/ and *.zip to the PARENT .gitignore - the downloaded notebook output is 863 MB,
                        836 MB of which is BERT checkpoints.
[2026-09-06 01:32] FIX    A7.3 CONFIRMED LEAK. build_musiccaps_tag_vocab counted aspect frequencies over the whole
                        musiccaps-public.csv (5,521 clips), so the test split helped choose the label space. Restricting the
                        count to the 2,095 train clips swaps 7 of the 50 tags: in calming/classical/joyful/keyboard/lively/
                        melodic singing/no other instruments, out e-guitar/fun/keyboard harmony/loud/poor audio quality/
                        spirited/youthful. Every MusicCaps Task 1 number is therefore invalidated and re-run.
[2026-09-06 01:32] NOTE   A7.3 Audited the MTAT vocabulary the same way: reduce_to_top_k_tags was also reading all 25,863
                        annotation rows. Restricting it to the 16,881 train rows leaves the top-50 SET identical - only the
                        frequency ordering moves. MTAT-scored results (Task 2, B2, B4) are therefore NOT invalidated. Fixed
                        anyway for provenance, and both vocab files now record split_used/n_clips_counted.
[2026-09-06 01:34] DONE   A7.3 Six provenance tests added, including one that asserts train-only and all-split selection still
                        DISAGREE - without it the other five would pass whether or not the fix were in place.
[2026-09-06 01:28] DONE   A7.2 Measured before building: decode+mel is 0.226 s/clip single-threaded, and gzip-4 with the HDF5
                        shuffle filter compresses float16 log-mel 1.63x at 2.2 ms/clip to read back. That takes the full-
                        resolution cache from 9.5 GB to ~5.8 GB, which fits the 12 GB free on this disk.
[2026-09-06 02:05] DONE   A7.2 scripts/build_mel_cache.py writes mels_full_{corpus}.h5 at the native 43.07 frames/second.
                        MTAT: 21,358 clips, 4.0 GB. The old cache was mean-pooled to 256 columns (8.8 fps), so a 3 s window
                        was 26 columns wide - that is the input B2 was being asked to convolve over.
[2026-09-06 02:00] DONE   A7.2 MelCNN rebuilt as the standard short-chunk CNN (7 conv blocks, 3.43M params for tagging,
                        2.89M for genre). target_params now RAISES rather than silently accepting - parameter matching to the
                        GNN's 432,690 is what produced the 0.1654, and the constraint should not be reachable by accident.
                        ChunkedMelDataset serves random 3 s excerpts in training and 9 evenly-spaced ones at inference, with
                        per-chunk PROBABILITIES averaged (not logits: a logit average lets one confident chunk dominate).
[2026-09-06 01:54] DONE   A7.1 FMA-small 8-way genre wired end to end as the Task 2 headline: masked_genre_loss (-1 rows
                        excluded, never trained as class 0), M.multiclass_metrics (accuracy IS legitimate here - single label,
                        8 near-balanced classes, chance 12.5%), GNNClassifier accepts n_tags=0 so the parameter count
                        describes the model actually trained. One-epoch smoke: 38.8% test accuracy, 30 s/epoch.
[2026-09-06 01:55] DONE   A7.4 _fit now dumps val/test score matrices to results/scores/*.npz. Bootstrapping the threshold
                        tuner needs the raw scores of the SELECTED model, and re-running training 100 times to get them would
                        be absurd. The val pass is repeated after early stopping rolls the weights back, not cached from the
                        loop, or the matrices would belong to the wrong epoch.
[2026-09-06 01:56] DONE   A7.4 scripts/threshold_bootstrap.py + M.bootstrap_thresholds. Decision rule fixed IN ADVANCE:
                        spread > 0.02 in test macro-F1 means tuned numbers may not stand alone and every table carries the
                        fixed-0.5 number beside them. The verdict is written into the JSON so it cannot be reinterpreted later.
[2026-09-06 02:01] FIX    A7.x --run-tag added to src.train. Two configurations of the same task were writing to the same
                        result file; the genre smoke run overwrote the MTAT Task 2 result and it had to be restored from git.
[2026-09-06 02:01] DONE   A7.x Task 1 sweep re-launched LOCALLY against the corrected vocabulary. This GPU is sm_75, so the
                        runs that had to go to Kaggle for wall-clock reasons can be reproduced here; ~2 min/epoch against
                        Kaggle's 12 s while the mel cache build competes for CPU. First log line confirms the freeze-mode fix
                        is live: "freeze_mode=frozen_probe -> 0 trainable" where the Kaggle run said "full_ft -> 109482240".
[2026-09-06 02:10] DONE   A7.x report/final_report.tex started in IEEEtran two-column form, with report/fill_report.py
                        injecting every number from results/*.json into an AUTOGEN macro block. The prose contains no
                        literal figures, so a stale number cannot survive a re-run. Missing results render as \textit{pending}
                        and the script names them, rather than leaving a plausible-looking placeholder.

[2026-09-06 04:29] NOTE   PHASE B begins. A7's code is pushed; its runs (Task 1 sweep 4-5 of 5, then B2) were still executing,
                        so B0 started with the work that does not contend for the GPU.
[2026-09-06 04:28] DONE   B0.1 graph.rewire wired into MusicGraphDataset. Seeded per track with crc32, NOT hash - Python
                        randomises string hashing per process, so a hash-seeded control would silently differ between runs
                        and measure nothing. Edge-type ablation needed no new code: graph.temporal_edges/similarity_edges
                        are already honoured at build time and graphs are built on the fly, so --override suffices.
[2026-09-06 04:32] FIX    B0.1 rewiring on the fly cost 43.7 s/epoch against a 9.5 s baseline - millions of Python-level
                        double-edge swaps per epoch recomputing an identical answer. Cached edge_index/edge_attr per track:
                        epoch 1 now 39.5 s, epochs 2-3 back to 8.7 s. Over 6 runs x 30 epochs that is ~1.7 h saved.
[2026-09-06 04:20] DONE   B0.2 verified rather than asserted: the old MTAT Task 2 run's best epoch was 8 of a 10-epoch CAP,
                        i.e. still improving when the budget ran out; the new run reached epoch 11 before early stopping.
                        Same architecture, same parameter count. So 0.3692 -> 0.3737 is the epoch budget, NOT the vocabulary
                        fix - A7.3's finding that MTAT's top-50 set is unchanged still stands. Stronger reading: +0.0045 is
                        six times below the 0.0288 noise floor and micro-F1 moved the OTHER way (-0.0022). Indistinguishable.
[2026-09-06 04:25] DONE   B0.5 mood sets in config.yaml. Of 15 proposed affect words, 7 are absent from MTAT's top-50
                        (sad, happy, mellow, calm, dark, upbeat, eerie). What survives - soft/hard/ambient/quiet/loud/slow/
                        fast/weird - is texture and dynamics, not affect, so DEAM quadrants carry the mood story and the
                        MTAT panel is labelled for what it actually is. No-mood clips are grey and EXCLUDED from the k-NN
                        probe rather than pooled into an "other" class that would inflate it.
[2026-09-06 04:27] DONE   B0.6 check_tex.py fails non-zero above 10 pages and counts anything after \appendix separately.
                        Three tests: fires at 12 pages, passes at 7, appendix material not charged to the limit.

[2026-09-06 04:36] DONE   B1.2 DEAM imbalance handled explicitly. multitask.batch_ratio [4,1] wired through
                        _AlternatingTrainLoader - 1:1 would show the emotion head every one of ~1,200 DEAM tracks fourteen
                        times per MTAT epoch while the tag head sees each example once.
[2026-09-06 04:35] DONE   B1.2 valence/arousal standardised with TRAIN-split statistics (valence mean 4.901 sd 1.206,
                        arousal mean 4.861 sd 1.273 over 1,277 tracks), cached to data/splits/emotion_stats.json.
                        Raw 1-9 targets give squared errors of 4-10 against per-tag BCE near 0.2; auto_balance reacts to
                        that after the fact, standardising removes it at the source. CRITICAL companion change: predictions
                        are INVERTED before MAE/RMSE so the report stays on the 1-9 scale. R2 is affine-invariant, MAE is
                        not - a test asserts the inversion happens and that skipping it looks wrong.
[2026-09-06 04:38] DONE   B1.2 per-term losses now print inline in the epoch line (tag / val / aro). Task 3 optimises two
                        jobs against very different data volumes; the total can keep falling while one head quietly stops
                        learning, and that is invisible without the breakdown.
[2026-09-06 04:38] DONE   B1.2 Task 3 early-stops on the TAGGING metric with emotion auxiliary, per the PDF's L_aux framing.
                        A blended criterion would let a collapsing tag head hide behind a good regression fit.
[2026-09-06 04:39] DONE   B1.3 scripts/fusion_ablation.py. Two documented budgets (distilbert + reduced epochs held
                        IDENTICAL across all seven modes for the sweep; bert-base full budget for the headline rows), and
                        the budget is stamped into every result file so the two tables cannot be silently merged.
                        summarise() applies the noise-floor rule mechanically: delta-vs-best column, every row inside
                        0.0288 flagged, and it REFUSES to name a best_mode when the top rows overlap. Three tests cover it,
                        including one asserting no winner is named among indistinguishable rows.
[2026-09-06 04:40] NOTE   Task 3 validated end to end on CPU with --dry-run: emotion stats computed, 4:1 alternation active,
                        epoch completes. 197 fast tests pass.

[2026-09-06 04:48] DONE   B2.2 random_retrieval_reference folded into every retrieval_metrics payload, so no table can
                        report R@K without its chance row. Gallery 2,503 -> chance R@10 = 0.4%; an R@10 of 0.05 reads as
                        failure alone and as 12x chance beside the reference. The multiple is reported too.
[2026-09-06 04:50] DONE   B2.3 scripts/zero_shot_eval.py: 4 prompt templates, per-template macro-F1, spread, and a template
                        ensemble, all with thresholds tuned on val. The MusicCaps corpus/vocabulary are PINNED - an
                        --override that would change them is refused, because the supervised reference is the Task 3
                        MusicCaps run and comparing against an MTAT-vocabulary model would be two different problems.
                        Runs whose recorded tag_vocab does not match are skipped with a warning.
[2026-09-06 04:52] DONE   B3 t-SNE mood colouring driven by the B0.5 config instead of literals. The DEAM midpoint was
                        hardcoded at 5 - correct for a 1-9 scale and silently wrong for any other range. Added a third
                        panel for MTAT mood tags, reported separately because it is a different construct (texture, not
                        affect). Quadrant names reordered in config to match the index the code assigns; keeping them in
                        circumplex order would have mislabelled the legend, which a plot never reveals.
[2026-09-06 04:55] DONE   B4 listening study: scripts/make_listening_page.py writes sheet_key.json, a self-contained HTML
                        page with base64-embedded 10 s MP3s (numbered, no rating widgets - ratings go to a form), and the
                        form question list in page order. Controls use REAL audio from an unrelated clip; a silent control
                        would be identifiable without listening.
[2026-09-06 04:56] DONE   B4 scripts/analyse_human_eval.py adapts the wide Google Forms export to long format keyed on the
                        CLIP NUMBER in the question text, not column order. An off-by-one here would swap real pairs with
                        controls and invert the headline finding, so it REFUSES a mismatched export rather than guessing.
                        Five tests, including shuffled columns, out-of-scale answers, and an uninformative-study case.
[2026-09-06 04:56] NOTE   Corrected-vocabulary leakage gap: raw 0.5670 vs masked 0.3707 = +0.1963 (53% relative), against
                        +0.2214 (62%) under the leaked vocabulary. The leak was inflating the gap itself - masked went up,
                        raw went down. 202 fast tests pass.

[2026-09-06 04:57] DONE   B6 demo notebook executed on CPU: 16.7 s, zero errors, against a 2-minute budget - measured while
                        a training job competed for the same cores, so the clean figure is lower.
[2026-09-06 04:58] FIX    B6 fresh-clone test found THREE real defects, none visible from the working tree. data/raw/ and
                        results/retrieval_examples/ are absent from a checkout (gitignored / run-created), so the
                        prescribed-tree test failed on a clone - for a grader that is indistinguishable from a broken repo.
                        And the two MusicCaps vocab tests FAILED rather than skipped without the gitignored raw CSV, which
                        reports "the code is broken" when the truth is "the corpus is absent". Fixed with committed
                        placeholders (data/raw/README.md documents the expected layout) and explicit skip conditions.
                        Clone now runs 198 passed, 5 skipped. All 20 sample graphs verified real and contract-compliant.
[2026-09-06 04:53] DONE   B7 four sections written: the named noise-floor subsection (and what refusing to rank costs),
                        a section collecting the four protocol findings as one failure mode rather than scattered
                        footnotes, the full limitations list, and a reproducibility statement with exact commands.
                        Report 7.4 pages of a 6-10 limit.

[2026-09-06 05:47] NOTE   PHASE C begins. Resumed with the C1 block still executing: Task 1 MTAT metadata on epoch 3/8 at
                        1147 s/epoch. Measured cost of bert-base on this GPU is ~68 ms/row, which is what makes the C4
                        sweep ~26 h at full budget - decision rule 3.6 will fire and the budget will be set from the
                        Task 3 timing probe rather than guessed.
[2026-09-06 05:48] DONE   C queue chained in strict priority order behind C1: C2 Task 3 headline (bert-base, MTAT,
                        cross_attention, seed 42) then C3 Task 4 headline + retrieval export + listening study. C3 is not
                        allowed to wait behind the ablation sweep because it opens the human-eval gate.
[2026-09-06 05:52] DONE   Section 5 compression decided and wired BEFORE the content lands, not after check_tex fires.
                        Ten retrieval examples -> one multi-panel figure; three case studies -> one three-column figure;
                        per-tag threshold detail -> appendix (\appendix scaffolding added, counted separately).
                        scripts/make_compact_figures.py renders the retrieval panel as stems on a log rank axis: it shows
                        which queries failed AND by how far, which ten prose blocks do not, at a tenth of the page cost.
                        Rendered on synthetic data first and fixed three defects only visible once drawn - rank-1 bars are
                        invisible on a log axis, the legend collided with a value label, and "failure" mislabelled rank 12
                        of 2,503 (now "outside top-10").
[2026-09-06 05:52] DONE   Related work now names MuLan alongside CLAP, with the scale gap made explicit: MuLan trains on
                        ~44M audio-text pairs against our 2,095, four orders of magnitude, which is the right frame for
                        reading the retrieval numbers.

[2026-09-06 05:56] FIX    Checkpoints were written as task{N}_seed{S}_{best,last}.pt with NO run tag, while results have
                        carried one since A7. Consequences, both silent: the C4 ablation's seven fusion modes would each
                        overwrite the previous mode's checkpoint, and the C6 case studies - which need the MusicCaps
                        Task 3 model specifically - would load whichever run finished last and produce plausible output
                        from the wrong weights. Found by asking what 4.4 would load, before running C2, not after C4.
                        Checkpoints now carry the tag; utils.find_checkpoint resolves exact tag -> untagged (older
                        artifacts) -> newest tagged, and returns None rather than a wrong-task file. Four tests, one of
                        which greps src/train.py to assert the write paths still interpolate the suffix.

[2026-09-06 06:45] DONE   C4 PREPARED FOR KAGGLE, moving it off the local GPU entirely. bert-base measures 68 ms/row here,
                        so 21 runs is ~26 h locally and decision rule 3.6 would have cut rows from the ablation table.
                        Two T4s run two shards concurrently instead, and C5-C7 proceed locally in parallel.
                        Payload kaggle_payload_ablation.tar.gz = 105.0 MB (237.6 MB raw, 71 files).
[2026-09-06 06:41] NOTE   The payload carries features_*.h5, NOT the exported .pt graphs. Task 3 builds graphs on the fly
                        (DataBundle sets graph_dir=None for real data), so a graph payload would be 450 MB AND unusable.
                        Checked before building rather than after uploading.
[2026-09-06 06:40] DONE   Sharding verified BEFORE upload: shard i takes every Nth run from a fixed-order list; for 1-4
                        shards every run appears in exactly one shard with no overlap. This matters because the two shards
                        are separate processes with no runtime coordination. Five tests cover it.
[2026-09-06 06:43] DONE   Import guard extended: the ordered tag vocabulary is now hashed into every result, and the
                        importer refuses any file whose hash matches no current vocabulary. A7.3 changed 7 of 50 MusicCaps
                        tags, so a pre-fix result is numerically fine and semantically incompatible - and nothing about the
                        file would reveal it. Hash is order-sensitive because order fixes which column is which tag.
[2026-09-06 06:45] FIX    Replaced the bash chains with scripts/run_queue.py, one process owning the whole queue. The
                        chains waited on log markers and twice produced two waiters on one job, hence two concurrent
                        training runs, once corrupting a result. Watchdog not halt; resume; commit+push after EVERY step;
                        wall-clock recorded per step and the remaining estimate rescaled from it.
[2026-09-06 06:44] FIX    The resume check was skipping on file EXISTENCE, which would have silently dropped two Task 2
                        deliverables: results/baselines_seed42.json exists from before the A7.2 rework holding the old
                        parameter-matched B2_mel_cnn, and structural_controls.json exists holding a --dry-run. Steps now
                        declare content that must be present for the artifact to count as fresh.
[2026-09-06 06:45] NOTE   Local queue re-planned as C1 -> C2 -> C3 -> C5 -> C6 -> C7 = 11.1 h estimated. One handover
                        waiter is live, waiting for the in-flight Task 1 sweep before starting the runner.
[2026-09-06 06:57] QUEUE  started with 21 step(s)
[2026-09-06 07:44] DONE   C1 C1a baselines B1/B2/B4 in 46.7 min
[2026-09-06 08:05] DONE   C1 C1b structural controls in 20.8 min
[2026-09-06 09:14] DONE   C2 C2 Task 3 headline (MTAT, bert-base) in 68.9 min
[2026-09-06 09:18] DONE   C3 C3 Task 4 headline (MusicCaps dual encoder) in 4.5 min
[2026-09-06 09:57] DONE   C3 C3 retrieval export in 38.7 min
[2026-09-06 09:57] DONE   C3 C3 listening study in 0.0 min
[2026-09-06 10:16] DONE   C5 C5 Task 3 MusicCaps bert_only in 19.2 min
[2026-09-06 10:18] DONE   C5 C5 Task 3 MusicCaps gnn_only in 2.2 min
[2026-09-06 10:38] DONE   C5 C5 Task 3 MusicCaps cross_attention in 19.7 min
[2026-09-06 10:39] DONE   C6 C6 zero-shot vs supervised in 0.3 min
[2026-09-06 11:17] BLOCKED C6 C6 full evaluation (t-SNE, S_graph, case studies) after 38.7 min: produced no artifact
         last lines of queue_C6_444137.log:
             "confusion_topk": "B:\\CSE425_Project\\gnn-bert-music-context\\results\\plots\\confusion_topk.png",
             "ablation": "B:\\CSE425_Project\\gnn-bert-music-context\\results\\plots\\ablation.png",
             "tsne_genre": "B:\\CSE425_Project\\gnn-bert-music-context\\results\\plots\\tsne_genre.png",
             "tsne_mood": "B:\\CSE425_Project\\gnn-bert-music-context\\results\\plots\\tsne_mood.png",
             "tsne_mood_mtat": "B:\\CSE425_Project\\gnn-bert-music-context\\results\\plots\\tsne_mood_mtat.png",
             "retrieval": "B:\\CSE425_Project\\gnn-bert-music-context\\results\\plots\\retrieval.png",
             "graph_coherence": "B:\\CSE425_Project\\gnn-bert-music-context\\results\\plots\\graph_coherence.png",
             "bert_attention": "B:\\CSE425_Project\\gnn-bert-music-context\\results\\plots\\bert_attention_00.png",
             "case_study": "B:\\CSE425_Project\\gnn-bert-music-context\\results\\plots\\case_study_0_graph.png",
             "seed_summary": "B:\\CSE425_Project\\gnn-bert-music-context\\results\\plots\\seed_summary.png"
           }
         }

[2026-09-06 11:22] BLOCKED C6 C6 threshold bootstrap after 4.3 min: produced no artifact
         last lines of queue_C6_753870.log:
         task2_seed42_mtat_tags                  0.3737    0.2910     0.3710   0.0288   0.071  UNSTABLE
         task2_seed42_tags_baseline              0.3591    0.2775     0.3604   0.0305   0.065  UNSTABLE
         task2_seed42_tags_rewired               0.3751    0.2823     0.3673   0.0332   0.072  UNSTABLE
         task2_seed42_tags_similarity_only       0.3617    0.2764     0.3611   0.0322   0.065  UNSTABLE
         task2_seed42_tags_temporal_only         0.3635    0.2726     0.3626   0.0332   0.064  UNSTABLE
         task3_seed42_mtat_cross_attention_hea   0.2479    0.2097     0.2487   0.0253   0.063  UNSTABLE
         task3_seed42_musiccaps_bert_only        0.3073    0.2287     0.2986   0.0551   0.083  UNSTABLE
         task3_seed42_musiccaps_cross_attentio   0.3188    0.2401     0.3033   0.0610   0.098  UNSTABLE
         task3_seed42_musiccaps_gnn_only         0.1120    0.0007     0.1036   0.0116   0.027  stable
         
         report fixed-0.5 numbers alongside tuned ones in all tables
         wrote B:\CSE425_Project\gnn-bert-music-context\results\threshold_bootstrap.json

[2026-09-06 11:22] BLOCKED C6 C6 compact figures after 0.0 min: produced no artifact
         last lines of queue_C6_395681.log:
         $ B:\CSE425_Project\gnn-bert-music-context\.venv\Scripts\python.exe -u scripts/make_compact_figures.py
         
         11:22:24 | INFO    | gbmc.figures | wrote B:\CSE425_Project\gnn-bert-music-context\results\plots\retrieval_examples.png (10 examples, 2 failures, gallery 2503)
         11:22:24 | INFO    | gbmc.figures | wrote B:\CSE425_Project\gnn-bert-music-context\results\plots\case_studies.png (3 panels)
         retrieval: {'path': 'B:\\CSE425_Project\\gnn-bert-music-context\\results\\plots\\retrieval_examples.png', 'n_examples': 10, 'n_failures': 2, 'gallery_size': 2503, 'median_rank': 3.5}
         cases: {'path': 'B:\\CSE425_Project\\gnn-bert-music-context\\results\\plots\\case_studies.png', 'n_panels': 3}

[2026-09-06 11:22] BLOCKED C6 C6 genre confusion figure after 0.0 min: produced no artifact
         last lines of queue_C6_345064.log:
         $ B:\CSE425_Project\gnn-bert-music-context\.venv\Scripts\python.exe -u scripts/plot_genre_confusion.py
         
         11:22:28 | INFO    | gbmc.genreplot | wrote B:\CSE425_Project\gnn-bert-music-context\results\plots\genre_confusion.png (2 panel(s))
         B:\CSE425_Project\gnn-bert-music-context\results\plots\genre_confusion.png

[2026-09-06 11:23] DONE   C7 C7 Task 2 genre seed 1337 in 1.0 min
[2026-09-06 11:24] DONE   C7 C7 Task 2 genre seed 2024 in 1.1 min
[2026-09-06 11:28] DONE   C7 C7 Task 2 tags seed 1337 in 3.7 min
[2026-09-06 11:33] DONE   C7 C7 Task 2 tags seed 2024 in 4.9 min
[2026-09-06 11:37] DONE   C7 C7 Task 4 seed 1337 in 3.8 min
[2026-09-06 11:40] BLOCKED C7 C7 Task 4 seed 2024 after 3.6 min: exit code 1
         last lines of queue_C7_209196.log:
                                    np.asarray(thresholds).tolist(),
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    "provenance": provenance})
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^
           File "B:\CSE425_Project\gnn-bert-music-context\src\train.py", line 354, in save_checkpoint
             return atomic_torch_save(payload, path)
           File "B:\CSE425_Project\gnn-bert-music-context\src\utils.py", line 308, in atomic_torch_save
             return atomic_write_bytes(path, buffer.getvalue())
           File "B:\CSE425_Project\gnn-bert-music-context\src\utils.py", line 293, in atomic_write_bytes
             fh.write(payload)
             ~~~~~~~~^^^^^^^^^
         OSError: [Errno 28] No space left on device

[2026-09-06 11:40] DONE   C7 report fill + structural check in 0.0 min
[2026-09-06 11:41] QUEUE  retrying 5 failed step(s) once
[2026-09-06 12:19] BLOCKED C6 full evaluation (t-SNE, S_graph, case studies) failed on retry too: 
[2026-09-06 12:24] BLOCKED C6 threshold bootstrap failed on retry too: 
[2026-09-06 12:24] BLOCKED C6 compact figures failed on retry too: 
[2026-09-06 12:24] BLOCKED C6 genre confusion figure failed on retry too: 
[2026-09-06 12:28] BLOCKED C7 Task 4 seed 2024 failed on retry too: exit code 1
[2026-09-06 12:28] QUEUE  finished: 16 done, 5 still failing (C6 full evaluation (t-SNE, S_graph, case studies), C6 threshold bootstrap, C6 compact figures, C6 genre confusion figure, C7 Task 4 seed 2024)

[2026-09-06 12:28] NOTE   Unattended queue finished: 16 of 21 steps done. Gate 1 reached 09:57. Measured wall-clock vs my
                        priors: B2 baselines 46.7 min (prior 1.5-2.5 h - the band was too pessimistic), C2 Task 3 headline
                        68.9 min (prior ~200 min - also pessimistic), C3 retrieval export 38.7 min (prior 10).
[2026-09-06 14:25] FIX   Four of the five "failures" were MY BUG, not failed runs. run_queue used done() as both the skip
                        test and the success test; done() returns False by design for always=True steps, so four
                        successful C6 steps were marked failed and 45 min was spent retrying them. All four artifacts
                        exist and are fresh. Success is now "process exited clean AND artifacts present".
[2026-09-06 14:25] FIX   The fifth failure was real: OSError errno 28, disk full at 100%. Cause was my own checkpoint-
                        tagging fix - runs stopped overwriting task{N}_seed{S}_last.pt and each kept its own ~440 MB copy,
                        44 files / 6.1 GB. Nothing ever reads _last back; it is a mid-training resume point. Deleted them
                        (freed 3.0 GB) and made _last a single rolling file again while _best stays tagged, because _best
                        IS loaded by name and the ablation would otherwise overwrite itself. Test updated to assert both
                        halves of that, since they pull in opposite directions.
[2026-09-06 14:27] FIX   TASK 4 WAS DEGENERATE, NOT WEAK. 2,095 training pairs at contrastive batch 512 gives 4 steps per
                        epoch; early stopping ended it at epoch 9, so the model got 36 optimiser steps and 15 SECONDS of
                        training. Its InfoNCE loss never left 6.4 = ln(512), which is exactly chance, and test R@10 came
                        out at 1.0x the random reference. Decision rule 3.5 covers a weak retrieval model; it does not
                        cover an untrained one, and reporting 36 steps as "small-scale contrastive learning is hard" would
                        have been wrong. Re-ran with epochs=150 and patience=15 (val R@10 on 232 clips is very noisy):
                        loss 6.08 -> 2.23, val mean_R@10 0.0625 -> 0.1703, early stop at 57 with best at 42. Costs 8 min.

[2026-09-06 15:31] DONE   Task 4 re-run complete, all three seeds. Test R@10 = 0.0135 +- 0.0006 against a chance reference
                        of 0.0040 on the 2,503-clip gallery = 3.4x chance, median rank ~675. Consistent across seeds.
                        That is decision rule 3.5: low but well above chance, with 2,095 training pairs as the limiting
                        factor. The previous run's 1.0x chance was an untrained model, not this finding.
[2026-09-06 15:43] FIX   ZERO-SHOT WAS SCORED ON A PARTLY RANDOM ENCODER. zero_shot_eval built its text tower from
                        config.yaml, which defaults to distilbert (6 layers), while the Task 4 checkpoint is bert-base
                        (12). load_state_dict(strict=False) left 96 tensors randomly initialised and returned a perfectly
                        plausible number - the log said "96 parameter(s) missing" and nothing else would have shown it.
                        Now reads the architecture from the checkpoint's own stored config, and RAISES rather than
                        warning if anything fails to load. Corrected: ensemble macro-F1 0.0914, template spread 0.0051
                        (below the 0.0288 floor, so phrasing is not the dominant effect), against the Task 3 MusicCaps
                        supervised reference of 0.3188 -- a gap of 0.2274 on the same corpus, vocabulary and split.
[2026-09-06 15:45] NOTE   Report now 9.0 pages + 0.1 appendix of a 6-10 limit. One macro pending (CaseStudyNote), which
                        needs the three MusicCaps case studies from C6.

[2026-09-06 16:44] FIX   THE SAME ARCHITECTURE MISMATCH IN THREE MORE PLACES. config.yaml defaults bert.model_name to
                        distilbert (6 layers) while the Task 3 and Task 4 headline runs use bert-base (12). Building from
                        config and then loading the checkpoint restored 48 of 144 tensors in evaluate.py - the rest stayed
                        random - and attention_viz would have drawn the Task 1 attention maps and the Task 3 case studies
                        the same way. Every symptom is a log line nobody reads: "96 parameter(s) missing", "restored
                        48/144 tensors". Nothing raises, and the figures look fine.
                        utils.encoder_name_from_checkpoint now reads the architecture out of the checkpoint's own stored
                        config, and evaluate/attention_viz build the encoder from that rather than from config.yaml.
                        evaluate also warns loudly when under half the tensors restore. Fixed once, in a shared helper,
                        because this is the fourth site and site-by-site was clearly not working.
[2026-09-06 16:05] FIX   The listening study was unratable and undersized: raters were shown caption_masked (descriptive
                        terms stripped) and only 14 of the intended 24 clips. Now 24 clips, 4 controls, 24/24 unmasked,
                        with random draws added beyond the curated best-and-worst so the sample is not biased upward.

[2026-09-06 18:15] DONE   C4 IMPORTED FROM KAGGLE. All 21 runs present (7 modes x 3 seeds), 0 rejected. The vocabulary
                        guard passed by recomputing the hash from each result's tag_vocab, since the payload predated the
                        tag_vocab_hash stamp - incoming 200f934da671 matched the local MTAT vocabulary exactly.
[2026-09-06 18:15] NOTE   Kaggle throughput 5.72 ms/row against 68 ms/row locally = 11.9x. The sweep took ~1.6 h of wall
                        clock instead of the ~26 h it would have cost here, so no ablation rows had to be cut and the
                        local GPU ran C5-C7 in parallel. Moving C4 was the right call and the numbers say so.
[2026-09-06 18:15] DONE   ABLATION RESULT: bert_only 0.1529 is the only mode outside the noise floor - roughly 0.12 below
                        everything else. The other SIX modes, gnn_only included, lie within 0.0288 of one another, so no
                        ordering is claimed. Critically gnn_only (0.2720) sits inside the band with every fusion variant:
                        on MTAT, fusion buys nothing over the graph alone. Decision rule 3.3, as predicted.
[2026-09-06 18:15] DONE   THE CONTRAST IS THE FINDING. Same architecture, different corpus:
                          MTAT (metadata) : bert_only 0.1529  gnn_only 0.2720  cross_attention 0.2562
                          MusicCaps (cap.): bert_only 0.3073  gnn_only 0.1117  cross_attention 0.3188
                        Which modality dominates FLIPS with the corpus, by a margin an order of magnitude larger than any
                        of the within-corpus gaps. Fusion is not intrinsically worthwhile; it is worthwhile in proportion
                        to what the second modality knows. A single-corpus study would have concluded the opposite.
[2026-09-06 18:15] DONE   Page guard fired at 10.1 pages, exactly as designed. Applied compression step 4: Reproducibility
                        moved behind \appendix (its standard venue placement anyway). Now 9.8 + 0.4 appendix.
[2026-09-06 18:15] FIX   The retrieval figure grew to 30 rows once the exporter started adding random draws for the
                        listening study - fourteen inches tall. The figure keeps the ten curated extremes it was designed
                        for; the study keeps the random sample it needs to be unbiased.
[2026-09-06 18:15] DONE   Cleaned 11.9 GB: the Kaggle output was 11 GB of ablation checkpoints nothing reads, plus an
                        echo of the payload. Kept the 21 results, 21 score matrices, 2 shard summaries and both logs.
                        Also removed the superseded 863 MB Task 1 output folder. Free space 26 GB -> 38 GB.

[2026-09-06 20:33] FIX   THE CASE STUDIES WERE ON MTAT, which Phase C 4.3 explicitly forbids. I had fixed half of it -
                        the checkpoint was correctly the MusicCaps model - but generate_case_studies still drew its ROWS
                        from TAG_DATASETS = mtat, so the attention maps were over "8 seconds. 8 Seconds. Pain Factor" and
                        "Fantasia (del segundo tono). Alonso Mudarra songs and solos": a title, an album and an artist.
                        That is precisely the uninformative map the instruction exists to prevent. Now reads the caption
                        corpus, and evaluate builds a MusicCaps-configured bundle so the vocabulary matches the head.
[2026-09-06 20:35] DONE   Case studies regenerated on MusicCaps, 2 successes + 1 failure as required. Both successes are
                        captions describing RECORDING CONDITIONS rather than musical content - "amateur recording", and
                        "low quality/noisy/mono" at 1.00/1.00/0.96 - consistent with segment features carrying spectral
                        character more readily than structure.
                        The failure is the interesting one: for a clip described as a triangle wave with randomly placed
                        samples, the model's TOP prediction is the correct tag (instrumental) at 0.11, against that tag's
                        validation-tuned threshold of 0.46. The representation ranked it first; the operating point
                        rejected it. A calibration failure, not a perceptual one - and a direct instance of the threshold
                        instability measured in A7.4.
[2026-09-06 20:35] GATE  REPORT COMPLETE except human eval: zero pending macros, check_tex clean, 9.9 pages + 0.4
                        appendix against a 6-10 limit. 211 fast tests pass.

[2026-09-08 00:15] DONE  B4 responses arrived: 10 raters, 24 items, 240 ratings.
[2026-09-08 00:17] FIND  THE STUDY RETURNS A NULL. Controls scored 3.65 against genuine pairs' 3.62 - a gap of -0.03 in
                        the WRONG direction, p = 0.582. The listening study validates nothing about retrieval quality.
                        Reported as inconclusive, which is what the protocol fixed in advance requires.
[2026-09-08 00:18] FIND  But "raters weren't listening" and "the controls weren't controls" are different claims with
                        different costs, and the headline gap cannot tell them apart. Added two diagnostics to
                        analyse_human_eval.py. Kruskal-Wallis across items: H = 99.8, p = 1.5e-11, 41.8% of variance
                        BETWEEN items, real-item means spanning 1.70-4.90. The panel was attending. What failed was the
                        control construction: 3 of 4 controls reuse a caption that a real item in the same sheet also
                        carries, so a rater meeting one description twice over different audio has no basis for calling
                        either pairing wrong. One such control scored 4.70 against its genuine counterpart's 4.10.
                        The fourth control's unique caption described RECORDING CONDITIONS and scored 4.20 - the same
                        property that makes the case-study successes succeed. Two judges, one learned and one human,
                        limited by the same corpus property.
[2026-09-08 00:30] REPORT Human Evaluation section written from real numbers, control discrimination first. 21 macros.
[2026-09-08 00:35] FIND  TASK 4 HAD NO RESULTS SUBSECTION. Three of four tasks had result tables; the contrastive
                        retrieval numbers sat in results/ and appeared nowhere in the report except a qualitative
                        figure. Added sec:retrieval + 18 macros: R@10 0.0135 +- 0.0006 vs analytic chance 0.0040
                        (3.4x), median rank 686 of 2,503 vs chance 1,252, both directions agreeing.
[2026-09-08 00:50] FIX   THE PAGE ESTIMATOR WAS WRONG BY A FULL PAGE, and I nearly cut real prose to satisfy it.
                        check_tex split at \appendix and counted everything above as body - including the AUTOGEN
                        macro block, which typesets nothing where it sits, and \BootWorstTagsTable, a table that only
                        renders INSIDE the appendix. Meanwhile a use site counted as one word whether it expanded to a
                        digit or a 130-word note. Macros now expand at their use sites, as LaTeX does. Two tests guard
                        it; both fail against the previous version. 10.4 -> 9.6 pages with no content removed.
[2026-09-08 00:55] REPORT Consolidated real duplication found while compressing: the four integrity defects were each
                        stated twice (Section 3 with effect sizes, then "What the Assertions Caught") and three of them
                        a third time in Limitations. Kept Section 3, which absorbed the one paragraph unique to the
                        removed section. ~900 words recovered, no fact lost.
[2026-09-08 01:00] FIX   .gitignore line 134 had "data/human_eval/*.wav" and "*.npz" concatenated with a comment
                        fragment, so neither pattern worked. Nothing had leaked into the index. Repaired.
[2026-09-08 01:05] GATE  REPORT COMPLETE. Zero pending macros, check_tex clean, 9.7 pages + 0.5 appendix against 6-10.
                        215 fast tests pass, 1 skipped. Every deliverable section now carries real numbers.

[2026-09-08 01:30] FIND  THE HEADLINE TASK 2 TABLE WAS WRONG IN THREE WAYS AT ONCE, and the bolding was the tell:
                        it bolded T2 GNN as the winner while B2 mel CNN scored HIGHER on both metrics (46.3%/0.4435
                        against 42.3%/0.4277). It also reported seed 42 alone while three GNN seeds exist, and the
                        Reproducibility section claimed "42, 1337 and 2024 for every reported ablation row and headline
                        result" - which was therefore false as written. And B0.3 had fixed in advance that if B2 won it
                        becomes the headline and the GNN is not re-tuned; that commitment had never reached the prose.
                        Fixed all three: the GNN row now carries the 3-seed mean 43.7% +- 1.8 / 0.4315 +- 0.0128, B2 is
                        labelled single-seed, neither row is bolded, and the text says plainly that the CNN outscores
                        the GNN. Reporting the spread also changes the reading - +-1.8 points covers most of the gap, so
                        no ordering is claimed either way, and B2 needs ~10x the parameters and ~10x the wall-clock.
[2026-09-08 01:35] DONE  Five PROGRESS.md items were still open against work that had landed days earlier (B0.3, B0.4,
                        B1.1, B2.1, the MusicCaps case studies). Verified each against its artifacts, then closed them
                        with outcomes rather than ticks. Zero open checkboxes remain.
[2026-09-08 01:36] NOTE  results/metrics.json still holds the pre-fix MTAT case studies. Nothing in the LaTeX pipeline
                        reads it - fill_report takes case studies from results/case_studies.json, which is the corrected
                        MusicCaps run - so it is inert, but it is stale and would mislead anyone who opened it.

[2026-09-08 01:50] FIX   THE RETRIEVAL FIGURE'S CAPTION DESCRIBED A DIFFERENT SET THAN THE FIGURE. The exporter's pool
                        grew to 30 when the listening study needed an unbiased random sample, but the figure still plots
                        only the 10 curated extremes - and the caption macros kept summarising all 30. A reader saw ten
                        stems captioned "median rank 451 over 30 queries" and "21 of the 30 queries place the true clip
                        outside the top ten". The word "Both" made it worse: it referred to the two worst-ranked queries
                        from when the note was written, and after the pool grew it dangled off a sentence about 21.
                        The notes now apply the figure's own selection rule, so caption and picture describe the same
                        rows, and both medians are given - 2 among the curated 10, 451 over the full 30 - because the
                        curated median is not a performance estimate and should not be readable as one.
[2026-09-08 01:55] REPORT Report at 9.94 pages + 0.5 appendix. Was 9.99 after the retrieval fix, which is not a margin;
                        recovered it by collapsing the B2-granularity caveat, which I had just duplicated into Results,
                        back to a pointer from Limitations. 119 macros, zero pending. 215 fast tests pass.

[2026-09-08 03:00] D0    QUARANTINED THE SYNTHETIC PDF. report/final_report.pdf was Creator: Matplotlib v3.11.1, written
                        2026-09-05, 24 pages against a 6-10 limit, built by build_report.py from metrics.json when its
                        synthetic flag was true. It sat at the exact deliverable path for three revisions - worse than
                        missing, because it looked satisfied. Moved with the markdown report and its generator to
                        results/_synthetic_smoke/, renamed, and a test keeps them there.
[2026-09-08 03:02] D0.2  THE FOUR C6 "produced no artifact" MARKERS WERE WRONG. Every artifact exists with real
                        provenance - three t-SNE panels, F1-vs-epoch, five attention examples, S_graph, the bootstrap,
                        20 real sample graphs, metrics.json itself. Stale bookkeeping, not missing work. D0.3: all
                        three Task 4 seeds exist and are real; the "exit code 1" entry did not match disk.
[2026-09-08 03:04] D1    TASK 4 WAS UNDERTRAINED, NOT DATA-LIMITED. Batch 512 over 2,095 pairs is 4 optimiser steps per
                        epoch. Re-run at batch 128, architecture untouched: R@10 0.0135 -> 0.0175 +- 0.0016, lift
                        3.4x -> 4.4x, median rank 686 -> 631, val peaking at epoch 35 of 75. Three seeds, 2.8 min each.
                        Both numbers reported: the difference between "too little data" and "too little training" is
                        exactly the claim that should be tested before it is made.
[2026-09-08 03:06] D2    DEAM emotion resolved at rung 1 - every Task 3 run had already written valence/arousal metrics
                        and no macro had ever read them. Valence MAE 0.714 / R2 0.311, arousal 0.753 / 0.478, n=275.
                        Valence swings 0.247 +- 0.134 across seeds against arousal's 0.522 +- 0.060, so the spread is
                        reported: a single valence number from this setup is not reproducible.
[2026-09-08 03:08] D3    TWO DEFECTS IN THE HEADLINE TABLE. The T3 row read "Phase B" - a placeholder, so the 22-mark
                        hard task had no row in the table comparing it to everything else. And B1/B4 were hardcoded
                        from baselines_seed42_PRENORM.json, the run predating the normalisation fix, in a report whose
                        reproducibility claim is that no literal figures appear in the prose. B4 printed 0.3239 against
                        a real 0.3146, understating the B4 -> GNN delta that carries the structural claim. Now macros:
                        delta 0.0591, about 2.1x the floor - separable, and modest, and we say both.
[2026-09-08 03:13] D4    Figures composited from existing real runs, not regenerated. Also built the architecture
                        diagram the report never had - every figure was a result, while the rubric asks for diagrams.
[2026-09-08 03:25] D5    11.9 -> 9.96 pages by the pre-registered compression order. Bootstrap table, attention figure,
                        diagnostics figure, confusion matrix, corpora table and the whole Qualitative Analysis section
                        moved behind \appendix; prose trimmed from Related Work, Setup, Intro, Method and Data only.
                        Results, Limitations, the floor, the protocol findings, human eval and reproducibility untouched.
[2026-09-08 03:30] D6    FROZEN. 173 macros, zero pending, check_tex clean, 216 tests pass, fresh clone 212 pass /
                        5 skip with no defects found. Submission ZIP 303 files / 10.2 MB.
                        ONE OPEN ACTION: report/final_report.pdf must be compiled on Overleaf by the operator.

[2026-09-08 04:10] FIND  RUBRIC AUDIT FOUND A THIRD WIRING GAP. Zero-shot tag prediction is an explicit Task 4
                        deliverable in the project PDF; results/zero_shot_seed42.json has existed since Phase C and no
                        macro had ever read it - the same failure as the DEAM metrics and the Task 4 results table.
                        Added: 0.0914 macro-F1 zero-shot against the supervised model's 0.3188 over 2,503 clips, with
                        a template spread of 0.0051 (inside the floor), so the number reflects the representation
                        rather than prompt wording. 33 of 34 deliverables now present; the one gap is the operator's
                        Overleaf compile. Report re-frozen at exactly 10.00 pages.
