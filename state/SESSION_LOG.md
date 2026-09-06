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
