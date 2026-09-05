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
