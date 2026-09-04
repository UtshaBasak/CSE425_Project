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
