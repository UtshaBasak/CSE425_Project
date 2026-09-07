# Submission checklist — CSE425 GNN-BERT Music Context

Generated on submission day (Phase D). Maps each of the assignment's five
required items (project PDF §10) to its path in this repository.

| # | Required item | Status | Path |
|---|---|---|---|
| 1 | GitHub repository / ZIP with full source code | **done** | `https://github.com/UtshaBasak/CSE425_Project` — `src/`, `scripts/`, `tests/`, `config.yaml`, `Makefile`, `requirements.txt` |
| 2 | Preprocessed graph samples (≥20 `.pt`/`.json`) | **done** — 20 files, all real provenance, 0 synthetic | `data/processed/sample_graphs/` |
| 3 | Evaluation tables + plots (F1, AUC-PR, t-SNE, retrieval) | **done** — 25 PNGs + 30 retrieval examples + consolidated metrics | `results/plots/`, `results/retrieval_examples/`, `results/metrics.json` |
| 4 | Final report PDF (6–10 pp, IEEE Overleaf template) | **SOURCE READY — operator must compile** | source: `report/final_report.tex` (9.96 pp + 3.5 pp appendix); figures: `report/figures/`; output goes to `report/final_report.pdf` |
| 5 | Demo notebook, one end-to-end inference example | **done** — 16.7 s on CPU against a 2-minute budget | `notebooks/demo_context.ipynb` |

## Item 4 is the only open action

`report/final_report.pdf` **does not exist right now, and that is deliberate.**
The file that previously sat there was a 24-page PDF written by Matplotlib on
2026-09-05 from *synthetic* metrics — not a build of the `.tex` at all. It was
quarantined in D0.1 to `results/_synthetic_smoke/final_report_SYNTHETIC.pdf`,
because a wrong PDF at the deliverable path is worse than a missing one. A test
keeps it there.

See `report/README.md` for the Overleaf steps.

## Report state at freeze

- **9.96 pages** main body + 3.5 pages appendix, against a 6–10 limit
- **173 macros, zero pending** — every number is read from `results/*.json`
- `report/check_tex.py`: no structural problems
- **216 fast tests pass**, 1 skipped

## Assignment directory tree

Every path in the project PDF's prescribed tree exists, with the single
exception of `report/final_report.pdf` above.

## Results the report carries

| Task | Headline | Reference |
|---|---|---|
| 1 (BERT tags) | MusicCaps masked 0.3707 macro-F1; **leakage gap +0.1963 (+53%)** | `sec:leakage` |
| 2 (GNN genre) | GNN 43.7% ± 1.8 acc; **B2 CNN beats it at 46.3%** | `tab:genre` |
| 2 (GNN tags) | 0.3737 macro-F1; **B4→GNN delta 0.0591 ≈ 2.1× the floor** | `tab:mtat` |
| 3 (fusion) | 6 of 7 modes inside the measurement floor; **no ordering claimed** | `tab:ablation` |
| 3 (emotion) | valence R² 0.311, arousal R² 0.478, 275 DEAM test tracks | `tab:emotion` |
| 4 (retrieval) | R@10 0.0175 ± 0.0016, **4.4× chance** (was 3.4× before the D1 re-run) | `tab:retrieval` |
| Human eval | **null**: controls −0.03 below real pairs (p = 0.582), cause diagnosed | `sec:human` |
