# Report

`final_report.tex` is the living report and the only source of truth. Every
number in it is written by `report/fill_report.py` from `results/*.json`; the
prose contains no literal figures, so a stale number cannot survive a re-run.

## Producing `final_report.pdf`

This machine has no LaTeX toolchain, so the PDF has to be built on Overleaf.

1. New Project -> Upload Project, or start from the **IEEE Conference**
   template (https://www.overleaf.com/latex/templates/ieee-conference-template),
   which supplies `IEEEtran.cls`.
2. Upload `final_report.tex` and the `figures/` folder beside it. The three
   PNGs there are copies of `results/plots/{genre_confusion,
   retrieval_examples,case_studies}.png`, staged so the upload is one folder.
3. Compile with pdfLaTeX. No `.bib` is needed -- the bibliography is inline
   `\bibitem`.
4. Download the PDF and replace `final_report.pdf` here.

## Checks before submitting

    python report/fill_report.py --check   # fails if any macro is still pending
    python report/check_tex.py             # structure + page budget (6-10 pages)

## A warning about the committed PDF

`final_report.pdf` in this directory is **stale**: it was written by Matplotlib
on 2026-09-05, is 24 pages, and predates every Phase B and C result. It is not
a build of this `.tex` and must be replaced before submission. `final_report.md`
is likewise superseded -- it is the pre-LaTeX draft, kept only for history.
