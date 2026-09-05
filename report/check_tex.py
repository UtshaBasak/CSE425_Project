#!/usr/bin/env python
"""Structural checks on the report source, because there is no LaTeX here.

    python report/check_tex.py [--tex report/final_report.tex]

This machine has no TeX toolchain, so the report is written to be compiled
elsewhere. That makes a whole class of mistake invisible until someone pastes it
into Overleaf and gets an error instead of a PDF. These are the checks that
catch the ones worth catching without a compiler:

* every ``\\Macro`` used in the body is defined by a ``\\newcommand``;
* every ``\\newcommand`` is actually used (a defined-but-unused macro usually
  means a table column was renamed and one half of the rename was missed);
* ``\\begin{x}``/``\\end{x}`` nest and match;
* braces and dollar signs balance;
* every ``\\ref`` resolves to a ``\\label``, and every ``\\cite`` to a
  ``\\bibitem``;
* every ``\\includegraphics`` names a file that exists somewhere in the repo;
* no ``pending`` markers or leftover ``[TBD]`` remain.

Exit code is non-zero if anything fails, so it can gate a commit.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import project_root  # noqa: E402

ROOT = project_root()

#: macros LaTeX and IEEEtran already define, which must not be reported missing
KNOWN = {
    "begin", "end", "documentclass", "usepackage", "newcommand", "def", "title",
    "author", "maketitle", "section", "subsection", "subsubsection", "paragraph",
    "textbf", "textit", "emph", "texttt", "item", "label", "ref", "cite",
    "caption", "centering", "small", "footnotesize", "toprule", "midrule",
    "bottomrule", "multicolumn", "includegraphics", "columnwidth", "textwidth",
    "bibitem", "thebibliography", "IEEEauthorblockN", "IEEEauthorblockA",
    "IEEEkeywords", "IEEEoverridecommandlockouts", "BibTeX", "kern", "sc",
    "lower", "hbox", "rm", "url", "sigma", "alpha", "lambda", "sum", "frac",
    "cdot", "leq", "geq", "times", "approx", "pm", "in", "text", "mathcal",
    "hat", "quad", "qquad", ",", ";", ":", "!", "%", "&", "_", "#", "$", "{",
    "}", "\\", "left", "right", "abstract", "equation", "figure", "table",
    "tabular", "enumerate", "itemize", "document", "suptitle", "tightlist",
    "linewidth", "hline", "footnote", "mathrm", "leftarrow", "rightarrow",
    # maths used in the method section
    "ell", "neq", "tau", "mu", "theta", "phi", "beta", "gamma", "delta",
    "epsilon", "infty", "partial", "nabla", "log", "exp", "max", "min",
    "argmax", "argmin", "mathbb", "mathbf", "operatorname", "top", "bot",
}

MACRO = re.compile(r"\\([A-Za-z]+)")
NEWCOMMAND = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}")
ENVIRON = re.compile(r"\\(begin|end)\{([A-Za-z*]+)\}")
LABEL = re.compile(r"\\label\{([^}]+)\}")
REF = re.compile(r"\\(?:ref|eqref)\{([^}]+)\}")
CITE = re.compile(r"\\cite\{([^}]+)\}")
BIBITEM = re.compile(r"\\bibitem\{([^}]+)\}")
GRAPHIC = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
TABLE = re.compile(r"\\begin\{table\}")
EQUATION = re.compile(r"\\begin\{equation\}")

PAGE_BEGIN = "%% PAGEBUDGET:BEGIN"
PAGE_END = "%% PAGEBUDGET:END"

#: IEEEtran two-column, 10pt: measured against typical conference papers. These
#: are estimates for steering, not a substitute for compiling -- the point is to
#: notice a 12-page draft early, not to predict the page count to two decimals.
WORDS_PER_PAGE = 950
PAGE_PER_TABLE = 0.22
PAGE_PER_FIGURE = 0.35
PAGE_PER_EQUATION = 0.05
PAGE_LIMIT = (6, 10)


def strip_comments(text: str) -> str:
    """Drop % comments, honouring \\% escapes."""
    out = []
    for line in text.splitlines():
        cleaned, i = [], 0
        while i < len(line):
            if line[i] == "\\" and i + 1 < len(line):
                cleaned.append(line[i:i + 2])
                i += 2
                continue
            if line[i] == "%":
                break
            cleaned.append(line[i])
            i += 1
        out.append("".join(cleaned))
    return "\n".join(out)


def check(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    body = strip_comments(raw)
    problems: list[str] = []

    defined = set(NEWCOMMAND.findall(raw))
    # \newcommand{\X} itself matches both \newcommand and \X; the latter is a
    # definition, not a use, so the definition sites are stripped before
    # counting uses
    definition_sites = set(NEWCOMMAND.findall(body))
    body_without_defs = NEWCOMMAND.sub("", body)
    genuinely_used = {m for m in MACRO.findall(body_without_defs) if m not in KNOWN}

    missing = sorted(genuinely_used - defined - KNOWN)
    unknown = [m for m in missing if m not in ("newcommand",)]
    if unknown:
        problems.append(f"undefined macros used: {', '.join(unknown)}")

    unused = sorted(definition_sites - genuinely_used)
    if unused:
        problems.append(f"macros defined but never used: {', '.join(unused)}")

    stack = []
    for kind, name in ENVIRON.findall(body):
        if kind == "begin":
            stack.append(name)
        elif not stack:
            problems.append(f"\\end{{{name}}} with no matching \\begin")
        elif stack[-1] != name:
            problems.append(f"\\end{{{name}}} closes \\begin{{{stack[-1]}}}")
            stack.pop()
        else:
            stack.pop()
    if stack:
        problems.append(f"unclosed environments: {', '.join(stack)}")

    # Skip escape PAIRS, not "any character preceded by a backslash": in a
    # table, "\\" is an escaped backslash and the brace after it is real, so
    # the naive rule silently swallows an opening brace and reports a phantom
    # imbalance.
    depth, i, negative = 0, 0, False
    while i < len(body):
        if body[i] == "\\" and i + 1 < len(body):
            i += 2
            continue
        depth += (body[i] == "{") - (body[i] == "}")
        if depth < 0 and not negative:
            problems.append(f"a closing brace at offset {i} precedes its opener")
            negative = True
        i += 1
    if depth > 0:
        problems.append(f"{depth} unclosed brace(s)")

    dollars = len([i for i, c in enumerate(body)
                   if c == "$" and (i == 0 or body[i - 1] != "\\")])
    if dollars % 2:
        problems.append(f"odd number of unescaped $ ({dollars}); math mode unbalanced")

    labels = set(LABEL.findall(body))
    for target in sorted(set(REF.findall(body))):
        if target not in labels:
            problems.append(f"\\ref{{{target}}} has no \\label")

    keys = set(BIBITEM.findall(body))
    for group in CITE.findall(body):
        for key in (k.strip() for k in group.split(",")):
            if key not in keys:
                problems.append(f"\\cite{{{key}}} has no \\bibitem")

    for graphic in GRAPHIC.findall(body):
        name = Path(graphic).name
        if not list(ROOT.rglob(name)):
            problems.append(f"\\includegraphics{{{graphic}}}: {name} not found in the repo")

    for marker in ("[TBD]", "TODO", "XXX"):
        if marker in body:
            problems.append(f"leftover marker {marker!r} in the body")
    if "pending" in body.lower().replace("\\textit{pending}", ""):
        pass                       # the pending macro itself is expected
    n_pending = raw.count(r"\textit{pending}")
    if n_pending:
        problems.append(f"{n_pending} macro(s) still render as pending "
                        "(run report/fill_report.py once the runs finish)")
    return problems


def estimate(path: Path) -> dict:
    """Rough page count: prose words plus fixed allowances for floats."""
    body = strip_comments(path.read_text(encoding="utf-8"))
    words = len(body.split())
    tables = len(TABLE.findall(body))
    figures = len(GRAPHIC.findall(body))
    equations = len(EQUATION.findall(body))
    prose = words / WORDS_PER_PAGE
    floats = (tables * PAGE_PER_TABLE + figures * PAGE_PER_FIGURE
              + equations * PAGE_PER_EQUATION)
    return {"words": words, "tables": tables, "figures": figures,
            "equations": equations, "prose_pages": prose,
            "float_pages": floats, "total_pages": prose + floats}


def render_budget(est: dict) -> str:
    low, high = PAGE_LIMIT
    total = est["total_pages"]
    if total < low:
        verdict = f"UNDER the {low}-page minimum -- Phase B sections still to come"
    elif total > high:
        verdict = f"OVER the {high}-page limit -- cut Sec. II first, then Table III"
    else:
        verdict = f"inside the {low}-{high} page limit"
    return "\n".join([
        PAGE_BEGIN + "  (regenerated by `python report/check_tex.py --update`)",
        "%%",
        "%% PAGE BUDGET, measured rather than guessed:",
        f"%%   {est['words']:>6,} words of prose        -> {est['prose_pages']:5.2f} pages",
        f"%%   {est['tables']:>6} tables x {PAGE_PER_TABLE}       -> "
        f"{est['tables'] * PAGE_PER_TABLE:5.2f} pages",
        f"%%   {est['figures']:>6} figures x {PAGE_PER_FIGURE}      -> "
        f"{est['figures'] * PAGE_PER_FIGURE:5.2f} pages",
        f"%%   {est['equations']:>6} equations x {PAGE_PER_EQUATION}    -> "
        f"{est['equations'] * PAGE_PER_EQUATION:5.2f} pages",
        "%%   " + "-" * 44,
        f"%%   TOTAL                        -> {total:5.2f} pages  ({verdict})",
        "%%",
        PAGE_END,
    ])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", default=str(ROOT / "report" / "final_report.tex"))
    parser.add_argument("--update", action="store_true",
                        help="rewrite the PAGEBUDGET block at the top of the file")
    args = parser.parse_args(argv)

    path = Path(args.tex)
    problems = check(path)
    est = estimate(path)

    if args.update:
        text = path.read_text(encoding="utf-8")
        if PAGE_BEGIN in text and PAGE_END in text:
            head = text[:text.index(PAGE_BEGIN)]
            tail = text[text.index(PAGE_END) + len(PAGE_END):]
            path.write_text(head + render_budget(est) + tail, encoding="utf-8")
            print("page budget block updated")
        else:
            problems.append("no PAGEBUDGET markers to update")

    print(f"{path.name}: {est['words']:,} words, {est['tables']} tables, "
          f"{est['figures']} figures -> ~{est['total_pages']:.1f} pages "
          f"(limit {PAGE_LIMIT[0]}-{PAGE_LIMIT[1]})")

    if not problems:
        print("no structural problems found")
        return 0
    print(f"\n{len(problems)} problem(s):")
    for problem in problems:
        print(f"  - {problem}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
