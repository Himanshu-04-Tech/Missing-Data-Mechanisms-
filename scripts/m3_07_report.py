#!/usr/bin/env python3
"""Module M7 entry point: generate the results and discussion text from the saved tables.

    python scripts/m3_07_report.py

Reads whatever Member 3 outputs exist (stat_recovery_runs.csv, predictive_results.csv, Table 6, Case labels, ...).
Parts whose tables do not exist yet are marked PENDING with the command that produces them. No number is ever
typed by hand or filled in as a placeholder.

Outputs (results/report/):
    results_discussion.md   the text (OBSERVED RESULT / POSSIBLE INTERPRETATION labels, limitations)
    findings.csv            one row per finding: id, section, label, text, source table

Regenerate after every change to the results. Exit code 0: written. Exit code 1: no result tables found, or the
generated text contains a claim the team agreed not to make.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.m3_report import load_inputs, render_markdown, save_report  # noqa: E402


def main() -> int:
    inp = load_inputs()
    if all(x is None for x in (inp.stat_runs, inp.pred_runs, inp.table6)):
        print("ERROR: no result tables found. Run scripts/m3_02_stat_track.py first (it needs no imputed files for the complete-case part).")
        return 1
    try:
        text, findings = render_markdown(inp)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    written = save_report(text, findings)
    n_obs = sum(f.label == "OBSERVED RESULT" for f in findings)
    n_int = len(findings) - n_obs
    print(f"{len(findings)} findings: {n_obs} OBSERVED RESULT, {n_int} POSSIBLE INTERPRETATION")
    if "PENDING" in text:
        print(f"{text.count('PENDING.')} section(s) are PENDING (their input tables do not exist yet); see the report.")
    for name, path in written.items():
        print(f"  {name:<9} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
