#!/usr/bin/env python3
"""Module M5 entry point: Case A/B/C/D flags, central figure, sensitivity check, supporting plots.

Run Modules 2, 3 and 4 first:

    python scripts/m3_02_stat_track.py
    python scripts/m3_03_predictive.py
    python scripts/m3_04_merge.py

Then:

    python scripts/m3_05_cases.py
    python scripts/m3_05_cases.py --no-figures

The thresholds are read from config.yaml (``cases:``). Fix them and commit BEFORE running this on real
imputed results (decision D3). While ``thresholds_locked`` is false, every output says PROVISIONAL.

Outputs (results/):
    tables/cases_by_condition.csv         Table 6 plus Case label for every method x mechanism x level
    tables/table7_case_counts.csv         counts of Cases A to D by method x mechanism
    tables/case_sensitivity.csv           one threshold moved at a time (FR-5.4)
    tables/case_sensitivity_grid.csv      both thresholds moved together
    experiment_results/case_thresholds_used.json
    figures/fig_member3_central_stat_vs_pred.png, fig_member3_pred_rmse_gap_vs_level.png,
            fig_member3_pred_recovery_heatmap.png, fig_member3_case_sensitivity.png

Exit code 0: finished. Exit code 1: inputs missing or thresholds invalid.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402

import pandas as pd  # noqa: E402

from src.m3_case_plots import plot_case_figures  # noqa: E402
from src.m3_cases import CASE_MEANING, get_thresholds, run_cases, save_case_results  # noqa: E402
from src.m3_merge import MergeError, load_merged  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Member 3 / Module 5: Cases, central figure, sensitivity")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    try:
        thresholds = get_thresholds()
        merged = load_merged()
        result = run_cases(merged, thresholds)
    except (FileNotFoundError, MergeError) as exc:
        print(f"ERROR: {exc}")
        return 1

    labelled, counts, sens = result["labelled"], result["counts"], result["sensitivity"]
    status = "LOCKED" if thresholds.locked else "PROVISIONAL"
    print(f"Case thresholds [{status}]:")
    print(f"  statistically good  : {thresholds.stat_metric} <= {thresholds.stat_threshold}")
    print(f"  predictively good   : {thresholds.pred_metric} <= {thresholds.pred_threshold}")
    if not thresholds.locked:
        print("  WARNING: thresholds are not locked. Agree them with the team (D3), commit config.yaml,")
        print("           set cases.thresholds_locked: true, and only then read these Cases as results.")

    print(f"\nLabelled {len(labelled)} conditions:")
    for case, n in labelled["case"].value_counts().reindex(list("ABCD"), fill_value=0).items():
        print(f"  Case {case} ({CASE_MEANING[case]}): {n}")

    with pd.option_context("display.width", 200):
        print("\nCase counts by method x mechanism (over the missingness levels):")
        print(counts.drop(columns=["thresholds_status"]).to_string(index=False))
        print("\nSensitivity (one threshold moved at a time; n_changed = labels that differ from the baseline):")
        print(sens[["varied", "factor", "A", "B", "C", "D", "n_changed"]].to_string(index=False))

    written = save_case_results(result, thresholds)
    print("\nWritten:")
    for name, path in written.items():
        print(f"  {name:<10} {path}")
    if not args.no_figures:
        for path in plot_case_figures(labelled, result["grid"]):
            print(f"  figure     {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
