#!/usr/bin/env python3
"""Module M2 entry point: statistical-recovery track.

Run Module 1 first so that a validation report exists:

    python scripts/m3_01_validate_imputed.py

Then:

    python scripts/m3_02_stat_track.py                       # all validated runs + complete-case baseline
    python scripts/m3_02_stat_track.py --methods mean knn --mechanisms MAR --levels 0.3
    python scripts/m3_02_stat_track.py --check-complete-case # compare with Member 1's complete-case results

Outputs (results/):
    experiment_results/stat_recovery_runs.csv           one row per run (summary metrics)
    experiment_results/stat_recovery_coefficients.csv   one row per run x model term
    experiment_results/stat_recovery_correlations.csv   one row per run x variable pair
    experiment_results/stat_recovery_errors.csv         only if some runs failed
    tables/table4_statistical_recovery.csv              Table 4: mean and sd over repeats
    tables/table4b_bias_by_coefficient.csv              signed z bias and noise per coefficient
    figures/fig_member3_stat_*.png

Exit code 0: finished with no failed runs. Exit code 1: no runs evaluated, or some runs failed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.m3_common import PROJECT_ROOT, load_config  # noqa: E402
from src.m3_stat_plots import plot_statistical_figures  # noqa: E402
from src.m3_stat_track import (  # noqa: E402
    build_bias_table,
    build_table4,
    run_statistical_track,
    save_results,
)


def check_complete_case(cfg: dict, args: argparse.Namespace) -> int:
    """Compute the complete-case baseline with this module and show it next to Member 1's saved file."""
    results = run_statistical_track(
        cfg,
        mechanisms=args.mechanisms,
        levels=args.levels,
        include_imputed=False,
        include_complete_case=True,
        require_validation=False,
    )
    if results.runs.empty:
        print("No complete-case runs were evaluated.")
        return 1
    table = build_table4(results.runs)
    cols = ["mechanism", "level", "n_repeats", "n_obs_mean", "stat_mean_abs_z_all_mean",
            "stat_mean_abs_z_affected_mean", "stat_ci_coverage_all_mean"]
    nobs = results.runs.groupby(["mechanism", "level"], as_index=False)["n_obs"].mean()
    table = table.merge(nobs.rename(columns={"n_obs": "n_obs_mean"}), on=["mechanism", "level"])
    print("\nThis module's complete-case results (mean over repeats):")
    print(table[cols].round(4).to_string(index=False))

    theirs = PROJECT_ROOT / "results" / "tables" / "complete_case_bias_noise.csv"
    print(f"\nMember 1's saved file: {theirs}")
    if theirs.exists():
        other = pd.read_csv(theirs)
        print(f"columns: {list(other.columns)}")
        print(other.head(10).to_string(index=False))
        print(
            "\nNEXT STEP: compare one or two numbers by hand (for example MAR at 0.30, mean |z| over "
            "the affected coefficients). Small differences are possible if the definitions differ "
            "(rows used, train vs full data, |z| averaging). Tell the team which definition is final."
        )
    else:
        print("That file does not exist in this checkout. Pull Member 1's latest results.")
    return 0


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Member 3 / Module 2: statistical-recovery track")
    parser.add_argument("--methods", nargs="+", choices=cfg["imputation_methods"])
    parser.add_argument("--mechanisms", nargs="+", choices=cfg["mechanisms"])
    parser.add_argument("--levels", nargs="+", type=float, help="for example 0.1 0.3")
    parser.add_argument("--imputed-root", help="override data/imputed")
    parser.add_argument("--no-complete-case", action="store_true", help="skip the complete-case baseline")
    parser.add_argument(
        "--no-require-validation",
        action="store_true",
        help="use every imputed file that exists, without the Module 1 validation report (not recommended)",
    )
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--check-complete-case", action="store_true",
                        help="only compute the complete-case baseline and show Member 1's results for comparison")
    args = parser.parse_args()

    if args.check_complete_case:
        return check_complete_case(cfg, args)

    if args.no_require_validation:
        print("WARNING: running without validation. Files that failed Module 1 checks may be used.")

    try:
        results = run_statistical_track(
            cfg,
            imputed_root=args.imputed_root,
            methods=args.methods,
            mechanisms=args.mechanisms,
            levels=args.levels,
            include_complete_case=not args.no_complete_case,
            require_validation=not args.no_require_validation,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    if results.runs.empty:
        print("No runs were evaluated. Are there validated imputed files? (python scripts/m3_01_validate_imputed.py)")
        return 1

    table4 = build_table4(results.runs)
    bias = build_bias_table(results.coefficients)
    written = save_results(results, table4, bias)

    counts = results.runs.groupby("method").size()
    print("\nRuns evaluated per method:")
    print(counts.to_string())
    print("\nWritten:")
    for name, path in written.items():
        print(f"  {name:<13} {path}")

    if not args.no_figures:
        for path in plot_statistical_figures(table4):
            print(f"  figure        {path}")

    if not results.errors.empty:
        print(f"\n{len(results.errors)} runs FAILED (see stat_recovery_errors.csv). First few:")
        print(results.errors.head(5).to_string(index=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
