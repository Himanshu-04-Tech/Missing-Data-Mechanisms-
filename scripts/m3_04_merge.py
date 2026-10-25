#!/usr/bin/env python3
"""Module M4 entry point: merge the statistical and predictive tracks, build Table 6.

Run Modules 2 and 3 first:

    python scripts/m3_02_stat_track.py
    python scripts/m3_03_predictive.py

Then:

    python scripts/m3_04_merge.py                  # strict: both tracks must cover exactly the same runs
    python scripts/m3_04_merge.py --allow-partial  # merge only runs present in both (while files are still arriving)

Outputs (results/):
    experiment_results/merged_results.csv        one row per run, both tracks + rmse_gap / r2_gap
    tables/table6_overall_recovery.csv           Table 6: mean and sd by method x mechanism x level
    tables/table6b_best_methods.csv              best predictive vs best statistical method per mechanism x level

Exit code 0: merged. Exit code 1: inputs missing, or keys do not match one-to-one (strict mode).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402

import pandas as pd  # noqa: E402

from src.m3_merge import RANK_PRED, RANK_STAT, MergeError, run_merge, save_merge_results  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Member 3 / Module 4: merge and Table 6")
    parser.add_argument("--allow-partial", action="store_true",
                        help="merge only runs found in both tracks and list the unmatched ones")
    args = parser.parse_args()

    try:
        result = run_merge(strict=not args.allow_partial)
    except (FileNotFoundError, MergeError) as exc:
        print(f"ERROR: {exc}")
        return 1

    merged, table6, best, unmatched = result["merged"], result["table6"], result["best"], result["unmatched"]
    print(f"Merged {len(merged)} runs across methods: {merged.groupby('method').size().to_dict()}")
    if len(unmatched):
        print(f"WARNING: {len(unmatched)} runs are in only one track (partial merge):")
        print(unmatched.groupby(["only_in", "method"]).size().to_string())

    ref_rmse, ref_r2 = merged["rmse_ref"].iloc[0], merged["r2_ref"].iloc[0]
    print(f"\nComplete-data reference: RMSE={ref_rmse:.4f}  R2={ref_r2:.4f}")
    print(f"Ranking metrics (provisional, decision D2): statistical={RANK_STAT}, predictive={RANK_PRED}; smaller is better.")

    cols = ["mechanism", "level", "best_predictive_method", "best_statistical_method", "same_method",
            "predictive_margin", "statistical_margin"]
    with pd.option_context("display.width", 200):
        print("\nBest method per mechanism x level (these conditions only, not a universal ranking):")
        print(best[cols].round(4).to_string(index=False))
    print(f"\nSame method wins both tracks in {int(best['same_method'].sum())} of {len(best)} mechanism x level cells.")
    print("Margins are runner-up minus winner. Compare them with the sd columns in Table 6 before reading a winner as real.")

    print("\nWritten:")
    for name, path in save_merge_results(result).items():
        print(f"  {name:<8} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
