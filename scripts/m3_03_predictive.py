#!/usr/bin/env python3
"""Module M3 entry point: predictive track (Random Forest).

Step 1 needs nothing from Member 2 (complete-data reference = upper bound):

    python3 scripts/m3_03_predictive.py --reference-only

Step 2 needs Member 2's imputed files and the Module 1 validation report:

    python3 scripts/m3_01_validate_imputed.py
    python3 scripts/m3_03_predictive.py                        # all validated runs (+ reference)
    python3 scripts/m3_03_predictive.py --methods mean --mechanisms MAR --levels 0.3
    python3 scripts/m3_03_predictive.py --resume               # continue an interrupted sweep
    python3 scripts/m3_03_predictive.py --n-estimators 50      # quick smoke test (NOT for final results)
    python3 scripts/m3_03_predictive.py --reference-seeds 5    # also measure seed-to-seed spread of the reference

Outputs (results/):
    experiment_results/predictive_results.csv     one row per run (rmse, mae, r2, truth-test diagnostic)
    experiment_results/predictive_reference.csv   complete-data reference (upper bound)
    experiment_results/predictive_settings.json   exact Random Forest settings used
    experiment_results/predictive_errors.csv      only if some runs failed
    tables/table5_predictive_recovery.csv         mean and sd over repeats

Exit code 0: finished with no failed runs. Exit code 1: no runs evaluated, or some runs failed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.m3_common import load_config  # noqa: E402
from src.m3_predictive import (  # noqa: E402
    PREDICTIVE_RESULTS_PATH,
    PredictiveResults,
    build_predictive_table,
    get_settings,
    run_predictive_track,
    run_reference,
    save_predictive_results,
)


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Member 3 / Module 3: predictive track")
    parser.add_argument("--methods", nargs="+", choices=cfg["imputation_methods"])
    parser.add_argument("--mechanisms", nargs="+", choices=cfg["mechanisms"])
    parser.add_argument("--levels", nargs="+", type=float, help="for example 0.1 0.3")
    parser.add_argument("--imputed-root", help="override data/imputed")
    parser.add_argument("--reference-only", action="store_true", help="only fit the complete-data reference")
    parser.add_argument("--reference-seeds", type=int, default=1,
                        help="fit the reference with this many forest seeds (spread, for the Case thresholds)")
    parser.add_argument("--resume", action="store_true", help="skip runs already in predictive_results.csv")
    parser.add_argument("--n-estimators", type=int, help="override the number of trees (smoke tests only)")
    parser.add_argument("--n-jobs", type=int, help="override n_jobs (default from config; -1 = all cores)")
    parser.add_argument("--no-truth-score", action="store_true", help="skip the FR-3.7 diagnostic on complete test rows")
    parser.add_argument("--no-require-validation", action="store_true",
                        help="use every imputed file that exists, without the validation report (not recommended)")
    args = parser.parse_args()

    settings = get_settings(
        cfg,
        {
            "n_estimators": args.n_estimators,
            "n_jobs": args.n_jobs,
            "score_on_truth": False if args.no_truth_score else None,
        },
    )
    print("Random Forest settings (identical for every condition):")
    print(f"  {settings.rf_params}, random_state={settings.random_state}, score_on_truth={settings.score_on_truth}")
    print(f"  features ({len(settings.features)}): {settings.features}")
    print(f"  target: {settings.target}")
    if args.n_estimators:
        print("WARNING: --n-estimators overrides the config. Do not use these numbers as final results.")
    if args.no_require_validation:
        print("WARNING: running without validation. Files that failed Module 1 checks may be used.")

    # --- Step 1: reference (upper bound)
    t0 = time.perf_counter()
    try:
        reference = run_reference(cfg, settings, n_seeds=args.reference_seeds)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    primary = reference[reference["is_primary"]].iloc[0]
    print(
        f"\nComplete-data reference (train on complete train, score on complete test): "
        f"RMSE={primary['rmse']:.4f}  MAE={primary['mae']:.4f}  R2={primary['r2']:.4f}  "
        f"({time.perf_counter() - t0:.0f}s)"
    )
    if len(reference) > 1:
        print(f"  seed-to-seed spread over {len(reference)} seeds: "
              f"RMSE sd={reference['rmse'].std(ddof=1):.5f}, R2 sd={reference['r2'].std(ddof=1):.5f}")

    if args.reference_only:
        written = save_predictive_results(
            PredictiveResults(pd.DataFrame(), pd.DataFrame()), reference, settings, pd.DataFrame()
        )
        for name, path in written.items():
            print(f"  {name:<10} {path}")
        return 0

    # --- Step 2: imputed runs
    done = None
    if args.resume and PREDICTIVE_RESULTS_PATH.exists():
        done = pd.read_csv(PREDICTIVE_RESULTS_PATH)

    def checkpoint(rows: pd.DataFrame) -> None:
        PREDICTIVE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(PREDICTIVE_RESULTS_PATH, index=False)

    try:
        results = run_predictive_track(
            cfg,
            settings,
            imputed_root=args.imputed_root,
            methods=args.methods,
            mechanisms=args.mechanisms,
            levels=args.levels,
            require_validation=not args.no_require_validation,
            done=done,
            checkpoint=checkpoint,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    if results.runs.empty:
        print("No runs were evaluated. Are there validated imputed files? (python3 scripts/m3_01_validate_imputed.py)")
        return 1

    table = build_predictive_table(results.runs)
    written = save_predictive_results(results, reference, settings, table)

    print("\nRuns evaluated per method:")
    print(results.runs.groupby("method").size().to_string())
    print(f"\nTotal Random Forest fit time: {results.runs['fit_seconds'].sum() / 60:.1f} min")
    print("\nWritten:")
    for name, path in written.items():
        print(f"  {name:<10} {path}")

    if not results.errors.empty:
        print(f"\n{len(results.errors)} runs FAILED (see predictive_errors.csv). First few:")
        print(results.errors.head(10).to_string(index=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
