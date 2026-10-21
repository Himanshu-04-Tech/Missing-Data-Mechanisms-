#!/usr/bin/env python3
"""Module M1 entry point: validate Member 2's imputed files.

Usage (from the repo root, virtual environment active):

    python3 scripts/m3_01_validate_imputed.py                      # all 300 runs, strict
    python3 scripts/m3_01_validate_imputed.py --allow-missing      # only check files delivered so far
    python3 scripts/m3_01_validate_imputed.py --methods mean knn --mechanisms MAR --levels 0.3

Exit code 0: every checked file passed (and, unless --allow-missing, none is missing).
Exit code 1: at least one file failed, or files are missing in strict mode.

Output: results/experiment_results/validation_report.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.m3_common import load_config  # noqa: E402
from src.m3_validate_imputed import (  # noqa: E402
    FAIL,
    MISSING,
    PASS,
    all_passed,
    failed_checks,
    run_validation,
    save_validation_report,
)

MAX_FAILURES_SHOWN = 20


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Member 3 / Module 1: validate imputed files")
    parser.add_argument("--methods", nargs="+", choices=cfg["imputation_methods"])
    parser.add_argument("--mechanisms", nargs="+", choices=cfg["mechanisms"])
    parser.add_argument("--levels", nargs="+", type=float, help="for example 0.1 0.3")
    parser.add_argument("--imputed-root", help="override data/imputed (for example a copy Member 2 sent)")
    parser.add_argument("--out", help="override the report path")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="do not treat files that have not been delivered yet as an error",
    )
    args = parser.parse_args()

    report = run_validation(
        cfg,
        imputed_root=args.imputed_root,
        methods=args.methods,
        mechanisms=args.mechanisms,
        levels=args.levels,
    )
    out = save_validation_report(report, args.out)

    total = len(report)
    counts = report["status"].value_counts().to_dict()
    print(
        f"Validated {total} expected runs: "
        f"{counts.get(PASS, 0)} PASS, {counts.get(FAIL, 0)} FAIL, {counts.get(MISSING, 0)} MISSING"
    )

    if total:
        table = pd.crosstab(report["method"], report["status"]).reindex(
            columns=[PASS, FAIL, MISSING], fill_value=0
        )
        print("\nBy method:")
        print(table.to_string())

    failures = report[report["status"] == FAIL]
    if len(failures):
        print(f"\nFailures (first {MAX_FAILURES_SHOWN} of {len(failures)}):")
        for _, row in failures.head(MAX_FAILURES_SHOWN).iterrows():
            names = ", ".join(failed_checks(row)) or "see notes"
            print(f"  {row['path']}\n    failed: {names}\n    notes:  {row['notes']}")

    n_missing = counts.get(MISSING, 0)
    if n_missing and not args.allow_missing:
        print(
            f"\n{n_missing} expected files are missing. Use --allow-missing to validate "
            "only what has been delivered, or ask Member 2 for the rest."
        )

    print(f"\nReport saved to: {out}")
    ok = all_passed(report, allow_missing=args.allow_missing)
    print("RESULT: OK" if ok else "RESULT: NOT OK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
