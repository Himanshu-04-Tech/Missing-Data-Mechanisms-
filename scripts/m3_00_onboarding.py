#!/usr/bin/env python3
"""Module M0 entry point: Member 3 setup and onboarding checks.

Usage (from the repo root, with the virtual environment active):

    python3 scripts/m3_00_onboarding.py
    python3 scripts/m3_00_onboarding.py --mechanism MNAR --level 0.5 --repeat 3

Exit code 0 means no blocking failures. Exit code 1 means at least one FAIL.
A JSON report is saved to results/experiment_results/m3_00_onboarding_report.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `import src...` work when running this file directly from scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.m3_onboarding import run_all, save_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Member 3 / Module 0 onboarding checks")
    parser.add_argument("--mechanism", default="MAR", choices=["MCAR", "MAR", "MNAR"])
    parser.add_argument("--level", type=float, default=0.30, help="missingness level, e.g. 0.30")
    parser.add_argument("--repeat", type=int, default=0, help="repeat index 0..4")
    args = parser.parse_args()

    report = run_all(mechanism=args.mechanism, level=args.level, repeat=args.repeat)
    report.print()
    out = save_report(report)
    print(f"\nReport saved to: {out}")
    return 1 if report.has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
