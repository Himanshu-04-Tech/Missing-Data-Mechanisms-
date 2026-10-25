#!/usr/bin/env python3
"""Module M6: audit the Member 3 outputs, and compare two pipeline runs.

    python scripts/m3_06_audit.py                          # checklist of the deliverables M1 to M5
    python scripts/m3_06_audit.py --compare OLD_MANIFEST   # is the latest run identical to an earlier one?

The audit does not recompute anything. It reads the saved outputs and checks that they are complete
(300 runs, 60 conditions), consistent with each other and with config.yaml, free of failed runs, and
not left over from a smoke test.

Exit code 0: no FAIL (WARN items still need a human look). Exit code 1: at least one FAIL.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402
import json  # noqa: E402

from src.m3_reproduce import FAIL, MANIFEST_PATH, WARN, audit, compare_manifests, threshold_commit_check  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Member 3 / Module 6: output audit")
    parser.add_argument("--compare", help="an earlier pipeline_manifest.json to compare the latest manifest with")
    args = parser.parse_args()

    if args.compare:
        old_path = Path(args.compare)
        if not old_path.exists() or not MANIFEST_PATH.exists():
            print(f"ERROR: need both {old_path} and {MANIFEST_PATH}")
            return 1
        old = json.loads(old_path.read_text(encoding="utf-8"))
        new = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        problems = compare_manifests(old, new)
        if problems:
            print("The two runs are NOT identical:")
            for p in problems:
                print(f"  - {p}")
            return 1
        print(f"The two runs are identical: {len(new['outputs'])} outputs have the same content digest (timings ignored).")
        return 0

    checks = audit() + [threshold_commit_check()]
    width = max(len(c.name) for c in checks)
    for c in checks:
        print(f"[{c.status:<4}] {c.name:<{width}}  {c.detail}")
    n_fail = sum(c.status == FAIL for c in checks)
    n_warn = sum(c.status == WARN for c in checks)
    print(f"\n{len(checks) - n_fail - n_warn} PASS, {n_warn} WARN, {n_fail} FAIL")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
