#!/usr/bin/env python3
"""Module M6 runner: one command per stage, or the whole Member 3 pipeline in order.

    python scripts/m3_06_run_all.py                      # stages 1 to 5 (needs Member 2's imputed files)
    python scripts/m3_06_run_all.py --stages 4 5         # only merge and cases
    python scripts/m3_06_run_all.py --from-stage 3       # stage 3 onwards
    python scripts/m3_06_run_all.py --smoke              # fast, NON-FINAL: 30 trees, partial data allowed
    python scripts/m3_06_run_all.py --with-tests         # run pytest at the end (FR-6.5)
    python scripts/m3_06_run_all.py --imputed-root PATH  # imputed files somewhere else

Stages (each is also runnable on its own):
    1 validate     scripts/m3_01_validate_imputed.py
    2 statistical  scripts/m3_02_stat_track.py
    3 predictive   scripts/m3_03_predictive.py
    4 merge        scripts/m3_04_merge.py
    5 cases        scripts/m3_05_cases.py

The runner stops at the first stage that fails. It writes results/experiment_results/pipeline_manifest.json
(exit codes, timings, git commit, config digest and a content digest of every output). To prove that a
re-run reproduces the same numbers, copy the manifest, run again, and compare:

    python scripts/m3_06_audit.py --compare results/experiment_results/pipeline_manifest_previous.json

All seeds come from config.yaml, so the same inputs give the same outputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

from src.m3_common import PROJECT_ROOT  # noqa: E402
from src.m3_reproduce import MANIFEST_PATH, STAGES, build_manifest  # noqa: E402


def stage_args(number: int, args: argparse.Namespace) -> list[str]:
    """Extra command-line arguments for one stage."""
    extra: list[str] = []
    root = ["--imputed-root", args.imputed_root] if args.imputed_root else []
    if number == 1:
        extra += root + (["--allow-missing"] if args.smoke else [])
    elif number == 2:
        extra += root + (["--no-figures"] if args.smoke else [])
    elif number == 3:
        extra += root + (["--n-estimators", "30"] if args.smoke else [])
    elif number == 4:
        extra += ["--allow-partial"] if args.smoke else []
    elif number == 5:
        extra += ["--no-figures"] if args.smoke and args.no_figures else []
    return extra


def main() -> int:
    parser = argparse.ArgumentParser(description="Member 3 / Module 6: run the pipeline stage by stage")
    parser.add_argument("--stages", nargs="+", type=int, choices=[s.number for s in STAGES])
    parser.add_argument("--from-stage", type=int, choices=[s.number for s in STAGES])
    parser.add_argument("--imputed-root")
    parser.add_argument("--smoke", action="store_true",
                        help="fast check of the plumbing (30 trees, partial data allowed). NOT for final results.")
    parser.add_argument("--no-figures", action="store_true", help="with --smoke: also skip the Case figures")
    parser.add_argument("--with-tests", action="store_true", help="run pytest after the stages")
    args = parser.parse_args()

    if args.stages and args.from_stage:
        parser.error("use either --stages or --from-stage")
    chosen = [s for s in STAGES if (not args.stages or s.number in args.stages)
              and (not args.from_stage or s.number >= args.from_stage)]
    if args.smoke:
        print("SMOKE MODE: 30 trees and partial data allowed. Do not use these outputs as results.\n")

    records: list[dict] = []
    exit_code = 0
    for stage in chosen:
        cmd = stage.command(extra=stage_args(stage.number, args))
        print(f"=== Stage {stage.number}: {stage.name}\n$ {' '.join(cmd)}", flush=True)
        start = time.perf_counter()
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
        records.append({"stage": stage.number, "name": stage.name, "command": " ".join(cmd),
                        "exit_code": proc.returncode, "seconds": round(time.perf_counter() - start, 1)})
        if proc.returncode != 0:
            print(f"\nStage {stage.number} ({stage.name}) FAILED with exit code {proc.returncode}. Stopping.")
            exit_code = proc.returncode
            break

    if exit_code == 0 and args.with_tests:
        print("=== Tests\n$ pytest -q", flush=True)
        start = time.perf_counter()
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=PROJECT_ROOT)
        records.append({"stage": "tests", "name": "pytest", "command": "pytest -q",
                        "exit_code": proc.returncode, "seconds": round(time.perf_counter() - start, 1)})
        exit_code = proc.returncode

    manifest = build_manifest(records)
    manifest["smoke_mode"] = bool(args.smoke)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nSummary:")
    for r in records:
        print(f"  stage {r['stage']!s:<6} {r['name']:<12} exit {r['exit_code']}  {r['seconds']}s")
    print(f"Manifest: {MANIFEST_PATH}")
    print("RESULT: OK" if exit_code == 0 else "RESULT: NOT OK")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
