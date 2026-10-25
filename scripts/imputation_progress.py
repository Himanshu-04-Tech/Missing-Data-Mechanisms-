"""Live progress view of the imputation runs (Member 2).

Counts the saved imputed files per method, for both variants, and refreshes every
few seconds. Press Ctrl+C to stop.

Usage:
  python scripts/imputation_progress.py          # refresh every 5 s
  python scripts/imputation_progress.py --once   # print once and exit
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.imputation import load_config

VARIANTS = {"imputed": "without target (prediction + stats)", "imputed_with_target": "with log_price (stats only)"}
BAR_WIDTH = 30


def show(config: dict) -> None:
    n_expected = len(config["mechanisms"]) * len(config["levels"]) * int(config["n_repeats"])
    print(f"Imputation progress   {datetime.now():%H:%M:%S}   (Ctrl+C to stop)\n")
    for folder, label in VARIANTS.items():
        print(f"data/{folder}/  - {label}")
        for method in config["imputation_methods"]:
            method_dir = PROJECT_ROOT / "data" / folder / method
            files = list(method_dir.rglob("*.csv.gz")) if method_dir.is_dir() else []
            done = len(files)
            filled = int(BAR_WIDTH * done / n_expected)
            bar = "#" * filled + "-" * (BAR_WIDTH - filled)
            status = "done" if done == n_expected else ""
            latest = ""
            if files and done < n_expected:
                newest = max(files, key=lambda p: p.stat().st_mtime)
                latest = "last: " + "/".join(newest.relative_to(method_dir).parts)
            print(f"  {method:<14} [{bar}] {done:>2}/{n_expected}  {status}{latest}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--every", type=float, default=5.0, help="seconds between refreshes")
    args = parser.parse_args()
    config = load_config()

    if args.once:
        show(config)
        return
    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            show(config)
            time.sleep(args.every)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
