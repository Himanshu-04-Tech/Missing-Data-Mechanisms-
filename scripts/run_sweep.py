"""Script to execute the complete missing data sweep.

Generates 75 conditions (3 mechanisms x 5 levels x 5 repeats), ground truth,
split copy, and manifest into data/generated_missing.
"""

from pathlib import Path
import sys
import time

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.sweep import run_sweep


def main():
    start_time = time.time()
    print("=" * 65)
    print("RUNNING MISSINGNESS SIMULATION SWEEP (MEMBER 2 HANDOFF DATASETS)")
    print("=" * 65)

    manifest_df = run_sweep(config_path="config.yaml", project_root=project_root, verbose=True)

    elapsed = time.time() - start_time
    print(f"Elapsed time: {elapsed:.2f}s | Total conditions generated: {len(manifest_df)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
