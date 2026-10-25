"""Run imputation on every corrupted condition and score it (Member 2).

For each method x mechanism x level x repeat:
  1. load the corrupted dataset (Member 1's handoff loader),
  2. fit the imputer on train rows only and fill all rows,
  3. check the output (no NaN, observed cells unchanged, same rows/columns),
  4. save it to data/imputed/{method}/{mechanism}/rate_XX/rep_k.csv.gz,
  5. score the filled cells against ground truth (scoring only).

Outputs:
  results/experiment_results/imputation_scores.csv   one row per run x column x split
  results/tables/imputation_error_summary.csv        mean and sd over the 5 repeats

With --use-target, log_price is an imputer input and everything goes to
data/imputed_with_target/ and *_with_target.csv instead. Those files are for the
STATISTICAL track only and must never be used for prediction (target leakage).

With --rescore, saved files are re-scored without imputing again (timings are kept
from the previous scores file).

Usage:
  python scripts/run_imputation.py --methods mean knn
  python scripts/run_imputation.py --use-target
  python scripts/run_imputation.py --methods knn --rescore
"""

import argparse
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.handoff import load_condition
from src.imputation import impute, load_config, score_imputation

RUN_KEYS = ["method", "mechanism", "level", "repeat"]


def output_paths(use_target: bool) -> dict:
    suffix = "_with_target" if use_target else ""
    return {
        "imputed_dir": PROJECT_ROOT / "data" / f"imputed{suffix}",
        "scores": PROJECT_ROOT / "results" / "experiment_results" / f"imputation_scores{suffix}.csv",
        "summary": PROJECT_ROOT / "results" / "tables" / f"imputation_error_summary{suffix}.csv",
    }


def imputed_file(imputed_dir: Path, method: str, mechanism: str, level: float, repeat: int) -> Path:
    pct = int(round(level * 100))
    return imputed_dir / method / mechanism / f"rate_{pct:02d}" / f"rep_{repeat}.csv.gz"


def check_output(filled: pd.DataFrame, corrupted: pd.DataFrame) -> None:
    """Fail loudly if the imputed file breaks any handoff rule."""
    assert list(filled.columns) == list(corrupted.columns), "column order changed"
    assert filled.index.equals(corrupted.index), "rows changed"
    assert not filled.isna().any().any(), "NaN left after imputation"
    assert np.isfinite(filled.to_numpy(dtype=float)).all(), "infinite values"
    observed = corrupted.notna().to_numpy()
    n_changed = int((filled.to_numpy()[observed] != corrupted.to_numpy()[observed]).sum())
    assert n_changed == 0, f"{n_changed} observed cells were changed"


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    """Average the per-run scores over repeats."""
    keys = ["method", "mechanism", "level", "column", "split"]
    metrics = ["rmse", "mae", "nrmse", "r2", "bias", "sd_ratio", "seconds"]
    summary = scores.groupby(keys, sort=False)[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{stat.replace('std', 'sd')}" for m, stat in summary.columns]
    summary["n"] = scores.groupby(keys, sort=False).size()
    return summary.reset_index()


def previous_seconds(old_scores: pd.DataFrame, method, mechanism, level, repeat) -> float:
    match = old_scores[
        (old_scores["method"] == method)
        & (old_scores["mechanism"] == mechanism)
        & np.isclose(old_scores["level"], level)
        & (old_scores["repeat"] == repeat)
    ]
    return float(match["seconds"].iloc[0]) if len(match) else np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--methods", nargs="+", default=None, help="default: all in config.yaml")
    parser.add_argument("--use-target", action="store_true", help="use log_price as imputer input")
    parser.add_argument("--rescore", action="store_true", help="re-score saved files only")
    args = parser.parse_args()

    config = load_config()
    paths = output_paths(args.use_target)
    methods = args.methods or list(config["imputation_methods"])
    conditions = list(
        product(config["mechanisms"], config["levels"], range(int(config["n_repeats"])))
    )
    old_scores = pd.read_csv(paths["scores"]) if paths["scores"].is_file() else None
    if args.rescore and old_scores is None:
        raise FileNotFoundError(f"--rescore needs an existing {paths['scores']}")

    all_scores = []
    for method in methods:
        for mechanism, level, repeat in conditions:
            data = load_condition(mechanism, level, repeat)
            corrupted = data["corrupted"]
            out_path = imputed_file(paths["imputed_dir"], method, mechanism, level, repeat)

            if args.rescore:
                filled = pd.read_csv(out_path, index_col="row_index")
                seconds = previous_seconds(old_scores, method, mechanism, level, repeat)
            else:
                start = time.perf_counter()
                filled = impute(corrupted, data["train_idx"], method, config, args.use_target)
                seconds = time.perf_counter() - start

            check_output(filled, corrupted)

            if not args.rescore:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                filled.to_csv(out_path, index_label="row_index")

            scores = score_imputation(
                filled, data["truth"], data["mask"], data["train_idx"], data["test_idx"]
            )
            scores.insert(0, "repeat", repeat)
            scores.insert(0, "level", level)
            scores.insert(0, "mechanism", mechanism)
            scores.insert(0, "method", method)
            scores["seconds"] = seconds
            all_scores.append(scores)
            print(f"{method:>13} {mechanism:<4} {level:.2f} rep {repeat}  done ({seconds:.2f}s)", flush=True)

    new_scores = pd.concat(all_scores, ignore_index=True)

    # Keep earlier results for methods that were not re-run this time
    if old_scores is not None:
        old_scores = old_scores[~old_scores["method"].isin(methods)]
        new_scores = pd.concat([old_scores, new_scores], ignore_index=True)

    order = {m: i for i, m in enumerate(config["imputation_methods"])}
    new_scores = new_scores.sort_values(
        "method", key=lambda s: s.map(order), kind="stable"
    ).reset_index(drop=True)

    paths["scores"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    new_scores.to_csv(paths["scores"], index=False)
    summarize(new_scores).to_csv(paths["summary"], index=False)
    print(f"\nSaved {paths['scores'].relative_to(PROJECT_ROOT)}")
    print(f"Saved {paths['summary'].relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
