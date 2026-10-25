"""Sensitivity check for the number of KNN neighbours k (Member 2).

Runs KNN imputation with several k values at 30% missingness (repeat 0) for each
mechanism and reports the imputation error on TRAIN cells only, so the choice of k
never looks at the test rows.

Output: results/tables/knn_k_check.csv

Usage:
  python scripts/check_knn_k.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.handoff import load_condition
from src.imputation import impute, load_config, score_imputation

K_VALUES = [3, 5, 10, 20]
LEVEL = 0.30
REPEAT = 0
OUT_PATH = PROJECT_ROOT / "results" / "tables" / "knn_k_check.csv"


def main() -> None:
    config = load_config()
    rows = []
    for mechanism in config["mechanisms"]:
        data = load_condition(mechanism, LEVEL, REPEAT)
        for k in K_VALUES:
            filled = impute(data["corrupted"], data["train_idx"], "knn", config, n_neighbors=k)
            scores = score_imputation(
                filled, data["truth"], data["mask"], data["train_idx"], data["test_idx"]
            )
            train_scores = scores[scores["split"] == "train"]
            for _, r in train_scores.iterrows():
                rows.append({"mechanism": mechanism, "k": k, "column": r["column"], "nrmse": r["nrmse"]})
            print(f"{mechanism:<4} k={k:<3} mean nrmse = {train_scores['nrmse'].mean():.3f}", flush=True)

    table = pd.DataFrame(rows)
    table = table.pivot_table(index=["mechanism", "k"], columns="column", values="nrmse", sort=False)
    table["mean_nrmse"] = table.mean(axis=1)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.reset_index().to_csv(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
