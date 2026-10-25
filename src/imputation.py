"""Imputation module (Member 2).

Fills the NaNs that Member 1 injected into the 4 missingness columns.

Leakage rules followed by every method:
- The imputer is FIT on the TRAIN rows only, then used to fill train AND test rows.
- By default the target (price, log_price) is never used as an imputer input.
  With use_target=True, log_price IS used as an input; those files are for the
  statistical track ONLY and must never be fed to the prediction models.
- Ground truth is never used here; it is only used later for scoring.
- Observed cells are never changed; only NaN cells are filled.

Methods:
- mean: each missing value is replaced by the train-set mean of its column.
- knn: each missing value is the average of the 5 most similar TRAIN houses
  (similarity measured on standardized columns).
- regression: iterative regression imputation (scikit-learn IterativeImputer with
  its default BayesianRidge linear model). Each missing column is predicted from the
  others, cycling until the fills stabilize. Deterministic (no random noise added).
- random_forest: same iterative scheme with a random forest as the model
  (similar in spirit to missForest).

Regression and random-forest fills are clipped to the observed train range of each
column, so they cannot produce impossible values such as negative bathrooms.
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (required import)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
import yaml

from src.evaluation import evaluate_imputation

KNN_NEIGHBORS = 5
ITERATIVE_MAX_ITER = 10
RF_N_ESTIMATORS = 50
RF_MIN_SAMPLES_LEAF = 3
RF_MAX_ITER = 5


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration YAML file from path or project root."""
    path = Path(config_path)
    if not path.is_file():
        project_root = Path(__file__).resolve().parent.parent
        path = project_root / config_path
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def feature_columns(df: pd.DataFrame, config: dict, use_target: bool = False) -> List[str]:
    """Columns an imputer may look at.

    Default: everything except price and log_price.
    use_target=True: also log_price (price itself is left out, it is redundant with it).
    """
    excluded = {config["target"]} if use_target else {config["target"], config["log_target_name"]}
    return [c for c in df.columns if c not in excluded]


class ScaledKNNImputer:
    """KNN imputation on standardized columns.

    Without scaling, distance would be dominated by large-unit columns such as
    sqft_lot (tens of thousands) and ignore small ones such as bathrooms. The scaler
    is fit on train rows only (it ignores NaN), and results are converted back to
    original units afterwards.
    """

    def __init__(self, n_neighbors: int = KNN_NEIGHBORS):
        self.scaler = StandardScaler()
        self.knn = KNNImputer(n_neighbors=n_neighbors)

    def fit(self, X: pd.DataFrame) -> "ScaledKNNImputer":
        self.knn.fit(self.scaler.fit_transform(X))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.scaler.inverse_transform(self.knn.transform(self.scaler.transform(X)))


def _make_imputer(method: str, train: pd.DataFrame, seed: int, n_neighbors: int = KNN_NEIGHBORS):
    """Return an unfitted imputer for the given method name.

    Args:
        method: 'mean', 'knn', 'regression' or 'random_forest'.
        train: Train rows of the imputer input (used only for the clipping range).
        seed: Random seed for the random forest.
        n_neighbors: k for KNN.
    """
    if method == "mean":
        return SimpleImputer(strategy="mean")
    if method == "knn":
        return ScaledKNNImputer(n_neighbors=n_neighbors)

    col_min = train.min().to_numpy(dtype=float)
    col_max = train.max().to_numpy(dtype=float)
    if method == "regression":
        return IterativeImputer(
            max_iter=ITERATIVE_MAX_ITER,
            min_value=col_min,
            max_value=col_max,
            random_state=seed,
        )
    if method == "random_forest":
        forest = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            n_jobs=-1,
            random_state=seed,
        )
        return IterativeImputer(
            estimator=forest,
            max_iter=RF_MAX_ITER,
            min_value=col_min,
            max_value=col_max,
            random_state=seed,
        )
    raise ValueError(f"Unknown imputation method: '{method}'")


def impute(
    corrupted: pd.DataFrame,
    train_idx: np.ndarray,
    method: str,
    config: Optional[dict] = None,
    use_target: bool = False,
    n_neighbors: int = KNN_NEIGHBORS,
) -> pd.DataFrame:
    """Fill the missing values of one corrupted dataset.

    Args:
        corrupted: Corrupted DataFrame from load_condition (index = row_index).
        train_idx: Row indices of the train split; the imputer is fit on these only.
        method: Imputation method name ('mean', 'knn', 'regression', 'random_forest').
        config: Configuration dictionary (loaded from config.yaml if None).
        use_target: If True, log_price is an imputer input (STATISTICS ONLY output).
        n_neighbors: k for KNN (ignored by other methods).

    Returns:
        pd.DataFrame: Same shape, index and column order as `corrupted`, with no NaN
        in the missingness columns. All other cells are identical to `corrupted`.
    """
    if config is None:
        config = load_config()

    missingness_cols = list(config["missingness_columns"])
    features = feature_columns(corrupted, config, use_target)
    train = corrupted.loc[train_idx, features]

    imputer = _make_imputer(method, train, int(config["base_seed"]), n_neighbors)
    imputer.fit(train)
    filled_values = pd.DataFrame(
        imputer.transform(corrupted[features]),
        index=corrupted.index,
        columns=features,
    )

    # Copy back only the cells that were missing, so observed values stay byte-identical
    filled = corrupted.copy()
    for col in missingness_cols:
        nan_rows = filled[col].isna()
        filled.loc[nan_rows, col] = filled_values.loc[nan_rows, col]

    assert not filled[missingness_cols].isna().any().any(), "Imputation left NaN values"
    return filled


def score_imputation(
    filled: pd.DataFrame,
    truth: pd.DataFrame,
    mask: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> pd.DataFrame:
    """Compare filled values with the true values, on the imputed cells only.

    Args:
        filled: Imputed DataFrame.
        truth: Ground-truth DataFrame (SCORING ONLY).
        mask: Boolean DataFrame, True where a value had been removed.
        train_idx: Row indices of the train split.
        test_idx: Row indices of the test split.

    Returns:
        pd.DataFrame: One row per (column, split) with n_imputed, rmse, mae, r2,
        nrmse (rmse divided by the column's true standard deviation, so columns with
        different units can be compared; nrmse = 1 means "as bad as guessing the mean"),
        bias (mean of filled minus true; negative = fills are too low) and
        sd_ratio (sd of the filled values / sd of the true values at those cells;
        below 1 means the method shrinks the natural spread of the data).
    """
    rows: List[Dict] = []
    for col in mask.columns:
        col_sd = float(truth[col].std())
        for split_name, idx in [("train", train_idx), ("test", test_idx), ("all", truth.index)]:
            cells = mask.loc[idx, col]
            cells = cells[cells].index
            true_vals = truth.loc[cells, col]
            filled_vals = filled.loc[cells, col]
            metrics = evaluate_imputation(true_vals, filled_vals)
            true_sd = float(true_vals.std()) if len(cells) > 1 else np.nan
            rows.append(
                {
                    "column": col,
                    "split": split_name,
                    "n_imputed": len(cells),
                    **metrics,
                    "nrmse": metrics["rmse"] / col_sd if col_sd > 0 else np.nan,
                    "bias": float((filled_vals - true_vals).mean()) if len(cells) else np.nan,
                    "sd_ratio": float(filled_vals.std()) / true_sd if true_sd else np.nan,
                }
            )
    return pd.DataFrame(rows)
