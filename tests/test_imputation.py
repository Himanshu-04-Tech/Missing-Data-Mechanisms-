"""Pytest suite for the imputation module (Member 2).

Tests cover:
1. Every method fills every NaN and leaves observed cells, rows and columns untouched.
2. No train/test leakage: changing test rows does not change how train rows are filled.
3. No target leakage: price/log_price are ignored unless use_target=True.
4. Mean fills equal the train-set mean.
5. Determinism: the same input always gives the same output.
6. Regression / random-forest fills stay within the observed train range.
7. score_imputation: a perfect fill scores 0 error; mean imputation scores nrmse ~ 1.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.handoff import load_condition
from src.imputation import impute, load_config, score_imputation

METHODS = ["mean", "knn", "regression", "random_forest"]
N_SAMPLE = 3000  # subsample so the slower methods keep the suite fast


@pytest.fixture(scope="session")
def config():
    return load_config(str(PROJECT_ROOT / "config.yaml"))


@pytest.fixture(scope="session")
def condition():
    """A small, fixed subsample of MAR 30% repeat 0."""
    data = load_condition("MAR", 0.30, 0)
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(data["corrupted"].index, size=N_SAMPLE, replace=False))
    train_idx = np.intersect1d(data["train_idx"], rows)
    test_idx = np.intersect1d(data["test_idx"], rows)
    return {
        "corrupted": data["corrupted"].loc[rows],
        "truth": data["truth"].loc[rows],
        "mask": data["mask"].loc[rows],
        "train_idx": train_idx,
        "test_idx": test_idx,
    }


@pytest.fixture(scope="session")
def filled_by_method(condition, config):
    return {
        m: impute(condition["corrupted"], condition["train_idx"], m, config) for m in METHODS
    }


@pytest.mark.parametrize("method", METHODS)
def test_fills_all_and_keeps_observed(method, condition, filled_by_method):
    corrupted = condition["corrupted"]
    filled = filled_by_method[method]
    assert list(filled.columns) == list(corrupted.columns)
    assert filled.index.equals(corrupted.index)
    assert not filled.isna().any().any()
    observed = corrupted.notna().to_numpy()
    assert (filled.to_numpy()[observed] == corrupted.to_numpy()[observed]).all()


@pytest.mark.parametrize("method", ["mean", "knn", "regression"])
def test_no_train_test_leakage(method, condition, config):
    """Scrambling the test rows must not change the fills of train rows."""
    corrupted = condition["corrupted"]
    train_idx, test_idx = condition["train_idx"], condition["test_idx"]
    base = impute(corrupted, train_idx, method, config)

    altered = corrupted.copy()
    for col in ["sqft_lot", "bedrooms", "sqft_living15", "sqft_living"]:
        altered.loc[test_idx, col] = altered.loc[test_idx, col] * 10 + 1
    after = impute(altered, train_idx, method, config)

    pd.testing.assert_frame_equal(base.loc[train_idx], after.loc[train_idx])


@pytest.mark.parametrize("method", ["mean", "knn", "regression"])
def test_target_not_used_by_default(method, condition, config):
    corrupted = condition["corrupted"]
    altered = corrupted.copy()
    altered["price"] = altered["price"].sample(frac=1, random_state=1).to_numpy()
    altered["log_price"] = np.log(altered["price"])
    base = impute(corrupted, condition["train_idx"], method, config)
    after = impute(altered, condition["train_idx"], method, config)
    cols = config["missingness_columns"]
    pd.testing.assert_frame_equal(base[cols], after[cols])


def test_target_used_when_requested(condition, config):
    corrupted = condition["corrupted"]
    altered = corrupted.copy()
    altered["log_price"] = altered["log_price"].sample(frac=1, random_state=1).to_numpy()
    base = impute(corrupted, condition["train_idx"], "regression", config, use_target=True)
    after = impute(altered, condition["train_idx"], "regression", config, use_target=True)
    assert not base["sqft_living"].equals(after["sqft_living"])


def test_mean_fill_equals_train_mean(condition, filled_by_method, config):
    corrupted = condition["corrupted"]
    filled = filled_by_method["mean"]
    for col in config["missingness_columns"]:
        nan_rows = corrupted[col].isna()
        expected = corrupted.loc[condition["train_idx"], col].mean()
        assert np.allclose(filled.loc[nan_rows, col], expected)


@pytest.mark.parametrize("method", METHODS)
def test_deterministic(method, condition, config, filled_by_method):
    again = impute(condition["corrupted"], condition["train_idx"], method, config)
    pd.testing.assert_frame_equal(filled_by_method[method], again)


@pytest.mark.parametrize("method", ["regression", "random_forest"])
def test_fills_within_train_range(method, condition, filled_by_method, config):
    corrupted = condition["corrupted"]
    filled = filled_by_method[method]
    for col in config["missingness_columns"]:
        train_vals = corrupted.loc[condition["train_idx"], col]
        fills = filled.loc[corrupted[col].isna(), col]
        assert fills.min() >= train_vals.min() - 1e-9
        assert fills.max() <= train_vals.max() + 1e-9


def test_score_perfect_and_mean(condition, filled_by_method):
    truth, mask = condition["truth"], condition["mask"]
    tr, te = condition["train_idx"], condition["test_idx"]

    perfect = score_imputation(truth, truth, mask, tr, te)
    assert np.allclose(perfect["rmse"], 0.0)
    assert np.allclose(perfect["bias"], 0.0)

    mcar = load_condition("MCAR", 0.30, 0)
    filled = impute(mcar["corrupted"], mcar["train_idx"], "mean")
    scores = score_imputation(filled, mcar["truth"], mcar["mask"], mcar["train_idx"], mcar["test_idx"])
    all_rows = scores[scores["split"] == "all"]
    assert all_rows["n_imputed"].tolist() == mcar["mask"].sum().tolist()
    assert ((all_rows["nrmse"] > 0.95) & (all_rows["nrmse"] < 1.05)).all()
