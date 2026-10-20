"""Pytest suite for missing data sweep generation and Member 2 handoff loader.

Covers:
- Integrity check (b): Non-missingness columns preserved, non-NaN values unmodified,
  exact round(rate * n_split) NaN counts across all 75 conditions.
- Seed distinctness (c): All 150 seeds in manifest are unique.
- Loader functionality: list_conditions() and load_condition() return expected keys,
  shapes, masks, and split indices, and raise appropriate errors for invalid inputs.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.handoff import list_conditions, load_condition
from src.sweep import load_config


@pytest.fixture(scope="session")
def config():
    return load_config(str(PROJECT_ROOT / "config.yaml"))


@pytest.fixture(scope="session")
def manifest():
    return list_conditions(str(PROJECT_ROOT / "config.yaml"))


@pytest.fixture(scope="session")
def ground_truth(config):
    gen_dir = PROJECT_ROOT / config.get("generated_missing_dir", "data/generated_missing")
    gt_path = gen_dir / "ground_truth.csv.gz"
    return pd.read_csv(gt_path, index_col="row_index")


@pytest.fixture(scope="session")
def splits(config):
    gen_dir = PROJECT_ROOT / config.get("generated_missing_dir", "data/generated_missing")
    split_path = gen_dir / "split_indices.csv"
    split_df = pd.read_csv(split_path)
    train_idx = split_df.loc[split_df["split"] == "train", "row_index"].to_numpy()
    test_idx = split_df.loc[split_df["split"] == "test", "row_index"].to_numpy()
    return train_idx, test_idx


def test_seed_distinctness_in_manifest(manifest):
    """Confirm all 150 seeds (75 train + 75 test) in the manifest are unique (Check c)."""
    assert len(manifest) == 75, f"Expected 75 manifest rows, got {len(manifest)}"
    train_seeds = manifest["seed_train"].tolist()
    test_seeds = manifest["seed_test"].tolist()
    all_seeds = train_seeds + test_seeds
    assert len(all_seeds) == 150, f"Expected 150 seeds, got {len(all_seeds)}"
    unique_seeds = set(all_seeds)
    assert len(unique_seeds) == 150, f"Expected 150 distinct seeds, found {len(unique_seeds)}"


def test_all_conditions_data_integrity(config, manifest, ground_truth, splits):
    """Verify data integrity for all 75 conditions (Check b)."""
    train_idx, test_idx = splits
    n_train = len(train_idx)
    n_test = len(test_idx)

    missingness_cols = config["missingness_columns"]
    non_missing_cols = [c for c in ground_truth.columns if c not in missingness_cols]

    for _, row in manifest.iterrows():
        mech = row["mechanism"]
        lvl = float(row["level"])
        rep = int(row["repeat"])

        data = load_condition(mech, lvl, rep)
        corr = data["corrupted"]

        # 1. Non-missingness columns (including always_observed) match truth exactly
        assert (corr[non_missing_cols] == ground_truth[non_missing_cols]).all().all(), (
            f"Non-missingness columns modified in condition ({mech}, {lvl}, {rep})"
        )

        # 2. Non-NaN values in the 4 missingness columns equal truth exactly
        for col in missingness_cols:
            not_na = ~corr[col].isna()
            assert (corr.loc[not_na, col] == ground_truth.loc[not_na, col]).all(), (
                f"Non-NaN values altered in column '{col}' for condition ({mech}, {lvl}, {rep})"
            )

        # 3. Exact NaN counts per split: round(rate * n_split)
        exp_train_nans = int(round(lvl * n_train))
        exp_test_nans = int(round(lvl * n_test))

        for col in missingness_cols:
            train_nans = corr.loc[train_idx, col].isna().sum()
            test_nans = corr.loc[test_idx, col].isna().sum()
            assert train_nans == exp_train_nans, (
                f"Expected {exp_train_nans} train NaNs for {col} ({mech}, {lvl}, {rep}), got {train_nans}"
            )
            assert test_nans == exp_test_nans, (
                f"Expected {exp_test_nans} test NaNs for {col} ({mech}, {lvl}, {rep}), got {test_nans}"
            )


def test_handoff_loader(config):
    """Test handoff loader functions: list_conditions and load_condition."""
    manifest = list_conditions()
    assert isinstance(manifest, pd.DataFrame)
    assert len(manifest) == 75
    expected_cols = [
        "mechanism", "level", "repeat", "path", "seed_train", "seed_test",
        "train_nan_sqft_living", "train_nan_grade", "train_nan_bathrooms", "train_nan_yr_built",
        "test_nan_sqft_living", "test_nan_grade", "test_nan_bathrooms", "test_nan_yr_built",
        "train_no_missing_fraction", "test_no_missing_fraction",
    ]
    for col in expected_cols:
        assert col in manifest.columns, f"Expected column '{col}' in manifest"

    # Test load_condition on specific condition
    cond = load_condition("MNAR", 0.30, 2)
    assert isinstance(cond, dict)
    for key in ["corrupted", "truth", "mask", "train_idx", "test_idx"]:
        assert key in cond, f"Missing key '{key}' in load_condition output"

    corrupted = cond["corrupted"]
    truth = cond["truth"]
    mask = cond["mask"]
    train_idx = cond["train_idx"]
    test_idx = cond["test_idx"]

    assert corrupted.shape == (21436, 17)
    assert truth.shape == (21436, 17)
    assert mask.shape == (21436, 4)
    assert len(train_idx) == 17148
    assert len(test_idx) == 4288

    # Mask correctness
    assert mask.columns.tolist() == config["missingness_columns"]
    assert mask.equals(corrupted[mask.columns].isna())
    assert corrupted.drop(columns=mask.columns).isna().sum().sum() == 0

    # Error handling for invalid arguments
    with pytest.raises(ValueError, match="Unknown mechanism"):
        load_condition("INVALID_MECH", 0.10, 0)

    with pytest.raises(ValueError, match="Invalid level"):
        load_condition("MCAR", 0.99, 0)

    with pytest.raises(ValueError, match="Invalid repeat"):
        load_condition("MCAR", 0.10, 99)
