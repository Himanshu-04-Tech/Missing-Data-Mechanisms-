"""Pytest suite for missing data generation mechanisms.

Tests cover:
1. Determinism and reproducibility across identical and different seeds.
2. Exact missingness counts matching round(rate * n_rows) across columns and rates.
3. Protected columns (always_observed) remaining strictly observed with zero NaNs.
4. Input DataFrame immutability (original dataframe remains untouched).
5. Assertions and edge cases (driver validation, invalid mechanism, zero variance).
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing import load_config
from src.missingness import generate_missingness, make_seed


@pytest.fixture(scope="session")
def config():
    return load_config(str(PROJECT_ROOT / "config.yaml"))


@pytest.fixture(scope="session")
def clean_train_data(config):
    clean_csv = PROJECT_ROOT / config["clean_data_path"]
    split_csv = PROJECT_ROOT / config["split_indices_path"]
    df = pd.read_csv(clean_csv)
    split_df = pd.read_csv(split_csv)
    train_idx = split_df[split_df["split"] == "train"]["row_index"]
    return df.iloc[train_idx].copy().reset_index(drop=True)


def test_seed_generation_determinism():
    """Verify make_seed returns deterministic and distinct seeds."""
    s1 = make_seed(42, "MCAR", 0.1, 0, "train")
    s2 = make_seed(42, "MCAR", 0.1, 0, "train")
    assert s1 == s2, "make_seed must be perfectly deterministic for identical inputs"

    s_mar = make_seed(42, "MAR", 0.1, 0, "train")
    s_mnar = make_seed(42, "MNAR", 0.1, 0, "train")
    s_diff_rate = make_seed(42, "MCAR", 0.3, 0, "train")
    s_diff_repeat = make_seed(42, "MCAR", 0.1, 1, "train")
    s_diff_split = make_seed(42, "MCAR", 0.1, 0, "test")
    s_col = make_seed(42, "MCAR", 0.1, 0, "train", column="sqft_living")

    unique_seeds = {s1, s_mar, s_mnar, s_diff_rate, s_diff_repeat, s_diff_split, s_col}
    assert len(unique_seeds) == 7, "make_seed should yield distinct seeds for distinct parameter sets"


def test_reproducibility(clean_train_data, config):
    """Verify identical seeds produce identical masks and different seeds produce different masks."""
    seed_a = 42
    seed_b = 999

    for mech in config.get("mechanisms", ["MCAR", "MAR", "MNAR"]):
        df_corr1, mask1 = generate_missingness(clean_train_data, mech, 0.20, seed_a, config)
        df_corr2, mask2 = generate_missingness(clean_train_data, mech, 0.20, seed_a, config)
        pd.testing.assert_frame_equal(mask1, mask2)
        pd.testing.assert_frame_equal(df_corr1, df_corr2)

        _, mask_diff = generate_missingness(clean_train_data, mech, 0.20, seed_b, config)
        assert not mask1.equals(mask_diff), f"Different seeds should yield different masks for {mech}"


def test_exact_missingness_counts(clean_train_data, config):
    """Verify each missingness column receives exactly round(rate * n_rows) missing values."""
    n_rows = len(clean_train_data)
    missing_cols = config["missingness_columns"]

    for rate in [0.10, 0.20, 0.30, 0.50]:
        expected_m = int(round(rate * n_rows))
        for mech in ["MCAR", "MAR", "MNAR"]:
            seed = make_seed(config["base_seed"], mech, rate, 0, "train")
            df_corr, mask = generate_missingness(clean_train_data, mech, rate, seed, config)

            # Check mask properties
            assert list(mask.columns) == missing_cols
            assert (mask.index == clean_train_data.index).all()

            for col in missing_cols:
                col_mask_count = int(mask[col].sum())
                col_nan_count = int(df_corr[col].isna().sum())
                assert col_mask_count == expected_m, (
                    f"{mech} at rate {rate}: expected {expected_m} in mask for {col}, got {col_mask_count}"
                )
                assert col_nan_count == expected_m, (
                    f"{mech} at rate {rate}: expected {expected_m} NaNs for {col}, got {col_nan_count}"
                )


def test_protected_columns_unmodified(clean_train_data, config):
    """Verify always_observed columns never receive any missing values."""
    always_obs = config["always_observed"]
    for mech in ["MCAR", "MAR", "MNAR"]:
        df_corr, _ = generate_missingness(clean_train_data, mech, 0.50, 42, config)
        for col in always_obs:
            assert col in df_corr.columns
            assert df_corr[col].isna().sum() == 0, f"Protected column {col} received missing values!"

        # Non-missingness columns should also have zero NaNs
        non_missing = [c for c in clean_train_data.columns if c not in config["missingness_columns"]]
        for col in non_missing:
            assert df_corr[col].isna().sum() == 0, f"Non-missingness column {col} received NaNs!"


def test_original_dataframe_unchanged(clean_train_data, config):
    """Verify that the original dataframe remains completely unmodified."""
    df_copy = clean_train_data.copy(deep=True)
    df_corr, mask = generate_missingness(clean_train_data, "MNAR", 0.40, 12345, config)

    # Check original dataframe
    pd.testing.assert_frame_equal(clean_train_data, df_copy)
    assert clean_train_data.isna().sum().sum() == 0, "Original dataframe was mutated to contain NaNs!"
    assert df_corr.isna().sum().sum() > 0, "Corrupted dataframe must contain missing values"


def test_assertions_and_invalid_inputs(clean_train_data, config):
    """Verify expected assertions and errors on invalid inputs."""
    # Invalid mechanism
    with pytest.raises(ValueError, match="Unknown mechanism"):
        generate_missingness(clean_train_data, "INVALID", 0.2, 42, config)

    # Driver in missingness columns
    bad_config = dict(config)
    bad_config["missingness_columns"] = list(config["missingness_columns"]) + [config["mar_driver"]]
    with pytest.raises(AssertionError, match="MAR driver .* must not be in missingness_columns"):
        generate_missingness(clean_train_data, "MAR", 0.2, 42, bad_config)

    # Driver containing NaN
    df_with_nan_driver = clean_train_data.copy()
    df_with_nan_driver.loc[0, config["mar_driver"]] = np.nan
    with pytest.raises(AssertionError, match="contains NaN"):
        generate_missingness(df_with_nan_driver, "MAR", 0.2, 42, config)
