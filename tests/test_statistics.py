"""Pytest suite for statistical recovery track and OLS inference module.

Tests cover:
1. Comparing the reference with ITSELF produces zero differences, no flips, CI covers = 1.
2. Parameter recovery on synthetic data with known coefficients within a few standard errors.
3. fit_ols strictly raises a ValueError when NaN values are present.
4. Complete-case n_rows_used equals the exact count of rows without any NaN.
5. In-memory smoke test confirming statistical_recovery runs on mean-imputed corrupted condition.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.handoff import load_condition
from src.statistics import (
    complete_case_recovery,
    fit_ols,
    get_model_spec,
    load_config,
    reference_correlations,
    statistical_recovery,
)


@pytest.fixture(scope="session")
def config():
    return load_config(str(PROJECT_ROOT / "config.yaml"))


@pytest.fixture(scope="session")
def clean_data(config):
    clean_path = PROJECT_ROOT / config["clean_data_path"]
    return pd.read_csv(clean_path)


@pytest.fixture(scope="session")
def reference_model(clean_data, config):
    ref_table = fit_ols(clean_data, config)
    ref_p, ref_s = reference_correlations(clean_data, config)
    return ref_table, (ref_p, ref_s)


def test_reference_compared_with_itself(clean_data, reference_model, config):
    """Verify comparing the reference with ITSELF yields zero diffs, no flips, CI covers = 1."""
    ref_table, ref_corr = reference_model

    per_coef, summary = statistical_recovery(
        df_other=clean_data,
        reference_table=ref_table,
        reference_corr=ref_corr,
        config=config,
    )

    assert len(per_coef) == 16, f"Expected 16 coefficient rows, got {len(per_coef)}"

    # All 15 predictors differences must be exactly 0
    assert summary["mean_abs_z_diff"] == 0.0
    assert summary["max_abs_z_diff"] == 0.0
    assert summary["median_se_ratio"] == 1.0
    assert summary["share_ci_covers_ref"] == 1.0
    assert summary["mean_ci_overlap_frac"] == 1.0
    assert summary["n_sig_flips"] == 0
    assert summary["n_sign_flips"] == 0
    assert summary["mean_neglog10p_diff"] == 0.0

    # 4 affected missingness columns differences must also be 0
    assert summary["mean_abs_z_diff_affected"] == 0.0
    assert summary["max_abs_z_diff_affected"] == 0.0
    assert summary["share_ci_covers_ref_affected"] == 1.0
    assert summary["n_sig_flips_affected"] == 0
    assert summary["n_sign_flips_affected"] == 0

    # Correlation diffs must be exactly 0
    assert summary["mean_abs_corr_diff_pearson"] == 0.0
    assert summary["mean_abs_corr_diff_spearman"] == 0.0
    assert summary["mean_abs_corr_diff_pearson_affected"] == 0.0
    assert summary["mean_abs_corr_diff_spearman_affected"] == 0.0

    assert summary["n_rows_used"] == len(clean_data)


def test_fit_ols_recovers_synthetic_coefficients(config):
    """Verify fit_ols recovers known synthetic coefficients within a few standard errors."""
    response, predictors, _, _ = get_model_spec(config)
    rng = np.random.default_rng(12345)
    n_samples = 20000

    # Create synthetic feature matrix
    X_dict = {}
    for col in predictors:
        X_dict[col] = rng.normal(loc=10.0, scale=3.0, size=n_samples)

    synth_df = pd.DataFrame(X_dict)

    # True parameter vector
    true_intercept = 12.5
    true_slopes = {col: float((i + 1) * 0.15) for i, col in enumerate(predictors)}

    # Generate response with Gaussian error
    linear_pred = true_intercept + sum(true_slopes[col] * synth_df[col] for col in predictors)
    synth_df[response] = linear_pred + rng.normal(loc=0.0, scale=0.5, size=n_samples)

    # Add dummy price column so schema matches expected_columns if needed
    synth_df[config.get("target", "price")] = np.exp(synth_df[response])

    fitted_table = fit_ols(synth_df, config)

    # Check intercept recovery
    se_const = fitted_table.loc["const", "se"]
    est_const = fitted_table.loc["const", "coef"]
    assert abs(est_const - true_intercept) <= 4.0 * se_const, (
        f"Intercept recovery error: est={est_const}, true={true_intercept}, se={se_const}"
    )

    # Check slope recovery across all 15 predictors
    for col in predictors:
        se_col = fitted_table.loc[col, "se"]
        est_col = fitted_table.loc[col, "coef"]
        true_col = true_slopes[col]
        assert abs(est_col - true_col) <= 4.0 * se_col, (
            f"Slope recovery error for {col}: est={est_col}, true={true_col}, se={se_col}"
        )


def test_fit_ols_raises_error_on_nan(clean_data, config):
    """Verify fit_ols strictly raises a ValueError when any NaN values are present."""
    # NaN in a predictor
    df_nan_pred = clean_data.copy()
    df_nan_pred.loc[10, "sqft_living"] = np.nan
    with pytest.raises(ValueError, match="DataFrame contains NaN values"):
        fit_ols(df_nan_pred, config)

    # NaN in response
    response, _, _, _ = get_model_spec(config)
    df_nan_resp = clean_data.copy()
    df_nan_resp.loc[25, response] = np.nan
    with pytest.raises(ValueError, match="DataFrame contains NaN values"):
        fit_ols(df_nan_resp, config)


def test_complete_case_n_rows_used(reference_model, config):
    """Verify complete_case_recovery n_rows_used equals the number of rows without any NaN."""
    ref_table, ref_corr = reference_model

    # Test condition 1: MCAR 0.10 repeat 0
    c1 = load_condition("MCAR", 0.10, 0)
    df_corr1 = c1["corrupted"]
    expected_rows_1 = len(df_corr1.dropna())
    _, summary1 = complete_case_recovery(df_corr1, ref_table, ref_corr, config)
    assert summary1["n_rows_used"] == expected_rows_1

    # Test condition 2: MAR 0.30 repeat 1
    c2 = load_condition("MAR", 0.30, 1)
    df_corr2 = c2["corrupted"]
    expected_rows_2 = len(df_corr2.dropna())
    _, summary2 = complete_case_recovery(df_corr2, ref_table, ref_corr, config)
    assert summary2["n_rows_used"] == expected_rows_2


def test_smoke_simple_mean_imputation_statistical_recovery(reference_model, config):
    """Smoke test: apply column-mean imputation strictly in memory to confirm recovery runs."""
    ref_table, ref_corr = reference_model

    c = load_condition("MAR", 0.30, 0)
    corrupted = c["corrupted"].copy()

    # Apply simple column-mean imputation inside test memory only (do not save to disk)
    missing_cols = config["missingness_columns"]
    for col in missing_cols:
        corrupted[col] = corrupted[col].fillna(corrupted[col].mean())

    assert corrupted.isna().sum().sum() == 0, "Imputed dataframe should have 0 NaNs"

    per_coef, summary = statistical_recovery(
        df_other=corrupted,
        reference_table=ref_table,
        reference_corr=ref_corr,
        config=config,
    )

    assert isinstance(per_coef, pd.DataFrame)
    assert len(per_coef) == 16
    assert isinstance(summary, dict)

    expected_keys = [
        "mean_abs_z_diff", "max_abs_z_diff", "median_se_ratio", "share_ci_covers_ref",
        "mean_ci_overlap_frac", "n_sig_flips", "n_sign_flips", "mean_neglog10p_diff",
        "mean_abs_z_diff_affected", "max_abs_z_diff_affected", "median_se_ratio_affected",
        "share_ci_covers_ref_affected", "mean_ci_overlap_frac_affected", "n_sig_flips_affected",
        "n_sign_flips_affected", "mean_neglog10p_diff_affected", "mean_abs_corr_diff_pearson",
        "mean_abs_corr_diff_spearman", "mean_abs_corr_diff_pearson_affected",
        "mean_abs_corr_diff_spearman_affected", "r_squared_other", "n_rows_used",
    ]
    for key in expected_keys:
        assert key in summary, f"Key '{key}' missing from summary dictionary"

    assert summary["n_rows_used"] == len(corrupted)
    assert 0.0 <= summary["r_squared_other"] <= 1.0
