"""Statistical recovery and OLS inference module.

Evaluates statistical parameter recovery on the full dataset without train/test splits.
Provides OLS regression fitting, classical inference diagnostics, correlation matrices,
and comparative statistical recovery metrics against baseline ground truth.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import yaml


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


def get_model_spec(config: Optional[dict] = None) -> Tuple[str, List[str], List[str], float]:
    """Extract model response, predictors, missingness columns, and alpha from config.

    Args:
        config: Configuration dictionary (loaded from config.yaml if None).

    Returns:
        Tuple of (response_col, predictors_list, missingness_cols_list, alpha).
    """
    if config is None:
        config = load_config()

    response = config.get("log_target_name", "log_price")
    target_raw = config.get("target", "price")
    expected_cols = list(config.get("expected_columns", []))

    # 15 predictors = expected_columns excluding price and log_price
    predictors = [c for c in expected_cols if c not in [target_raw, response]]
    missingness_cols = list(config.get("missingness_columns", []))
    alpha = float(config.get("alpha", 0.05))

    return response, predictors, missingness_cols, alpha


def fit_ols(df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
    """Fit OLS regression with intercept on log_price using statsmodels.

    Predictors are the 15 features defined in config.expected_columns (excluding
    price and log_price). Uses classical standard errors.

    Args:
        df: Input DataFrame containing response and all 15 predictors.
        config: Configuration dictionary (loaded from config.yaml if None).

    Returns:
        pd.DataFrame: Table with one row per coefficient (including intercept 'const'):
            - 'coef': Estimated parameter value
            - 'se': Classical standard error
            - 'ci_low': Lower bound of 95% (or 1-alpha) confidence interval
            - 'ci_high': Upper bound of 95% (or 1-alpha) confidence interval
            - 'p_value': Two-tailed p-value
            - 'significant': Boolean indicator (p_value < alpha)

    Raises:
        ValueError: If df contains any NaN values in response or predictors.
    """
    response, predictors, _, alpha = get_model_spec(config)

    needed_cols = [response] + predictors
    for col in needed_cols:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing from DataFrame.")

    if df[needed_cols].isna().any().any():
        nan_counts = df[needed_cols].isna().sum()
        cols_with_nan = nan_counts[nan_counts > 0].to_dict()
        raise ValueError(f"DataFrame contains NaN values in model columns: {cols_with_nan}")

    y = df[response].astype(float)
    X = df[predictors].astype(float)
    X = sm.add_constant(X, has_constant="add")

    model = sm.OLS(y, X)
    results = model.fit()

    ci = results.conf_int(alpha=alpha)

    table = pd.DataFrame(
        {
            "coef": results.params,
            "se": results.bse,
            "ci_low": ci[0],
            "ci_high": ci[1],
            "p_value": results.pvalues,
            "significant": (results.pvalues < alpha),
        },
        index=results.params.index,
    )
    table.index.name = "variable"

    # Store fit attributes for recovery consumers
    table.attrs["rsquared"] = float(results.rsquared)
    table.attrs["nobs"] = int(results.nobs)
    table.attrs["condition_number"] = float(results.condition_number)

    return table


def compute_model_diagnostics(
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    """Compute OLS fit diagnostics including R2, condition number, and VIFs."""
    response, predictors, _, alpha = get_model_spec(config)
    y = df[response].astype(float)
    X = df[predictors].astype(float)
    X = sm.add_constant(X, has_constant="add")

    model = sm.OLS(y, X)
    results = model.fit()

    vifs: Dict[str, float] = {}
    for i, col in enumerate(X.columns):
        if col != "const":
            vifs[col] = float(variance_inflation_factor(X.values, i))

    max_vif_col = max(vifs, key=vifs.get)

    return {
        "nobs": int(results.nobs),
        "rsquared": float(results.rsquared),
        "condition_number": float(results.condition_number),
        "vifs": vifs,
        "max_vif": float(vifs[max_vif_col]),
        "max_vif_variable": str(max_vif_col),
    }


def reference_correlations(
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Pearson and Spearman correlation matrices over 15 predictors plus log_price.

    Args:
        df: Input DataFrame.
        config: Optional configuration dictionary.

    Returns:
        Tuple of (pearson_df, spearman_df), both (16 x 16) symmetric DataFrames.
    """
    response, predictors, _, _ = get_model_spec(config)
    corr_cols = [response] + predictors

    for col in corr_cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found for correlation computation.")

    sub_df = df[corr_cols].astype(float)
    pearson_corr = sub_df.corr(method="pearson")
    spearman_corr = sub_df.corr(method="spearman")

    return pearson_corr, spearman_corr


def statistical_recovery(
    df_other: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, int]],
    reference_table: pd.DataFrame,
    reference_corr: Union[Tuple[pd.DataFrame, pd.DataFrame], Dict[str, pd.DataFrame]],
    config: Optional[dict] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compare a recovered/imputed dataset against reference OLS and correlation estimates.

    Args:
        df_other: Imputed/complete-case DataFrame, or precomputed tuple
                  (other_table, other_pearson, other_spearman, rsquared, nobs).
        reference_table: Reference OLS table from clean baseline data.
        reference_corr: Tuple of (ref_pearson, ref_spearman) or dict with 'pearson' and 'spearman'.
        config: Optional configuration dictionary.

    Returns:
        Tuple of (per_coefficient_table, summary_dict):
            - per_coefficient_table: DataFrame indexed by coefficient with comparison metrics.
            - summary_dict: Dictionary of aggregate recovery metrics across all 15 predictors,
                            the 4 affected missingness columns, correlation errors, and fit stats.
    """
    response, predictors, missingness_cols, alpha = get_model_spec(config)

    # Unpack reference correlations
    if isinstance(reference_corr, dict):
        ref_p = reference_corr["pearson"]
        ref_s = reference_corr["spearman"]
    elif isinstance(reference_corr, (tuple, list)):
        ref_p, ref_s = reference_corr[0], reference_corr[1]
    else:
        raise TypeError("reference_corr must be a tuple/list or dict containing pearson and spearman matrices.")

    # Fit other or use precomputed stats
    if isinstance(df_other, pd.DataFrame):
        table_other = fit_ols(df_other, config)
        corr_other_p, corr_other_s = reference_correlations(df_other, config)
        r_squared_other = float(table_other.attrs.get("rsquared", np.nan))
        n_rows_used = int(len(df_other))
    elif isinstance(df_other, (tuple, list)):
        table_other, corr_other_p, corr_other_s, r_squared_other, n_rows_used = df_other
    else:
        raise TypeError("df_other must be a pd.DataFrame or precomputed tuple.")

    # -------------------------------------------------------------------------
    # 1. Per-coefficient comparison metrics
    # -------------------------------------------------------------------------
    rows = []
    all_vars = list(reference_table.index)

    for var in all_vars:
        c_ref = float(reference_table.loc[var, "coef"])
        se_ref = float(reference_table.loc[var, "se"])
        ci_low_ref = float(reference_table.loc[var, "ci_low"])
        ci_high_ref = float(reference_table.loc[var, "ci_high"])
        p_ref = float(reference_table.loc[var, "p_value"])
        sig_ref = bool(reference_table.loc[var, "significant"])

        c_oth = float(table_other.loc[var, "coef"])
        se_oth = float(table_other.loc[var, "se"])
        ci_low_oth = float(table_other.loc[var, "ci_low"])
        ci_high_oth = float(table_other.loc[var, "ci_high"])
        p_oth = float(table_other.loc[var, "p_value"])
        sig_oth = bool(table_other.loc[var, "significant"])

        # Metric formulas specified in prompt:
        # z_diff = (coef_other - coef_ref) / se_ref
        z_diff = (c_oth - c_ref) / se_ref if se_ref > 0 else 0.0

        # se_ratio = se_other / se_ref
        se_ratio = se_oth / se_ref if se_ref > 0 else np.nan

        # ci_covers_ref = reference coefficient lies inside other CI
        ci_covers_ref = bool(ci_low_oth <= c_ref <= ci_high_oth)

        # ci_overlap_frac = length of CI intersection / length of reference CI (0 if none)
        width_ref = ci_high_ref - ci_low_ref
        width_oth = ci_high_oth - ci_low_oth

        inter_low = max(ci_low_ref, ci_low_oth)
        inter_high = min(ci_high_ref, ci_high_oth)
        inter_len = max(0.0, inter_high - inter_low)
        ci_overlap_frac = inter_len / width_ref if width_ref > 0 else 0.0

        # ci_width_ratio = width_other / width_ref
        ci_width_ratio = width_oth / width_ref if width_ref > 0 else np.nan

        # sig_flip = significance differs from the reference
        sig_flip = bool(sig_oth != sig_ref)

        # sign_flip = coefficient sign differs from the reference
        sign_flip = bool(np.sign(c_oth) != np.sign(c_ref))

        # neglog10p_diff = |-log10(p_other) - (-log10(p_ref))| with p floored at 1e-300
        p_oth_floor = max(p_oth, 1e-300)
        p_ref_floor = max(p_ref, 1e-300)
        neglog10p_diff = float(abs(-np.log10(p_oth_floor) - (-np.log10(p_ref_floor))))

        rows.append(
            {
                "variable": var,
                "coef_other": c_oth,
                "se_other": se_oth,
                "ci_low_other": ci_low_oth,
                "ci_high_other": ci_high_oth,
                "p_value_other": p_oth,
                "significant_other": sig_oth,
                "coef_ref": c_ref,
                "se_ref": se_ref,
                "ci_low_ref": ci_low_ref,
                "ci_high_ref": ci_high_ref,
                "p_value_ref": p_ref,
                "significant_ref": sig_ref,
                "z_diff": float(z_diff),
                "se_ratio": float(se_ratio),
                "ci_covers_ref": bool(ci_covers_ref),
                "ci_overlap_frac": float(ci_overlap_frac),
                "ci_width_ratio": float(ci_width_ratio),
                "sig_flip": bool(sig_flip),
                "sign_flip": bool(sign_flip),
                "neglog10p_diff": float(neglog10p_diff),
            }
        )

    per_coef_table = pd.DataFrame(rows).set_index("variable")

    # -------------------------------------------------------------------------
    # 2. Correlation error metrics
    # -------------------------------------------------------------------------
    corr_cols = [response] + predictors
    all_pairs = []
    affected_pairs = []

    for i in range(len(corr_cols)):
        for j in range(i + 1, len(corr_cols)):
            c1, c2 = corr_cols[i], corr_cols[j]
            all_pairs.append((c1, c2))
            if c1 in missingness_cols or c2 in missingness_cols:
                affected_pairs.append((c1, c2))

    diffs_p_all = [abs(float(corr_other_p.loc[c1, c2]) - float(ref_p.loc[c1, c2])) for c1, c2 in all_pairs]
    diffs_s_all = [abs(float(corr_other_s.loc[c1, c2]) - float(ref_s.loc[c1, c2])) for c1, c2 in all_pairs]

    diffs_p_aff = [abs(float(corr_other_p.loc[c1, c2]) - float(ref_p.loc[c1, c2])) for c1, c2 in affected_pairs]
    diffs_s_aff = [abs(float(corr_other_s.loc[c1, c2]) - float(ref_s.loc[c1, c2])) for c1, c2 in affected_pairs]

    # -------------------------------------------------------------------------
    # 3. Aggregate summary metrics (15 predictors & 4 missingness columns)
    # -------------------------------------------------------------------------
    t_pred = per_coef_table.loc[predictors]
    t_aff = per_coef_table.loc[missingness_cols]

    summary_dict: Dict[str, Any] = {
        # Over all 15 predictors:
        "mean_abs_z_diff": float(np.mean(np.abs(t_pred["z_diff"]))),
        "max_abs_z_diff": float(np.max(np.abs(t_pred["z_diff"]))),
        "median_se_ratio": float(np.median(t_pred["se_ratio"])),
        "share_ci_covers_ref": float(np.mean(t_pred["ci_covers_ref"])),
        "mean_ci_overlap_frac": float(np.mean(t_pred["ci_overlap_frac"])),
        "n_sig_flips": int(np.sum(t_pred["sig_flip"])),
        "n_sign_flips": int(np.sum(t_pred["sign_flip"])),
        "mean_neglog10p_diff": float(np.mean(t_pred["neglog10p_diff"])),
        # Restricted to the 4 missingness columns (_affected):
        "mean_abs_z_diff_affected": float(np.mean(np.abs(t_aff["z_diff"]))),
        "max_abs_z_diff_affected": float(np.max(np.abs(t_aff["z_diff"]))),
        "median_se_ratio_affected": float(np.median(t_aff["se_ratio"])),
        "share_ci_covers_ref_affected": float(np.mean(t_aff["ci_covers_ref"])),
        "mean_ci_overlap_frac_affected": float(np.mean(t_aff["ci_overlap_frac"])),
        "n_sig_flips_affected": int(np.sum(t_aff["sig_flip"])),
        "n_sign_flips_affected": int(np.sum(t_aff["sign_flip"])),
        "mean_neglog10p_diff_affected": float(np.mean(t_aff["neglog10p_diff"])),
        # Correlation metrics:
        "mean_abs_corr_diff_pearson": float(np.mean(diffs_p_all)),
        "mean_abs_corr_diff_spearman": float(np.mean(diffs_s_all)),
        "mean_abs_corr_diff_pearson_affected": float(np.mean(diffs_p_aff)),
        "mean_abs_corr_diff_spearman_affected": float(np.mean(diffs_s_aff)),
        # Fit stats:
        "r_squared_other": float(r_squared_other),
        "n_rows_used": int(n_rows_used),
    }

    return per_coef_table, summary_dict


def complete_case_recovery(
    df_corrupted: pd.DataFrame,
    reference_table: pd.DataFrame,
    reference_corr: Union[Tuple[pd.DataFrame, pd.DataFrame], Dict[str, pd.DataFrame]],
    config: Optional[dict] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Evaluate complete-case analysis (listwise deletion) against the reference model.

    Drops all rows containing any NaN values across model features, fits the OLS
    regression on remaining rows, and returns statistical recovery metrics.

    Args:
        df_corrupted: Corrupted DataFrame with missing values.
        reference_table: Baseline reference OLS table.
        reference_corr: Baseline correlation matrices.
        config: Optional configuration dictionary.

    Returns:
        Tuple of (per_coefficient_table, summary_dict).
    """
    response, predictors, _, _ = get_model_spec(config)
    needed_cols = [response] + predictors

    df_cc = df_corrupted.dropna(subset=needed_cols).copy()

    return statistical_recovery(
        df_other=df_cc,
        reference_table=reference_table,
        reference_corr=reference_corr,
        config=config,
    )
