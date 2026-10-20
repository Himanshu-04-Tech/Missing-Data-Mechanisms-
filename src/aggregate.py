"""Repeat-aggregation module for missingness simulation and statistical recovery tracks.

Computes repeat-level aggregations (mean, sample standard deviation, sample size n)
across simulation replications, and produces summary tables for downstream analysis,
reporting, and plotting.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import yaml


def _load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration YAML file from path or project root."""
    path = Path(config_path)
    if not path.is_file():
        project_root = Path(__file__).resolve().parent.parent
        path = project_root / config_path
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def summarize_repeats(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    metrics: Union[str, List[str]],
) -> pd.DataFrame:
    """Summarize simulation replications returning mean, sd, and n per group.

    Args:
        df: Input DataFrame containing grouping columns and metric columns across repeats.
        group_cols: Column name or list of column names to group by.
        metrics: Column name or list of metric column names to summarize.

    Returns:
        pd.DataFrame with one row per unique group combination. Contains:
            - The group columns in original order
            - For each metric '{m}': '{m}_mean' and '{m}_sd'
            - 'n': Count of replications per group
            - If a single metric was provided: also includes direct 'mean' and 'sd' columns.
    """
    if isinstance(group_cols, str):
        g_cols = [group_cols]
    else:
        g_cols = list(group_cols)

    if isinstance(metrics, str):
        m_cols = [metrics]
        single_metric = True
    else:
        m_cols = list(metrics)
        single_metric = len(m_cols) == 1

    for c in g_cols:
        if c not in df.columns:
            raise KeyError(f"Grouping column '{c}' not found in DataFrame.")
    for m in m_cols:
        if m not in df.columns:
            raise KeyError(f"Metric column '{m}' not found in DataFrame.")

    grouped = df.groupby(g_cols, as_index=False, sort=False)

    # Base result with unique group values
    first_metric = m_cols[0]
    counts = grouped[first_metric].count()
    result = counts[g_cols].copy()

    for m in m_cols:
        m_mean = grouped[m].mean()[m]
        m_sd = grouped[m].std()[m]
        result[f"{m}_mean"] = m_mean
        result[f"{m}_sd"] = m_sd

    result["n"] = counts[first_metric]

    if single_metric:
        result["mean"] = result[f"{first_metric}_mean"]
        result["sd"] = result[f"{first_metric}_sd"]

    return result


def generate_missingness_summary(
    config_path: str = "config.yaml",
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Generate and optionally save missingness_summary.csv.

    Computes per mechanism x level mean and sd over the 5 repeats of:
    - Train complete-row fraction (from manifest)
    - Percent bias of observed sqft_living mean vs true train mean.
    """
    project_root = Path(__file__).resolve().parent.parent
    config = _load_config(config_path)

    gen_dir = project_root / config.get("generated_missing_dir", "data/generated_missing")
    manifest_path = gen_dir / "manifest.csv"
    truth_path = gen_dir / "ground_truth.csv.gz"
    split_path = gen_dir / "split_indices.csv"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found at '{manifest_path}'.")
    manifest = pd.read_csv(manifest_path)
    truth = pd.read_csv(truth_path, index_col="row_index")
    splits = pd.read_csv(split_path)

    train_idx = splits.loc[splits["split"] == "train", "row_index"].to_numpy()
    true_train_sqft_mean = float(truth.loc[train_idx, "sqft_living"].mean())

    records = []
    for _, row in manifest.iterrows():
        mech = row["mechanism"]
        lvl = float(row["level"])
        rep = int(row["repeat"])

        pct = int(round(lvl * 100))
        corr_path = gen_dir / mech / f"rate_{pct:02d}" / f"rep_{rep}.csv.gz"
        corr_df = pd.read_csv(corr_path, index_col="row_index")

        obs_train_sqft_mean = float(corr_df.loc[train_idx, "sqft_living"].mean())
        pct_bias = 100.0 * (obs_train_sqft_mean - true_train_sqft_mean) / true_train_sqft_mean
        complete_frac = float(row["train_no_missing_fraction"])

        records.append(
            {
                "mechanism": mech,
                "level": lvl,
                "repeat": rep,
                "train_complete_row_frac": complete_frac,
                "sqft_living_pct_bias": pct_bias,
            }
        )

    records_df = pd.DataFrame(records)
    summary_df = summarize_repeats(
        records_df,
        group_cols=["mechanism", "level"],
        metrics=["train_complete_row_frac", "sqft_living_pct_bias"],
    )

    if output_path is None:
        output_path = project_root / "results" / "tables" / "missingness_summary.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)

    return summary_df


def generate_complete_case_summary(
    stats_csv_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Generate and optionally save complete_case_summary.csv.

    Computes per method x mechanism x level mean and sd of:
    - mean_abs_z_diff
    - mean_abs_z_diff_affected
    - share_ci_covers_ref
    - n_sig_flips
    - mean_abs_corr_diff_pearson
    - n_rows_used
    """
    project_root = Path(__file__).resolve().parent.parent
    if stats_csv_path is None:
        stats_csv_path = project_root / "results" / "experiment_results" / "complete_case_stats.csv"
    stats_csv_path = Path(stats_csv_path)

    if not stats_csv_path.is_file():
        raise FileNotFoundError(f"Stats file not found at '{stats_csv_path}'.")

    stats_df = pd.read_csv(stats_csv_path)
    metrics = [
        "mean_abs_z_diff",
        "mean_abs_z_diff_affected",
        "share_ci_covers_ref",
        "n_sig_flips",
        "mean_abs_corr_diff_pearson",
        "n_rows_used",
    ]
    summary_df = summarize_repeats(
        stats_df,
        group_cols=["method", "mechanism", "level"],
        metrics=metrics,
    )

    if output_path is None:
        output_path = project_root / "results" / "tables" / "complete_case_summary.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)

    return summary_df


def generate_complete_case_bias_noise(
    per_coef_csv_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Generate and optionally save complete_case_bias_noise.csv.

    Computes per method x mechanism x level x coefficient:
    - mean of signed z_diff over repeats (bias)
    - sd of signed z_diff over repeats (noise)
    """
    project_root = Path(__file__).resolve().parent.parent
    if per_coef_csv_path is None:
        per_coef_csv_path = project_root / "results" / "experiment_results" / "complete_case_per_coef.csv"
    per_coef_csv_path = Path(per_coef_csv_path)

    if not per_coef_csv_path.is_file():
        raise FileNotFoundError(f"Per-coefficient stats file not found at '{per_coef_csv_path}'.")

    per_coef_df = pd.read_csv(per_coef_csv_path)
    summary_df = summarize_repeats(
        per_coef_df,
        group_cols=["method", "mechanism", "level", "variable"],
        metrics=["z_diff"],
    )

    summary_df["bias"] = summary_df["z_diff_mean"]
    summary_df["noise"] = summary_df["z_diff_sd"]

    # Reorder columns logically
    cols = [
        "method",
        "mechanism",
        "level",
        "variable",
        "bias",
        "noise",
        "z_diff_mean",
        "z_diff_sd",
        "n",
    ]
    summary_df = summary_df[cols]

    if output_path is None:
        output_path = project_root / "results" / "tables" / "complete_case_bias_noise.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)

    return summary_df
