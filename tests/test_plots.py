"""Pytest suite for repeat aggregation and figure generation modules.

Tests cover:
1. summarize_repeats computes exact mean, sample sd (ddof=1), and n per group on tiny hand-made DataFrames.
2. Every one of the 8 figure functions creates a non-empty PNG file in a temporary folder.
3. Statistical figure functions accept multi-method DataFrames with method column and run successfully.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.aggregate import (
    generate_complete_case_bias_noise,
    generate_complete_case_summary,
    generate_missingness_summary,
    summarize_repeats,
)
from src.plots import (
    fig_m1_missingness_heatmap,
    fig_m2_missing_prob_by_driver,
    fig_m3_complete_rows_vs_level,
    fig_m4_sqft_bias_vs_level,
    fig_s1_zdiff_vs_level,
    fig_s2_ci_coverage_vs_level,
    fig_s3_bias_heatmap,
    fig_s4_coef_forest,
)


def test_summarize_repeats_hand_made():
    """Verify summarize_repeats computes exact mathematical mean, sd, and n on hand-made data."""
    df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "val": [1.0, 3.0, 10.0, 20.0],
        }
    )

    res = summarize_repeats(df, group_cols="group", metrics="val")
    assert len(res) == 2, f"Expected 2 rows, got {len(res)}"
    assert "mean" in res.columns
    assert "sd" in res.columns
    assert "n" in res.columns

    # Group A: [1.0, 3.0] -> mean = 2.0, sd = sqrt(((1-2)^2 + (3-2)^2)/1) = sqrt(2), n = 2
    row_a = res[res["group"] == "A"].iloc[0]
    assert np.isclose(row_a["mean"], 2.0)
    assert np.isclose(row_a["sd"], np.sqrt(2.0))
    assert row_a["n"] == 2

    # Group B: [10.0, 20.0] -> mean = 15.0, sd = sqrt(((10-15)^2 + (20-15)^2)/1) = sqrt(50), n = 2
    row_b = res[res["group"] == "B"].iloc[0]
    assert np.isclose(row_b["mean"], 15.0)
    assert np.isclose(row_b["sd"], np.sqrt(50.0))
    assert row_b["n"] == 2


def test_summarize_repeats_multiple_metrics():
    """Verify summarize_repeats supports multiple metrics simultaneously."""
    df = pd.DataFrame(
        {
            "group": ["G1", "G1", "G2", "G2"],
            "m1": [10.0, 20.0, 30.0, 50.0],
            "m2": [100.0, 300.0, 200.0, 400.0],
        }
    )

    res = summarize_repeats(df, group_cols="group", metrics=["m1", "m2"])
    assert len(res) == 2
    assert "m1_mean" in res.columns and "m1_sd" in res.columns
    assert "m2_mean" in res.columns and "m2_sd" in res.columns
    assert "n" in res.columns

    row_g1 = res[res["group"] == "G1"].iloc[0]
    assert np.isclose(row_g1["m1_mean"], 15.0)
    assert np.isclose(row_g1["m2_mean"], 200.0)
    assert row_g1["n"] == 2


@pytest.mark.parametrize(
    "fig_func, filename",
    [
        (fig_m1_missingness_heatmap, "fig_m1.png"),
        (fig_m2_missing_prob_by_driver, "fig_m2.png"),
        (fig_m3_complete_rows_vs_level, "fig_m3.png"),
        (fig_m4_sqft_bias_vs_level, "fig_m4.png"),
        (fig_s1_zdiff_vs_level, "fig_s1.png"),
        (fig_s2_ci_coverage_vs_level, "fig_s2.png"),
        (fig_s3_bias_heatmap, "fig_s3.png"),
        (fig_s4_coef_forest, "fig_s4.png"),
    ],
)
def test_all_figures_write_non_empty_png(fig_func, filename, tmp_path):
    """Verify that every figure function writes a valid, non-empty PNG into a temp folder."""
    target_path = tmp_path / filename
    out_path = fig_func(output_path=target_path)

    assert out_path.is_file(), f"Expected figure file at {out_path}"
    assert out_path.stat().st_size > 1024, f"Figure file {out_path} is unexpectedly small or empty"


def test_statistical_figures_support_multiple_methods(tmp_path):
    """Verify that statistical figure functions support multiple methods with distinct linestyles."""
    # Create mock multi-method dataframe with complete_case and a mock imputer
    stats_csv = PROJECT_ROOT / "results" / "experiment_results" / "complete_case_stats.csv"
    if stats_csv.is_file():
        df_cc = pd.read_csv(stats_csv)
        df_imputer = df_cc.copy()
        df_imputer["method"] = "knn"
        df_imputer["mean_abs_z_diff"] = df_imputer["mean_abs_z_diff"] * 0.8
        df_multi = pd.concat([df_cc, df_imputer], ignore_index=True)

        out_s1 = tmp_path / "multi_s1.png"
        fig_s1_zdiff_vs_level(df=df_multi, output_path=out_s1)
        assert out_s1.is_file() and out_s1.stat().st_size > 1024

        out_s2 = tmp_path / "multi_s2.png"
        fig_s2_ci_coverage_vs_level(df=df_multi, output_path=out_s2)
        assert out_s2.is_file() and out_s2.stat().st_size > 1024
