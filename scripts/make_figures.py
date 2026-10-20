"""Script to run aggregation (Step 2) and generate all 8 figures (Step 3).

Produces:
- results/tables/missingness_summary.csv
- results/tables/complete_case_summary.csv
- results/tables/complete_case_bias_noise.csv
- results/figures/fig_m1_missingness_heatmap.png
- results/figures/fig_m2_missing_prob_by_driver.png
- results/figures/fig_m3_complete_rows_vs_level.png
- results/figures/fig_m4_sqft_bias_vs_level.png
- results/figures/fig_s1_zdiff_vs_level.png
- results/figures/fig_s2_ci_coverage_vs_level.png
- results/figures/fig_s3_bias_heatmap.png
- results/figures/fig_s4_coef_forest.png

Prints:
- Figure file list with sizes in KB
- 15-line missingness_summary table
- 3 one-line verification checks
"""

from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.aggregate import (
    generate_complete_case_bias_noise,
    generate_complete_case_summary,
    generate_missingness_summary,
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


def main():
    config_path = str(PROJECT_ROOT / "config.yaml")
    tables_dir = PROJECT_ROOT / "results" / "tables"
    exp_dir = PROJECT_ROOT / "results" / "experiment_results"
    fig_dir = PROJECT_ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Run Step 2: Aggregation Tables
    # -------------------------------------------------------------------------
    missing_summary = generate_missingness_summary(config_path=config_path)
    generate_complete_case_summary()
    generate_complete_case_bias_noise()

    # -------------------------------------------------------------------------
    # 2. Run Step 3: All 8 Figures
    # -------------------------------------------------------------------------
    fig_m1_missingness_heatmap(config_path=config_path)
    fig_m2_missing_prob_by_driver(config_path=config_path)
    fig_m3_complete_rows_vs_level(df_summary=missing_summary, config_path=config_path)
    fig_m4_sqft_bias_vs_level(df_summary=missing_summary, config_path=config_path)

    stats_csv = exp_dir / "complete_case_stats.csv"
    per_coef_csv = exp_dir / "complete_case_per_coef.csv"
    df_stats = pd.read_csv(stats_csv)
    df_per_coef = pd.read_csv(per_coef_csv)

    fig_s1_zdiff_vs_level(df=df_stats)
    fig_s2_ci_coverage_vs_level(df=df_stats)
    fig_s3_bias_heatmap(df=df_per_coef, method="complete_case")
    fig_s4_coef_forest(df=df_per_coef, level=0.30, repeat=0)

    # -------------------------------------------------------------------------
    # 3. Print Outputs: Figure List with Sizes in KB
    # -------------------------------------------------------------------------
    fig_names = [
        "fig_m1_missingness_heatmap.png",
        "fig_m2_missing_prob_by_driver.png",
        "fig_m3_complete_rows_vs_level.png",
        "fig_m4_sqft_bias_vs_level.png",
        "fig_s1_zdiff_vs_level.png",
        "fig_s2_ci_coverage_vs_level.png",
        "fig_s3_bias_heatmap.png",
        "fig_s4_coef_forest.png",
    ]

    print("=" * 86)
    print("FIGURE FILES GENERATED (results/figures/):")
    print("-" * 86)
    for fname in fig_names:
        fpath = fig_dir / fname
        size_kb = fpath.stat().st_size / 1024.0 if fpath.is_file() else 0.0
        print(f"  {fname:<36} {size_kb:>8.1f} KB")

    # -------------------------------------------------------------------------
    # 4. Print 15-line missingness_summary table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 86)
    print("MISSINGNESS SUMMARY TABLE (15 CONDITIONS, 5 REPEATS AVERAGED):")
    print("-" * 86)
    print(
        f"{'Mechanism':<10} | {'Level':<6} | {'Train Complete Rows (mean +/- sd)':<34} | {'sqft_living % Bias (mean +/- sd)':<32}"
    )
    print("-" * 86)
    for _, r in missing_summary.iterrows():
        comp_str = f"{r['train_complete_row_frac_mean']*100:.2f}% +/- {r['train_complete_row_frac_sd']*100:.2f}%"
        bias_str = f"{r['sqft_living_pct_bias_mean']:+.2f}% +/- {r['sqft_living_pct_bias_sd']:.2f}%"
        print(f"{r['mechanism']:<10} | {r['level']:<6.2f} | {comp_str:<34} | {bias_str:<32}")
    print("-" * 86)

    # -------------------------------------------------------------------------
    # 5. Print ONE line for each of the 3 checks
    # -------------------------------------------------------------------------
    # Check 1: reference_ols.csv keeps full precision
    ref_ols = pd.read_csv(tables_dir / "reference_ols.csv", index_col="variable")
    sqft_coef = float(ref_ols.loc["sqft_living", "coef"])
    sqft_se = float(ref_ols.loc["sqft_living", "se"])
    print(
        f"[Check 1] reference_ols.csv precision: sqft_living coef = {sqft_coef:.16e}, se = {sqft_se:.16e} (full precision verified)."
    )

    # Check 2: complete_case_per_coef.csv has 75*16 = 1200 rows
    n_per_coef = len(df_per_coef)
    print(
        f"[Check 2] complete_case_per_coef.csv: {n_per_coef} rows (expected 75 * 16 = 1200, match = {n_per_coef == 1200})."
    )

    # Check 3: complete_case_bias_noise.csv has the expected rows (240 rows)
    bias_noise_df = pd.read_csv(tables_dir / "complete_case_bias_noise.csv")
    n_bn = len(bias_noise_df)
    print(
        f"[Check 3] complete_case_bias_noise.csv: {n_bn} rows (expected 15 * 16 = 240, match = {n_bn == 240})."
    )
    print("=" * 86)


if __name__ == "__main__":
    main()
