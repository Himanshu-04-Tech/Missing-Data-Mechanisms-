"""Build baseline reference OLS model and correlation matrices on clean data.

Saves:
- results/tables/reference_ols.csv
- results/tables/reference_corr_pearson.csv
- results/tables/reference_corr_spearman.csv

Prints:
- n, R-squared, condition number, max VIF (and which predictor)
- Compact reference coefficient table (16 rows)
"""

from pathlib import Path
import sys
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.statistics import (
    compute_model_diagnostics,
    fit_ols,
    load_config,
    reference_correlations,
)


def main():
    config = load_config(str(project_root / "config.yaml"))
    clean_csv_path = project_root / config["clean_data_path"]
    df_clean = pd.read_csv(clean_csv_path)

    # 1. Fit reference OLS table and compute diagnostics
    ref_table = fit_ols(df_clean, config)
    diagnostics = compute_model_diagnostics(df_clean, config)

    # 2. Compute reference correlation matrices
    ref_corr_p, ref_corr_s = reference_correlations(df_clean, config)

    # 3. Save outputs to results/tables/
    out_dir = project_root / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_table.to_csv(out_dir / "reference_ols.csv")
    ref_corr_p.to_csv(out_dir / "reference_corr_pearson.csv")
    ref_corr_s.to_csv(out_dir / "reference_corr_spearman.csv")

    # 4. Print requested diagnostics and compact 16-row reference table
    print("=" * 86)
    print("BASELINE REFERENCE OLS MODEL & DIAGNOSTICS")
    print("=" * 86)
    print(f"Sample size (n):         {diagnostics['nobs']:,}")
    print(f"R-squared:               {diagnostics['rsquared']:.4f}")
    print(f"Condition number:        {diagnostics['condition_number']:.2e}")
    print(f"Max VIF:                 {diagnostics['max_vif']:.2f} ({diagnostics['max_vif_variable']})")
    print("\nReference Coefficient Table (16 coefficients):")
    print("-" * 86)
    print(f"{'Variable':<16} | {'Coef':>10} | {'Std.Err':>8} | {'95% CI Low':>11} | {'95% CI High':>11} | {'p-value':>10} | {'Sig (alpha=0.05)':<16}")
    print("-" * 86)

    for var, row in ref_table.iterrows():
        p_val_str = f"{row['p_value']:.2e}" if row['p_value'] < 1e-4 else f"{row['p_value']:.4f}"
        sig_str = "True" if row['significant'] else "False"
        print(f"{var:<16} | {row['coef']:>+10.4f} | {row['se']:>8.4f} | {row['ci_low']:>+11.4f} | {row['ci_high']:>+11.4f} | {p_val_str:>10} | {sig_str:<16}")

    print("-" * 86)
    print("Saved reference tables to 'results/tables/'.")
    print("=" * 86)


if __name__ == "__main__":
    main()
