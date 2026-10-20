"""Run complete-case analysis across all 75 missingness conditions.

Saves:
- results/experiment_results/complete_case_stats.csv

Prints:
- A 15-line summary table (mechanism x level averaged over repeats) reporting:
  rows used, mean_abs_z_diff, mean_abs_z_diff_affected, n_sig_flips,
  share_ci_covers_ref, and mean_abs_corr_diff_pearson.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.handoff import list_conditions, load_condition
from src.statistics import complete_case_recovery, load_config


def main():
    config = load_config(str(project_root / "config.yaml"))
    tables_dir = project_root / "results" / "tables"
    exp_dir = project_root / "results" / "experiment_results"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load baseline reference
    ref_table_path = tables_dir / "reference_ols.csv"
    ref_corr_p_path = tables_dir / "reference_corr_pearson.csv"
    ref_corr_s_path = tables_dir / "reference_corr_spearman.csv"

    if not ref_table_path.is_file():
        raise FileNotFoundError(
            f"Reference table not found at '{ref_table_path}'. "
            "Please run 'python scripts/build_reference.py' first."
        )

    ref_table = pd.read_csv(ref_table_path, index_col="variable")
    ref_p = pd.read_csv(ref_corr_p_path, index_col=0)
    ref_s = pd.read_csv(ref_corr_s_path, index_col=0)
    ref_corr = (ref_p, ref_s)

    # 2. Iterate through all 75 conditions
    manifest = list_conditions()
    records = []
    per_coef_records = []

    print("=" * 86)
    print("COMPLETE-CASE STATISTICAL RECOVERY SWEEP (75 CONDITIONS)")
    print("=" * 86)

    for _, row in manifest.iterrows():
        mech = row["mechanism"]
        lvl = float(row["level"])
        rep = int(row["repeat"])

        data = load_condition(mechanism=mech, level=lvl, repeat=rep)
        df_corr = data["corrupted"]

        per_coef, summary = complete_case_recovery(
            df_corrupted=df_corr,
            reference_table=ref_table,
            reference_corr=ref_corr,
            config=config,
        )

        record = {
            "method": "complete_case",
            "mechanism": mech,
            "level": lvl,
            "repeat": rep,
            **summary,
        }
        records.append(record)

        for var, crow in per_coef.iterrows():
            per_coef_records.append(
                {
                    "method": "complete_case",
                    "mechanism": mech,
                    "level": lvl,
                    "repeat": rep,
                    "variable": var,
                    "z_diff": float(crow["z_diff"]),
                    "se_ratio": float(crow["se_ratio"]),
                    "ci_covers_ref": bool(crow["ci_covers_ref"]),
                    "sig_flip": bool(crow["sig_flip"]),
                    "sign_flip": bool(crow["sign_flip"]),
                }
            )

    # 3. Save full experimental results
    results_df = pd.DataFrame(records)
    out_csv = exp_dir / "complete_case_stats.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"Saved complete-case stats for {len(results_df)} conditions to '{out_csv}'.\n")

    per_coef_df = pd.DataFrame(per_coef_records)
    per_coef_csv = exp_dir / "complete_case_per_coef.csv"
    per_coef_df.to_csv(per_coef_csv, index=False)
    print(f"Saved complete-case per-coefficient stats ({len(per_coef_df)} rows) to '{per_coef_csv}'.\n")

    # 4. Print 15-line summary table (mechanism x level averaged over repeats)
    print("Complete-Case Statistical Recovery (Averaged Over 5 Repeats):")
    print("-" * 86)
    print(
        f"{'Mechanism':<10} | {'Level':<6} | {'Rows Used':>9} | {'Mean |z| (All)':>14} | "
        f"{'Mean |z| (Aff)':>14} | {'Sig Flips':>9} | {'CI Coverage':>11} | {'Pearson Diff':>12}"
    )
    print("-" * 86)

    for mech in config["mechanisms"]:
        for lvl in config["levels"]:
            sub = results_df[(results_df["mechanism"] == mech) & np.isclose(results_df["level"], lvl)]
            avg_rows = sub["n_rows_used"].mean()
            avg_z_all = sub["mean_abs_z_diff"].mean()
            avg_z_aff = sub["mean_abs_z_diff_affected"].mean()
            avg_sig_flips = sub["n_sig_flips"].mean()
            avg_ci_cov = sub["share_ci_covers_ref"].mean()
            avg_corr_p = sub["mean_abs_corr_diff_pearson"].mean()

            print(
                f"{mech:<10} | {lvl:<6.2f} | {avg_rows:>9.0f} | {avg_z_all:>14.4f} | "
                f"{avg_z_aff:>14.4f} | {avg_sig_flips:>9.2f} | {avg_ci_cov * 100:>10.1f}% | {avg_corr_p:>12.4f}"
            )

    print("-" * 86)
    print("=" * 86)


if __name__ == "__main__":
    main()
