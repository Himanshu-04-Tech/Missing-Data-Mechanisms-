"""Figures and comparison table for the imputation track (Member 2).

Reads results/tables/imputation_error_summary.csv (from run_imputation.py) and writes:
  results/figures/fig_i1_nrmse_vs_level.png      imputation error vs missingness level
  results/figures/fig_i2_sd_ratio_vs_level.png   how much natural spread the fills keep
  results/tables/imputation_method_comparison.csv  Table 3: method x mechanism overview

All numbers are on the imputed cells only, averaged over the 4 missingness columns,
then over the 5 repeats (error bars = sd over repeats).

Usage:
  python scripts/make_imputation_figures.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.imputation import load_config

SCORES_PATH = PROJECT_ROOT / "results" / "experiment_results" / "imputation_scores.csv"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLE_PATH = PROJECT_ROOT / "results" / "tables" / "imputation_method_comparison.csv"

# Fixed colour per method (validated colourblind-safe order); marker as second cue
METHOD_COLORS = {"mean": "#2a78d6", "knn": "#eb6834", "regression": "#1baf7a", "random_forest": "#eda100"}
METHOD_MARKERS = {"mean": "s", "knn": "^", "regression": "D", "random_forest": "o"}
METHOD_LABELS = {"mean": "Mean", "knn": "KNN", "regression": "Regression", "random_forest": "Random forest"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def per_run_averages(scores: pd.DataFrame) -> pd.DataFrame:
    """Average the 4 columns within each run (all cells), keeping one row per run."""
    run_keys = ["method", "mechanism", "level", "repeat"]
    scores = scores[scores["split"] == "all"]
    return scores.groupby(run_keys, sort=False)[["nrmse", "sd_ratio", "seconds"]].mean().reset_index()


def plot_metric(runs: pd.DataFrame, metric: str, ylabel: str, title: str, out_name: str, config: dict) -> Path:
    methods = [m for m in config["imputation_methods"] if m in set(runs["method"])]
    mechanisms = config["mechanisms"]
    agg = runs.groupby(["method", "mechanism", "level"])[metric].agg(["mean", "std"]).reset_index()

    y_span = agg["mean"].max() - agg["mean"].min()
    min_gap = 0.055 * (y_span if y_span > 0 else 1.0)

    fig, axes = plt.subplots(1, len(mechanisms), figsize=(13, 4.2), sharey=True)
    for ax, mech in zip(axes, mechanisms):
        ax.axhline(1.0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ends = []
        for method in methods:
            d = agg[(agg["method"] == method) & (agg["mechanism"] == mech)].sort_values("level")
            ax.errorbar(
                d["level"] * 100, d["mean"], yerr=d["std"].fillna(0),
                color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], ms=7,
                mec="white", mew=1.2, lw=2, capsize=0, label=METHOD_LABELS[method], zorder=3,
            )
            ends.append([d["mean"].iloc[-1], method, d["level"].iloc[-1] * 100])

        # End-of-line labels, pushed apart so lines that finish close together stay readable
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + min_gap)
        for y, method, x in ends:
            ax.annotate(
                METHOD_LABELS[method], (x, y), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8, color=TEXT,
            )
        ax.set_title(mech, color=TEXT, fontsize=11, loc="left")
        ax.set_xlabel("Missingness level (%)", color=MUTED)
        ax.set_xticks([lvl * 100 for lvl in config["levels"]])
        ax.set_xlim(5, 64)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED)
    axes[0].set_ylabel(ylabel, color=MUTED)
    axes[0].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle(title, x=0.01, ha="left", color=TEXT, fontsize=12)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / out_name
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    return out_path


def comparison_table(runs: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Table 3: per method and mechanism, averaged over levels and repeats."""
    table = (
        runs.groupby(["method", "mechanism"], sort=False)
        .agg(nrmse_mean=("nrmse", "mean"), sd_ratio_mean=("sd_ratio", "mean"), seconds_per_run=("seconds", "mean"))
        .reset_index()
    )
    order = {m: i for i, m in enumerate(config["imputation_methods"])}
    mech_order = {m: i for i, m in enumerate(config["mechanisms"])}
    return table.sort_values(
        ["method", "mechanism"], key=lambda s: s.map(order) if s.name == "method" else s.map(mech_order)
    ).reset_index(drop=True)


def main() -> None:
    config = load_config()
    runs = per_run_averages(pd.read_csv(SCORES_PATH))

    p1 = plot_metric(
        runs, "nrmse", "NRMSE (1 = no better than the mean)",
        "Imputation error on the filled cells (lower is better)", "fig_i1_nrmse_vs_level.png", config,
    )
    p2 = plot_metric(
        runs, "sd_ratio", "SD of fills / SD of true values",
        "Spread kept by the fills (1 = natural spread; below 1 = shrunk toward the average)",
        "fig_i2_sd_ratio_vs_level.png", config,
    )
    table = comparison_table(runs, config)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_PATH, index=False)

    print(table.round(3).to_string(index=False))
    for p in [p1, p2, TABLE_PATH]:
        print(f"Saved {p.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
