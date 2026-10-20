"""Visualization module for missingness simulation and statistical recovery tracks.

Implements colorblind-safe publication-ready figures:
- fig_m1_missingness_heatmap: Missing mask heatmap for 300 rows sorted by longitude.
- fig_m2_missing_prob_by_driver: Missingness probability of sqft_living by deciles of long and sqft_living.
- fig_m3_complete_rows_vs_level: Train complete-row retention vs missingness level with theoretical curve.
- fig_m4_sqft_bias_vs_level: Observed sqft_living mean percent bias vs level.
- fig_s1_zdiff_vs_level: Mean absolute z-score difference vs level across methods and mechanisms.
- fig_s2_ci_coverage_vs_level: Share of 95% CIs covering reference parameters vs level.
- fig_s3_bias_heatmap: Heatmap of mean signed z-differences for all 15 predictors across 15 conditions.
- fig_s4_coef_forest: Forest plot of 4 affected coefficients on reference-SE scale at 30% missingness.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from src.aggregate import summarize_repeats
from src.handoff import load_condition

# Colorblind-safe fixed palette for mechanisms (Okabe-Ito inspired)
MECHANISM_COLORS: Dict[str, str] = {
    "MCAR": "#0072B2",  # Blue
    "MAR": "#E69F00",   # Orange
    "MNAR": "#009E73",  # Green
}

# Line styles and markers per imputation method for extensible multi-method plots
METHOD_STYLES: Dict[str, str] = {
    "complete_case": "-",
    "mean": "--",
    "knn": "-.",
    "regression": ":",
    "random_forest": (0, (3, 1, 1, 1)),
}

METHOD_MARKERS: Dict[str, str] = {
    "complete_case": "o",
    "mean": "s",
    "knn": "^",
    "regression": "D",
    "random_forest": "v",
}

FALLBACK_LINESTYLES = ["-", "--", "-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1))]
FALLBACK_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def _load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration YAML file."""
    path = Path(config_path)
    if not path.is_file():
        project_root = Path(__file__).resolve().parent.parent
        path = project_root / config_path
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_output_path(filename: str, output_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve destination path and ensure parent directory exists."""
    if output_path is not None:
        p = Path(output_path)
    else:
        project_root = Path(__file__).resolve().parent.parent
        p = project_root / "results" / "figures" / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# =============================================================================
# 1. MISSINGNESS TRACK FIGURES
# =============================================================================

def fig_m1_missingness_heatmap(
    config_path: str = "config.yaml",
    output_path: Optional[Union[str, Path]] = None,
    seed: int = 42,
) -> Path:
    """3 panels (MCAR, MAR, MNAR) at level 0.30, repeat 0, TRAIN rows.

    Shows the 4-column missing mask for 300 randomly sampled rows (fixed seed),
    with rows sorted by longitude (west to east).
    """
    config = _load_config(config_path)
    missing_cols = list(config.get("missingness_columns", ["sqft_living", "grade", "bathrooms", "yr_built"]))
    mechanisms = ["MCAR", "MAR", "MNAR"]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.5), sharey=True)

    for i, mech in enumerate(mechanisms):
        ax = axes[i]
        c = load_condition(mechanism=mech, level=0.30, repeat=0, config_path=config_path)
        train_idx = c["train_idx"]

        rng = np.random.RandomState(seed)
        sampled_rows = rng.choice(train_idx, size=300, replace=False)

        long_vals = c["truth"].loc[sampled_rows, "long"]
        sorted_rows = long_vals.sort_values().index

        mask_sub = c["mask"].loc[sorted_rows, missing_cols].astype(int).values

        # Binary colormap: 0 = observed (neutral light gray), 1 = missing (mechanism color)
        cmap = ListedColormap(["#F2F2F2", MECHANISM_COLORS[mech]])
        ax.imshow(mask_sub, aspect="auto", cmap=cmap, interpolation="nearest", vmin=0, vmax=1)

        ax.set_xticks(range(len(missing_cols)))
        ax.set_xticklabels(missing_cols, rotation=35, ha="right", fontsize=9)
        ax.set_title(f"{mech} (30% missing)", fontsize=11, fontweight="bold", color=MECHANISM_COLORS[mech])

        if i == 0:
            ax.set_ylabel("300 Sampled Train Rows\n(sorted by longitude: west $\\to$ east)", fontsize=9.5)
        else:
            ax.set_ylabel("")

    fig.suptitle("Missingness Mask Patterns (Level 0.30, Repeat 0)", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = _resolve_output_path("fig_m1_missingness_heatmap.png", output_path)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_file


def fig_m2_missing_prob_by_driver(
    config_path: str = "config.yaml",
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """2 panels for sqft_living at level 0.30 (average over the 5 repeats).

    Plots fraction missing by DECILE of long, and by DECILE of sqft_living
    (its own true value); one line per mechanism.
    """
    config = _load_config(config_path)
    mechanisms = ["MCAR", "MAR", "MNAR"]
    n_repeats = int(config.get("n_repeats", 5))

    # Pre-collect data averaged across 5 repeats
    deciles = np.arange(1, 11)
    results_long = {m: np.zeros(10) for m in mechanisms}
    results_sqft = {m: np.zeros(10) for m in mechanisms}

    for mech in mechanisms:
        long_reps = []
        sqft_reps = []
        for rep in range(n_repeats):
            c = load_condition(mechanism=mech, level=0.30, repeat=rep, config_path=config_path)
            train_idx = c["train_idx"]
            truth = c["truth"].loc[train_idx]
            mask_sqft = c["mask"].loc[train_idx, "sqft_living"].astype(float)

            # Decile binning
            dec_long = pd.qcut(truth["long"], q=10, labels=False, duplicates="drop") + 1
            dec_sqft = pd.qcut(truth["sqft_living"], q=10, labels=False, duplicates="drop") + 1

            p_long = mask_sqft.groupby(dec_long).mean().reindex(deciles, fill_value=0.0).values
            p_sqft = mask_sqft.groupby(dec_sqft).mean().reindex(deciles, fill_value=0.0).values

            long_reps.append(p_long)
            sqft_reps.append(p_sqft)

        results_long[mech] = np.mean(long_reps, axis=0)
        results_sqft[mech] = np.mean(sqft_reps, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 4.5), sharey=True)

    markers = {"MCAR": "o", "MAR": "s", "MNAR": "^"}
    for mech in mechanisms:
        col = MECHANISM_COLORS[mech]
        m = markers[mech]
        ax1.plot(deciles, results_long[mech], label=mech, color=col, marker=m, linewidth=2, markersize=6)
        ax2.plot(deciles, results_sqft[mech], label=mech, color=col, marker=m, linewidth=2, markersize=6)

    # Reference dashed line at 30% nominal level
    for ax in (ax1, ax2):
        ax.axhline(0.30, color="gray", linestyle=":", linewidth=1.2, label="Nominal Rate (30%)")
        ax.set_xticks(deciles)
        ax.set_xlabel("Decile (1 = lowest, 10 = highest)", fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.4)

    ax1.set_ylabel("Fraction Missing in sqft_living", fontsize=9.5)
    ax1.set_title("By Decile of Longitude (MAR Driver)", fontsize=10.5, fontweight="bold")
    ax2.set_title("By Decile of sqft_living (MNAR Target)", fontsize=10.5, fontweight="bold")

    ax1.legend(loc="upper left", framealpha=0.9, fontsize=8.5)
    fig.suptitle("Missing Probability for sqft_living (Level 0.30, 5-Repeat Mean)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    out_file = _resolve_output_path("fig_m2_missing_prob_by_driver.png", output_path)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_file


def fig_m3_complete_rows_vs_level(
    df_summary: Optional[pd.DataFrame] = None,
    config_path: str = "config.yaml",
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Train complete-row fraction vs level per mechanism (mean with sd band).

    Includes the theoretical (1-level)^4 curve as a dashed grey line.
    """
    project_root = Path(__file__).resolve().parent.parent
    if df_summary is None:
        table_path = project_root / "results" / "tables" / "missingness_summary.csv"
        if not table_path.is_file():
            from src.aggregate import generate_missingness_summary
            df_summary = generate_missingness_summary(config_path=config_path)
        else:
            df_summary = pd.read_csv(table_path)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    levels_grid = np.linspace(0.10, 0.50, 100)
    theo_curve = (1.0 - levels_grid) ** 4
    ax.plot(
        levels_grid,
        theo_curve,
        linestyle="--",
        color="#7F7F7F",
        linewidth=1.8,
        label="Theoretical $(1 - p)^4$",
        zorder=2,
    )

    markers = {"MCAR": "o", "MAR": "s", "MNAR": "^"}
    for mech in ["MCAR", "MAR", "MNAR"]:
        sub = df_summary[df_summary["mechanism"] == mech].sort_values("level")
        lvls = sub["level"].values
        means = sub["train_complete_row_frac_mean"].values
        sds = sub["train_complete_row_frac_sd"].values
        col = MECHANISM_COLORS[mech]

        ax.plot(lvls, means, label=mech, color=col, marker=markers[mech], linewidth=2, markersize=6, zorder=3)
        ax.fill_between(lvls, means - sds, means + sds, color=col, alpha=0.18, zorder=1)

    ax.set_xlabel("Missingness Level ($p$)", fontsize=10)
    ax.set_ylabel("Train Complete-Row Fraction", fontsize=10)
    ax.set_title("Train Complete-Row Retention vs. Missingness Level", fontsize=11.5, fontweight="bold")
    ax.set_xticks([0.10, 0.20, 0.30, 0.40, 0.50])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    out_file = _resolve_output_path("fig_m3_complete_rows_vs_level.png", output_path)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_file


def fig_m4_sqft_bias_vs_level(
    df_summary: Optional[pd.DataFrame] = None,
    config_path: str = "config.yaml",
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Percent bias of the observed sqft_living mean vs level per mechanism (mean with sd band)."""
    project_root = Path(__file__).resolve().parent.parent
    if df_summary is None:
        table_path = project_root / "results" / "tables" / "missingness_summary.csv"
        if not table_path.is_file():
            from src.aggregate import generate_missingness_summary
            df_summary = generate_missingness_summary(config_path=config_path)
        else:
            df_summary = pd.read_csv(table_path)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    ax.axhline(0, color="gray", linestyle=":", linewidth=1.4, label="Unbiased (0%)")

    markers = {"MCAR": "o", "MAR": "s", "MNAR": "^"}
    for mech in ["MCAR", "MAR", "MNAR"]:
        sub = df_summary[df_summary["mechanism"] == mech].sort_values("level")
        lvls = sub["level"].values
        means = sub["sqft_living_pct_bias_mean"].values
        sds = sub["sqft_living_pct_bias_sd"].values
        col = MECHANISM_COLORS[mech]

        ax.plot(lvls, means, label=mech, color=col, marker=markers[mech], linewidth=2, markersize=6)
        ax.fill_between(lvls, means - sds, means + sds, color=col, alpha=0.18)

    ax.set_xlabel("Missingness Level ($p$)", fontsize=10)
    ax.set_ylabel("Observed sqft_living Mean Bias (%)", fontsize=10)
    ax.set_title("Percent Bias of Observed sqft_living Mean vs. Level", fontsize=11.5, fontweight="bold")
    ax.set_xticks([0.10, 0.20, 0.30, 0.40, 0.50])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    out_file = _resolve_output_path("fig_m4_sqft_bias_vs_level.png", output_path)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_file


# =============================================================================
# 2. STATISTICAL RECOVERY TRACK FIGURES
# =============================================================================

def _get_method_linestyle(method: str, idx: int = 0) -> str:
    """Return configured or cyclic fallback linestyle for a given method."""
    if method in METHOD_STYLES:
        return METHOD_STYLES[method]
    return FALLBACK_LINESTYLES[idx % len(FALLBACK_LINESTYLES)]


def _get_method_marker(method: str, idx: int = 0) -> str:
    """Return configured or cyclic fallback marker for a given method."""
    if method in METHOD_MARKERS:
        return METHOD_MARKERS[method]
    return FALLBACK_MARKERS[idx % len(FALLBACK_MARKERS)]


def fig_s1_zdiff_vs_level(
    df: Optional[pd.DataFrame] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """2 panels (all 15 coefficients; the 4 affected), mean_abs_z_diff vs level.

    Plots mean with sd band per mechanism, with a different line style per method.
    """
    project_root = Path(__file__).resolve().parent.parent
    if df is None:
        csv_path = project_root / "results" / "experiment_results" / "complete_case_stats.csv"
        df = pd.read_csv(csv_path)

    # Check if aggregation over repeats is needed
    if "repeat" in df.columns:
        agg_df = summarize_repeats(
            df,
            group_cols=["method", "mechanism", "level"],
            metrics=["mean_abs_z_diff", "mean_abs_z_diff_affected"],
        )
    else:
        agg_df = df.copy()

    methods = list(agg_df["method"].unique())
    mechanisms = ["MCAR", "MAR", "MNAR"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 4.5), sharey=True)

    for m_idx, method in enumerate(methods):
        ls = _get_method_linestyle(method, m_idx)
        marker = _get_method_marker(method, m_idx)

        for mech in mechanisms:
            sub = agg_df[(agg_df["method"] == method) & (agg_df["mechanism"] == mech)].sort_values("level")
            if sub.empty:
                continue

            lvls = sub["level"].values
            col = MECHANISM_COLORS[mech]
            lbl = f"{mech}" if len(methods) == 1 else f"{mech} ({method})"

            # Panel 1: All 15 predictors
            m_all = sub["mean_abs_z_diff_mean"].values if "mean_abs_z_diff_mean" in sub.columns else sub["mean_abs_z_diff"].values
            sd_all = sub["mean_abs_z_diff_sd"].values if "mean_abs_z_diff_sd" in sub.columns else np.zeros_like(m_all)
            ax1.plot(lvls, m_all, linestyle=ls, color=col, marker=marker, label=lbl, linewidth=2, markersize=5)
            ax1.fill_between(lvls, m_all - sd_all, m_all + sd_all, color=col, alpha=0.15)

            # Panel 2: 4 affected predictors
            m_aff = sub["mean_abs_z_diff_affected_mean"].values if "mean_abs_z_diff_affected_mean" in sub.columns else sub["mean_abs_z_diff_affected"].values
            sd_aff = sub["mean_abs_z_diff_affected_sd"].values if "mean_abs_z_diff_affected_sd" in sub.columns else np.zeros_like(m_aff)
            ax2.plot(lvls, m_aff, linestyle=ls, color=col, marker=marker, label=lbl, linewidth=2, markersize=5)
            ax2.fill_between(lvls, m_aff - sd_aff, m_aff + sd_aff, color=col, alpha=0.15)

    for ax in (ax1, ax2):
        ax.set_xlabel("Missingness Level", fontsize=9.5)
        ax.set_xticks([0.10, 0.20, 0.30, 0.40, 0.50])
        ax.grid(True, linestyle="--", alpha=0.4)

    ax1.set_ylabel("Mean Absolute $z$-difference ($|z_{\\mathrm{diff}}| $)", fontsize=9.5)
    ax1.set_title("All 15 Predictors", fontsize=10.5, fontweight="bold")
    ax2.set_title("4 Affected Predictors", fontsize=10.5, fontweight="bold")
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=8.5)

    fig.suptitle("OLS Parameter Divergence ($|z_{\\mathrm{diff}}|$) vs. Level", fontsize=12, fontweight="bold")
    plt.tight_layout()

    out_file = _resolve_output_path("fig_s1_zdiff_vs_level.png", output_path)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_file


def fig_s2_ci_coverage_vs_level(
    df: Optional[pd.DataFrame] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """share_ci_covers_ref vs level per mechanism across methods."""
    project_root = Path(__file__).resolve().parent.parent
    if df is None:
        csv_path = project_root / "results" / "experiment_results" / "complete_case_stats.csv"
        df = pd.read_csv(csv_path)

    if "repeat" in df.columns:
        agg_df = summarize_repeats(
            df,
            group_cols=["method", "mechanism", "level"],
            metrics=["share_ci_covers_ref"],
        )
    else:
        agg_df = df.copy()

    methods = list(agg_df["method"].unique())
    mechanisms = ["MCAR", "MAR", "MNAR"]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.axhline(0.95, color="gray", linestyle=":", linewidth=1.5, label="Nominal 95% Coverage")

    for m_idx, method in enumerate(methods):
        ls = _get_method_linestyle(method, m_idx)
        marker = _get_method_marker(method, m_idx)

        for mech in mechanisms:
            sub = agg_df[(agg_df["method"] == method) & (agg_df["mechanism"] == mech)].sort_values("level")
            if sub.empty:
                continue

            lvls = sub["level"].values
            col = MECHANISM_COLORS[mech]
            lbl = f"{mech}" if len(methods) == 1 else f"{mech} ({method})"

            cov_mean = sub["share_ci_covers_ref_mean"].values if "share_ci_covers_ref_mean" in sub.columns else sub["share_ci_covers_ref"].values
            cov_sd = sub["share_ci_covers_ref_sd"].values if "share_ci_covers_ref_sd" in sub.columns else np.zeros_like(cov_mean)

            ax.plot(lvls, cov_mean, linestyle=ls, color=col, marker=marker, label=lbl, linewidth=2, markersize=6)
            ax.fill_between(lvls, cov_mean - cov_sd, cov_mean + cov_sd, color=col, alpha=0.15)

    ax.set_xlabel("Missingness Level", fontsize=10)
    ax.set_ylabel("Share CI Covers Reference", fontsize=10)
    ax.set_title("Reference Parameter 95% CI Coverage vs. Level", fontsize=11.5, fontweight="bold")
    ax.set_xticks([0.10, 0.20, 0.30, 0.40, 0.50])
    ax.set_ylim(0.30, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    out_file = _resolve_output_path("fig_s2_ci_coverage_vs_level.png", output_path)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_file


def fig_s3_bias_heatmap(
    df: Optional[pd.DataFrame] = None,
    method: str = "complete_case",
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Heatmap of the mean signed z_diff over repeats.

    Rows = 15 predictors (excluding intercept 'const'),
    Columns = 15 mechanism-level conditions,
    Diverging colour map centered at 0 (clipped at +/-10), annotated.
    """
    project_root = Path(__file__).resolve().parent.parent
    if df is None:
        csv_path = project_root / "results" / "experiment_results" / "complete_case_per_coef.csv"
        df = pd.read_csv(csv_path)

    # Filter by method if method column present
    if "method" in df.columns:
        if method in df["method"].values:
            df_sub = df[df["method"] == method].copy()
        else:
            first_m = df["method"].iloc[0]
            df_sub = df[df["method"] == first_m].copy()
            method = first_m
    else:
        df_sub = df.copy()

    # Filter to the 15 predictors (exclude const)
    df_sub = df_sub[df_sub["variable"] != "const"].copy()

    # Aggregate over repeats if needed
    if "repeat" in df_sub.columns:
        mean_df = df_sub.groupby(["variable", "mechanism", "level"], as_index=False)["z_diff"].mean()
    elif "bias" in df_sub.columns:
        mean_df = df_sub.rename(columns={"bias": "z_diff"})
    elif "z_diff_mean" in df_sub.columns:
        mean_df = df_sub.rename(columns={"z_diff_mean": "z_diff"})
    else:
        mean_df = df_sub

    # Column ordering: MCAR, MAR, MNAR by level
    mechanisms = ["MCAR", "MAR", "MNAR"]
    levels = [0.10, 0.20, 0.30, 0.40, 0.50]
    ordered_cols = [f"{m} {l:.2f}" for m in mechanisms for l in levels]

    mean_df["cond"] = mean_df["mechanism"] + " " + mean_df["level"].map(lambda x: f"{x:.2f}")

    pivot_table = mean_df.pivot(index="variable", columns="cond", values="z_diff")
    pivot_table = pivot_table.reindex(columns=ordered_cols)

    # Order rows sensibly (affected predictors first, then remaining)
    affected = ["sqft_living", "grade", "bathrooms", "yr_built"]
    other_preds = [v for v in pivot_table.index if v not in affected]
    pivot_table = pivot_table.reindex(affected + sorted(other_preds))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    # Clip values at +/- 10 for visual clarity
    clipped_data = pivot_table.clip(lower=-10.0, upper=10.0)

    sns.heatmap(
        clipped_data,
        ax=ax,
        cmap="coolwarm",
        center=0.0,
        vmin=-10.0,
        vmax=10.0,
        annot=True,
        fmt=".1f",
        annot_kws={"size": 6.0},
        cbar_kws={"label": "Mean Signed $z_{\\mathrm{diff}}$ (clipped at $\\pm 10$)", "shrink": 0.8},
        linewidths=0.5,
        linecolor="#E0E0E0",
    )

    ax.set_title(f"Predictor Bias Heatmap: Mean Signed $z_{{\\mathrm{{diff}}}}$ ({method})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Mechanism & Missingness Level", fontsize=9.5)
    ax.set_ylabel("Predictor", fontsize=9.5)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    out_file = _resolve_output_path("fig_s3_bias_heatmap.png", output_path)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_file


def fig_s4_coef_forest(
    df: Optional[pd.DataFrame] = None,
    level: float = 0.30,
    repeat: int = 0,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Forest plot for MCAR, MAR, MNAR at level 0.30, repeat 0.

    Shows the 4 affected coefficients on the reference-SE scale:
    point = (coef_other - coef_ref) / se_ref  (= z_diff)
    bar = +/- 1.96 * se_other / se_ref        (= +/- 1.96 * se_ratio)
    Includes vertical line at 0 and a reference 95% CI band at +/- 1.96.
    """
    project_root = Path(__file__).resolve().parent.parent
    if df is None:
        csv_path = project_root / "results" / "experiment_results" / "complete_case_per_coef.csv"
        df = pd.read_csv(csv_path)

    # Filter to condition level=0.30, repeat=0
    sub = df[np.isclose(df["level"], level) & (df["repeat"] == repeat)].copy()
    affected_vars = ["sqft_living", "grade", "bathrooms", "yr_built"]
    sub = sub[sub["variable"].isin(affected_vars)]

    mechanisms = ["MCAR", "MAR", "MNAR"]
    methods = list(sub["method"].unique()) if "method" in sub.columns else ["complete_case"]

    fig, ax = plt.subplots(figsize=(7.2, 4.5))

    # Reference baseline band and zero line
    ax.axvspan(-1.96, 1.96, color="#D9D9D9", alpha=0.45, label="Reference 95% CI", zorder=1)
    ax.axvline(0, color="black", linestyle="-", linewidth=1.2, zorder=2)

    # Layout: 4 variables on y-axis, with mechanisms slightly offset
    y_base = np.arange(len(affected_vars))
    mech_offsets = {"MCAR": -0.22, "MAR": 0.0, "MNAR": 0.22}

    for m_idx, method in enumerate(methods):
        sub_m = sub[sub["method"] == method] if "method" in sub.columns else sub
        marker = _get_method_marker(method, m_idx)

        for i, var in enumerate(affected_vars):
            for mech in mechanisms:
                row = sub_m[(sub_m["variable"] == var) & (sub_m["mechanism"] == mech)]
                if row.empty:
                    continue

                point = float(row["z_diff"].iloc[0])
                se_ratio = float(row["se_ratio"].iloc[0])
                half_width = 1.96 * se_ratio

                y_pos = y_base[i] + mech_offsets[mech]
                col = MECHANISM_COLORS[mech]

                # Label only once per mechanism for a clean legend
                lbl = mech if (i == 0 and m_idx == 0) else None

                ax.errorbar(
                    point,
                    y_pos,
                    xerr=half_width,
                    fmt=marker,
                    color=col,
                    ecolor=col,
                    elinewidth=2.0,
                    capsize=4.0,
                    capthick=1.5,
                    markersize=6,
                    label=lbl,
                    zorder=3,
                )

    ax.set_yticks(y_base)
    ax.set_yticklabels(affected_vars, fontsize=9.5, fontweight="normal")
    ax.set_xlabel("Normalized Estimate Error $(\\hat{\\beta} - \\beta_{\\mathrm{ref}}) / \\mathrm{SE}_{\\mathrm{ref}}$", fontsize=9.5)
    ax.set_title(
        f"Forest Plot: Affected Predictors at Level {level:.2f} (Repeat {repeat})",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.4, axis="x")
    ax.legend(loc="lower left", framealpha=0.9, fontsize=8.5)

    plt.tight_layout()
    out_file = _resolve_output_path("fig_s4_coef_forest.png", output_path)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_file
