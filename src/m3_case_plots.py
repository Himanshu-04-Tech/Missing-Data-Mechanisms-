"""Module M5 (part 2): central figure, supporting plots and sensitivity figure.

NEW file; does not modify any existing module in ``src/``. All figures are drawn from tables,
so they regenerate from one script (``scripts/m3_05_cases.py``).

Figures written to ``results/figures/``:
  fig_member3_central_stat_vs_pred.png         FR-5.1: statistical vs predictive recovery, one point per condition
  fig_member3_pred_rmse_gap_vs_level.png       FR-5.5: predictive RMSE gap vs missingness level, per mechanism, line per method
  fig_member3_pred_recovery_heatmap.png        FR-5.5: predictive recovery by mechanism x level, one panel per method
  fig_member3_case_sensitivity.png             FR-5.4: number of Case B conditions as both thresholds move
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.m3_common import FIGURES_DIR  # noqa: E402
from src.m3_merge import METHOD_ORDER, MECH_ORDER  # noqa: E402
from src.m3_stat_plots import _panel_figure  # noqa: E402

# Okabe-Ito colour-blind-safe palette; markers also differ by mechanism so colour is never the only cue.
METHOD_COLOURS = {"mean": "#0072B2", "knn": "#E69F00", "regression": "#009E73", "random_forest": "#CC79A7"}
MECH_MARKERS = {"MCAR": "o", "MAR": "s", "MNAR": "^"}


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """A column, or zeros if absent (so error bars are optional)."""
    return df[name].fillna(0.0) if name in df.columns else pd.Series(0.0, index=df.index)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_central_figure(labelled: pd.DataFrame, out_dir: Path | str | None = None) -> Path:
    """FR-5.1. x = statistical recovery, y = predictive recovery (both smaller = better)."""
    out = Path(out_dir) if out_dir else FIGURES_DIR
    st, pt = float(labelled["stat_threshold"].iloc[0]), float(labelled["pred_threshold"].iloc[0])
    status = labelled["thresholds_status"].iloc[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    x_max = max(labelled["stat_value"].max() + _col(labelled, "stat_mean_abs_z_affected_sd").max() * 1.05, st * 1.6)
    y_max = max(labelled["pred_value"].max() + _col(labelled, "rmse_gap_rel_sd").max() * 1.05, pt * 1.6)
    ax.axvline(st, color="grey", linestyle="--", linewidth=1)
    ax.axhline(pt, color="grey", linestyle="--", linewidth=1)
    corner = dict(fontsize=11, color="grey", alpha=0.85, fontweight="bold")
    ax.text(0.01 * x_max, 0.02 * y_max, "A  good on both", va="bottom", **corner)
    ax.text(0.99 * x_max, 0.02 * y_max, "B  predicts well,\nstatistically distorted", va="bottom", ha="right", **corner)
    ax.text(0.01 * x_max, 0.98 * y_max, "C  statistically good,\npredicts poorly", va="top", **corner)
    ax.text(0.99 * x_max, 0.98 * y_max, "D  poor on both", va="top", ha="right", **corner)

    for (method, mech), grp in labelled.groupby(["method", "mechanism"]):
        ax.errorbar(
            grp["stat_value"], grp["pred_value"],
            xerr=_col(grp, "stat_mean_abs_z_affected_sd"),
            yerr=_col(grp, "rmse_gap_rel_sd"),
            fmt=MECH_MARKERS.get(mech, "o"), color=METHOD_COLOURS.get(method, "black"),
            ecolor=METHOD_COLOURS.get(method, "black"), elinewidth=0.7, capsize=2, markersize=6, alpha=0.85,
        )
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Statistical recovery: mean |z difference| on affected coefficients\n(reference-SE units; smaller = better)")
    ax.set_ylabel("Predictive recovery: RMSE gap / complete-data RMSE\n(smaller = better)")
    title = "Does good prediction imply good statistical recovery?"
    if status != "LOCKED":
        title += "\nTHRESHOLDS PROVISIONAL (not yet locked by the team)"
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25)

    method_handles = [plt.Line2D([], [], color=c, marker="o", linestyle="", label=m)
                      for m, c in METHOD_COLOURS.items() if m in set(labelled["method"])]
    mech_handles = [plt.Line2D([], [], color="black", marker=mk, linestyle="", label=m)
                    for m, mk in MECH_MARKERS.items() if m in set(labelled["mechanism"])]
    leg1 = ax.legend(handles=method_handles, title="method", loc="center left", bbox_to_anchor=(1.01, 0.75), fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=mech_handles, title="mechanism", loc="center left", bbox_to_anchor=(1.01, 0.35), fontsize=8)
    ax.text(1.01, 0.02, "points = condition means\nbars = sd over repeats\ndashed = Case thresholds",
            transform=ax.transAxes, fontsize=7, color="grey", va="bottom")
    return _save(fig, out / "fig_member3_central_stat_vs_pred.png")


def plot_predictive_vs_level(table6: pd.DataFrame, out_dir: Path | str | None = None) -> Path:
    """FR-5.5: RMSE gap versus missingness level per mechanism, one line per method."""
    out = Path(out_dir) if out_dir else FIGURES_DIR
    return _panel_figure(
        table6, "rmse_gap", "RMSE gap vs complete-data model (log-price units)",
        "Predictive recovery: smaller is better\n(bars = sd over repeats)",
        out / "fig_member3_pred_rmse_gap_vs_level.png", hline=0.0,
    )


def plot_predictive_heatmap(table6: pd.DataFrame, out_dir: Path | str | None = None) -> Path:
    """FR-5.5: heatmap of the relative RMSE gap by mechanism x level, one panel per method."""
    out = Path(out_dir) if out_dir else FIGURES_DIR
    methods = [m for m in METHOD_ORDER if m in set(table6["method"])] or list(dict.fromkeys(table6["method"]))
    mechs = [m for m in MECH_ORDER if m in set(table6["mechanism"])]
    levels = sorted(table6["level"].unique())
    vmax = float(table6["rmse_gap_rel_mean"].max())
    vmin = min(0.0, float(table6["rmse_gap_rel_mean"].min()))

    fig, axes = plt.subplots(1, len(methods), figsize=(3.6 * len(methods) + 1.2, 3.2), sharey=True, squeeze=False)
    image = None
    for ax, method in zip(axes[0], methods):
        grid = (
            table6[table6["method"] == method]
            .pivot(index="mechanism", columns="level", values="rmse_gap_rel_mean")
            .reindex(index=mechs, columns=levels)
        )
        image = ax.imshow(grid.to_numpy(dtype=float), aspect="auto", cmap="viridis_r", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(levels)), [f"{int(round(l * 100))}%" for l in levels])
        ax.set_yticks(range(len(mechs)), mechs)
        ax.set_title(method, fontsize=10)
        ax.set_xlabel("Missingness level")
        for i in range(len(mechs)):
            for j in range(len(levels)):
                v = grid.iloc[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                            color="white" if v > (vmin + vmax) / 2 else "black")
    fig.colorbar(image, ax=axes[0].tolist(), label="RMSE gap / complete-data RMSE\n(smaller = better)", shrink=0.9)
    fig.suptitle("Predictive recovery by mechanism and level", fontsize=11)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "fig_member3_pred_recovery_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sensitivity(grid: pd.DataFrame, out_dir: Path | str | None = None) -> Path:
    """FR-5.4: number of Case B conditions as both thresholds are scaled."""
    out = Path(out_dir) if out_dir else FIGURES_DIR
    piv = grid.pivot(index="pred_factor", columns="stat_factor", values="B").sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    im = ax.imshow(piv.to_numpy(dtype=float), cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), [f"x{f:g}" for f in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"x{f:g}" for f in piv.index])
    ax.set_xlabel("statistical threshold factor")
    ax.set_ylabel("predictive threshold factor")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, int(piv.iloc[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title("Case B count when both thresholds are scaled\n(x1 / x1 = configured thresholds)", fontsize=10)
    fig.colorbar(im, ax=ax, label="conditions in Case B")
    return _save(fig, out / "fig_member3_case_sensitivity.png")


def plot_case_figures(labelled: pd.DataFrame, grid: pd.DataFrame, out_dir: Path | str | None = None) -> list[Path]:
    """All Module M5 figures. ``labelled`` is Table 6 plus Case labels."""
    if labelled.empty:
        return []
    return [
        plot_central_figure(labelled, out_dir),
        plot_predictive_vs_level(labelled, out_dir),
        plot_predictive_heatmap(labelled, out_dir),
        plot_sensitivity(grid, out_dir),
    ]
