"""Module M2 figures for the statistical track.

NEW file. These figures are Member 3's own versions built from Table 4. They do not
replace Member 1's ``src/plots.py`` figures; folding the new methods into those is a
separate integration step once the output schema of Member 1's tables is known.

Figures written to ``results/figures/``:
  fig_member3_stat_abs_z_vs_level.png      mean |z_diff| on the 4 affected coefficients vs missingness level
  fig_member3_stat_ci_coverage_vs_level.png share of imputed 95% CIs that cover the reference coefficient
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.m3_common import FIGURES_DIR  # noqa: E402

MECHANISMS = ["MCAR", "MAR", "MNAR"]


def _panel_figure(
    table4: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    hline: float | None = None,
    ylim: tuple[float, float] | None = None,
) -> Path:
    mean_col, sd_col = f"{metric}_mean", f"{metric}_sd"
    mechanisms = [m for m in MECHANISMS if m in set(table4["mechanism"])]
    methods = list(dict.fromkeys(table4["method"]))
    fig, axes = plt.subplots(1, len(mechanisms), figsize=(4.6 * len(mechanisms), 4), sharey=True, squeeze=False)

    for ax, mech in zip(axes[0], mechanisms):
        sub = table4[table4["mechanism"] == mech]
        for method in methods:
            line = sub[sub["method"] == method].sort_values("level")
            if line.empty:
                continue
            x = line["level"].to_numpy() * 100
            y = line[mean_col].to_numpy()
            sd = np.nan_to_num(line[sd_col].to_numpy())
            style = "--" if method == "complete_case" else "-"
            ax.errorbar(x, y, yerr=sd, marker="o", capsize=3, linestyle=style, label=method)
        if hline is not None:
            ax.axhline(hline, color="grey", linewidth=0.8, linestyle=":")
        ax.set_title(mech)
        ax.set_xlabel("Missingness level (%)")
        ax.grid(alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
    axes[0][0].set_ylabel(ylabel)
    axes[0][-1].legend(fontsize=8, title="method")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_statistical_figures(table4: pd.DataFrame, out_dir: Path | str | None = None) -> list[Path]:
    """Create the statistical-track figures from Table 4. Returns the written paths."""
    if table4.empty:
        return []
    out = Path(out_dir) if out_dir else FIGURES_DIR
    return [
        _panel_figure(
            table4,
            "stat_mean_abs_z_affected",
            "mean |z difference| (reference-SE units)",
            "Affected coefficients: lower is better\n(bars = sd over repeats)",
            out / "fig_member3_stat_abs_z_vs_level.png",
            hline=0.0,
        ),
        _panel_figure(
            table4,
            "stat_ci_coverage_affected",
            "share of 95% CIs covering reference",
            "CI coverage of reference: higher is better\n(bars = sd over repeats)",
            out / "fig_member3_stat_ci_coverage_vs_level.png",
            hline=0.95,
            ylim=(-0.05, 1.05),
        ),
    ]
