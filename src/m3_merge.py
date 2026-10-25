"""Module M4: merge the statistical and predictive tracks and build Table 6.

NEW file; does not modify any existing module in ``src/``.

Inputs (produced by Modules M2 and M3)
--------------------------------------
* ``results/experiment_results/stat_recovery_runs.csv``  one row per run (and complete_case)
* ``results/experiment_results/predictive_results.csv``  one row per imputed run
* ``results/experiment_results/predictive_reference.csv`` complete-data Random Forest (upper bound)

What it does
------------
1. FR-4.1  Merge on method + mechanism + level + repeat. Asserts exactly one row per key on both
           sides and no unmatched rows. ``complete_case`` rows exist only in the statistical track
           (FR-3.8), so they are set aside (and reported), never treated as unmatched.
2. FR-4.2  Predictive recovery (decision D2, PROVISIONAL):
              rmse_gap = rmse_imputed_model - rmse_reference        (>= 0 normally; smaller is better)
              rmse_gap_rel = rmse_gap / rmse_reference              (the same gap as a share; used for the Case threshold in M5)
              r2_gap   = r2_reference       - r2_imputed_model      (>= 0 normally; smaller is better)
           The reference is the complete-data Random Forest (primary seed).
3. FR-4.3  Statistical recovery (decision D2, PROVISIONAL), taken from Module M2:
              stat_mean_abs_z_affected    mean |z_diff| over the 4 affected coefficients (smaller is better;
                                          mixes bias and noise, in reference-SE units)
              stat_ci_coverage_affected   share of the 4 affected CIs covering the reference (larger is better)
              stat_mean_signed_z_affected the signed average = bias (repeat-averaging keeps the sign)
4. FR-4.4  Mean and sd (ddof=1) over the 5 repeats by method x mechanism x level.
5. FR-4.5  Table 6: statistical and predictive recovery side by side.
6. FR-4.6  Per mechanism x level: best predictive method, best statistical method, and whether they
           are the same. The ranking columns are fixed in ``RANK_STAT`` / ``RANK_PRED`` below (and
           recorded in the output), not chosen after looking at results.

Honesty notes
-------------
* Which method is "best" is a statement about THESE conditions only, never a universal ranking.
* ``margin`` columns show how far the winner is ahead of the runner-up; compare them with the
  seed-to-seed spread (the ``_sd`` columns and the reference spread) before reading a winner
  as meaningful.
* This module does not label Cases A to D. That is Module M5, with thresholds fixed beforehand (D3).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.m3_common import EXPERIMENT_RESULTS_DIR, TABLES_DIR
from src.m3_predictive import (
    PREDICTIVE_REFERENCE_PATH,
    PREDICTIVE_RESULTS_PATH,
    load_predictive_reference,
    load_predictive_runs,
)
from src.m3_stat_track import COMPLETE_CASE, RUNS_PATH as STAT_RUNS_PATH, load_stat_runs

KEY = ["method", "mechanism", "level", "repeat"]
CONDITION = ["method", "mechanism", "level"]

MERGED_PATH = EXPERIMENT_RESULTS_DIR / "merged_results.csv"
TABLE6_PATH = TABLES_DIR / "table6_overall_recovery.csv"
TABLE6B_PATH = TABLES_DIR / "table6b_best_methods.csv"

# D2 (PROVISIONAL): the one number per track used to rank methods. Lower is better for both.
RANK_STAT = "stat_mean_abs_z_affected"
RANK_PRED = "rmse_gap"

STAT_COLUMNS = [
    "stat_mean_abs_z_affected",
    "stat_mean_signed_z_affected",
    "stat_ci_coverage_affected",
    "stat_mean_abs_z_all",
    "stat_ci_coverage_all",
    "stat_n_sig_flips",
]
PRED_COLUMNS = ["rmse", "mae", "r2", "rmse_gap", "rmse_gap_rel", "r2_gap"]
PRED_TRUTH_COLUMNS = ["rmse_truth_test", "r2_truth_test", "rmse_gap_truth_test", "r2_gap_truth_test"]

MECH_ORDER = ["MCAR", "MAR", "MNAR"]
METHOD_ORDER = ["mean", "knn", "regression", "random_forest"]


class MergeError(ValueError):
    """Raised when the two tracks cannot be merged one-to-one."""


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------
def _clean_keys(df: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = [c for c in KEY if c not in df.columns]
    if missing:
        raise MergeError(f"{name} is missing key columns: {missing}")
    out = df.copy()
    out["level"] = out["level"].astype(float).round(4)  # 0.1 vs 0.10000000000000001
    out["repeat"] = out["repeat"].astype(int)
    return out


def _assert_unique(df: pd.DataFrame, name: str) -> None:
    dup = df[df.duplicated(KEY, keep=False)]
    if len(dup):
        raise MergeError(
            f"{name} has {len(dup)} rows sharing a key (expected exactly one per key). "
            f"First: {dup[KEY].head(3).to_dict('records')}"
        )


# ---------------------------------------------------------------------------
# FR-4.1 merge
# ---------------------------------------------------------------------------
def merge_tracks(
    stat_runs: pd.DataFrame,
    pred_runs: pd.DataFrame,
    strict: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge the tracks one-to-one on KEY.

    Returns ``(merged, unmatched)``. With ``strict`` (default) any key present in only one track
    raises ``MergeError``. With ``strict=False`` the intersection is returned and the unmatched
    keys are listed in ``unmatched`` (columns: KEY + ``only_in``), for partial runs while
    Member 2 is still delivering.
    """
    stat = _clean_keys(stat_runs, "statistical results")
    pred = _clean_keys(pred_runs, "predictive results")
    stat = stat[stat["method"] != COMPLETE_CASE]  # statistical-only baseline (FR-3.8)
    _assert_unique(stat, "statistical results")
    _assert_unique(pred, "predictive results")

    stat_keys = stat[KEY].assign(only_in="statistical")
    pred_keys = pred[KEY].assign(only_in="predictive")
    outer = stat_keys.merge(pred_keys, on=KEY, how="outer", suffixes=("_s", "_p"), indicator=True)
    unmatched = outer[outer["_merge"] != "both"].copy()
    unmatched["only_in"] = np.where(unmatched["_merge"] == "left_only", "statistical", "predictive")
    unmatched = unmatched[KEY + ["only_in"]].reset_index(drop=True)

    if strict and len(unmatched):
        raise MergeError(
            f"{len(unmatched)} keys are in only one track "
            f"({unmatched['only_in'].value_counts().to_dict()}). "
            f"First: {unmatched.head(5).to_dict('records')}. "
            "Run both tracks on the same runs, or use --allow-partial."
        )

    overlap = [c for c in pred.columns if c in stat.columns and c not in KEY]
    merged = stat.merge(pred.drop(columns=overlap), on=KEY, how="inner", validate="one_to_one")
    if strict and len(merged) != len(stat):
        raise MergeError(f"merge kept {len(merged)} of {len(stat)} statistical rows")
    if merged.empty:
        raise MergeError("no run is present in both tracks")
    return merged.reset_index(drop=True), unmatched


# ---------------------------------------------------------------------------
# FR-4.2 / FR-4.3 recovery scores
# ---------------------------------------------------------------------------
def reference_row(reference: pd.DataFrame) -> pd.Series:
    """The primary complete-data reference row (``is_primary`` if present, otherwise the only row)."""
    if "is_primary" in reference.columns:
        primary = reference[reference["is_primary"].astype(bool)]
    else:
        primary = reference
    if len(primary) != 1:
        raise MergeError(f"expected exactly one primary reference row, found {len(primary)}")
    return primary.iloc[0]


def add_recovery_scores(merged: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Add rmse_gap and r2_gap versus the complete-data model (D2, PROVISIONAL)."""
    ref = reference_row(reference)
    out = merged.copy()
    out["rmse_ref"] = float(ref["rmse"])
    out["r2_ref"] = float(ref["r2"])
    out["rmse_gap"] = out["rmse"] - out["rmse_ref"]
    out["rmse_gap_rel"] = out["rmse_gap"] / out["rmse_ref"]  # gap as a share of the complete-data RMSE (used by M5)
    out["r2_gap"] = out["r2_ref"] - out["r2"]
    # Diagnostic (FR-3.7): model trained on imputed data, scored on COMPLETE test inputs.
    if {"rmse_truth_test", "r2_truth_test"} <= set(out.columns):
        out["rmse_gap_truth_test"] = out["rmse_truth_test"] - out["rmse_ref"]
        out["r2_gap_truth_test"] = out["r2_ref"] - out["r2_truth_test"]
    return out


# ---------------------------------------------------------------------------
# FR-4.4 / FR-4.5 Table 6
# ---------------------------------------------------------------------------
def _order(table: pd.DataFrame) -> pd.DataFrame:
    """Sort by mechanism, level and (if present) method in the project's standard order."""
    keys = table["mechanism"].map({m: i for i, m in enumerate(MECH_ORDER)}).fillna(99).to_frame("_k")
    keys["_l"] = table["level"]
    sort_cols = ["_k", "_l"]
    if "method" in table.columns:
        keys["_m"] = table["method"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(99)
        sort_cols.append("_m")
    order = keys.sort_values(sort_cols).index
    return table.loc[order].reset_index(drop=True)


def build_table6(merged: pd.DataFrame) -> pd.DataFrame:
    """Table 6: mean and sd over repeats by method x mechanism x level (statistical and predictive side by side)."""
    metrics = [c for c in STAT_COLUMNS + PRED_COLUMNS + PRED_TRUTH_COLUMNS if c in merged.columns]
    grouped = merged.groupby(CONDITION, sort=False)
    mean = grouped[metrics].mean().add_suffix("_mean")
    sd = grouped[metrics].std(ddof=1).add_suffix("_sd")
    n = grouped.size().rename("n_repeats")
    table = pd.concat([n, mean, sd], axis=1).reset_index()
    return _order(table)


# ---------------------------------------------------------------------------
# FR-4.6 best method per mechanism x level
# ---------------------------------------------------------------------------
def best_methods(
    table6: pd.DataFrame,
    stat_metric: str = RANK_STAT,
    pred_metric: str = RANK_PRED,
) -> pd.DataFrame:
    """Per mechanism x level: best predictive method, best statistical method, and whether they agree.

    Both metrics are "smaller is better" and are ranked on the repeat-averaged value.
    ``*_margin`` is runner-up minus winner (0 means a tie; compare it with the sd columns).
    """
    s_col, p_col = f"{stat_metric}_mean", f"{pred_metric}_mean"
    for col in (s_col, p_col):
        if col not in table6.columns:
            raise MergeError(f"Table 6 has no column '{col}'")

    rows = []
    for (mech, level), grp in table6.groupby(["mechanism", "level"], sort=False):
        grp = grp.sort_values(s_col)
        stat_best, stat_margin = _winner(grp, s_col)
        grp = grp.sort_values(p_col)
        pred_best, pred_margin = _winner(grp, p_col)
        rows.append(
            {
                "mechanism": mech,
                "level": level,
                "n_methods": len(grp),
                "best_predictive_method": pred_best,
                "best_statistical_method": stat_best,
                "same_method": pred_best == stat_best,
                "predictive_metric": pred_metric,
                "statistical_metric": stat_metric,
                "predictive_margin": pred_margin,
                "statistical_margin": stat_margin,
            }
        )
    out = pd.DataFrame(rows)
    return _order(out)


def _winner(sorted_grp: pd.DataFrame, col: str) -> tuple[str, float]:
    """Winner after sorting ascending; exact ties are reported as 'a|b' with margin 0."""
    best = sorted_grp[col].iloc[0]
    tied = sorted_grp.loc[np.isclose(sorted_grp[col], best, rtol=0, atol=1e-12), "method"].tolist()
    margin = float(sorted_grp[col].iloc[1] - best) if len(sorted_grp) > 1 else float("nan")
    tied.sort(key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)  # deterministic label
    return "|".join(tied), margin


# ---------------------------------------------------------------------------
# Pipeline and saving
# ---------------------------------------------------------------------------
def run_merge(
    stat_runs: pd.DataFrame | None = None,
    pred_runs: pd.DataFrame | None = None,
    reference: pd.DataFrame | None = None,
    strict: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load (unless injected), merge, score, aggregate. Returns merged, table6, best, unmatched."""
    stat_runs = load_stat_runs() if stat_runs is None else stat_runs
    pred_runs = load_predictive_runs() if pred_runs is None else pred_runs
    reference = load_predictive_reference() if reference is None else reference

    merged, unmatched = merge_tracks(stat_runs, pred_runs, strict=strict)
    merged = add_recovery_scores(merged, reference)
    table6 = build_table6(merged)
    best = best_methods(table6)
    return {"merged": merged, "table6": table6, "best": best, "unmatched": unmatched}


def save_merge_results(result: dict[str, pd.DataFrame]) -> dict[str, Path]:
    EXPERIMENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, key, path in [
        ("merged", "merged", MERGED_PATH),
        ("table6", "table6", TABLE6_PATH),
        ("table6b", "best", TABLE6B_PATH),
    ]:
        result[key].to_csv(path, index=False)
        written[name] = path
    return written


def load_merged(path: Path | str | None = None) -> pd.DataFrame:
    """Merged per-run table; used by Module M5 (central figure and Case flags)."""
    path = Path(path) if path else MERGED_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python3 scripts/m3_04_merge.py")
    return pd.read_csv(path)
