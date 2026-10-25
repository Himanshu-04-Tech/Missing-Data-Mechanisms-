"""Module M5 (part 1): Case A/B/C/D flagging, case counts and the threshold sensitivity check.

NEW file; does not modify any existing module in ``src/``.

Cases (decision D3; thresholds live in ``config.yaml`` under ``cases:``)
------------------------------------------------------------------------
A condition is one method x mechanism x level. Its statistical score and predictive score are the
repeat-averaged values from Table 6. Both scores are "smaller is better".

  statistically good :  stat_mean_abs_z_affected_mean <= stat_threshold     (reference-SE units)
  predictively good  :  rmse_gap_rel_mean             <= pred_threshold     (RMSE gap / complete-data RMSE)

  Case A  statistically good and predictively good
  Case B  predictively good but statistically distorted   (the central finding we are looking for)
  Case C  statistically good but predicts poorly
  Case D  poor on both

``rmse_gap_rel = rmse_gap / rmse_ref`` where ``rmse_gap`` and ``rmse_ref`` come from Module M4.

Honesty notes
-------------
* The thresholds must be committed BEFORE the imputed results are viewed. ``cases.thresholds_locked``
  in the config records that the team agreed; while it is false every output is marked PROVISIONAL.
* A Case label describes one condition under these two thresholds; it is not a ranking of methods.
* The sensitivity check (FR-5.4) shows how many labels depend on the exact threshold values.
  Report it next to any Case count.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.m3_common import EXPERIMENT_RESULTS_DIR, PROJECT_ROOT, TABLES_DIR, load_config
from src.m3_merge import METHOD_ORDER, MECH_ORDER, MergeError, build_table6

CASES = ["A", "B", "C", "D"]
CASE_MEANING = {
    "A": "good on both",
    "B": "predicts well, statistically distorted",
    "C": "statistically good, predicts poorly",
    "D": "poor on both",
}

CASES_BY_CONDITION_PATH = TABLES_DIR / "cases_by_condition.csv"
CASE_COUNTS_PATH = TABLES_DIR / "table7_case_counts.csv"
SENSITIVITY_PATH = TABLES_DIR / "case_sensitivity.csv"
SENSITIVITY_GRID_PATH = TABLES_DIR / "case_sensitivity_grid.csv"
THRESHOLDS_USED_PATH = EXPERIMENT_RESULTS_DIR / "case_thresholds_used.json"

DEFAULT_FACTORS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Thresholds:
    stat_metric: str
    stat_threshold: float
    pred_metric: str
    pred_threshold: float
    locked: bool
    sensitivity_factors: tuple[float, ...] = tuple(DEFAULT_FACTORS)

    def scaled(self, stat_factor: float = 1.0, pred_factor: float = 1.0) -> "Thresholds":
        return Thresholds(
            self.stat_metric, self.stat_threshold * stat_factor,
            self.pred_metric, self.pred_threshold * pred_factor,
            self.locked, self.sensitivity_factors,
        )


def get_thresholds(cfg: dict | None = None) -> Thresholds:
    """Read the ``cases:`` block of the config. Fails loudly if it is missing (no silent defaults)."""
    cfg = load_config() if cfg is None else cfg
    block = cfg.get("cases")
    if not block:
        raise MergeError("config.yaml has no 'cases:' block. Add the Case thresholds (decision D3) and commit them first.")
    for key in ("stat_metric", "stat_threshold", "pred_metric", "pred_threshold"):
        if key not in block:
            raise MergeError(f"config.yaml cases: block is missing '{key}'")
    thr = Thresholds(
        stat_metric=str(block["stat_metric"]),
        stat_threshold=float(block["stat_threshold"]),
        pred_metric=str(block["pred_metric"]),
        pred_threshold=float(block["pred_threshold"]),
        locked=bool(block.get("thresholds_locked", False)),
        sensitivity_factors=tuple(float(f) for f in block.get("sensitivity_factors", DEFAULT_FACTORS)),
    )
    if thr.stat_threshold <= 0 or thr.pred_threshold <= 0:
        raise MergeError("Case thresholds must be positive")
    return thr


# ---------------------------------------------------------------------------
# Condition table and labels
# ---------------------------------------------------------------------------
def build_condition_scores(merged: pd.DataFrame) -> pd.DataFrame:
    """Table 6 (mean and sd over repeats per method x mechanism x level), from the Module M4 merged runs."""
    if "rmse_gap_rel" not in merged.columns:
        raise MergeError("merged results have no 'rmse_gap_rel'; re-run Module M4 (python scripts/m3_04_merge.py)")
    return build_table6(merged)


def label_cases(scores: pd.DataFrame, thresholds: Thresholds) -> pd.DataFrame:
    """Return ``scores`` with the Case label for every condition (FR-5.3).

    Adds: stat_value, pred_value, stat_good, pred_good, case, case_meaning, thresholds_status.
    """
    s_col, p_col = f"{thresholds.stat_metric}_mean", f"{thresholds.pred_metric}_mean"
    for col in (s_col, p_col):
        if col not in scores.columns:
            raise MergeError(f"condition table has no column '{col}'")
    out = scores.copy()
    if out[[s_col, p_col]].isna().any().any():
        raise MergeError("NaN in the Case scores; cannot label (a repeat may have failed upstream)")
    out["stat_value"] = out[s_col]
    out["pred_value"] = out[p_col]
    out["stat_good"] = out["stat_value"] <= thresholds.stat_threshold
    out["pred_good"] = out["pred_value"] <= thresholds.pred_threshold
    out["case"] = np.select(
        [out["stat_good"] & out["pred_good"], ~out["stat_good"] & out["pred_good"], out["stat_good"] & ~out["pred_good"]],
        ["A", "B", "C"],
        default="D",
    )
    out["case_meaning"] = out["case"].map(CASE_MEANING)
    out["stat_threshold"] = thresholds.stat_threshold
    out["pred_threshold"] = thresholds.pred_threshold
    out["thresholds_status"] = "LOCKED" if thresholds.locked else "PROVISIONAL"
    return out


def case_counts(labelled: pd.DataFrame) -> pd.DataFrame:
    """Counts of Cases A to D by method x mechanism (over the missingness levels)."""
    counts = (
        labelled.groupby(["method", "mechanism"])["case"].value_counts().unstack(fill_value=0)
        .reindex(columns=CASES, fill_value=0)
    )
    counts["n_conditions"] = counts.sum(axis=1)
    counts = counts.reset_index()
    counts.columns.name = None
    counts["thresholds_status"] = labelled["thresholds_status"].iloc[0]
    return _order(counts)


def _order(table: pd.DataFrame) -> pd.DataFrame:
    keys = pd.DataFrame(index=table.index)
    keys["_k"] = table["mechanism"].map({m: i for i, m in enumerate(MECH_ORDER)}).fillna(99)
    keys["_m"] = table["method"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(99)
    return table.loc[keys.sort_values(["_m", "_k"]).index].reset_index(drop=True)


# ---------------------------------------------------------------------------
# FR-5.4 sensitivity
# ---------------------------------------------------------------------------
def _counts_row(labelled: pd.DataFrame) -> dict:
    vc = labelled["case"].value_counts()
    return {c: int(vc.get(c, 0)) for c in CASES}


def sensitivity(scores: pd.DataFrame, thresholds: Thresholds) -> tuple[pd.DataFrame, pd.DataFrame]:
    """How the Case counts change when each threshold is moved up and down.

    Returns ``(one_at_a_time, grid)``:
      * one_at_a_time: one threshold is multiplied by each factor while the other stays at its configured value;
        ``n_changed`` = conditions whose label differs from the baseline labelling.
      * grid: both thresholds multiplied at once; number of Case A..D for every factor pair.
    """
    base = label_cases(scores, thresholds)["case"].to_numpy()
    rows = []
    for which in ("stat", "pred"):
        for f in thresholds.sensitivity_factors:
            t = thresholds.scaled(stat_factor=f if which == "stat" else 1.0, pred_factor=f if which == "pred" else 1.0)
            lab = label_cases(scores, t)
            rows.append(
                {
                    "varied": which,
                    "factor": f,
                    "stat_threshold": t.stat_threshold,
                    "pred_threshold": t.pred_threshold,
                    **_counts_row(lab),
                    "n_changed": int((lab["case"].to_numpy() != base).sum()),
                    "is_baseline": bool(np.isclose(f, 1.0)),
                }
            )
    grid_rows = []
    for fs in thresholds.sensitivity_factors:
        for fp in thresholds.sensitivity_factors:
            t = thresholds.scaled(fs, fp)
            lab = label_cases(scores, t)
            grid_rows.append(
                {"stat_factor": fs, "pred_factor": fp, "stat_threshold": t.stat_threshold,
                 "pred_threshold": t.pred_threshold, **_counts_row(lab),
                 "n_changed": int((lab["case"].to_numpy() != base).sum())}
            )
    one = pd.DataFrame(rows)
    one["thresholds_status"] = "LOCKED" if thresholds.locked else "PROVISIONAL"
    return one, pd.DataFrame(grid_rows)


# ---------------------------------------------------------------------------
# Pipeline and saving
# ---------------------------------------------------------------------------
def run_cases(merged: pd.DataFrame, thresholds: Thresholds | None = None) -> dict[str, pd.DataFrame]:
    """Merged runs (from M4) -> labelled conditions, counts by method x mechanism, sensitivity tables."""
    thresholds = get_thresholds() if thresholds is None else thresholds
    scores = build_condition_scores(merged)
    labelled = label_cases(scores, thresholds)
    one, grid = sensitivity(scores, thresholds)
    return {
        "labelled": labelled,
        "counts": case_counts(labelled),
        "sensitivity": one,
        "grid": grid,
    }


def _config_commit() -> str | None:
    """Best effort: the last git commit that touched config.yaml (for traceability of the thresholds)."""
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H %cI", "--", "config.yaml"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10, check=False,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 (git may be absent; this is only a convenience)
        return None


def save_case_results(result: dict[str, pd.DataFrame], thresholds: Thresholds) -> dict[str, Path]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, key, path in [
        ("cases", "labelled", CASES_BY_CONDITION_PATH),
        ("table7", "counts", CASE_COUNTS_PATH),
        ("sens", "sensitivity", SENSITIVITY_PATH),
        ("grid", "grid", SENSITIVITY_GRID_PATH),
    ]:
        result[key].to_csv(path, index=False)
        written[name] = path
    record = {**asdict(thresholds), "sensitivity_factors": list(thresholds.sensitivity_factors),
              "config_last_commit": _config_commit()}
    THRESHOLDS_USED_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")
    written["thresholds"] = THRESHOLDS_USED_PATH
    return written
