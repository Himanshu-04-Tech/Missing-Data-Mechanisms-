"""Tests for Member 3 / Module 5 (Case flags, sensitivity, figures). Hand-built data; no real files needed."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.m3_case_plots import plot_case_figures  # noqa: E402
from src.m3_cases import (  # noqa: E402
    Thresholds,
    build_condition_scores,
    case_counts,
    get_thresholds,
    label_cases,
    run_cases,
    save_case_results,
    sensitivity,
)
from src.m3_common import load_config  # noqa: E402
from src.m3_merge import MergeError, add_recovery_scores, merge_tracks  # noqa: E402

REF = pd.DataFrame([{"model": "complete_data", "seed": 42, "is_primary": True, "rmse": 0.20, "mae": 0.1, "r2": 0.9}])
THR = Thresholds("stat_mean_abs_z_affected", 1.0, "rmse_gap_rel", 0.05, locked=False)

# (method, stat |z|, rmse) chosen so each method lands in a known Case under THR (rmse_gap_rel = (rmse-0.2)/0.2)
#   mean: z 3.0, rmse 0.205 -> rel 0.025  => stat bad, pred good  => B
#   knn : z 0.5, rmse 0.205 -> rel 0.025  => stat good, pred good => A
#   regression: z 0.5, rmse 0.25 -> rel 0.25 => stat good, pred bad => C
#   random_forest: z 3.0, rmse 0.25      => D
PLAN = {"mean": (3.0, 0.205), "knn": (0.5, 0.205), "regression": (0.5, 0.25), "random_forest": (3.0, 0.25)}
EXPECTED = {"mean": "B", "knn": "A", "regression": "C", "random_forest": "D"}


def make_merged(plan=PLAN, mechs=("MCAR", "MNAR"), levels=(0.1, 0.5), reps=3):
    stat, pred = [], []
    for m, (z, rmse) in plan.items():
        for mech in mechs:
            for lv in levels:
                for r in range(reps):
                    stat.append(dict(method=m, mechanism=mech, level=lv, repeat=r,
                                     stat_mean_abs_z_affected=z + 0.01 * r, stat_ci_coverage_affected=0.5))
                    pred.append(dict(method=m, mechanism=mech, level=lv, repeat=r, rmse=rmse, mae=0.1, r2=0.85))
    merged, _ = merge_tracks(pd.DataFrame(stat), pd.DataFrame(pred))
    return add_recovery_scores(merged, REF)


@pytest.fixture()
def scores():
    return build_condition_scores(make_merged())


def test_cases_match_hand_computed_labels(scores):
    lab = label_cases(scores, THR)
    assert len(lab) == 4 * 2 * 2
    for method, case in EXPECTED.items():
        assert (lab.loc[lab["method"] == method, "case"] == case).all()


def test_threshold_boundary_counts_as_good():
    merged = make_merged({"mean": (1.0, 0.21)})  # |z| mean slightly above 1 (repeat offsets), rmse rel exactly 0.05
    s = build_condition_scores(merged)
    s["stat_mean_abs_z_affected_mean"] = 1.0
    s["rmse_gap_rel_mean"] = 0.05
    assert (label_cases(s, THR)["case"] == "A").all()
    s["stat_mean_abs_z_affected_mean"] = 1.0000001
    assert (label_cases(s, THR)["case"] == "B").all()


def test_label_marks_provisional_until_locked(scores):
    assert (label_cases(scores, THR)["thresholds_status"] == "PROVISIONAL").all()
    locked = Thresholds(THR.stat_metric, 1.0, THR.pred_metric, 0.05, locked=True)
    assert (label_cases(scores, locked)["thresholds_status"] == "LOCKED").all()


def test_nan_scores_are_refused(scores):
    bad = scores.copy()
    bad.loc[0, "rmse_gap_rel_mean"] = np.nan
    with pytest.raises(MergeError, match="NaN"):
        label_cases(bad, THR)


def test_unknown_metric_is_refused(scores):
    with pytest.raises(MergeError, match="no column"):
        label_cases(scores, Thresholds("nonsense", 1.0, "rmse_gap_rel", 0.05, False))


def test_merged_without_relative_gap_is_refused():
    with pytest.raises(MergeError, match="rmse_gap_rel"):
        build_condition_scores(make_merged().drop(columns=["rmse_gap_rel"]))


def test_case_counts_by_method_and_mechanism(scores):
    counts = case_counts(label_cases(scores, THR))
    assert len(counts) == 4 * 2 and (counts["n_conditions"] == 2).all()
    row = counts[(counts["method"] == "knn") & (counts["mechanism"] == "MCAR")].iloc[0]
    assert (row["A"], row["B"], row["C"], row["D"]) == (2, 0, 0, 0)
    assert counts[list("ABCD")].to_numpy().sum() == 16


def test_sensitivity_is_baseline_consistent_and_monotone(scores):
    one, grid = sensitivity(scores, THR)
    base = one[one["is_baseline"]]
    assert (base["n_changed"] == 0).all()
    assert (one[list("ABCD")].sum(axis=1) == 16).all()
    stat = one[one["varied"] == "stat"].sort_values("factor")
    # loosening the statistical threshold can only move conditions into 'statistically good' (A or C)
    assert (stat["A"] + stat["C"]).is_monotonic_increasing
    pred = one[one["varied"] == "pred"].sort_values("factor")
    assert (pred["A"] + pred["B"]).is_monotonic_increasing
    assert len(grid) == len(THR.sensitivity_factors) ** 2
    assert grid.loc[(grid.stat_factor == 1) & (grid.pred_factor == 1), "n_changed"].iloc[0] == 0


def test_sensitivity_shows_a_flip(scores):
    # tighten the statistical threshold below knn's |z| of 0.5: A -> B and C -> D for the good-|z| methods
    one, _ = sensitivity(scores, THR)
    row = one[(one["varied"] == "stat") & (one["factor"] == 0.5)].iloc[0]
    assert row["n_changed"] == 8 and row["A"] == 0 and row["C"] == 0


def test_get_thresholds_from_real_config():
    thr = get_thresholds(load_config())
    assert thr.stat_threshold > 0 and thr.pred_threshold > 0
    assert thr.stat_metric == "stat_mean_abs_z_affected" and thr.pred_metric == "rmse_gap_rel"


def test_missing_cases_block_fails_loudly():
    with pytest.raises(MergeError, match="cases"):
        get_thresholds({"base_seed": 1})
    with pytest.raises(MergeError, match="stat_threshold"):
        get_thresholds({"cases": {"stat_metric": "a", "pred_metric": "b", "pred_threshold": 1}})
    with pytest.raises(MergeError, match="positive"):
        get_thresholds({"cases": {"stat_metric": "a", "stat_threshold": 0, "pred_metric": "b", "pred_threshold": 1}})


def test_run_cases_save_and_figures(tmp_path, monkeypatch):
    import src.m3_cases as mod

    result = run_cases(make_merged(), THR)
    assert set(result) == {"labelled", "counts", "sensitivity", "grid"}
    for name in ("TABLES_DIR", "EXPERIMENT_RESULTS_DIR"):
        monkeypatch.setattr(mod, name, tmp_path)
    for name in ("CASES_BY_CONDITION_PATH", "CASE_COUNTS_PATH", "SENSITIVITY_PATH", "SENSITIVITY_GRID_PATH", "THRESHOLDS_USED_PATH"):
        monkeypatch.setattr(mod, name, tmp_path / f"{name}.out")
    written = save_case_results(result, THR)
    assert all(p.exists() for p in written.values())
    figs = plot_case_figures(result["labelled"], result["grid"], out_dir=tmp_path)
    assert len(figs) == 4 and all(p.exists() and p.stat().st_size > 5000 for p in figs)
    again = run_cases(make_merged(), THR)
    pd.testing.assert_frame_equal(result["labelled"], again["labelled"])  # reproducible
