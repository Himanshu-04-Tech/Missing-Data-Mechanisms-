"""Tests for Member 3 / Module 4 (merge, recovery scores, Table 6). Hand-built data, no real files needed."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.m3_merge import (  # noqa: E402
    MergeError,
    add_recovery_scores,
    best_methods,
    build_table6,
    merge_tracks,
    reference_row,
    run_merge,
    save_merge_results,
)

REF = pd.DataFrame(
    [
        {"model": "complete_data", "seed": 42, "is_primary": True, "rmse": 0.17, "mae": 0.12, "r2": 0.89},
        {"model": "complete_data", "seed": 43, "is_primary": False, "rmse": 0.18, "mae": 0.13, "r2": 0.88},
    ]
)


def make_tracks(methods=("mean", "knn"), mechs=("MCAR", "MAR"), levels=(0.1, 0.5), reps=3):
    """Small hand-built tracks where the answer is known.

    mean: bad statistically (z = 2.0 + 0.1*rep) but best predictively (rmse = 0.18)
    knn : good statistically (z = 0.5 + 0.1*rep) but worse predictively (rmse = 0.20)
    """
    stat, pred = [], []
    for m in methods:
        for mech in mechs:
            for lv in levels:
                for r in range(reps):
                    base_z = 2.0 if m == "mean" else 0.5
                    stat.append(dict(method=m, mechanism=mech, level=lv, repeat=r,
                                     stat_mean_abs_z_affected=base_z + 0.1 * r,
                                     stat_mean_signed_z_affected=-base_z, stat_ci_coverage_affected=0.5))
                    pred.append(dict(method=m, mechanism=mech, level=lv, repeat=r,
                                     rmse=0.18 if m == "mean" else 0.20, mae=0.13, r2=0.86 if m == "mean" else 0.84,
                                     rmse_truth_test=0.17, r2_truth_test=0.88))
    return pd.DataFrame(stat), pd.DataFrame(pred)


# --- FR-4.1 merge uniqueness ------------------------------------------------
def test_merge_is_one_to_one():
    stat, pred = make_tracks()
    merged, unmatched = merge_tracks(stat, pred)
    assert len(merged) == len(stat) == len(pred) and unmatched.empty
    assert not merged.duplicated(["method", "mechanism", "level", "repeat"]).any()


def test_duplicate_key_is_rejected():
    stat, pred = make_tracks()
    with pytest.raises(MergeError, match="sharing a key"):
        merge_tracks(pd.concat([stat, stat.head(1)]), pred)
    with pytest.raises(MergeError, match="sharing a key"):
        merge_tracks(stat, pd.concat([pred, pred.head(1)]))


def test_unmatched_rows_fail_in_strict_mode():
    stat, pred = make_tracks()
    with pytest.raises(MergeError, match="only one track"):
        merge_tracks(stat, pred.iloc[1:])
    with pytest.raises(MergeError, match="only one track"):
        merge_tracks(stat.iloc[1:], pred)


def test_partial_mode_reports_unmatched():
    stat, pred = make_tracks()
    merged, unmatched = merge_tracks(stat, pred.iloc[2:], strict=False)
    assert len(merged) == len(pred) - 2 and len(unmatched) == 2
    assert set(unmatched["only_in"]) == {"statistical"}


def test_complete_case_is_set_aside_not_unmatched():
    stat, pred = make_tracks()
    cc = stat[stat["method"] == "mean"].assign(method="complete_case")
    merged, unmatched = merge_tracks(pd.concat([stat, cc]), pred)
    assert "complete_case" not in set(merged["method"]) and unmatched.empty


def test_float_level_noise_does_not_break_matching():
    stat, pred = make_tracks()
    pred = pred.assign(level=pred["level"] + 1e-12)
    merged, _ = merge_tracks(stat, pred)
    assert len(merged) == len(stat)


def test_missing_key_column_is_rejected():
    stat, pred = make_tracks()
    with pytest.raises(MergeError, match="missing key"):
        merge_tracks(stat.drop(columns=["repeat"]), pred)


# --- FR-4.2 gaps ------------------------------------------------------------
def test_reference_row_is_the_primary_seed():
    assert reference_row(REF)["seed"] == 42
    with pytest.raises(MergeError):
        reference_row(REF.assign(is_primary=True))


def test_gaps_use_primary_reference():
    stat, pred = make_tracks()
    merged, _ = merge_tracks(stat, pred)
    scored = add_recovery_scores(merged, REF)
    mean_rows = scored[scored["method"] == "mean"]
    assert np.allclose(mean_rows["rmse_gap"], 0.18 - 0.17)
    assert np.allclose(mean_rows["r2_gap"], 0.89 - 0.86)
    assert np.allclose(scored["rmse_gap_truth_test"], 0.17 - 0.17)
    assert np.allclose(mean_rows["rmse_gap_rel"], 0.01 / 0.17)


# --- FR-4.4 / 4.5 Table 6 ---------------------------------------------------
def test_table6_mean_and_sd_over_repeats():
    stat, pred = make_tracks()
    merged, _ = merge_tracks(stat, pred)
    table = build_table6(add_recovery_scores(merged, REF))
    assert len(table) == 2 * 2 * 2 and (table["n_repeats"] == 3).all()
    row = table[(table["method"] == "mean") & (table["mechanism"] == "MAR") & (table["level"] == 0.5)].iloc[0]
    assert row["stat_mean_abs_z_affected_mean"] == pytest.approx(2.1)       # mean of 2.0, 2.1, 2.2
    assert row["stat_mean_abs_z_affected_sd"] == pytest.approx(0.1)         # sd (ddof=1) of 2.0, 2.1, 2.2
    assert row["rmse_gap_mean"] == pytest.approx(0.01)
    # statistical and predictive columns are side by side
    assert {"stat_ci_coverage_affected_mean", "r2_gap_mean", "rmse_gap_sd"} <= set(table.columns)


def test_table_order_is_mechanism_level_method():
    stat, pred = make_tracks(methods=("knn", "mean"), mechs=("MAR", "MCAR"))
    merged, _ = merge_tracks(stat, pred)
    table = build_table6(add_recovery_scores(merged, REF))
    assert list(table["mechanism"].unique()) == ["MCAR", "MAR"]
    assert list(table["method"].iloc[:2]) == ["mean", "knn"]


# --- FR-4.6 best methods ----------------------------------------------------
def test_best_methods_disagree_when_tracks_disagree():
    stat, pred = make_tracks()
    merged, _ = merge_tracks(stat, pred)
    best = best_methods(build_table6(add_recovery_scores(merged, REF)))
    assert len(best) == 4
    assert (best["best_predictive_method"] == "mean").all()        # lowest rmse_gap
    assert (best["best_statistical_method"] == "knn").all()        # lowest |z|
    assert not best["same_method"].any()
    assert np.allclose(best["predictive_margin"], 0.02)
    assert np.allclose(best["statistical_margin"], 1.5)


def test_best_methods_agree_when_one_method_wins_both():
    stat, pred = make_tracks()
    pred = pred.assign(rmse=np.where(pred["method"] == "knn", 0.175, 0.19))
    merged, _ = merge_tracks(stat, pred)
    best = best_methods(build_table6(add_recovery_scores(merged, REF)))
    assert best["same_method"].all() and (best["best_predictive_method"] == "knn").all()


def test_exact_ties_are_reported_not_hidden():
    stat, pred = make_tracks()
    pred = pred.assign(rmse=0.19)
    merged, _ = merge_tracks(stat, pred)
    best = best_methods(build_table6(add_recovery_scores(merged, REF)))
    assert (best["best_predictive_method"] == "mean|knn").all() and (best["predictive_margin"] == 0).all()


def test_unknown_rank_metric_is_rejected():
    stat, pred = make_tracks()
    merged, _ = merge_tracks(stat, pred)
    with pytest.raises(MergeError, match="no column"):
        best_methods(build_table6(add_recovery_scores(merged, REF)), pred_metric="nonsense")


# --- pipeline and saving ----------------------------------------------------
def test_run_merge_and_save(tmp_path, monkeypatch):
    import src.m3_merge as mod

    stat, pred = make_tracks()
    out = run_merge(stat_runs=stat, pred_runs=pred, reference=REF)
    assert set(out) == {"merged", "table6", "best", "unmatched"}
    monkeypatch.setattr(mod, "EXPERIMENT_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(mod, "TABLES_DIR", tmp_path)
    monkeypatch.setattr(mod, "MERGED_PATH", tmp_path / "merged.csv")
    monkeypatch.setattr(mod, "TABLE6_PATH", tmp_path / "t6.csv")
    monkeypatch.setattr(mod, "TABLE6B_PATH", tmp_path / "t6b.csv")
    written = save_merge_results(out)
    assert all(p.exists() for p in written.values())
    assert len(mod.load_merged(tmp_path / "merged.csv")) == len(stat)
    again = run_merge(stat_runs=stat, pred_runs=pred, reference=REF)
    pd.testing.assert_frame_equal(out["table6"], again["table6"])  # reproducible, no manual edits
