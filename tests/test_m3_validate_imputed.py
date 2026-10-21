"""Tests for Member 3 / Module 1 (imputed-file validation).

These tests use small synthetic data and an injected fake loader, so they do not
need the generated missingness data or Member 2's files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.m3_common import imputed_path, load_config  # noqa: E402
from src.m3_validate_imputed import (  # noqa: E402
    FAIL,
    FRAME_CHECKS,
    MISSING,
    PASS,
    all_passed,
    passed_runs,
    run_validation,
    validate_imputed_frame,
)

N = 200


@pytest.fixture(scope="module")
def cfg() -> dict:
    cfg = dict(load_config())
    cfg["expected_rows"] = N  # small synthetic data
    return cfg


def make_corrupted(cfg: dict, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        rng.normal(size=(N, len(cfg["expected_columns"]))), columns=cfg["expected_columns"]
    )
    frame.index.name = "row_index"
    for col in cfg["missingness_columns"]:
        frame.loc[rng.choice(N, 60, replace=False), col] = np.nan
    return frame


def mean_impute(corrupted: pd.DataFrame) -> pd.DataFrame:
    return corrupted.fillna(corrupted.mean())


# ---------------------------------------------------------------------------
# validate_imputed_frame
# ---------------------------------------------------------------------------
def test_valid_frame_passes_every_check(cfg):
    corrupted = make_corrupted(cfg)
    result = validate_imputed_frame(mean_impute(corrupted), corrupted, cfg)
    assert all(result[c] for c in FRAME_CHECKS), result["notes"]
    assert result["n_cells_to_impute"] == 240
    assert result["notes"] == []


def test_remaining_nan_is_detected(cfg):
    corrupted = make_corrupted(cfg)
    result = validate_imputed_frame(corrupted, corrupted, cfg)  # nothing imputed
    assert not result["no_nan"]
    assert result["n_nan"] == 240


def test_changed_observed_value_is_detected(cfg):
    corrupted = make_corrupted(cfg)
    imputed = mean_impute(corrupted)
    row = corrupted["grade"].dropna().index[0]  # an observed cell
    imputed.loc[row, "grade"] += 1.0
    result = validate_imputed_frame(imputed, corrupted, cfg)
    assert not result["observed_unchanged"]
    assert result["n_observed_changed"] == 1
    assert result["protected_unchanged"]


@pytest.mark.parametrize("col", ["price", "log_price", "lat", "long"])
def test_changed_protected_column_is_detected(cfg, col):
    corrupted = make_corrupted(cfg)
    imputed = mean_impute(corrupted)
    imputed.loc[3, col] += 0.5
    result = validate_imputed_frame(imputed, corrupted, cfg)
    assert not result["protected_unchanged"]
    assert result["n_protected_changed"] == 1


def test_changed_other_observed_column_is_detected(cfg):
    corrupted = make_corrupted(cfg)
    imputed = mean_impute(corrupted)
    imputed.loc[7, "bedrooms"] += 1.0
    result = validate_imputed_frame(imputed, corrupted, cfg)
    assert not result["other_unchanged"]


def test_wrong_columns_stop_early(cfg):
    corrupted = make_corrupted(cfg)
    imputed = mean_impute(corrupted).drop(columns=["lat"])
    result = validate_imputed_frame(imputed, corrupted, cfg)
    assert not result["columns_ok"]
    assert not result["observed_unchanged"]  # not evaluated
    assert any("columns differ" in n for n in result["notes"])


def test_reordered_columns_flagged_but_values_still_checked(cfg):
    corrupted = make_corrupted(cfg)
    imputed = mean_impute(corrupted)[list(reversed(cfg["expected_columns"]))]
    result = validate_imputed_frame(imputed, corrupted, cfg)
    assert not result["columns_ok"]
    assert result["observed_unchanged"] and result["protected_unchanged"]


def test_wrong_row_count(cfg):
    corrupted = make_corrupted(cfg)
    imputed = mean_impute(corrupted).iloc[:-1]
    result = validate_imputed_frame(imputed, corrupted, cfg)
    assert not result["row_count_ok"]
    assert not result["index_ok"]


def test_shuffled_rows_flagged_but_values_still_checked(cfg):
    corrupted = make_corrupted(cfg)
    imputed = mean_impute(corrupted).sample(frac=1.0, random_state=1)
    result = validate_imputed_frame(imputed, corrupted, cfg)
    assert not result["index_ok"]
    assert result["observed_unchanged"]  # compared after re-aligning
    assert any("different order" in n for n in result["notes"])


def test_different_index_values(cfg):
    corrupted = make_corrupted(cfg)
    imputed = mean_impute(corrupted)
    imputed.index = imputed.index + 1000
    result = validate_imputed_frame(imputed, corrupted, cfg)
    assert not result["index_ok"]
    assert not result["observed_unchanged"]  # not evaluated


def test_infinite_value_is_detected(cfg):
    corrupted = make_corrupted(cfg)
    imputed = mean_impute(corrupted)
    row = corrupted["sqft_living"].index[corrupted["sqft_living"].isna()][0]
    imputed.loc[row, "sqft_living"] = np.inf
    result = validate_imputed_frame(imputed, corrupted, cfg)
    assert not result["no_inf"]


def test_non_numeric_values_are_detected(cfg):
    corrupted = make_corrupted(cfg)
    imputed = mean_impute(corrupted).astype(object)
    imputed.loc[0, "grade"] = "high"
    result = validate_imputed_frame(imputed, corrupted, cfg)
    assert not result["numeric_ok"]


# ---------------------------------------------------------------------------
# run_validation (files on disk, fake loader)
# ---------------------------------------------------------------------------
def _write(path: Path, frame: pd.DataFrame, with_index_column: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.reset_index() if with_index_column else frame.reset_index(drop=True)
    out.to_csv(path, index=False, compression="gzip")


def _loader_for(corrupted: pd.DataFrame):
    def loader(mechanism, level, repeat):
        return {"corrupted": corrupted}

    return loader


FILTERS = dict(methods=["mean"], mechanisms=["MAR"], levels=[0.30])


def test_run_validation_pass_fail_missing(cfg, tmp_path):
    corrupted = make_corrupted(cfg)
    good = mean_impute(corrupted)
    bad = good.copy()
    bad.loc[corrupted["grade"].dropna().index[0], "grade"] += 5

    _write(imputed_path("mean", "MAR", 0.30, 0, imputed_root=tmp_path), good)
    _write(imputed_path("mean", "MAR", 0.30, 1, imputed_root=tmp_path), bad)
    # repeats 2, 3, 4 are never written -> MISSING

    report = run_validation(cfg, imputed_root=tmp_path, loader=_loader_for(corrupted), **FILTERS)
    status = dict(zip(report["repeat"], report["status"]))
    assert status == {0: PASS, 1: FAIL, 2: MISSING, 3: MISSING, 4: MISSING}
    assert len(report) == 5
    assert "observed cells" in report.loc[report["repeat"] == 1, "notes"].iloc[0]


def test_run_validation_does_not_load_data_when_nothing_exists(cfg, tmp_path):
    def exploding_loader(**kwargs):
        raise AssertionError("loader must not be called when no imputed file exists")

    report = run_validation(cfg, imputed_root=tmp_path, loader=exploding_loader, **FILTERS)
    assert (report["status"] == MISSING).all()


def test_file_without_row_index_column_fails(cfg, tmp_path):
    corrupted = make_corrupted(cfg)
    _write(
        imputed_path("mean", "MAR", 0.30, 0, imputed_root=tmp_path),
        mean_impute(corrupted),
        with_index_column=False,
    )
    report = run_validation(cfg, imputed_root=tmp_path, loader=_loader_for(corrupted), **FILTERS)
    first = report[report["repeat"] == 0].iloc[0]
    assert first["status"] == FAIL
    assert "row_index" in first["notes"]


def test_passed_runs_and_all_passed(cfg, tmp_path):
    corrupted = make_corrupted(cfg)
    for rep in range(5):
        _write(imputed_path("mean", "MAR", 0.30, rep, imputed_root=tmp_path), mean_impute(corrupted))
    report = run_validation(cfg, imputed_root=tmp_path, loader=_loader_for(corrupted), **FILTERS)
    assert all_passed(report)
    assert len(passed_runs(report)) == 5
    assert list(passed_runs(report).columns) == ["method", "mechanism", "level", "repeat"]


def test_all_passed_respects_allow_missing(cfg, tmp_path):
    corrupted = make_corrupted(cfg)
    _write(imputed_path("mean", "MAR", 0.30, 0, imputed_root=tmp_path), mean_impute(corrupted))
    report = run_validation(cfg, imputed_root=tmp_path, loader=_loader_for(corrupted), **FILTERS)
    assert not all_passed(report)  # strict: 4 files missing
    assert all_passed(report, allow_missing=True)


def test_full_sweep_expects_300_rows(cfg, tmp_path):
    def exploding_loader(**kwargs):
        raise AssertionError("not needed")

    report = run_validation(cfg, imputed_root=tmp_path, loader=exploding_loader)
    assert len(report) == 300
    assert (report["status"] == MISSING).all()
