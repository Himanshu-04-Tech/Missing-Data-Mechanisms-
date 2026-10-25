"""Tests for Member 3 / Module 3 (predictive track).

Synthetic data only, so no real data or Member 2 files are needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.m3_common import imputed_path, load_config  # noqa: E402
from src.m3_predictive import (  # noqa: E402
    DEFAULT_RF_PARAMS,
    build_predictive_table,
    feature_columns,
    fit_and_score,
    get_settings,
    load_split,
    run_predictive_track,
    run_reference,
    save_predictive_results,
)
from src.m3_validate_imputed import PASS, run_validation  # noqa: E402

N = 400


@pytest.fixture(scope="module")
def cfg() -> dict:
    cfg = dict(load_config())
    cfg["expected_rows"] = N
    cfg["predictive"] = {"random_state": 7, "score_on_truth": True,
                         "rf_params": {"n_estimators": 20, "n_jobs": 1}}
    return cfg


@pytest.fixture(scope="module")
def settings(cfg):
    return get_settings(cfg)


@pytest.fixture(scope="module")
def complete(cfg) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    feats = feature_columns(cfg)
    X = pd.DataFrame(rng.normal(size=(N, len(feats))), columns=feats)
    y = 2.0 * X["sqft_living"] + X["grade"] - X["lat"] + rng.normal(scale=0.3, size=N)
    frame = X.copy()
    frame["log_price"] = y
    frame["price"] = np.exp(y)
    frame = frame[cfg["expected_columns"]]
    frame.index.name = "row_index"
    return frame


@pytest.fixture(scope="module")
def split():
    idx = np.random.default_rng(1).permutation(N)
    return np.sort(idx[: int(0.8 * N)]), np.sort(idx[int(0.8 * N):])


def corrupt_and_mean_impute(complete, cfg, train_idx, rate=0.3, seed=2):
    """Corrupt the 4 columns, then mean-impute with TRAIN means (like a correct imputer)."""
    rng = np.random.default_rng(seed)
    corrupted = complete.copy()
    for col in cfg["missingness_columns"]:
        corrupted.loc[rng.choice(N, int(rate * N), replace=False), col] = np.nan
    imputed = corrupted.copy()
    for col in cfg["missingness_columns"]:
        imputed[col] = imputed[col].fillna(corrupted.loc[train_idx, col].mean())
    return corrupted, imputed


# ---------------------------------------------------------------------------
# Settings and leakage guards
# ---------------------------------------------------------------------------
def test_features_are_the_15_predictors_without_target(cfg):
    feats = feature_columns(cfg)
    assert len(feats) == 15
    assert "price" not in feats and "log_price" not in feats


def test_settings_come_from_config_and_overrides(cfg):
    s = get_settings(cfg)
    assert s.rf_params["n_estimators"] == 20 and s.random_state == 7
    s2 = get_settings(cfg, {"n_estimators": 5, "random_state": None})
    assert s2.rf_params["n_estimators"] == 5 and s2.random_state == 7


def test_defaults_when_config_has_no_predictive_block():
    s = get_settings({"base_seed": 42, "target": "price", "log_target_name": "log_price",
                      "expected_columns": ["price", "log_price", "a", "b"]})
    assert s.rf_params == DEFAULT_RF_PARAMS and s.random_state == 42


def test_real_config_has_fixed_settings():
    s = get_settings(load_config())
    assert s.rf_params["n_estimators"] == 300 and s.random_state == 42


def test_real_split_is_disjoint_and_correct_size():
    train, test = load_split(load_config())
    assert len(train) == 17148 and len(test) == 4288
    assert not set(train) & set(test)


def test_nan_input_is_refused(cfg, settings, complete, split):
    bad = complete.copy()
    bad.iloc[0, bad.columns.get_loc("grade")] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        fit_and_score(bad, *split, settings)


# ---------------------------------------------------------------------------
# FR-6.2: fitted on TRAIN rows only
# ---------------------------------------------------------------------------
def test_changing_test_labels_does_not_change_the_model(cfg, settings, complete, split):
    train_idx, test_idx = split
    base = fit_and_score(complete, train_idx, test_idx, settings)
    changed = complete.copy()
    changed.loc[test_idx, "log_price"] += 5.0  # test labels only
    again = fit_and_score(changed, train_idx, test_idx, settings)
    # identical model => identical predictions => the error moves only because the labels moved
    assert again["rmse"] > base["rmse"]
    # directly compare predictions of the two fitted models
    from src.m3_predictive import make_model
    feats, target = settings.features, settings.target
    m1 = make_model(settings).fit(complete.loc[train_idx, feats], complete.loc[train_idx, target])
    m2 = make_model(settings).fit(changed.loc[train_idx, feats], changed.loc[train_idx, target])
    assert np.array_equal(m1.predict(complete.loc[test_idx, feats]), m2.predict(complete.loc[test_idx, feats]))


def test_changing_test_features_does_not_change_the_model_fit(cfg, settings, complete, split):
    """Test-row inputs must not influence training: training-row predictions stay identical."""
    train_idx, test_idx = split
    from src.m3_predictive import make_model
    feats, target = settings.features, settings.target
    changed = complete.copy()
    changed.loc[test_idx, feats] = changed.loc[test_idx, feats] * 100
    m1 = make_model(settings).fit(complete.loc[train_idx, feats], complete.loc[train_idx, target])
    m2 = make_model(settings).fit(changed.loc[train_idx, feats], changed.loc[train_idx, target])
    assert np.array_equal(m1.predict(complete.loc[train_idx, feats]), m2.predict(complete.loc[train_idx, feats]))


def test_price_is_not_used_as_a_feature(cfg, settings, complete, split):
    """Corrupting `price` (a leakage route to log_price) must not change anything."""
    train_idx, test_idx = split
    a = fit_and_score(complete, train_idx, test_idx, settings)
    leaky = complete.copy()
    leaky["price"] = 0.0
    b = fit_and_score(leaky, train_idx, test_idx, settings)
    assert a["rmse"] == b["rmse"] and a["r2"] == b["r2"]


def test_same_seed_reproduces_exactly(cfg, settings, complete, split):
    a = fit_and_score(complete, *split, settings)
    b = fit_and_score(complete, *split, settings)
    assert {k: a[k] for k in ("rmse", "mae", "r2")} == {k: b[k] for k in ("rmse", "mae", "r2")}


def test_metrics_are_sensible_and_truth_diagnostic_is_added(cfg, settings, complete, split):
    ref = fit_and_score(complete, *split, settings, truth=complete)
    assert ref["r2"] > 0.8 and ref["rmse"] > 0
    # model trained on complete data, scored on the same complete test rows => identical
    assert ref["rmse"] == pytest.approx(ref["rmse_truth_test"])


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------
def test_reference_has_one_primary_row_and_seed_spread(cfg, settings, complete, split):
    ref = run_reference(cfg, settings, complete=complete, split=split, n_seeds=3)
    assert len(ref) == 3 and ref["is_primary"].sum() == 1
    assert list(ref["seed"]) == [7, 8, 9]


# ---------------------------------------------------------------------------
# End-to-end sweep with fake Member 2 files
# ---------------------------------------------------------------------------
@pytest.fixture()
def sweep(tmp_path, cfg, settings, complete, split):
    """Write mean-imputed files for MCAR / 0.1 / repeats 0-1 and validate them."""
    train_idx, _ = split
    corrupted_by_rep = {}
    for rep in (0, 1):
        corrupted, imputed = corrupt_and_mean_impute(complete, cfg, train_idx, seed=10 + rep)
        corrupted_by_rep[rep] = corrupted
        path = imputed_path("mean", "MCAR", 0.1, rep, imputed_root=tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        imputed.to_csv(path)

    def loader(mechanism, level, repeat):
        return {"corrupted": corrupted_by_rep[repeat]}

    report = run_validation(cfg, imputed_root=tmp_path, loader=loader, methods=["mean"],
                            mechanisms=["MCAR"], levels=[0.1])
    return tmp_path, report


def test_sweep_scores_only_validated_runs(sweep, cfg, settings, complete, split):
    root, report = sweep
    assert (report["status"] == PASS).sum() == 2
    res = run_predictive_track(cfg, settings, imputed_root=root, validation_report=report,
                               complete=complete, split=split, progress=False)
    assert len(res.runs) == 2 and res.errors.empty
    assert set(res.runs["repeat"]) == {0, 1}
    for col in ("rmse", "mae", "r2", "rmse_truth_test"):
        assert res.runs[col].notna().all()
    # exactly one row per key
    assert not res.runs.duplicated(["method", "mechanism", "level", "repeat"]).any()
    # imputed data cannot beat the complete-data upper bound by much (and usually is worse)
    ref = fit_and_score(complete, *split, settings)
    assert (res.runs["rmse"] > ref["rmse"] * 0.9).all()


def test_failed_validation_run_is_not_scored(sweep, cfg, settings, complete, split):
    root, report = sweep
    report = report.copy()
    report.loc[report["repeat"] == 1, "status"] = "FAIL"
    res = run_predictive_track(cfg, settings, imputed_root=root, validation_report=report,
                               complete=complete, split=split, progress=False)
    assert list(res.runs["repeat"]) == [0]


def test_resume_skips_done_runs_and_matches_full_run(sweep, cfg, settings, complete, split):
    root, report = sweep
    kw = dict(imputed_root=root, validation_report=report, complete=complete, split=split, progress=False)
    full = run_predictive_track(cfg, settings, **kw)
    first = run_predictive_track(cfg, settings, **{**kw, "levels": [0.1]}).runs.head(1)
    resumed = run_predictive_track(cfg, settings, done=first, **kw)
    assert len(resumed.runs) == 2
    cols = ["rmse", "mae", "r2"]
    pd.testing.assert_frame_equal(full.runs[cols], resumed.runs[cols])


def test_bad_file_is_recorded_as_error_not_crash(sweep, cfg, settings, complete, split):
    root, report = sweep
    path = imputed_path("mean", "MCAR", 0.1, 1, imputed_root=root)
    frame = pd.read_csv(path)
    frame.loc[0, "grade"] = np.nan  # sneaks in after validation
    frame.to_csv(path, index=False)
    res = run_predictive_track(cfg, settings, imputed_root=root, validation_report=report,
                               complete=complete, split=split, progress=False)
    assert len(res.runs) == 1 and len(res.errors) == 1 and res.errors.iloc[0]["repeat"] == 1


def test_table_and_saving(sweep, cfg, settings, complete, split, tmp_path, monkeypatch):
    import src.m3_predictive as mod

    root, report = sweep
    res = run_predictive_track(cfg, settings, imputed_root=root, validation_report=report,
                               complete=complete, split=split, progress=False)
    table = build_predictive_table(res.runs)
    assert len(table) == 1 and table.loc[0, "n_repeats"] == 2
    assert {"rmse_mean", "rmse_sd", "r2_mean", "r2_sd"} <= set(table.columns)

    out = tmp_path / "out"
    monkeypatch.setattr(mod, "EXPERIMENT_RESULTS_DIR", out)
    monkeypatch.setattr(mod, "TABLES_DIR", out)
    monkeypatch.setattr(mod, "PREDICTIVE_RESULTS_PATH", out / "predictive_results.csv")
    monkeypatch.setattr(mod, "PREDICTIVE_REFERENCE_PATH", out / "predictive_reference.csv")
    monkeypatch.setattr(mod, "PREDICTIVE_ERRORS_PATH", out / "predictive_errors.csv")
    monkeypatch.setattr(mod, "PREDICTIVE_SETTINGS_PATH", out / "predictive_settings.json")
    monkeypatch.setattr(mod, "PREDICTIVE_TABLE_PATH", out / "table5.csv")
    ref = run_reference(cfg, settings, complete=complete, split=split)
    written = save_predictive_results(res, ref, settings, table)
    assert (out / "predictive_results.csv").exists() and (out / "predictive_settings.json").exists()
    back = mod.load_predictive_runs(out / "predictive_results.csv")
    assert len(back) == 2 and set(written) >= {"runs", "reference", "table5", "settings"}
