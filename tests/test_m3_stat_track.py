"""Tests for Member 3 / Module 2 (statistical-recovery track).

Synthetic data and an injected fake loader are used, so no real data or Member 2 files are needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.m3_common import imputed_path, load_config  # noqa: E402
from src.m3_stat_plots import plot_statistical_figures  # noqa: E402
from src.m3_stat_track import (  # noqa: E402
    COMPLETE_CASE,
    CONST,
    SUMMARY_METRICS,
    build_bias_table,
    build_reference,
    build_table4,
    evaluate_dataset,
    fit_ols,
    predictor_columns,
    run_statistical_track,
)
from src.m3_validate_imputed import run_validation  # noqa: E402

N = 2000


@pytest.fixture(scope="module")
def cfg() -> dict:
    cfg = dict(load_config())
    cfg["expected_rows"] = N
    return cfg


@pytest.fixture(scope="module")
def complete(cfg) -> pd.DataFrame:
    """Complete synthetic data with a known linear relationship."""
    rng = np.random.default_rng(0)
    preds = predictor_columns(cfg)
    X = pd.DataFrame(rng.normal(size=(N, len(preds))), columns=preds)
    true_beta = np.linspace(0.2, 1.0, len(preds))
    true_beta[preds.index("sqft_living")] = 1.0
    log_price = 3.0 + X.to_numpy() @ true_beta + rng.normal(scale=1.0, size=N)
    frame = X.copy()
    frame["log_price"] = log_price
    frame["price"] = np.exp(log_price)
    frame = frame[cfg["expected_columns"]]
    frame.index.name = "row_index"
    return frame


def corrupt(complete: pd.DataFrame, cfg: dict, rate: float = 0.5, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = complete.copy()
    for col in cfg["missingness_columns"]:
        out.loc[rng.choice(N, int(rate * N), replace=False), col] = np.nan
    return out


# ---------------------------------------------------------------------------
# OLS
# ---------------------------------------------------------------------------
def test_fit_ols_matches_normal_equations(cfg, complete):
    preds = predictor_columns(cfg)
    res = fit_ols(complete[preds], complete["log_price"])
    A = np.column_stack([np.ones(N), complete[preds].to_numpy()])
    beta = np.linalg.solve(A.T @ A, A.T @ complete["log_price"].to_numpy())
    resid = complete["log_price"].to_numpy() - A @ beta
    cov = (resid @ resid / (N - A.shape[1])) * np.linalg.inv(A.T @ A)
    assert np.allclose(res.beta, beta)
    assert np.allclose(res.se, np.sqrt(np.diag(cov)))
    assert res.terms[0] == CONST and res.n == N and res.df == N - A.shape[1]


def test_fit_ols_matches_statsmodels(cfg, complete):
    sm = pytest.importorskip("statsmodels.api")
    preds = predictor_columns(cfg)
    res = fit_ols(complete[preds], complete["log_price"], alpha=0.05)
    ref = sm.OLS(complete["log_price"], sm.add_constant(complete[preds])).fit()
    assert np.allclose(res.beta, ref.params.to_numpy())
    assert np.allclose(res.se, ref.bse.to_numpy())
    assert np.allclose(res.p, ref.pvalues.to_numpy(), atol=1e-12)
    assert np.allclose(np.column_stack([res.ci_lo, res.ci_hi]), ref.conf_int().to_numpy())
    assert res.r2 == pytest.approx(ref.rsquared)


def test_fit_ols_rejects_nan_and_rank_deficiency(cfg, complete):
    preds = predictor_columns(cfg)
    with_nan = complete[preds].copy()
    with_nan.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        fit_ols(with_nan, complete["log_price"])
    duplicated = complete[preds].copy()
    duplicated["copy"] = duplicated[preds[0]]
    with pytest.raises(ValueError):
        fit_ols(duplicated, complete["log_price"])


# ---------------------------------------------------------------------------
# Recovery metrics
# ---------------------------------------------------------------------------
def test_identical_data_gives_perfect_recovery(cfg, complete):
    ref = build_reference(cfg, complete)
    coef, corr, summary = evaluate_dataset(complete, ref)
    assert np.allclose(coef["z_diff"], 0)
    assert coef["ci_covers_ref"].all() and coef["ci_overlap"].all()
    assert not coef["sig_flip"].any()
    assert np.allclose(coef["se_ratio"], 1)
    assert np.allclose(corr["diff"], 0)
    assert summary["stat_mean_abs_z_affected"] == pytest.approx(0)
    assert summary["stat_ci_coverage_all"] == 1
    assert summary["stat_n_sig_flips"] == 0


def test_mean_imputation_attenuates_the_affected_coefficient(cfg, complete):
    ref = build_reference(cfg, complete)
    corrupted = corrupt(complete, cfg, rate=0.5)
    imputed = corrupted.fillna(corrupted.mean())
    coef, corr, summary = evaluate_dataset(imputed, ref)
    row = coef[coef["term"] == "sqft_living"].iloc[0]
    assert abs(row["beta_imp"]) < abs(row["beta_ref"])  # attenuation toward zero
    assert row["z_diff"] < 0  # true coefficient is positive
    assert summary["stat_mean_abs_z_affected"] > 1.0
    assert summary["stat_corr_mean_abs_diff_affected"] > 0
    assert summary["stat_mean_signed_z_affected"] < 0


def test_summary_excludes_the_intercept_and_has_all_metrics(cfg, complete):
    ref = build_reference(cfg, complete)
    imputed = corrupt(complete, cfg).fillna(complete.mean())
    coef, _, summary = evaluate_dataset(imputed, ref)
    assert (coef["term"] == CONST).sum() == 1
    assert coef["is_affected"].sum() == 4
    assert len(coef) == 16  # intercept + 15 predictors
    assert set(SUMMARY_METRICS) <= set(summary)
    assert all(np.isfinite(v) for v in summary.values())


def test_significance_flip_is_detected(cfg, complete):
    ref = build_reference(cfg, complete)
    broken = complete.copy()
    # Destroy the relationship for one predictor so it loses significance.
    broken["sqft_living"] = np.random.default_rng(5).normal(size=N)
    coef, _, summary = evaluate_dataset(broken, ref)
    assert coef.loc[coef["term"] == "sqft_living", "sig_flip"].iloc[0]
    assert summary["stat_n_sig_flips_affected"] >= 1


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def test_table4_mean_and_sd_use_sample_sd():
    runs = pd.DataFrame(
        {
            "method": ["mean", "mean"],
            "mechanism": ["MAR", "MAR"],
            "level": [0.3, 0.3],
            "repeat": [0, 1],
            **{m: [1.0, 3.0] for m in SUMMARY_METRICS},
        }
    )
    table = build_table4(runs)
    assert len(table) == 1
    row = table.iloc[0]
    assert row["n_repeats"] == 2
    assert row["stat_mean_abs_z_affected_mean"] == 2.0
    assert row["stat_mean_abs_z_affected_sd"] == pytest.approx(np.sqrt(2))


def test_bias_table_keeps_the_sign():
    coef = pd.DataFrame(
        {
            "method": "mean", "mechanism": "MAR", "level": 0.3,
            "term": ["const", "grade", "grade"],
            "is_affected": [False, True, True],
            "z_diff": [5.0, -2.0, -4.0],
            "abs_z_diff": [5.0, 2.0, 4.0],
            "ci_covers_ref": [True, False, False],
            "sig_flip": [False, False, False],
        }
    )
    table = build_bias_table(coef)
    assert list(table["term"]) == ["grade"]  # intercept dropped
    assert table.iloc[0]["bias_z"] == -3.0
    assert table.iloc[0]["mean_abs_z"] == 3.0


# ---------------------------------------------------------------------------
# End-to-end with files on disk (Module 1 report feeds Module 2)
# ---------------------------------------------------------------------------
def _setup_files(cfg, complete, tmp_path, tamper_repeat: int | None = None):
    corrupted = corrupt(complete, cfg, rate=0.3)
    imputed = corrupted.fillna(corrupted.mean())

    def loader(mechanism, level, repeat):
        return {"corrupted": corrupted}

    for rep in range(5):
        frame = imputed.copy()
        if rep == tamper_repeat:
            frame.loc[corrupted["grade"].dropna().index[0], "grade"] += 10  # changes an observed value
        path = imputed_path("mean", "MAR", 0.30, rep, imputed_root=tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.reset_index().to_csv(path, index=False, compression="gzip")
    report = run_validation(
        cfg, imputed_root=tmp_path, loader=loader, methods=["mean"], mechanisms=["MAR"], levels=[0.30]
    )
    return loader, report


FILTERS = dict(methods=["mean"], mechanisms=["MAR"], levels=[0.30], progress=False)


def test_end_to_end_with_complete_case(cfg, complete, tmp_path):
    loader, report = _setup_files(cfg, complete, tmp_path)
    results = run_statistical_track(
        cfg,
        imputed_root=tmp_path,
        loader=loader,
        validation_report=report,
        reference=build_reference(cfg, complete),
        **FILTERS,
    )
    assert results.errors.empty
    assert len(results.runs) == 10  # 5 mean + 5 complete_case
    assert set(results.runs["method"]) == {"mean", COMPLETE_CASE}
    assert len(results.coefficients) == 10 * 16
    assert len(results.correlations) == 10 * (16 * 15 // 2)
    cc = results.runs[results.runs["method"] == COMPLETE_CASE]
    assert (cc["n_obs"] < N).all()  # rows were dropped
    table = build_table4(results.runs)
    assert set(table["method"]) == {"mean", COMPLETE_CASE}
    assert (table["n_repeats"] == 5).all()
    # complete-case is noisier than the reference (SE ratio above 1)
    assert cc["stat_mean_se_ratio_affected"].mean() > 1.0


def test_runs_that_failed_validation_are_excluded(cfg, complete, tmp_path):
    loader, report = _setup_files(cfg, complete, tmp_path, tamper_repeat=2)
    assert (report["status"] == "FAIL").sum() == 1
    results = run_statistical_track(
        cfg,
        imputed_root=tmp_path,
        loader=loader,
        validation_report=report,
        reference=build_reference(cfg, complete),
        include_complete_case=False,
        **FILTERS,
    )
    assert len(results.runs) == 4
    assert 2 not in set(results.runs["repeat"])


def test_missing_validation_report_gives_clear_error(cfg, complete, tmp_path, monkeypatch):
    import src.m3_validate_imputed as validate

    monkeypatch.setattr(validate, "VALIDATION_REPORT_PATH", tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError, match="m3_01_validate_imputed"):
        run_statistical_track(
            cfg, reference=build_reference(cfg, complete), include_complete_case=False, **FILTERS
        )


def test_failure_of_one_run_does_not_stop_the_sweep(cfg, complete, tmp_path):
    loader, report = _setup_files(cfg, complete, tmp_path)
    # Corrupt one imputed file after validation: make it unreadable.
    bad = imputed_path("mean", "MAR", 0.30, 1, imputed_root=tmp_path)
    bad.write_bytes(b"not a gzip file")
    results = run_statistical_track(
        cfg,
        imputed_root=tmp_path,
        loader=loader,
        validation_report=report,
        reference=build_reference(cfg, complete),
        include_complete_case=False,
        **FILTERS,
    )
    assert len(results.runs) == 4
    assert len(results.errors) == 1
    assert results.errors.iloc[0]["repeat"] == 1


def test_figures_are_written(cfg, complete, tmp_path):
    loader, report = _setup_files(cfg, complete, tmp_path)
    results = run_statistical_track(
        cfg, imputed_root=tmp_path, loader=loader, validation_report=report,
        reference=build_reference(cfg, complete), **FILTERS,
    )
    paths = plot_statistical_figures(build_table4(results.runs), out_dir=tmp_path / "figs")
    assert len(paths) == 2
    assert all(p.exists() and p.stat().st_size > 1000 for p in paths)
