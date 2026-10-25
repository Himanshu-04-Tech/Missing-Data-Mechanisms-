"""Tests for Member 3 / Module 6 (reproducibility, seed hygiene, stage commands, output audit).

SRS: FR-6.1 to FR-6.3 are covered by the module test files (validation failures in test_m3_validate_imputed.py,
train-only fitting in test_m3_predictive.py, merge uniqueness and Case labels in test_m3_merge.py / test_m3_cases.py).
This file covers FR-6.4 (seeds from config, one command per stage, identical re-runs) and the output audit.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.m3_predictive as m3_predictive  # noqa: E402
import src.m3_stat_track as m3_stat_track  # noqa: E402
from src.m3_cases import get_thresholds, run_cases  # noqa: E402
from src.m3_common import PROJECT_ROOT, imputed_path, load_config  # noqa: E402
from src.m3_merge import run_merge  # noqa: E402
from src.m3_predictive import feature_columns, get_settings, run_predictive_track, run_reference  # noqa: E402
from src.m3_reproduce import (  # noqa: E402
    FAIL,
    PASS,
    STAGES,
    WARN,
    audit,
    build_manifest,
    compare_manifests,
    content_digest,
    expected_counts,
    stage_by_number,
)
from src.m3_stat_track import build_reference, run_statistical_track  # noqa: E402
from src.m3_validate_imputed import run_validation  # noqa: E402

M3_SOURCES = sorted((ROOT / "src").glob("m3_*.py")) + sorted((ROOT / "scripts").glob("m3_*.py"))


# ---------------------------------------------------------------------------
# FR-6.4 seed hygiene (static)
# ---------------------------------------------------------------------------
def code_lines(path: Path):
    """Source lines without comments and without docstring text (good enough for a pattern scan)."""
    text = re.sub(r'"""[\s\S]*?"""', "", path.read_text(encoding="utf-8"))
    for lineno, line in enumerate(text.splitlines(), start=1):
        yield lineno, line.split("#", 1)[0]


@pytest.mark.parametrize("path", M3_SOURCES, ids=lambda p: p.name)
def test_no_hard_coded_seed_in_member3_code(path):
    patterns = [r"random_state\s*=\s*\d", r"default_rng\(\s*\d", r"RandomState\(\s*\d",
                r"np\.random\.seed", r"\brandom\.seed", r"\bseed\s*=\s*\d"]
    hits = [(n, line.strip()) for n, line in code_lines(path) for p in patterns if re.search(p, line)]
    assert not hits, f"hard-coded seed in {path.name}: {hits}"


@pytest.mark.parametrize("name", ["m3_stat_track", "m3_merge", "m3_cases", "m3_case_plots"])
def test_deterministic_modules_use_no_randomness(name):
    src = (ROOT / "src" / f"{name}.py").read_text(encoding="utf-8")
    assert "np.random" not in src and "import random" not in src and "default_rng" not in src


def test_forest_seed_comes_from_config():
    cfg = load_config()
    assert get_settings(cfg).random_state == cfg["predictive"]["random_state"]
    changed = {**cfg, "predictive": {**cfg["predictive"], "random_state": 7}}
    assert get_settings(changed).random_state == 7
    assert cfg["predictive"]["random_state"] == cfg["base_seed"]  # one source of truth


# ---------------------------------------------------------------------------
# One command per stage
# ---------------------------------------------------------------------------
def test_every_stage_has_a_runnable_script_and_unique_number():
    assert [s.number for s in STAGES] == [1, 2, 3, 4, 5]
    for stage in STAGES:
        script = ROOT / "scripts" / stage.script
        assert script.exists(), script
        assert "__main__" in script.read_text(encoding="utf-8")
        assert stage.command(python="python")[:2] == ["python", str(Path("scripts") / stage.script)]


def test_stage_outputs_match_the_paths_the_modules_write():
    import src.m3_cases as cases
    import src.m3_merge as merge
    import src.m3_validate_imputed as validate

    code_paths = {
        1: [validate.VALIDATION_REPORT_PATH],
        2: [m3_stat_track.RUNS_PATH, m3_stat_track.COEFS_PATH, m3_stat_track.CORRS_PATH,
            m3_stat_track.TABLE4_PATH, m3_stat_track.TABLE4B_PATH],
        3: [m3_predictive.PREDICTIVE_RESULTS_PATH, m3_predictive.PREDICTIVE_REFERENCE_PATH,
            m3_predictive.PREDICTIVE_SETTINGS_PATH, m3_predictive.PREDICTIVE_TABLE_PATH],
        4: [merge.MERGED_PATH, merge.TABLE6_PATH, merge.TABLE6B_PATH],
        5: [cases.CASES_BY_CONDITION_PATH, cases.CASE_COUNTS_PATH, cases.SENSITIVITY_PATH,
            cases.SENSITIVITY_GRID_PATH, cases.THRESHOLDS_USED_PATH],
    }
    for number, paths in code_paths.items():
        declared = {Path(p) for p in stage_by_number(number).outputs}
        actual = {Path(p).relative_to(PROJECT_ROOT) for p in paths}
        assert declared == actual, (number, declared ^ actual)


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError):
        stage_by_number(9)


def load_runner():
    spec = importlib.util.spec_from_file_location("m3_06_run_all", ROOT / "scripts" / "m3_06_run_all.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runner_passes_smoke_flags_only_when_asked():
    runner = load_runner()
    import argparse

    final = argparse.Namespace(smoke=False, imputed_root=None, no_figures=False)
    smoke = argparse.Namespace(smoke=True, imputed_root="X", no_figures=False)
    assert all(runner.stage_args(n, final) == [] for n in range(1, 6))
    assert "--n-estimators" in runner.stage_args(3, smoke) and "--allow-partial" in runner.stage_args(4, smoke)
    assert runner.stage_args(3, final) == []           # a final run never shrinks the forest
    assert runner.stage_args(2, smoke)[:2] == ["--imputed-root", "X"]


# ---------------------------------------------------------------------------
# Digests and manifests
# ---------------------------------------------------------------------------
def test_content_digest_ignores_timings_but_not_values(tmp_path):
    a, b, c = (tmp_path / f"{n}.csv" for n in "abc")
    pd.DataFrame({"x": [1.0, 2.0], "fit_seconds": [1.1, 2.2]}).to_csv(a, index=False)
    pd.DataFrame({"x": [1.0, 2.0], "fit_seconds": [9.9, 8.8]}).to_csv(b, index=False)
    pd.DataFrame({"x": [1.0, 2.5], "fit_seconds": [1.1, 2.2]}).to_csv(c, index=False)
    assert content_digest(a) == content_digest(b)
    assert content_digest(a) != content_digest(c)


def test_content_digest_tolerates_last_bit_noise_but_not_real_changes(tmp_path):
    a, b, c = (tmp_path / f"{n}.csv" for n in "abc")
    pd.DataFrame({"rmse": [0.1725, 0.19]}).to_csv(a, index=False)
    pd.DataFrame({"rmse": [0.1725 + 5e-17, 0.19 - 1e-16]}).to_csv(b, index=False)   # parallel-sum noise
    pd.DataFrame({"rmse": [0.1725 + 1e-6, 0.19]}).to_csv(c, index=False)             # a real change
    assert content_digest(a) == content_digest(b)
    assert content_digest(a) != content_digest(c)


def test_compare_manifests_reports_differences():
    old = {"config_sha256": "1", "outputs": {"a.csv": {"digest": "x"}, "b.csv": {"digest": "y"}}}
    assert compare_manifests(old, json.loads(json.dumps(old))) == []
    new = {"config_sha256": "2", "outputs": {"a.csv": {"digest": "x"}, "b.csv": {"digest": "z"}, "c.csv": {"digest": "w"}}}
    problems = compare_manifests(old, new)
    assert any("config.yaml" in p for p in problems)
    assert any("b.csv" in p and "differs" in p for p in problems)
    assert any("c.csv" in p and "missing" in p for p in problems)


def test_build_manifest_on_empty_root_records_missing_outputs(tmp_path):
    manifest = build_manifest([], root=tmp_path)
    assert all(v["digest"] is None for v in manifest["outputs"].values())


# ---------------------------------------------------------------------------
# End-to-end determinism on synthetic data (M1 -> M5), and the audit on its outputs
# ---------------------------------------------------------------------------
N = 500


@pytest.fixture(scope="module")
def cfg() -> dict:
    cfg = dict(load_config())
    cfg.update(expected_rows=N, mechanisms=["MCAR"], levels=[0.1, 0.3], n_repeats=2)
    cfg["predictive"] = {"random_state": cfg["base_seed"], "score_on_truth": True,
                         "rf_params": {"n_estimators": 15, "n_jobs": 1}}
    return cfg


@pytest.fixture(scope="module")
def world(cfg, tmp_path_factory):
    """Synthetic complete data, corrupted frames, and fake Member 2 files for 4 imputers."""
    rng = np.random.default_rng(0)
    feats = feature_columns(cfg)
    X = pd.DataFrame(rng.normal(size=(N, len(feats))), columns=feats)
    y = 1.5 * X["sqft_living"] + X["grade"] - 0.5 * X["lat"] + rng.normal(scale=0.5, size=N)
    complete = X.assign(log_price=y, price=np.exp(y))[cfg["expected_columns"]]
    complete.index.name = "row_index"
    perm = np.random.default_rng(1).permutation(N)
    split = (np.sort(perm[: int(0.8 * N)]), np.sort(perm[int(0.8 * N):]))

    root = tmp_path_factory.mktemp("imputed")
    corrupted = {}
    fills = {"mean": lambda s: s.mean(), "knn": lambda s: s.median(),
             "regression": lambda s: s.mean() + 0.1 * s.std(), "random_forest": lambda s: s.mean() - 0.1 * s.std()}
    for level in cfg["levels"]:
        for rep in range(cfg["n_repeats"]):
            r = np.random.default_rng(int(level * 100) * 10 + rep)
            frame = complete.copy()
            for col in cfg["missingness_columns"]:
                frame.loc[r.choice(N, int(level * N), replace=False), col] = np.nan
            corrupted[("MCAR", level, rep)] = frame
            for method, fill in fills.items():
                imp = frame.copy()
                for col in cfg["missingness_columns"]:
                    imp[col] = imp[col].fillna(fill(frame.loc[split[0], col]))
                path = imputed_path(method, "MCAR", level, rep, imputed_root=root)
                path.parent.mkdir(parents=True, exist_ok=True)
                imp.to_csv(path)

    def loader(mechanism, level, repeat):
        return {"corrupted": corrupted[(mechanism, level, repeat)]}

    return {"complete": complete, "split": split, "root": root, "loader": loader}


def run_pipeline(cfg, world):
    report = run_validation(cfg, imputed_root=world["root"], loader=world["loader"])
    stat = run_statistical_track(cfg, imputed_root=world["root"], loader=world["loader"], validation_report=report,
                                 reference=build_reference(cfg, frame=world["complete"]), progress=False)
    settings = get_settings(cfg)
    pred = run_predictive_track(cfg, settings, imputed_root=world["root"], validation_report=report,
                                complete=world["complete"], split=world["split"], progress=False)
    ref = run_reference(cfg, settings, complete=world["complete"], split=world["split"])
    merged = run_merge(stat_runs=stat.runs, pred_runs=pred.runs, reference=ref)
    cases = run_cases(merged["merged"], get_thresholds(cfg))
    return {"report": report, "stat": stat.runs, "pred": pred.runs, "ref": ref, "settings": settings,
            "merged": merged, "cases": cases}


@pytest.fixture(scope="module")
def first_run(cfg, world):
    return run_pipeline(cfg, world)


def test_pipeline_runs_on_all_runs(cfg, first_run):
    n = expected_counts(cfg)
    assert (first_run["report"]["status"] == "PASS").sum() == n["imputed_runs"] == 16
    assert len(first_run["stat"]) == n["imputed_runs"] + n["cc_runs"]
    assert len(first_run["pred"]) == n["imputed_runs"] and len(first_run["merged"]["merged"]) == 16
    assert len(first_run["cases"]["labelled"]) == n["conditions"] == 8


def test_rerun_with_same_seeds_gives_identical_outputs(cfg, world, first_run):
    second = run_pipeline(cfg, world)
    drop = ["fit_seconds"]
    pd.testing.assert_frame_equal(first_run["report"], second["report"])
    pd.testing.assert_frame_equal(first_run["stat"], second["stat"])
    pd.testing.assert_frame_equal(first_run["pred"].drop(columns=drop), second["pred"].drop(columns=drop))
    pd.testing.assert_frame_equal(first_run["ref"].drop(columns=drop), second["ref"].drop(columns=drop))
    pd.testing.assert_frame_equal(first_run["merged"]["table6"], second["merged"]["table6"])
    pd.testing.assert_frame_equal(first_run["cases"]["labelled"], second["cases"]["labelled"])
    pd.testing.assert_frame_equal(first_run["cases"]["sensitivity"], second["cases"]["sensitivity"])


def test_different_forest_seed_changes_predictive_numbers_only(cfg, world, first_run):
    other_cfg = {**cfg, "predictive": {**cfg["predictive"], "random_state": 99}}
    other = run_pipeline(other_cfg, world)
    assert not np.allclose(first_run["pred"]["rmse"], other["pred"]["rmse"])           # the forest depends on the seed
    pd.testing.assert_frame_equal(first_run["stat"], other["stat"])                    # statistics are deterministic


def write_outputs(root: Path, run: dict, cfg: dict) -> None:
    exp, tab = root / "results" / "experiment_results", root / "results" / "tables"
    exp.mkdir(parents=True), tab.mkdir(parents=True)
    run["report"].to_csv(exp / "validation_report.csv", index=False)
    run["stat"].to_csv(exp / "stat_recovery_runs.csv", index=False)
    run["pred"].to_csv(exp / "predictive_results.csv", index=False)
    run["ref"].to_csv(exp / "predictive_reference.csv", index=False)
    (exp / "predictive_settings.json").write_text(json.dumps(run["settings"].as_record()))
    run["merged"]["merged"].to_csv(exp / "merged_results.csv", index=False)
    run["merged"]["table6"].to_csv(tab / "table6_overall_recovery.csv", index=False)
    run["cases"]["labelled"].to_csv(tab / "cases_by_condition.csv", index=False)
    thr = get_thresholds(cfg)
    (exp / "case_thresholds_used.json").write_text(json.dumps(asdict(thr)))


def statuses(checks):
    return {c.name: c.status for c in checks}


def test_audit_passes_on_a_complete_consistent_run(cfg, first_run, tmp_path):
    write_outputs(tmp_path, first_run, cfg)
    checks = audit(cfg, root=tmp_path)
    failed = [c for c in checks if c.status == FAIL]
    assert not failed, [(c.name, c.detail) for c in failed]
    assert statuses(checks)["M5 cases: thresholds locked by the team (D3)"] == WARN  # config is still provisional


def test_audit_fails_when_outputs_are_missing(cfg, tmp_path):
    checks = audit(cfg, root=tmp_path)
    assert all(c.status == FAIL for c in checks if "no failed runs" not in c.name and "locked" not in c.name)


def test_audit_catches_smoke_test_settings_left_in_results(cfg, first_run, tmp_path):
    write_outputs(tmp_path, first_run, cfg)
    full = {**cfg, "predictive": {**cfg["predictive"], "rf_params": {"n_estimators": 300, "n_jobs": 1}}}
    st = statuses(audit(full, root=tmp_path))
    assert st["M3 predictive: settings match config (no smoke-test leftovers)"] == FAIL


def test_audit_catches_duplicate_and_incomplete_runs(cfg, first_run, tmp_path):
    write_outputs(tmp_path, first_run, cfg)
    exp = tmp_path / "results" / "experiment_results"
    pred = first_run["pred"]
    pd.concat([pred, pred.head(1)]).to_csv(exp / "predictive_results.csv", index=False)
    assert statuses(audit(cfg, root=tmp_path))["M3 predictive: runs complete"] == FAIL
    first_run["merged"]["merged"].iloc[1:].to_csv(exp / "merged_results.csv", index=False)
    assert statuses(audit(cfg, root=tmp_path))["M4 merge: one row per run"] == FAIL


def test_audit_catches_failed_runs_and_threshold_mismatch(cfg, first_run, tmp_path):
    write_outputs(tmp_path, first_run, cfg)
    exp = tmp_path / "results" / "experiment_results"
    (exp / "predictive_errors.csv").write_text("method,error\nmean,boom\n")
    assert statuses(audit(cfg, root=tmp_path))["M3 predictive: no failed runs"] == FAIL
    moved = {**cfg, "cases": {**cfg["cases"], "stat_threshold": 3.0}}
    assert statuses(audit(moved, root=tmp_path))["M5 cases: thresholds used = config"] == FAIL


def test_audit_reports_locked_thresholds_as_pass(cfg, first_run, tmp_path):
    write_outputs(tmp_path, first_run, cfg)
    locked = {**cfg, "cases": {**cfg["cases"], "thresholds_locked": True}}
    assert statuses(audit(locked, root=tmp_path))["M5 cases: thresholds locked by the team (D3)"] == PASS
