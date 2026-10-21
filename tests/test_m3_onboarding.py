"""Tests for Member 3 / Module 0.

Pure tests (config, path convention, counts) always run.
Tests that need the generated missingness data are skipped automatically
if ``data/generated_missing`` has not been created yet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.m3_common import (  # noqa: E402
    corrupted_conditions,
    imputed_conditions,
    imputed_path,
    level_to_pct,
    load_config,
    project_path,
)
from src.m3_onboarding import (  # noqa: E402
    FAIL,
    REQUIRED_CONFIG_KEYS,
    Report,
    check_condition_load,
    check_config,
)


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


def test_config_has_required_keys(cfg):
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    assert not missing, f"config.yaml is missing keys: {missing}"


def test_condition_counts(cfg):
    assert sum(1 for _ in corrupted_conditions(cfg)) == 75
    assert sum(1 for _ in imputed_conditions(cfg)) == 300


def test_imputed_conditions_are_unique(cfg):
    keys = [tuple(c.values()) for c in imputed_conditions(cfg)]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize(
    "level, pct", [(0.10, 10), (0.20, 20), (0.30, 30), (0.40, 40), (0.50, 50)]
)
def test_level_to_pct(level, pct):
    assert level_to_pct(level) == pct


def test_imputed_path_convention(tmp_path):
    p = imputed_path("knn", "MAR", 0.30, 2, imputed_root=tmp_path)
    assert p == tmp_path / "knn" / "MAR" / "rate_30" / "rep_2.csv.gz"
    p10 = imputed_path("mean", "MCAR", 0.10, 0, imputed_root=tmp_path)
    assert p10.parent.name == "rate_10"  # zero-padded two digits


def test_check_config_passes_on_repo_config(cfg):
    report = Report()
    check_config(report, cfg)
    assert not [c for c in report.checks if c.status == FAIL]


def test_check_config_fails_on_missing_key(cfg):
    broken = {k: v for k, v in cfg.items() if k != "levels"}
    report = Report()
    check_config(report, broken)
    assert report.has_failures


@pytest.mark.skipif(
    not (project_path("data/generated_missing") / "manifest.csv").exists(),
    reason="generated missingness data not present (run python3 scripts/run_sweep.py)",
)
@pytest.mark.parametrize("mechanism, level, repeat", [("MCAR", 0.10, 0), ("MAR", 0.30, 2), ("MNAR", 0.50, 4)])
def test_condition_load_invariants(cfg, mechanism, level, repeat):
    report = Report()
    data = check_condition_load(report, cfg, mechanism, level, repeat)
    assert data is not None
    failures = [f"{c.name}: {c.detail}" for c in report.checks if c.status == FAIL]
    assert not failures, failures
