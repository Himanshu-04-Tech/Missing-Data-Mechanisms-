"""Shared helpers for Member 3 modules (M0 to M8).

This is a NEW file. It does not modify any existing module in ``src/``.

It only provides small, dependency-light utilities that every Member 3 module
needs: the project root, config loading, and the file-path convention for
Member 2's imputed datasets.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Iterator

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Where Member 2 delivers imputed files (agreed handoff layout):
#   data/imputed/{method}/{mechanism}/rate_{pct:02d}/rep_{k}.csv.gz
IMPUTED_DIR = PROJECT_ROOT / "data" / "imputed"

# Where Member 3 writes outputs (these folders already exist in the repo layout).
RESULTS_DIR = PROJECT_ROOT / "results"
EXPERIMENT_RESULTS_DIR = RESULTS_DIR / "experiment_results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    """Load ``config.yaml`` (the team's single source of truth)."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_path(relative: str | Path) -> Path:
    """Resolve a path from ``config.yaml`` (relative to the repo root)."""
    return PROJECT_ROOT / relative


def level_to_pct(level: float) -> int:
    """0.30 -> 30. Rounds to avoid float artefacts such as 0.3 * 100 = 30.000000000000004."""
    return int(round(float(level) * 100))


def imputed_path(
    method: str,
    mechanism: str,
    level: float,
    repeat: int,
    imputed_root: Path | None = None,
) -> Path:
    """Path of one imputed file, following the agreed handoff layout."""
    root = IMPUTED_DIR if imputed_root is None else Path(imputed_root)
    return (
        root
        / method
        / mechanism
        / f"rate_{level_to_pct(level):02d}"
        / f"rep_{int(repeat)}.csv.gz"
    )


def corrupted_conditions(cfg: dict) -> Iterator[dict]:
    """Yield every corrupted condition (mechanism x level x repeat). Expected: 75."""
    for mechanism, level, repeat in product(
        cfg["mechanisms"], cfg["levels"], range(int(cfg["n_repeats"]))
    ):
        yield {"mechanism": mechanism, "level": level, "repeat": repeat}


def imputed_conditions(cfg: dict) -> Iterator[dict]:
    """Yield every imputed run (method x mechanism x level x repeat). Expected: 300."""
    for method in cfg["imputation_methods"]:
        for cond in corrupted_conditions(cfg):
            yield {"method": method, **cond}
