"""Module M0: Member 3 setup and onboarding checks.

Run through ``scripts/m3_00_onboarding.py``. This module only READS the repo:
it never edits data, config, or existing source files.

What it verifies, in order:
  1. Environment: Python version and required packages.
  2. Config: required keys, and the condition counts (75 corrupted / 300 imputed).
  3. Repo files: processed data, generated_missing files, existing src modules.
  4. Condition load: ``load_condition`` returns what the handoff README promises,
     and the leakage-relevant invariants hold.
  5. Manifest: ``list_conditions`` lists all 75 conditions.
  6. Imputed files: how many of Member 2's 300 files exist yet (informational).

Status levels:
  PASS  the check succeeded
  WARN  something is unexpected but not blocking
  FAIL  blocking problem; fix before continuing
  INFO  status information only (for example, Member 2 files not delivered yet)
"""
from __future__ import annotations

import importlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.m3_common import (
    EXPERIMENT_RESULTS_DIR,
    IMPUTED_DIR,
    corrupted_conditions,
    imputed_conditions,
    imputed_path,
    load_config,
    project_path,
)

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"

MIN_PYTHON = (3, 9)

# import name -> (display name, required?)
PACKAGES = {
    "numpy": ("numpy", True),
    "pandas": ("pandas", True),
    "sklearn": ("scikit-learn", True),
    "yaml": ("PyYAML", True),
    "matplotlib": ("matplotlib", True),
    "scipy": ("scipy", False),
    "statsmodels": ("statsmodels", False),
    "pytest": ("pytest", False),
}

REQUIRED_CONFIG_KEYS = [
    "base_seed",
    "test_size",
    "log_target_name",
    "missingness_columns",
    "always_observed",
    "mechanisms",
    "levels",
    "n_repeats",
    "imputation_methods",
    "expected_columns",
    "expected_rows",
    "clean_data_path",
    "split_indices_path",
    "generated_missing_dir",
    "alpha",
]

# (file, blocking if missing?)
SRC_MODULES = [
    ("handoff.py", True),
    ("statistics.py", True),
    ("data_processing.py", False),
    ("missingness.py", False),
    ("aggregate.py", False),
    ("plots.py", False),
    ("evaluation.py", False),
]

LOADER_KEYS = ["corrupted", "truth", "mask", "train_idx", "test_idx"]

LOCKED_RULES = [
    "Fit imputers and models on TRAIN rows only, then apply to test.",
    "Ground truth is for scoring only. Never an input to an imputer or model.",
    "Do not edit existing files in src/ without telling the team. Add new files.",
    "Never invent results. Python only, no R.",
    "Always separate OBSERVED RESULT from POSSIBLE INTERPRETATION.",
]


# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------
@dataclass
class Check:
    section: str
    name: str
    status: str
    detail: str = ""


class Report:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(self, section: str, name: str, status: str, detail: str = "") -> None:
        self.checks.append(Check(section, name, status, detail))

    def count(self, status: str) -> int:
        return sum(1 for c in self.checks if c.status == status)

    @property
    def has_failures(self) -> bool:
        return self.count(FAIL) > 0

    def to_dict(self) -> dict:
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "summary": {s: self.count(s) for s in (PASS, WARN, FAIL, INFO)},
            "checks": [asdict(c) for c in self.checks],
        }

    def print(self) -> None:
        current = None
        for check in self.checks:
            if check.section != current:
                current = check.section
                print(f"\n== {current} ==")
            line = f"[{check.status:<4}] {check.name}"
            if check.detail:
                line += f": {check.detail}"
            print(line)
        print(
            "\nSummary: "
            f"{self.count(PASS)} PASS, {self.count(WARN)} WARN, "
            f"{self.count(FAIL)} FAIL, {self.count(INFO)} INFO"
        )
        print("\nLocked team rules (re-read before every module):")
        for rule in LOCKED_RULES:
            print(f"  - {rule}")


# ---------------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------------
def check_environment(report: Report) -> None:
    sec = "1. Environment"
    version = sys.version_info[:2]
    ok = version >= MIN_PYTHON
    report.add(
        sec,
        "Python version",
        PASS if ok else FAIL,
        f"{platform.python_version()} (need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})",
    )

    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    report.add(
        sec,
        "Virtual environment active",
        PASS if in_venv else WARN,
        "yes" if in_venv else "no (recommended: python3 -m venv .venv && source .venv/bin/activate)",
    )

    for import_name, (display, required) in PACKAGES.items():
        try:
            module = importlib.import_module(import_name)
            report.add(
                sec, display, PASS, f"version {getattr(module, '__version__', 'unknown')}"
            )
        except ImportError:
            if required:
                report.add(sec, display, FAIL, "not installed (pip install -r requirements.txt)")
            else:
                report.add(sec, display, WARN, "not installed (optional for Member 3)")


# ---------------------------------------------------------------------------
# 2. Config
# ---------------------------------------------------------------------------
def check_config(report: Report, cfg: dict) -> None:
    sec = "2. Config (config.yaml)"
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        report.add(sec, "Required keys", FAIL, f"missing: {missing}")
        return
    report.add(sec, "Required keys", PASS, f"{len(REQUIRED_CONFIG_KEYS)} keys present")

    n_corrupted = sum(1 for _ in corrupted_conditions(cfg))
    n_imputed = sum(1 for _ in imputed_conditions(cfg))
    report.add(
        sec,
        "Corrupted conditions (mechanisms x levels x repeats)",
        PASS if n_corrupted == 75 else WARN,
        f"{n_corrupted} (expected 75)",
    )
    report.add(
        sec,
        "Imputed runs (methods x conditions)",
        PASS if n_imputed == 300 else WARN,
        f"{n_imputed} (expected 300)",
    )
    report.add(
        sec,
        "Design summary",
        INFO,
        f"mechanisms={cfg['mechanisms']}, levels={cfg['levels']}, "
        f"repeats={cfg['n_repeats']}, methods={cfg['imputation_methods']}, "
        f"seed={cfg['base_seed']}, test_size={cfg['test_size']}",
    )


# ---------------------------------------------------------------------------
# 3. Repo files
# ---------------------------------------------------------------------------
def check_repo_files(report: Report, cfg: dict) -> None:
    sec = "3. Repo files"
    for rel in ("clean_data_path", "split_indices_path"):
        path = project_path(cfg[rel])
        report.add(
            sec,
            f"{cfg[rel]}",
            PASS if path.exists() else FAIL,
            "found" if path.exists() else "missing (run the data processing from README section 5)",
        )

    raw = project_path(cfg.get("raw_data_path", "data/raw/kc_house_data.csv"))
    report.add(
        sec,
        str(cfg.get("raw_data_path", "data/raw/kc_house_data.csv")),
        PASS if raw.exists() else WARN,
        "found" if raw.exists() else "not present locally (only needed to regenerate from scratch)",
    )

    gen_dir = project_path(cfg["generated_missing_dir"])
    for name in ("ground_truth.csv.gz", "split_indices.csv", "manifest.csv"):
        path = gen_dir / name
        report.add(
            sec,
            f"{cfg['generated_missing_dir']}/{name}",
            PASS if path.exists() else WARN,
            "found" if path.exists() else "missing (regenerate with: python3 scripts/run_sweep.py)",
        )

    src_dir = Path(project_path("src"))
    for filename, blocking in SRC_MODULES:
        path = src_dir / filename
        status = PASS if path.exists() else (FAIL if blocking else WARN)
        report.add(sec, f"src/{filename}", status, "found" if path.exists() else "missing")


# ---------------------------------------------------------------------------
# 4. Condition load and invariants
# ---------------------------------------------------------------------------
def check_condition_load(
    report: Report,
    cfg: dict,
    mechanism: str = "MAR",
    level: float = 0.30,
    repeat: int = 0,
) -> dict | None:
    """Load one condition and check what the handoff README promises."""
    sec = f"4. Condition load ({mechanism}, {level:.2f}, repeat {repeat})"

    try:
        from src.handoff import load_condition  # existing Member 1 module
    except Exception as exc:  # noqa: BLE001 (report any import problem)
        report.add(sec, "import src.handoff", FAIL, repr(exc))
        return None

    try:
        data = load_condition(mechanism=mechanism, level=level, repeat=repeat)
    except Exception as exc:  # noqa: BLE001
        report.add(
            sec,
            "load_condition()",
            FAIL,
            f"{exc!r} (generated data may be missing: python3 scripts/run_sweep.py)",
        )
        return None
    report.add(sec, "load_condition()", PASS, "returned without error")

    absent = [k for k in LOADER_KEYS if k not in data]
    if absent:
        report.add(sec, "Returned keys", FAIL, f"missing: {absent}")
        return None
    report.add(sec, "Returned keys", PASS, ", ".join(LOADER_KEYS))

    corrupted, truth, mask = data["corrupted"], data["truth"], data["mask"]
    train_idx = np.asarray(data["train_idx"])
    test_idx = np.asarray(data["test_idx"])
    miss_cols = list(cfg["missingness_columns"])
    protected = list(cfg["always_observed"])
    expected_cols = list(cfg["expected_columns"])
    n_rows = int(cfg["expected_rows"])

    # Shapes and columns
    shape_ok = corrupted.shape == (n_rows, len(expected_cols)) and truth.shape == corrupted.shape
    report.add(
        sec,
        "corrupted / truth shape",
        PASS if shape_ok else FAIL,
        f"corrupted={corrupted.shape}, truth={truth.shape}, expected=({n_rows}, {len(expected_cols)})",
    )
    report.add(
        sec,
        "Column order matches config expected_columns",
        PASS if list(corrupted.columns) == expected_cols else FAIL,
        "" if list(corrupted.columns) == expected_cols else f"got {list(corrupted.columns)}",
    )
    mask_cols_ok = set(mask.columns) == set(miss_cols) and len(mask) == n_rows
    report.add(
        sec,
        "mask covers the 4 missingness columns",
        PASS if mask_cols_ok else FAIL,
        f"mask shape={mask.shape}, columns={list(mask.columns)}",
    )
    if not (shape_ok and mask_cols_ok):
        return data
    mask = mask[miss_cols]
    index_ok = corrupted.index.equals(truth.index) and corrupted.index.equals(mask.index)
    report.add(
        sec,
        "corrupted, truth and mask share the same index (row_index)",
        PASS if index_ok else FAIL,
    )

    # NaN pattern
    nan_cols = [c for c in corrupted.columns if corrupted[c].isna().any()]
    nan_ok = set(nan_cols) <= set(miss_cols)
    report.add(
        sec,
        "NaNs appear only in missingness columns",
        PASS if nan_ok else FAIL,
        f"columns with NaN: {nan_cols}",
    )
    protected_ok = not corrupted[protected].isna().any().any()
    report.add(
        sec,
        f"Protected columns have 0 NaN {protected}",
        PASS if protected_ok else FAIL,
    )
    truth_ok = not truth.isna().any().any()
    report.add(sec, "truth has 0 NaN", PASS if truth_ok else FAIL)
    mask_match = np.array_equal(
        corrupted[miss_cols].isna().to_numpy(), mask[miss_cols].to_numpy(dtype=bool)
    )
    report.add(sec, "mask equals corrupted.isna() on missingness columns", PASS if mask_match else FAIL)

    # Observed values unchanged vs ground truth (scoring-only use of truth)
    unchanged = True
    for col in miss_cols:
        keep = ~mask[col].to_numpy(dtype=bool)
        a = corrupted[col].to_numpy(dtype=float)[keep]
        b = truth[col].to_numpy(dtype=float)[keep]
        if not np.allclose(a, b, equal_nan=False):
            unchanged = False
    other = [c for c in expected_cols if c not in miss_cols]
    others_ok = np.allclose(
        corrupted[other].to_numpy(dtype=float), truth[other].to_numpy(dtype=float)
    )
    report.add(
        sec,
        "Observed values equal truth (missingness columns and all other columns)",
        PASS if (unchanged and others_ok) else FAIL,
    )

    # Split
    n_train, n_test = len(train_idx), len(test_idx)
    disjoint = len(np.intersect1d(train_idx, test_idx)) == 0
    covers = n_train + n_test == n_rows and set(train_idx) | set(test_idx) == set(corrupted.index)
    frac = n_test / max(n_rows, 1)
    frac_ok = abs(frac - float(cfg["test_size"])) < 0.002
    report.add(
        sec,
        "Split sizes",
        PASS if (frac_ok and covers) else FAIL,
        f"train={n_train}, test={n_test}, test fraction={frac:.4f} (config test_size={cfg['test_size']})",
    )
    report.add(sec, "Train and test are disjoint and cover every row", PASS if (disjoint and covers) else FAIL)

    # Realised missingness rate per column (information; WARN only if far from target)
    rates = {c: float(mask[c].mean()) for c in miss_cols}
    far = {c: round(r, 4) for c, r in rates.items() if abs(r - level) > 0.01}
    report.add(
        sec,
        f"Realised missingness rate per column (target {level:.2f})",
        PASS if not far else WARN,
        ", ".join(f"{c}={r:.3f}" for c, r in rates.items()),
    )
    return data


# ---------------------------------------------------------------------------
# 5. Manifest
# ---------------------------------------------------------------------------
def check_manifest(report: Report) -> None:
    sec = "5. Manifest"
    try:
        from src.handoff import list_conditions
    except Exception as exc:  # noqa: BLE001
        report.add(sec, "import list_conditions", FAIL, repr(exc))
        return
    try:
        manifest = list_conditions()
    except Exception as exc:  # noqa: BLE001
        report.add(sec, "list_conditions()", FAIL, f"{exc!r}")
        return
    n = len(manifest)
    report.add(
        sec,
        "Conditions listed",
        PASS if n == 75 else WARN,
        f"{n} (expected 75); columns: {list(getattr(manifest, 'columns', []))}",
    )


# ---------------------------------------------------------------------------
# 6. Imputed files from Member 2
# ---------------------------------------------------------------------------
def check_imputed_status(report: Report, cfg: dict) -> None:
    sec = "6. Imputed files (Member 2 handoff)"
    expected = list(imputed_conditions(cfg))
    present = [c for c in expected if imputed_path(**c).exists()]
    report.add(
        sec,
        "Files present",
        INFO,
        f"{len(present)} of {len(expected)} expected under data/imputed/",
    )

    for method in cfg["imputation_methods"]:
        n_have = sum(1 for c in present if c["method"] == method)
        n_total = sum(1 for c in expected if c["method"] == method)
        report.add(sec, f"method '{method}'", INFO, f"{n_have}/{n_total}")

    if IMPUTED_DIR.exists():
        folders = sorted(p.name for p in IMPUTED_DIR.iterdir() if p.is_dir())
        unexpected = [f for f in folders if f not in cfg["imputation_methods"]]
        if unexpected:
            report.add(
                sec,
                "Unexpected folders in data/imputed/",
                WARN,
                f"{unexpected} (possible outcome-free / outcome-aware variant layout; "
                "confirm with Member 2, open decision D1)",
            )
    else:
        report.add(
            sec,
            "data/imputed/ folder",
            INFO,
            "does not exist yet; Member 2 has not delivered (this is expected before M1 can start)",
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_all(
    mechanism: str = "MAR", level: float = 0.30, repeat: int = 0
) -> Report:
    report = Report()
    check_environment(report)
    try:
        cfg = load_config()
    except Exception as exc:  # noqa: BLE001
        report.add("2. Config (config.yaml)", "Load config.yaml", FAIL, repr(exc))
        return report
    check_config(report, cfg)
    check_repo_files(report, cfg)
    check_condition_load(report, cfg, mechanism, level, repeat)
    check_manifest(report)
    check_imputed_status(report, cfg)
    return report


def save_report(report: Report, path: Path | None = None) -> Path:
    path = Path(path) if path else EXPERIMENT_RESULTS_DIR / "m3_00_onboarding_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2)
    return path
