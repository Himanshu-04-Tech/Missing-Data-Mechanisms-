"""Module M1: validation of Member 2's imputed files.

NEW file; does not modify any existing module in ``src/``.

For every expected imputed run (method x mechanism x level x repeat = 300) this
module checks the file against the CORRUPTED dataset of the same condition.
It never uses the ground truth: validation only needs to know that observed
values were left alone and that every missing cell was filled.

Per-file checks (SRS requirements FR-1.1 to FR-1.6):

  row_count_ok          21,436 rows (config ``expected_rows``)
  columns_ok            the 17 expected columns, in the expected order
  index_ok              ``row_index`` equals the corrupted file's index (same rows, same order)
  numeric_ok            every column converts to float
  no_nan                0 NaN anywhere (imputation must fill every gap)
  no_inf                0 infinite values
  observed_unchanged    cells that were observed in the 4 missingness columns are unchanged
  protected_unchanged   price, log_price, lat, long are identical to the corrupted file
  other_unchanged       the remaining fully observed columns are identical to the corrupted file

A file with status MISSING does not exist. A file is PASS only if every check is True.
A False check can also mean "could not be evaluated" (for example the columns were
wrong, so values could not be compared); the ``notes`` column says which.

Output: ``results/experiment_results/validation_report.csv`` (one row per expected run).
Downstream modules (M2 statistical track, M3 predictive track) should use
``passed_runs()`` so they only ever read files that passed validation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from src.m3_common import (
    EXPERIMENT_RESULTS_DIR,
    PROJECT_ROOT,
    imputed_conditions,
    imputed_path,
    load_config,
)

VALIDATION_REPORT_PATH = EXPERIMENT_RESULTS_DIR / "validation_report.csv"

PASS, FAIL, MISSING = "PASS", "FAIL", "MISSING"

FRAME_CHECKS = [
    "row_count_ok",
    "columns_ok",
    "index_ok",
    "numeric_ok",
    "no_nan",
    "no_inf",
    "observed_unchanged",
    "protected_unchanged",
    "other_unchanged",
]

KEY_COLUMNS = ["method", "mechanism", "level", "repeat"]

# Tolerance for "unchanged" (values pass through CSV, so allow tiny float noise).
RTOL = 1e-8
ATOL = 1e-8


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def read_imputed_file(path: Path | str) -> pd.DataFrame:
    """Read one imputed ``.csv.gz`` and index it by ``row_index``."""
    frame = pd.read_csv(path)
    if "row_index" not in frame.columns:
        raise ValueError("file has no 'row_index' column")
    return frame.set_index("row_index")


# ---------------------------------------------------------------------------
# Pure validation of one frame (no file access; easy to unit-test)
# ---------------------------------------------------------------------------
def _differs(a: np.ndarray, b: np.ndarray) -> int:
    """Number of positions where a and b differ beyond tolerance."""
    return int((~np.isclose(a, b, rtol=RTOL, atol=ATOL)).sum())


def validate_imputed_frame(
    imputed: pd.DataFrame, corrupted: pd.DataFrame, cfg: dict
) -> dict:
    """Validate one imputed frame against its corrupted counterpart.

    Returns a dict with the boolean checks in ``FRAME_CHECKS``, a few counts, and
    ``notes`` (a list of human-readable problems).
    """
    expected_cols = list(cfg["expected_columns"])
    miss_cols = list(cfg["missingness_columns"])
    protected = list(cfg["always_observed"])
    other_cols = [c for c in expected_cols if c not in miss_cols and c not in protected]

    result: dict = {name: False for name in FRAME_CHECKS}
    result.update(
        n_rows=len(imputed),
        n_cells_to_impute=int(corrupted[miss_cols].isna().sum().sum()),
        n_nan=-1,
        n_observed_changed=-1,
        n_protected_changed=-1,
        n_other_changed=-1,
        notes=[],
    )
    notes: list[str] = result["notes"]

    # --- row count
    result["row_count_ok"] = len(imputed) == int(cfg["expected_rows"])
    if not result["row_count_ok"]:
        notes.append(f"rows={len(imputed)}, expected {cfg['expected_rows']}")

    # --- columns
    if list(imputed.columns) == expected_cols:
        result["columns_ok"] = True
    elif set(imputed.columns) == set(expected_cols):
        notes.append("columns present but in a different order")
        imputed = imputed[expected_cols]  # continue evaluating the rest
    else:
        gone = sorted(set(expected_cols) - set(imputed.columns))
        extra = sorted(set(imputed.columns) - set(expected_cols))
        notes.append(f"columns differ: missing={gone}, unexpected={extra}")
        return result  # cannot compare values

    # --- index
    if imputed.index.equals(corrupted.index):
        result["index_ok"] = True
    elif imputed.index.is_unique and set(imputed.index) == set(corrupted.index):
        notes.append("row_index has the same rows but a different order")
        imputed = imputed.reindex(corrupted.index)  # continue evaluating the rest
    else:
        notes.append("row_index does not match the corrupted file's index")
        return result  # cannot compare values

    # --- numeric
    try:
        values = imputed[expected_cols].to_numpy(dtype=float)
        result["numeric_ok"] = True
    except (ValueError, TypeError) as exc:
        notes.append(f"non-numeric values: {exc}")
        return result
    frame = pd.DataFrame(values, index=imputed.index, columns=expected_cols)

    # --- NaN / inf
    result["n_nan"] = int(np.isnan(values).sum())
    result["no_nan"] = result["n_nan"] == 0
    if not result["no_nan"]:
        notes.append(f"{result['n_nan']} NaN values remain")
    result["no_inf"] = bool(np.isfinite(values[~np.isnan(values)]).all())
    if not result["no_inf"]:
        notes.append("infinite values present")

    corrupted_vals = corrupted[expected_cols].to_numpy(dtype=float)
    corrupted_frame = pd.DataFrame(corrupted_vals, index=corrupted.index, columns=expected_cols)

    # --- observed cells in the missingness columns must be untouched
    changed = 0
    for col in miss_cols:
        observed = ~corrupted_frame[col].isna().to_numpy()
        a = frame[col].to_numpy()[observed]
        b = corrupted_frame[col].to_numpy()[observed]
        # NaN in the imputed observed cell counts as a change.
        changed += int(np.isnan(a).sum()) + _differs(a[~np.isnan(a)], b[~np.isnan(a)])
    result["n_observed_changed"] = changed
    result["observed_unchanged"] = changed == 0
    if changed:
        notes.append(f"{changed} observed cells in missingness columns were changed")

    # --- protected columns and the other fully observed columns
    def _compare(cols: Iterable[str]) -> int:
        total = 0
        for col in cols:
            a, b = frame[col].to_numpy(), corrupted_frame[col].to_numpy()
            total += int(np.isnan(a).sum()) + _differs(a[~np.isnan(a)], b[~np.isnan(a)])
        return total

    result["n_protected_changed"] = _compare(protected)
    result["protected_unchanged"] = result["n_protected_changed"] == 0
    if not result["protected_unchanged"]:
        notes.append(f"protected columns {protected} changed in {result['n_protected_changed']} cells")

    result["n_other_changed"] = _compare(other_cols)
    result["other_unchanged"] = result["n_other_changed"] == 0
    if not result["other_unchanged"]:
        notes.append(f"{result['n_other_changed']} cells changed in other observed columns")

    return result


# ---------------------------------------------------------------------------
# Whole-sweep validation
# ---------------------------------------------------------------------------
def _default_loader() -> Callable[..., dict]:
    from src.handoff import load_condition  # existing Member 1 module

    return load_condition


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run_validation(
    cfg: dict | None = None,
    imputed_root: Path | str | None = None,
    loader: Callable[..., dict] | None = None,
    methods: Iterable[str] | None = None,
    mechanisms: Iterable[str] | None = None,
    levels: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Validate every expected imputed run and return one row per run.

    ``loader`` defaults to ``src.handoff.load_condition``; tests can inject a fake.
    Corrupted data is loaded once per (mechanism, level, repeat) and only if at least
    one imputed file exists for it.
    """
    cfg = load_config() if cfg is None else cfg
    methods = set(methods) if methods else None
    mechanisms = set(mechanisms) if mechanisms else None
    levels = {round(float(x), 4) for x in levels} if levels else None

    runs = [
        r
        for r in imputed_conditions(cfg)
        if (methods is None or r["method"] in methods)
        and (mechanisms is None or r["mechanism"] in mechanisms)
        and (levels is None or round(float(r["level"]), 4) in levels)
    ]

    # Group the runs that share a corrupted condition so it is loaded only once.
    groups: dict[tuple, list[dict]] = {}
    for run in runs:
        groups.setdefault((run["mechanism"], run["level"], run["repeat"]), []).append(run)

    rows: list[dict] = []
    load = loader
    for (mechanism, level, repeat), group in groups.items():
        corrupted = None
        for run in group:
            path = imputed_path(
                run["method"], mechanism, level, repeat, imputed_root=imputed_root
            )
            row = {**{k: run[k] for k in KEY_COLUMNS}, "path": _rel(path)}

            if not path.exists():
                row.update(status=MISSING, notes="file does not exist")
                rows.append(row)
                continue

            if corrupted is None:
                load = load or _default_loader()
                corrupted = load(mechanism=mechanism, level=level, repeat=repeat)["corrupted"]

            try:
                imputed = read_imputed_file(path)
            except Exception as exc:  # noqa: BLE001 (any read problem is a validation failure)
                row.update(status=FAIL, notes=f"unreadable: {exc}")
                rows.append(row)
                continue

            result = validate_imputed_frame(imputed, corrupted, cfg)
            notes = result.pop("notes")
            passed = all(result[name] for name in FRAME_CHECKS)
            row.update(result)
            row.update(status=PASS if passed else FAIL, notes="; ".join(notes))
            rows.append(row)

    report = pd.DataFrame(rows)
    ordered = KEY_COLUMNS + ["status", "path"] + FRAME_CHECKS + [
        "n_rows", "n_cells_to_impute", "n_nan", "n_observed_changed",
        "n_protected_changed", "n_other_changed", "notes",
    ]
    for col in ordered:
        if col not in report.columns:
            report[col] = np.nan
    count_cols = ["n_rows", "n_cells_to_impute", "n_nan", "n_observed_changed",
                  "n_protected_changed", "n_other_changed"]
    report[count_cols] = report[count_cols].astype("Int64")  # -1 means "not evaluated"
    return report[ordered]


# ---------------------------------------------------------------------------
# Saving / reading the report; used by later modules
# ---------------------------------------------------------------------------
def save_validation_report(report: pd.DataFrame, path: Path | str | None = None) -> Path:
    path = Path(path) if path else VALIDATION_REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False)
    return path


def load_validation_report(path: Path | str | None = None) -> pd.DataFrame:
    path = Path(path) if path else VALIDATION_REPORT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python3 scripts/m3_01_validate_imputed.py"
        )
    return pd.read_csv(path)


def passed_runs(report: pd.DataFrame | None = None) -> pd.DataFrame:
    """The runs that passed validation (key columns only). Use this in M2 and M3."""
    report = load_validation_report() if report is None else report
    return report.loc[report["status"] == PASS, KEY_COLUMNS].reset_index(drop=True)


def failed_checks(row: pd.Series) -> list[str]:
    """Names of the checks that are False in one report row."""
    return [name for name in FRAME_CHECKS if row.get(name) == False]  # noqa: E712 (also works for numpy bool)


def all_passed(report: pd.DataFrame, allow_missing: bool = False) -> bool:
    """True if no file failed (and, unless allowed, none is missing)."""
    if report.empty:
        return False
    bad = {FAIL} if allow_missing else {FAIL, MISSING}
    return not report["status"].isin(bad).any()
