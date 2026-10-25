"""Module M3: predictive track (Random Forest) on Member 2's validated imputed files.

NEW file; does not modify any existing module in ``src/``.

What it does
------------
1. REFERENCE (upper bound, FR-3.4): fit a RandomForestRegressor on the COMPLETE train
   rows and score it on the COMPLETE test rows. Needs nothing from Member 2.
2. For every validated imputed run (method x mechanism x level x repeat = 300):
   split the imputed file with the saved ``split_indices.csv``, fit the SAME Random Forest
   on the imputed TRAIN rows only, and score it on the imputed TEST rows
   (RMSE, MAE, R2 in log-price units). (FR-3.2, FR-3.3, FR-3.6)
3. Optional diagnostic (FR-3.7 / D5, ``score_on_truth``): predict the COMPLETE (truth)
   test rows with the same model. Compare with step 2 to separate
     * error from imputed TEST INPUTS      (rmse_imputed_test  vs  rmse_truth_test), and
     * error from a MODEL TRAINED ON IMPUTED DATA (rmse_truth_test vs reference rmse).
   Truth is only ever used to SCORE; it is never an input to fitting.
4. Aggregates over the 5 repeats (mean and sd) into a predictive summary table.

Complete-case is NOT scored here: a Random Forest needs complete inputs and complete-case
deletion leaves test rows without predictions (FR-3.8). It stays a statistical-track baseline.

Leakage rules enforced in code
------------------------------
* features = the 15 predictors; ``price`` and ``log_price`` are never features
  (``feature_columns`` asserts this).
* the model is fit on rows in ``train_idx`` only; test labels are used only to score.
* the same Random Forest settings are used for every condition (no tuning), read from
  the ``predictive:`` block of ``config.yaml`` (defaults below if the block is absent).
* every random seed comes from ``config.yaml`` (``base_seed`` unless overridden).

Metric definitions
------------------
* ``rmse``, ``mae``, ``r2``  : imputed model, imputed test inputs, vs the observed log_price
                               (log_price is a protected, always-observed column).
* ``rmse_truth_test`` ...    : imputed model, COMPLETE test inputs (diagnostic only).
The recovery GAPS (rmse - reference rmse, reference r2 - r2) are computed in Module M4
(merge), where decision D2 is applied; this module only produces the raw metrics.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.m3_common import (
    EXPERIMENT_RESULTS_DIR,
    TABLES_DIR,
    imputed_path,
    load_config,
    project_path,
)
from src.m3_validate_imputed import (
    KEY_COLUMNS,
    PASS,
    load_validation_report,
    read_imputed_file,
)

PREDICTIVE_RESULTS_PATH = EXPERIMENT_RESULTS_DIR / "predictive_results.csv"
PREDICTIVE_REFERENCE_PATH = EXPERIMENT_RESULTS_DIR / "predictive_reference.csv"
PREDICTIVE_ERRORS_PATH = EXPERIMENT_RESULTS_DIR / "predictive_errors.csv"
PREDICTIVE_SETTINGS_PATH = EXPERIMENT_RESULTS_DIR / "predictive_settings.json"
PREDICTIVE_TABLE_PATH = TABLES_DIR / "table5_predictive_recovery.csv"

# FR-3.5 (PROVISIONAL / D4): fixed, untuned. Overridden by config.yaml ``predictive:``.
DEFAULT_RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 1,
    "max_features": 1.0,
    "n_jobs": -1,
}

METRICS = ["rmse", "mae", "r2"]
TRUTH_METRICS = ["rmse_truth_test", "mae_truth_test", "r2_truth_test"]
SUMMARY_COLUMNS = METRICS + TRUTH_METRICS


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@dataclass
class PredictiveSettings:
    """Everything that defines the Random Forest, recorded with the results (FR-3.5)."""

    rf_params: dict
    random_state: int
    score_on_truth: bool
    target: str
    features: list[str]

    def as_record(self) -> dict:
        return {
            "rf_params": self.rf_params,
            "random_state": self.random_state,
            "score_on_truth": self.score_on_truth,
            "target": self.target,
            "features": self.features,
        }


def feature_columns(cfg: dict) -> list[str]:
    """The 15 predictors. ``price`` and ``log_price`` are never allowed in (leakage guard)."""
    banned = {cfg["target"], cfg["log_target_name"]}
    features = [c for c in cfg["expected_columns"] if c not in banned]
    assert not banned & set(features), "target leaked into the feature list"
    return features


def get_settings(cfg: dict, overrides: dict | None = None) -> PredictiveSettings:
    """Merge defaults, the ``predictive:`` block of the config, and optional overrides."""
    block = dict(cfg.get("predictive") or {})
    params = dict(DEFAULT_RF_PARAMS)
    params.update(block.get("rf_params") or {})
    random_state = int(block.get("random_state", cfg["base_seed"]))
    score_on_truth = bool(block.get("score_on_truth", True))
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key == "random_state":
            random_state = int(value)
        elif key == "score_on_truth":
            score_on_truth = bool(value)
        else:
            params[key] = value
    return PredictiveSettings(
        rf_params=params,
        random_state=random_state,
        score_on_truth=score_on_truth,
        target=cfg["log_target_name"],
        features=feature_columns(cfg),
    )


def make_model(settings: PredictiveSettings, random_state: int | None = None) -> RandomForestRegressor:
    """A fresh Random Forest. The same settings are used for every condition."""
    rs = settings.random_state if random_state is None else int(random_state)
    return RandomForestRegressor(random_state=rs, **settings.rf_params)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_split(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Train and test ``row_index`` arrays from the saved split (same split for every run)."""
    gen_dir = project_path(cfg.get("generated_missing_dir", "data/generated_missing"))
    path = gen_dir / "split_indices.csv"
    if not path.is_file():
        path = project_path(cfg["split_indices_path"])
    split = pd.read_csv(path)
    train = split.loc[split["split"] == "train", "row_index"].to_numpy()
    test = split.loc[split["split"] == "test", "row_index"].to_numpy()
    if len(np.intersect1d(train, test)):
        raise ValueError("train and test indices overlap")
    return train, test


def load_complete_data(cfg: dict) -> pd.DataFrame:
    """The complete (no-missing) data indexed by ``row_index``. Used for the reference and for scoring only."""
    gen_dir = project_path(cfg.get("generated_missing_dir", "data/generated_missing"))
    path = gen_dir / "ground_truth.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run: python3 scripts/run_sweep.py")
    return pd.read_csv(path, index_col="row_index")


# ---------------------------------------------------------------------------
# Core: fit on train, score on test
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def fit_and_score(
    data: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    settings: PredictiveSettings,
    truth: pd.DataFrame | None = None,
    random_state: int | None = None,
) -> dict:
    """Fit on ``data`` TRAIN rows only; score on ``data`` TEST rows (and on ``truth`` test rows if given).

    ``data`` can be the imputed file (an imputed run) or the complete data (the reference).
    ``truth`` is used only to score, never to fit.
    """
    feats, target = settings.features, settings.target
    missing = [c for c in feats + [target] if c not in data.columns]
    if missing:
        raise ValueError(f"data is missing columns: {missing}")
    if data[feats + [target]].isna().any().any():
        raise ValueError("data contains NaN; Random Forest needs complete inputs (validate the file first)")

    train = data.loc[train_idx]
    test = data.loc[test_idx]

    model = make_model(settings, random_state)
    start = time.perf_counter()
    model.fit(train[feats], train[target])
    fit_seconds = time.perf_counter() - start

    out = regression_metrics(test[target], model.predict(test[feats]))
    out.update(n_train=len(train), n_test=len(test), fit_seconds=round(fit_seconds, 3))

    if truth is not None:
        truth_test = truth.loc[test_idx]
        scored = regression_metrics(truth_test[target], model.predict(truth_test[feats]))
        out.update({f"{k}_truth_test": v for k, v in scored.items()})
    return out


def run_reference(
    cfg: dict,
    settings: PredictiveSettings,
    complete: pd.DataFrame | None = None,
    split: tuple[np.ndarray, np.ndarray] | None = None,
    n_seeds: int = 1,
) -> pd.DataFrame:
    """Complete-data Random Forest (upper bound). ``n_seeds`` > 1 also gives the seed-to-seed spread.

    Row 0 uses ``settings.random_state`` and is THE reference; further rows use the next seeds and
    only describe how much the reference moves when only the forest's seed changes (useful for
    anchoring the Case thresholds, decision D3, and for not over-reading small gaps).
    """
    complete = load_complete_data(cfg) if complete is None else complete
    train_idx, test_idx = load_split(cfg) if split is None else split
    rows = []
    for k in range(max(1, int(n_seeds))):
        seed = settings.random_state + k
        res = fit_and_score(complete, train_idx, test_idx, settings, random_state=seed)
        rows.append({"model": "complete_data", "seed": seed, "is_primary": k == 0, **res})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sweep over the imputed runs
# ---------------------------------------------------------------------------
@dataclass
class PredictiveResults:
    runs: pd.DataFrame
    errors: pd.DataFrame


def _level_key(x: float) -> float:
    return round(float(x), 4)


def _row_key(row) -> tuple:
    return (row["method"], row["mechanism"], _level_key(row["level"]), int(row["repeat"]))


def select_runs(
    cfg: dict,
    imputed_root: Path | str | None = None,
    validation_report: pd.DataFrame | None = None,
    methods: Iterable[str] | None = None,
    mechanisms: Iterable[str] | None = None,
    levels: Iterable[float] | None = None,
    require_validation: bool = True,
) -> list[dict]:
    """The imputed runs to score: validated PASS runs (default) filtered by method/mechanism/level."""
    methods = set(methods) if methods else None
    mechanisms = set(mechanisms) if mechanisms else None
    levels = {_level_key(x) for x in levels} if levels else None

    if require_validation:
        report = load_validation_report() if validation_report is None else validation_report
        runs = report.loc[report["status"] == PASS, KEY_COLUMNS]
    else:
        from src.m3_common import imputed_conditions

        runs = pd.DataFrame(
            [r for r in imputed_conditions(cfg) if imputed_path(**r, imputed_root=imputed_root).exists()],
            columns=KEY_COLUMNS,
        )
    return [
        {"method": r["method"], "mechanism": r["mechanism"], "level": float(r["level"]), "repeat": int(r["repeat"])}
        for _, r in runs.iterrows()
        if (methods is None or r["method"] in methods)
        and (mechanisms is None or r["mechanism"] in mechanisms)
        and (levels is None or _level_key(r["level"]) in levels)
    ]


def run_predictive_track(
    cfg: dict | None = None,
    settings: PredictiveSettings | None = None,
    imputed_root: Path | str | None = None,
    validation_report: pd.DataFrame | None = None,
    complete: pd.DataFrame | None = None,
    split: tuple[np.ndarray, np.ndarray] | None = None,
    methods: Iterable[str] | None = None,
    mechanisms: Iterable[str] | None = None,
    levels: Iterable[float] | None = None,
    require_validation: bool = True,
    done: pd.DataFrame | None = None,
    checkpoint: Callable[[pd.DataFrame], None] | None = None,
    checkpoint_every: int = 10,
    progress: bool = True,
) -> PredictiveResults:
    """Score every validated imputed run with the fixed Random Forest.

    ``done`` holds earlier results (resume): runs whose key is already there are skipped and
    their rows are kept. ``checkpoint`` is called with the rows so far every ``checkpoint_every``
    new runs, so a crash in a long sweep loses little work.
    One failing run is recorded in ``errors`` and does not stop the sweep.
    """
    cfg = load_config() if cfg is None else cfg
    settings = get_settings(cfg) if settings is None else settings
    train_idx, test_idx = load_split(cfg) if split is None else split
    need_truth = settings.score_on_truth
    truth = (load_complete_data(cfg) if complete is None else complete) if need_truth else None

    run_list = select_runs(cfg, imputed_root, validation_report, methods, mechanisms, levels, require_validation)

    rows: list[dict] = [] if done is None or done.empty else done.to_dict("records")
    already = {_row_key(r) for r in rows}
    todo = [r for r in run_list if _row_key(r) not in already]
    if progress and already:
        print(f"Resuming: {len(already)} runs already scored, {len(todo)} to do.")

    errors: list[dict] = []
    for i, key in enumerate(todo, start=1):
        path = imputed_path(key["method"], key["mechanism"], key["level"], key["repeat"], imputed_root=imputed_root)
        try:
            data = read_imputed_file(path)
            res = fit_and_score(data, train_idx, test_idx, settings, truth=truth)
        except Exception as exc:  # noqa: BLE001 (one bad run must not stop the sweep)
            errors.append({**key, "error": repr(exc)})
            continue
        rows.append({**key, **res})
        if progress and (i % 5 == 0 or i == len(todo)):
            print(f"  predictive runs: {i}/{len(todo)}  (last fit {res['fit_seconds']:.1f}s)", flush=True)
        if checkpoint is not None and i % checkpoint_every == 0:
            checkpoint(_tidy(pd.DataFrame(rows)))

    return PredictiveResults(runs=_tidy(pd.DataFrame(rows)), errors=pd.DataFrame(errors))


def _tidy(runs: pd.DataFrame) -> pd.DataFrame:
    """Key columns first, deterministic order."""
    if runs.empty:
        return runs
    runs = runs.copy()
    runs["repeat"] = runs["repeat"].astype(int)
    rest = [c for c in runs.columns if c not in KEY_COLUMNS]
    order = {m: i for i, m in enumerate(["mean", "knn", "regression", "random_forest"])}
    mech = {m: i for i, m in enumerate(["MCAR", "MAR", "MNAR"])}
    runs = runs.assign(
        _m=runs["method"].map(order).fillna(99),
        _k=runs["mechanism"].map(mech).fillna(99),
    ).sort_values(["_k", "level", "_m", "repeat"]).drop(columns=["_m", "_k"])
    return runs[KEY_COLUMNS + rest].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Aggregation: predictive summary table (mean and sd over repeats)
# ---------------------------------------------------------------------------
def build_predictive_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and sd (ddof=1) over repeats, by method x mechanism x level."""
    if runs.empty:
        return pd.DataFrame()
    metrics = [c for c in SUMMARY_COLUMNS if c in runs.columns]
    grouped = runs.groupby(["method", "mechanism", "level"], sort=False)
    mean = grouped[metrics].mean().add_suffix("_mean")
    sd = grouped[metrics].std(ddof=1).add_suffix("_sd")
    n = grouped.size().rename("n_repeats")
    table = pd.concat([n, mean, sd], axis=1).reset_index()
    return _tidy_table(table)


def _tidy_table(table: pd.DataFrame) -> pd.DataFrame:
    order = {m: i for i, m in enumerate(["mean", "knn", "regression", "random_forest"])}
    mech = {m: i for i, m in enumerate(["MCAR", "MAR", "MNAR"])}
    return (
        table.assign(_m=table["method"].map(order).fillna(99), _k=table["mechanism"].map(mech).fillna(99))
        .sort_values(["_k", "level", "_m"])
        .drop(columns=["_m", "_k"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Saving / loading (used by the merge module M4)
# ---------------------------------------------------------------------------
def save_predictive_results(
    results: PredictiveResults,
    reference: pd.DataFrame | None,
    settings: PredictiveSettings,
    table: pd.DataFrame,
) -> dict[str, Path]:
    import json

    EXPERIMENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, frame, path in [
        ("runs", results.runs, PREDICTIVE_RESULTS_PATH),
        ("reference", reference, PREDICTIVE_REFERENCE_PATH),
        ("table5", table, PREDICTIVE_TABLE_PATH),
    ]:
        if frame is not None and not frame.empty:
            frame.to_csv(path, index=False)
            written[name] = path
    PREDICTIVE_SETTINGS_PATH.write_text(json.dumps(settings.as_record(), indent=2), encoding="utf-8")
    written["settings"] = PREDICTIVE_SETTINGS_PATH
    if not results.errors.empty:
        results.errors.to_csv(PREDICTIVE_ERRORS_PATH, index=False)
        written["errors"] = PREDICTIVE_ERRORS_PATH
    elif PREDICTIVE_ERRORS_PATH.exists():
        PREDICTIVE_ERRORS_PATH.unlink()  # no stale error file from an earlier run
    return written


def load_predictive_runs(path: Path | str | None = None) -> pd.DataFrame:
    path = Path(path) if path else PREDICTIVE_RESULTS_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python3 scripts/m3_03_predictive.py")
    return pd.read_csv(path)


def load_predictive_reference(path: Path | str | None = None) -> pd.DataFrame:
    path = Path(path) if path else PREDICTIVE_REFERENCE_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python3 scripts/m3_03_predictive.py --reference-only")
    return pd.read_csv(path)
