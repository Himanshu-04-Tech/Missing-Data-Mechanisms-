"""Module M2: statistical-recovery track on Member 2's validated imputed files.

NEW file; does not modify any existing module in ``src/``.

What it does
------------
1. Fits the REFERENCE model: OLS of ``log_price`` on the 15 predictors using the
   complete data (``data/processed/clean_data.csv``).
2. For every validated imputed run (method x mechanism x level x repeat) it fits the
   same OLS on the full imputed dataset and compares it with the reference.
3. Repeats the same comparison for a ``complete_case`` baseline (rows with any NaN
   dropped from the corrupted data), so imputers can be compared with doing nothing clever.
4. Aggregates over the 5 repeats into Table 4 (mean and sd) and a bias table.

Metric definitions (all documented here so the team can check them)
-------------------------------------------------------------------
* ``z_diff``       = (beta_imputed - beta_reference) / SE_reference.
                     Measured in REFERENCE-SE units. |z_diff| mixes bias and noise;
                     the repeat-averaged SIGNED z_diff is the bias.
* ``ci_covers_ref``= the imputed 95% CI contains the reference coefficient.
* ``ci_overlap``   = the imputed CI and the reference CI intersect.
* ``sig_flip``     = significance (p < alpha) differs between reference and imputed.
* ``se_ratio``     = SE_imputed / SE_reference (below 1 means over-confident, above 1 means noisier).
* correlation differences: Pearson correlation among the 15 predictors and log_price,
  imputed minus reference.

This is an independent implementation written from the metric descriptions in the team
handover. It has NOT been compared with ``src/statistics.py`` yet; use
``scripts/m3_02_stat_track.py --check-complete-case`` to compare the complete-case numbers
with Member 1's saved complete-case results before trusting the two as identical.

Statistical track note: it uses the FULL imputed dataset (train + test rows), so no test
split applies here. Ground truth is never read. The reference comes from the clean
complete data, which is exactly what "compared with the complete-data result" means.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from src.m3_common import (
    EXPERIMENT_RESULTS_DIR,
    TABLES_DIR,
    corrupted_conditions,
    imputed_path,
    load_config,
    project_path,
)
from src.m3_validate_imputed import (
    PASS,
    KEY_COLUMNS,
    load_validation_report,
    read_imputed_file,
)

try:  # scipy gives exact Student-t quantiles; fall back to the normal approximation
    from scipy import stats as _scipy_stats
except ImportError:  # pragma: no cover
    _scipy_stats = None

COMPLETE_CASE = "complete_case"
CONST = "const"

RUNS_PATH = EXPERIMENT_RESULTS_DIR / "stat_recovery_runs.csv"
COEFS_PATH = EXPERIMENT_RESULTS_DIR / "stat_recovery_coefficients.csv"
CORRS_PATH = EXPERIMENT_RESULTS_DIR / "stat_recovery_correlations.csv"
ERRORS_PATH = EXPERIMENT_RESULTS_DIR / "stat_recovery_errors.csv"
TABLE4_PATH = TABLES_DIR / "table4_statistical_recovery.csv"
TABLE4B_PATH = TABLES_DIR / "table4b_bias_by_coefficient.csv"

SUMMARY_METRICS = [
    "stat_mean_abs_z_affected",
    "stat_mean_signed_z_affected",
    "stat_mean_abs_z_all",
    "stat_ci_coverage_affected",
    "stat_ci_coverage_all",
    "stat_n_sig_flips",
    "stat_n_sig_flips_affected",
    "stat_mean_se_ratio_affected",
    "stat_corr_mean_abs_diff",
    "stat_corr_mean_abs_diff_affected",
    "stat_corr_max_abs_diff",
    "r2_imputed",
]


# ---------------------------------------------------------------------------
# OLS
# ---------------------------------------------------------------------------
@dataclass
class OLSResult:
    terms: list[str]
    beta: np.ndarray
    se: np.ndarray
    p: np.ndarray
    ci_lo: np.ndarray
    ci_hi: np.ndarray
    df: int
    r2: float
    n: int


def _t_quantile(prob: float, df: int) -> float:
    if _scipy_stats is not None:
        return float(_scipy_stats.t.ppf(prob, df))
    # Normal approximation (accurate for the large n used in this project).
    from statistics import NormalDist

    return NormalDist().inv_cdf(prob)


def _two_sided_p(t_values: np.ndarray, df: int) -> np.ndarray:
    if _scipy_stats is not None:
        return 2.0 * _scipy_stats.t.sf(np.abs(t_values), df)
    return np.array([math.erfc(abs(t) / math.sqrt(2.0)) for t in t_values])


def fit_ols(X: pd.DataFrame, y: pd.Series | np.ndarray, alpha: float = 0.05) -> OLSResult:
    """OLS with intercept, via QR for numerical stability (``yr_built`` and ``sqft_*`` are on very different scales)."""
    A = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    yv = np.asarray(y, dtype=float)
    n, p = A.shape
    if n <= p:
        raise ValueError(f"not enough rows ({n}) for {p} parameters")
    if not (np.isfinite(A).all() and np.isfinite(yv).all()):
        raise ValueError("data contains NaN or infinite values")
    if np.linalg.matrix_rank(A) < p:
        raise ValueError("design matrix is rank deficient")

    Q, R = np.linalg.qr(A)
    beta = np.linalg.solve(R, Q.T @ yv)
    resid = yv - A @ beta
    rss = float(resid @ resid)
    df = n - p
    sigma2 = rss / df
    R_inv = np.linalg.inv(R)
    se = np.sqrt(sigma2 * np.einsum("ij,ij->i", R_inv, R_inv))
    t_values = beta / se
    tcrit = _t_quantile(1.0 - alpha / 2.0, df)
    tss = float(((yv - yv.mean()) ** 2).sum())
    return OLSResult(
        terms=[CONST] + list(X.columns),
        beta=beta,
        se=se,
        p=_two_sided_p(t_values, df),
        ci_lo=beta - tcrit * se,
        ci_hi=beta + tcrit * se,
        df=df,
        r2=1.0 - rss / tss if tss > 0 else float("nan"),
        n=n,
    )


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------
@dataclass
class Reference:
    ols: OLSResult
    corr: pd.DataFrame
    predictors: list[str]
    target: str
    variables: list[str]
    affected: list[str]
    alpha: float


def predictor_columns(cfg: dict) -> list[str]:
    """The 15 predictors: every expected column except price and log_price."""
    return [c for c in cfg["expected_columns"] if c not in (cfg["target"], cfg["log_target_name"])]


def build_reference(cfg: dict, frame: pd.DataFrame | None = None) -> Reference:
    """Fit the complete-data reference. ``frame`` can be injected (tests); default reads clean_data.csv."""
    if frame is None:
        frame = pd.read_csv(project_path(cfg["clean_data_path"]))
    target = cfg["log_target_name"]
    predictors = predictor_columns(cfg)
    missing = [c for c in predictors + [target] if c not in frame.columns]
    if missing:
        raise ValueError(f"reference data is missing columns: {missing}")
    alpha = float(cfg["alpha"])
    variables = predictors + [target]
    return Reference(
        ols=fit_ols(frame[predictors], frame[target], alpha),
        corr=frame[variables].corr(),
        predictors=predictors,
        target=target,
        variables=variables,
        affected=list(cfg["missingness_columns"]),
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Recovery metrics for one dataset
# ---------------------------------------------------------------------------
def coefficient_recovery(ref: OLSResult, imp: OLSResult, affected: list[str], alpha: float) -> pd.DataFrame:
    """One row per model term comparing the imputed fit with the reference fit."""
    if ref.terms != imp.terms:
        raise ValueError("reference and imputed models have different terms")
    out = pd.DataFrame(
        {
            "term": ref.terms,
            "is_affected": [t in affected for t in ref.terms],
            "beta_ref": ref.beta,
            "se_ref": ref.se,
            "p_ref": ref.p,
            "ci_lo_ref": ref.ci_lo,
            "ci_hi_ref": ref.ci_hi,
            "beta_imp": imp.beta,
            "se_imp": imp.se,
            "p_imp": imp.p,
            "ci_lo_imp": imp.ci_lo,
            "ci_hi_imp": imp.ci_hi,
        }
    )
    out["z_diff"] = (out["beta_imp"] - out["beta_ref"]) / out["se_ref"]
    out["abs_z_diff"] = out["z_diff"].abs()
    out["se_ratio"] = out["se_imp"] / out["se_ref"]
    out["ci_covers_ref"] = (out["ci_lo_imp"] <= out["beta_ref"]) & (out["beta_ref"] <= out["ci_hi_imp"])
    out["ci_overlap"] = (out["ci_lo_imp"] <= out["ci_hi_ref"]) & (out["ci_lo_ref"] <= out["ci_hi_imp"])
    out["sig_ref"] = out["p_ref"] < alpha
    out["sig_imp"] = out["p_imp"] < alpha
    out["sig_flip"] = out["sig_ref"] != out["sig_imp"]
    return out


def correlation_recovery(ref_corr: pd.DataFrame, imp_corr: pd.DataFrame, affected: list[str]) -> pd.DataFrame:
    """One row per unordered variable pair: reference, imputed and difference."""
    names = list(ref_corr.columns)
    rows, cols = np.triu_indices(len(names), k=1)
    a = [names[i] for i in rows]
    b = [names[j] for j in cols]
    ref_vals = ref_corr.to_numpy()[rows, cols]
    imp_vals = imp_corr.reindex(index=names, columns=names).to_numpy()[rows, cols]
    out = pd.DataFrame(
        {
            "var_a": a,
            "var_b": b,
            "corr_ref": ref_vals,
            "corr_imp": imp_vals,
        }
    )
    out["diff"] = out["corr_imp"] - out["corr_ref"]
    out["abs_diff"] = out["diff"].abs()
    out["involves_affected"] = out["var_a"].isin(affected) | out["var_b"].isin(affected)
    return out


def summarize_run(coef: pd.DataFrame, corr: pd.DataFrame, imp: OLSResult) -> dict:
    """One summary row for a run. Intercept is excluded from all summaries."""
    body = coef[coef["term"] != CONST]
    aff = body[body["is_affected"]]
    corr_aff = corr[corr["involves_affected"]]
    return {
        "n_obs": imp.n,
        "r2_imputed": imp.r2,
        "stat_mean_abs_z_affected": float(aff["abs_z_diff"].mean()),
        "stat_mean_signed_z_affected": float(aff["z_diff"].mean()),
        "stat_mean_abs_z_all": float(body["abs_z_diff"].mean()),
        "stat_ci_coverage_affected": float(aff["ci_covers_ref"].mean()),
        "stat_ci_coverage_all": float(body["ci_covers_ref"].mean()),
        "stat_n_sig_flips": int(body["sig_flip"].sum()),
        "stat_n_sig_flips_affected": int(aff["sig_flip"].sum()),
        "stat_mean_se_ratio_affected": float(aff["se_ratio"].mean()),
        "stat_corr_mean_abs_diff": float(corr["abs_diff"].mean()),
        "stat_corr_mean_abs_diff_affected": float(corr_aff["abs_diff"].mean()),
        "stat_corr_max_abs_diff": float(corr["abs_diff"].max()),
    }


def evaluate_dataset(data: pd.DataFrame, ref: Reference) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit OLS on ``data`` and compare it with the reference."""
    imp = fit_ols(data[ref.predictors], data[ref.target], ref.alpha)
    coef = coefficient_recovery(ref.ols, imp, ref.affected, ref.alpha)
    corr = correlation_recovery(ref.corr, data[ref.variables].corr(), ref.affected)
    return coef, corr, summarize_run(coef, corr, imp)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
@dataclass
class StatResults:
    runs: pd.DataFrame
    coefficients: pd.DataFrame
    correlations: pd.DataFrame
    errors: pd.DataFrame


def _level_key(x: float) -> float:
    return round(float(x), 4)


def _default_loader() -> Callable[..., dict]:
    from src.handoff import load_condition  # existing Member 1 module

    return load_condition


def run_statistical_track(
    cfg: dict | None = None,
    imputed_root: Path | str | None = None,
    loader: Callable[..., dict] | None = None,
    validation_report: pd.DataFrame | None = None,
    reference: Reference | None = None,
    methods: Iterable[str] | None = None,
    mechanisms: Iterable[str] | None = None,
    levels: Iterable[float] | None = None,
    include_complete_case: bool = True,
    include_imputed: bool = True,
    require_validation: bool = True,
    progress: bool = True,
) -> StatResults:
    """Evaluate every validated imputed run (and the complete-case baseline).

    With ``require_validation`` (default) only runs with status PASS in the Module 1
    validation report are used. If the report is missing a FileNotFoundError tells you to run
    Module 1 first.
    """
    cfg = load_config() if cfg is None else cfg
    ref = build_reference(cfg) if reference is None else reference
    methods = set(methods) if methods else None
    mechanisms = set(mechanisms) if mechanisms else None
    levels = {_level_key(x) for x in levels} if levels else None

    def wanted(mech: str, level: float, method: str | None = None) -> bool:
        return (
            (methods is None or method is None or method in methods)
            and (mechanisms is None or mech in mechanisms)
            and (levels is None or _level_key(level) in levels)
        )

    # --- which imputed runs to evaluate
    if not include_imputed:
        runs = pd.DataFrame(columns=KEY_COLUMNS)
    elif require_validation:
        report = load_validation_report() if validation_report is None else validation_report
        runs = report.loc[report["status"] == PASS, KEY_COLUMNS]
        not_passed = int((report["status"] != PASS).sum())
        if progress and not_passed:
            print(f"Skipping {not_passed} runs that did not pass validation (see validation_report.csv).")
    else:
        from src.m3_common import imputed_conditions

        runs = pd.DataFrame(
            [r for r in imputed_conditions(cfg) if imputed_path(**r, imputed_root=imputed_root).exists()],
            columns=KEY_COLUMNS,
        )
    run_list = [
        {k: row[k] for k in KEY_COLUMNS}
        for _, row in runs.iterrows()
        if wanted(row["mechanism"], row["level"], row["method"])
    ]

    run_rows: list[dict] = []
    coef_frames: list[pd.DataFrame] = []
    corr_frames: list[pd.DataFrame] = []
    errors: list[dict] = []

    def record(key: dict, data: pd.DataFrame) -> None:
        try:
            coef, corr, summary = evaluate_dataset(data, ref)
        except Exception as exc:  # noqa: BLE001 (one bad run must not stop the sweep)
            errors.append({**key, "error": repr(exc)})
            return
        run_rows.append({**key, **summary})
        coef_frames.append(coef.assign(**key))
        corr_frames.append(corr.assign(**key))

    for i, key in enumerate(run_list, start=1):
        path = imputed_path(key["method"], key["mechanism"], key["level"], int(key["repeat"]), imputed_root=imputed_root)
        key = {**key, "repeat": int(key["repeat"])}
        try:
            data = read_imputed_file(path)
        except Exception as exc:  # noqa: BLE001
            errors.append({**key, "error": f"unreadable: {exc}"})
            continue
        record(key, data)
        if progress and (i % 25 == 0 or i == len(run_list)):
            print(f"  imputed runs: {i}/{len(run_list)}")

    # --- complete-case baseline (rows with any NaN dropped from the corrupted data)
    if include_complete_case:
        load = loader or _default_loader()
        todo = [c for c in corrupted_conditions(cfg) if wanted(c["mechanism"], c["level"])]
        for j, cond in enumerate(todo, start=1):
            corrupted = load(mechanism=cond["mechanism"], level=cond["level"], repeat=cond["repeat"])["corrupted"]
            key = {"method": COMPLETE_CASE, **cond}
            record(key, corrupted.dropna())
            if progress and (j % 25 == 0 or j == len(todo)):
                print(f"  complete-case conditions: {j}/{len(todo)}")

    def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _front(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        return df[KEY_COLUMNS + [c for c in df.columns if c not in KEY_COLUMNS]]

    return StatResults(
        runs=_front(pd.DataFrame(run_rows)),
        coefficients=_front(_concat(coef_frames)),
        correlations=_front(_concat(corr_frames)),
        errors=pd.DataFrame(errors),
    )


# ---------------------------------------------------------------------------
# Aggregation: Table 4 and the bias table
# ---------------------------------------------------------------------------
def _method_order(methods: Iterable[str]) -> list[str]:
    preferred = [COMPLETE_CASE, "mean", "knn", "regression", "random_forest"]
    methods = list(dict.fromkeys(methods))
    return [m for m in preferred if m in methods] + sorted(m for m in methods if m not in preferred)


def build_table4(runs: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """Mean and sd (ddof=1) over repeats, by method x mechanism x level."""
    metrics = metrics or SUMMARY_METRICS
    if runs.empty:
        return pd.DataFrame()
    grouped = runs.groupby(["method", "mechanism", "level"], sort=False)
    mean = grouped[metrics].mean().add_suffix("_mean")
    sd = grouped[metrics].std(ddof=1).add_suffix("_sd")
    n = grouped.size().rename("n_repeats")
    table = pd.concat([n, mean, sd], axis=1).reset_index()
    order = {m: i for i, m in enumerate(_method_order(table["method"]))}
    mech_order = {m: i for i, m in enumerate(["MCAR", "MAR", "MNAR"])}
    table = table.assign(
        _m=table["method"].map(order), _k=table["mechanism"].map(mech_order).fillna(99)
    ).sort_values(["_k", "level", "_m"]).drop(columns=["_m", "_k"])
    return table.reset_index(drop=True)


def build_bias_table(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Per coefficient: repeat-averaged SIGNED z_diff (bias) and its sd (noise)."""
    if coefficients.empty:
        return pd.DataFrame()
    body = coefficients[coefficients["term"] != CONST]
    grouped = body.groupby(["method", "mechanism", "level", "term", "is_affected"], sort=False)
    table = grouped.agg(
        n_repeats=("z_diff", "size"),
        bias_z=("z_diff", "mean"),
        noise_sd_z=("z_diff", lambda s: s.std(ddof=1)),
        mean_abs_z=("abs_z_diff", "mean"),
        ci_coverage=("ci_covers_ref", "mean"),
        sig_flip_rate=("sig_flip", "mean"),
    ).reset_index()
    return table


# ---------------------------------------------------------------------------
# Saving / loading
# ---------------------------------------------------------------------------
def save_results(results: StatResults, table4: pd.DataFrame, bias: pd.DataFrame) -> dict[str, Path]:
    EXPERIMENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, frame, path in [
        ("runs", results.runs, RUNS_PATH),
        ("coefficients", results.coefficients, COEFS_PATH),
        ("correlations", results.correlations, CORRS_PATH),
        ("table4", table4, TABLE4_PATH),
        ("table4b", bias, TABLE4B_PATH),
    ]:
        if not frame.empty:
            frame.to_csv(path, index=False)
            written[name] = path
    if not results.errors.empty:
        results.errors.to_csv(ERRORS_PATH, index=False)
        written["errors"] = ERRORS_PATH
    elif ERRORS_PATH.exists():
        ERRORS_PATH.unlink()  # do not leave a stale error file from an earlier run
    return written


def load_stat_runs(path: Path | str | None = None) -> pd.DataFrame:
    """Per-run statistical summaries; used by the merge module (M4)."""
    path = Path(path) if path else RUNS_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/m3_02_stat_track.py")
    return pd.read_csv(path)
