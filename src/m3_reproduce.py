"""Module M6: reproducibility and readiness tooling (one command per stage, output audit).

NEW file; does not modify any existing module in ``src/``.

Covers SRS requirements
-----------------------
FR-6.4  "All seeds read from config.yaml; a single command per stage regenerates outputs."
        ``STAGES`` lists the one command per stage; ``scripts/m3_06_run_all.py`` runs them in order.
        The only random component in Member 3's code is the Random Forest, and its seed is
        ``config.yaml -> predictive.random_state`` (default ``base_seed``). The statistical track,
        merge and Case flagging are deterministic. ``tests/test_m3_reproducibility.py`` checks
        that no seed is hard-coded.
FR-6.5  ``scripts/m3_06_run_all.py --with-tests`` runs pytest as the last step.
NFR     Traceability / integrity: ``audit()`` checks that the outputs are complete, consistent with
        each other and with the config, and that no smoke-test setting leaked into final results.

Output manifest
---------------
After a run the runner writes ``results/experiment_results/pipeline_manifest.json`` with the exit code
and timing of each stage, the git commit, the config digest, and a CONTENT DIGEST of every output.
The content digest ignores run-time-only columns (``VOLATILE_COLUMNS``, for example ``fit_seconds``) and
rounds floats to ``DIGEST_DECIMALS`` decimals (parallel Random Forest predictions can differ in the last
bits), so two runs with the same inputs and seeds have identical manifests apart from timings.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.m3_common import EXPERIMENT_RESULTS_DIR, PROJECT_ROOT, TABLES_DIR, FIGURES_DIR, load_config

MANIFEST_PATH = EXPERIMENT_RESULTS_DIR / "pipeline_manifest.json"

# Columns that legitimately differ between two identical runs (wall-clock time).
VOLATILE_COLUMNS = {"fit_seconds"}

# A Random Forest with n_jobs > 1 adds the trees' predictions in a thread-dependent order, so two runs with
# the same seed can differ in the last bits (about 1e-16). Digests therefore round numbers to this many
# decimals. Real differences (a different seed, data or code) are many orders of magnitude larger.
DIGEST_DECIMALS = 10

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"


# ---------------------------------------------------------------------------
# One command per stage
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Stage:
    number: int
    name: str
    script: str
    outputs: tuple[str, ...]  # paths relative to the project root

    def command(self, python: str | None = None, extra: list[str] | None = None) -> list[str]:
        return [python or sys.executable, str(Path("scripts") / self.script), *(extra or [])]


STAGES: tuple[Stage, ...] = (
    Stage(1, "validate", "m3_01_validate_imputed.py", ("results/experiment_results/validation_report.csv",)),
    Stage(2, "statistical", "m3_02_stat_track.py", (
        "results/experiment_results/stat_recovery_runs.csv",
        "results/experiment_results/stat_recovery_coefficients.csv",
        "results/experiment_results/stat_recovery_correlations.csv",
        "results/tables/table4_statistical_recovery.csv",
        "results/tables/table4b_bias_by_coefficient.csv",
    )),
    Stage(3, "predictive", "m3_03_predictive.py", (
        "results/experiment_results/predictive_results.csv",
        "results/experiment_results/predictive_reference.csv",
        "results/experiment_results/predictive_settings.json",
        "results/tables/table5_predictive_recovery.csv",
    )),
    Stage(4, "merge", "m3_04_merge.py", (
        "results/experiment_results/merged_results.csv",
        "results/tables/table6_overall_recovery.csv",
        "results/tables/table6b_best_methods.csv",
    )),
    Stage(5, "cases", "m3_05_cases.py", (
        "results/tables/cases_by_condition.csv",
        "results/tables/table7_case_counts.csv",
        "results/tables/case_sensitivity.csv",
        "results/tables/case_sensitivity_grid.csv",
        "results/experiment_results/case_thresholds_used.json",
    )),
)


def stage_by_number(n: int) -> Stage:
    for stage in STAGES:
        if stage.number == n:
            return stage
    raise ValueError(f"no stage {n}; valid stages are {[s.number for s in STAGES]}")


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------
def content_digest(path: Path | str) -> str:
    """SHA-256 of a file's content. CSVs drop VOLATILE_COLUMNS and round floats to DIGEST_DECIMALS first."""
    path = Path(path)
    if path.suffix == ".csv":
        frame = pd.read_csv(path)
        frame = frame.drop(columns=[c for c in frame.columns if c in VOLATILE_COLUMNS])
        floats = frame.select_dtypes("float").columns
        frame[floats] = frame[floats].round(DIGEST_DECIMALS)
        payload = frame.to_csv(index=False).encode("utf-8")
    else:
        payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest()


def git_commit(root: Path = PROJECT_ROOT) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10, check=False)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 (git may be absent)
        return None


def build_manifest(stage_records: list[dict], root: Path = PROJECT_ROOT, stages: tuple[Stage, ...] = STAGES) -> dict:
    outputs = {}
    for stage in stages:
        for rel in stage.outputs:
            path = root / rel
            outputs[rel] = {"stage": stage.number, "digest": content_digest(path) if path.exists() else None}
    config_path = root / "config.yaml"
    return {
        "git_commit": git_commit(root),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path.exists() else None,
        "python": sys.version.split()[0],
        "stages": stage_records,
        "outputs": outputs,
    }


def compare_manifests(old: dict, new: dict) -> list[str]:
    """Differences between two manifests, ignoring timings. Empty list = the outputs are identical."""
    problems = []
    if old.get("config_sha256") != new.get("config_sha256"):
        problems.append("config.yaml differs between the two runs")
    for rel in sorted(set(old.get("outputs", {})) | set(new.get("outputs", {}))):
        a = old.get("outputs", {}).get(rel, {}).get("digest")
        b = new.get("outputs", {}).get(rel, {}).get("digest")
        if a != b:
            problems.append(f"{rel}: {'missing in one run' if a is None or b is None else 'content differs'}")
    return problems


# ---------------------------------------------------------------------------
# Output audit
# ---------------------------------------------------------------------------
@dataclass
class Check:
    name: str
    status: str
    detail: str


def expected_counts(cfg: dict) -> dict:
    n_methods, n_mech = len(cfg["imputation_methods"]), len(cfg["mechanisms"])
    n_levels, n_rep = len(cfg["levels"]), int(cfg["n_repeats"])
    return {
        "imputed_runs": n_methods * n_mech * n_levels * n_rep,
        "cc_runs": n_mech * n_levels * n_rep,
        "conditions": n_methods * n_mech * n_levels,
        "n_repeats": n_rep,
    }


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _missing(name: str, path: Path) -> Check:
    return Check(name, FAIL, f"{path.name} not found (run the stage that produces it)")


def audit(cfg: dict | None = None, root: Path = PROJECT_ROOT) -> list[Check]:
    """Check that the Member 3 outputs are complete, consistent and honest. Does not recompute anything."""
    cfg = load_config() if cfg is None else cfg
    exp_dir, tab_dir = root / "results" / "experiment_results", root / "results" / "tables"
    n = expected_counts(cfg)
    key = ["method", "mechanism", "level", "repeat"]
    checks: list[Check] = []

    def norm(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["level"] = out["level"].astype(float).round(4)
        out["repeat"] = out["repeat"].astype(int)
        return out

    # --- M1 validation
    path = exp_dir / "validation_report.csv"
    rep = _read(path)
    if rep is None:
        checks.append(_missing("M1 validation: all runs pass", path))
    else:
        vc = rep["status"].value_counts().to_dict()
        ok = len(rep) == n["imputed_runs"] and vc.get("PASS", 0) == n["imputed_runs"]
        checks.append(Check("M1 validation: all runs pass", PASS if ok else FAIL,
                            f"{len(rep)} rows, status counts {vc}; expected {n['imputed_runs']} PASS"))

    # --- M2 statistical
    path = exp_dir / "stat_recovery_runs.csv"
    stat = _read(path)
    if stat is None:
        checks.append(_missing("M2 statistical: runs complete", path))
    else:
        imp = stat[stat["method"] != "complete_case"]
        cc = stat[stat["method"] == "complete_case"]
        dup = norm(stat).duplicated(key).any()
        ok = len(imp) == n["imputed_runs"] and len(cc) == n["cc_runs"] and not dup
        checks.append(Check("M2 statistical: runs complete", PASS if ok else FAIL,
                            f"{len(imp)} imputed (expected {n['imputed_runs']}), {len(cc)} complete-case "
                            f"(expected {n['cc_runs']}), duplicate keys: {bool(dup)}"))
    err = exp_dir / "stat_recovery_errors.csv"
    checks.append(Check("M2 statistical: no failed runs", FAIL if err.exists() else PASS,
                        "stat_recovery_errors.csv exists" if err.exists() else "no error file"))

    # --- M3 predictive
    path = exp_dir / "predictive_results.csv"
    pred = _read(path)
    if pred is None:
        checks.append(_missing("M3 predictive: runs complete", path))
    else:
        bad_vals = pred[["rmse", "mae", "r2"]].isna().any().any()
        dup = norm(pred).duplicated(key).any()
        ok = len(pred) == n["imputed_runs"] and not dup and not bad_vals
        checks.append(Check("M3 predictive: runs complete", PASS if ok else FAIL,
                            f"{len(pred)} rows (expected {n['imputed_runs']}), duplicate keys: {bool(dup)}, NaN metrics: {bool(bad_vals)}"))
    err = exp_dir / "predictive_errors.csv"
    checks.append(Check("M3 predictive: no failed runs", FAIL if err.exists() else PASS,
                        "predictive_errors.csv exists" if err.exists() else "no error file"))

    ref = _read(exp_dir / "predictive_reference.csv")
    if ref is None:
        checks.append(_missing("M3 predictive: reference present", exp_dir / "predictive_reference.csv"))
    else:
        prim = ref[ref["is_primary"].astype(bool)] if "is_primary" in ref.columns else ref
        checks.append(Check("M3 predictive: reference present", PASS if len(prim) == 1 else FAIL,
                            f"{len(prim)} primary reference row(s), expected 1"))

    sp = exp_dir / "predictive_settings.json"
    if not sp.exists():
        checks.append(_missing("M3 predictive: settings match config (no smoke-test leftovers)", sp))
    else:
        used = json.loads(sp.read_text(encoding="utf-8"))
        want = dict(cfg.get("predictive") or {}).get("rf_params") or {}
        diffs = [f"{k}: used {used['rf_params'].get(k)!r}, config {v!r}" for k, v in want.items()
                 if k != "n_jobs" and used["rf_params"].get(k) != v]
        want_seed = (cfg.get("predictive") or {}).get("random_state", cfg["base_seed"])
        if used.get("random_state") != want_seed:
            diffs.append(f"random_state: used {used.get('random_state')!r}, config {want_seed!r}")
        checks.append(Check("M3 predictive: settings match config (no smoke-test leftovers)",
                            PASS if not diffs else FAIL, "identical to config" if not diffs else "; ".join(diffs)))

    # --- M4 merge
    path = exp_dir / "merged_results.csv"
    merged = _read(path)
    if merged is None:
        checks.append(_missing("M4 merge: one row per run", path))
    else:
        same = pred is not None and set(map(tuple, norm(merged)[key].values)) == set(map(tuple, norm(pred)[key].values))
        ok = len(merged) == n["imputed_runs"] and not norm(merged).duplicated(key).any() and same
        checks.append(Check("M4 merge: one row per run", PASS if ok else FAIL,
                            f"{len(merged)} rows (expected {n['imputed_runs']}); same keys as predictive results: {same}"))
    path = tab_dir / "table6_overall_recovery.csv"
    t6 = _read(path)
    if t6 is None:
        checks.append(_missing("M4 Table 6: every condition, 5 repeats each", path))
    else:
        ok = len(t6) == n["conditions"] and (t6["n_repeats"] == n["n_repeats"]).all()
        checks.append(Check("M4 Table 6: every condition, 5 repeats each", PASS if ok else FAIL,
                            f"{len(t6)} conditions (expected {n['conditions']}), n_repeats values {sorted(t6['n_repeats'].unique())}"))

    # --- M5 cases
    path = tab_dir / "cases_by_condition.csv"
    cases = _read(path)
    if cases is None:
        checks.append(_missing("M5 cases: every condition labelled", path))
    else:
        ok = len(cases) == n["conditions"] and cases["case"].isin(list("ABCD")).all()
        checks.append(Check("M5 cases: every condition labelled", PASS if ok else FAIL,
                            f"{len(cases)} labelled (expected {n['conditions']}), counts {cases['case'].value_counts().to_dict()}"))
    tp = exp_dir / "case_thresholds_used.json"
    block = cfg.get("cases") or {}
    if not tp.exists():
        checks.append(_missing("M5 cases: thresholds used = config", tp))
    else:
        used = json.loads(tp.read_text(encoding="utf-8"))
        same = (abs(used["stat_threshold"] - float(block.get("stat_threshold", -1))) < 1e-12
                and abs(used["pred_threshold"] - float(block.get("pred_threshold", -1))) < 1e-12)
        checks.append(Check("M5 cases: thresholds used = config", PASS if same else FAIL,
                            f"used stat {used['stat_threshold']}, pred {used['pred_threshold']}; "
                            f"config stat {block.get('stat_threshold')}, pred {block.get('pred_threshold')}"))
    locked = bool(block.get("thresholds_locked", False))
    checks.append(Check("M5 cases: thresholds locked by the team (D3)", PASS if locked else WARN,
                        "locked" if locked else "cases.thresholds_locked is false: Case results are PROVISIONAL"))
    return checks


def threshold_commit_check(root: Path = PROJECT_ROOT) -> Check:
    """Best effort (git needed): config.yaml is committed and unmodified, and was last committed before the results were written."""
    name = "D3: config.yaml committed before the results were produced"
    try:
        status = subprocess.run(["git", "status", "--porcelain", "--", "config.yaml"], cwd=root,
                                capture_output=True, text=True, timeout=10, check=False)
        log = subprocess.run(["git", "log", "-1", "--format=%ct", "--", "config.yaml"], cwd=root,
                             capture_output=True, text=True, timeout=10, check=False)
    except Exception:  # noqa: BLE001
        return Check(name, WARN, "git not available; cannot check")
    if status.stdout.strip():
        return Check(name, FAIL, "config.yaml has uncommitted changes; commit it before producing results")
    if not log.stdout.strip():
        return Check(name, WARN, "config.yaml has no git history here")
    committed = int(log.stdout.strip())
    results = [p for p in (root / "results" / "experiment_results").glob("*") if p.name.startswith(("stat_recovery", "predictive", "merged"))]
    if not results:
        return Check(name, WARN, "no result files to compare with")
    first = min(int(p.stat().st_mtime) for p in results)
    if committed <= first:
        return Check(name, PASS, "last config commit is earlier than the earliest result file")
    return Check(name, WARN, "config.yaml was committed after some result files were written "
                             "(file times can also change when files are copied or checked out; verify by hand)")
