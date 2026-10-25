"""Module M7: results and discussion text, generated from the saved result tables.

NEW file; does not modify any existing module in ``src/``.

Why a generator and not hand-typed text
---------------------------------------
Rule: never invent results. Every number in the Results sections is computed here from a saved table, and
every finding carries its source table (traceability). Re-running after the tables change gives a new
report with no manual editing. A section whose tables do not exist yet says PENDING and names the command
that produces them; it never contains placeholder numbers.

Structure (SRS FR-7.1 to FR-7.4)
--------------------------------
1. Statistical track         OBSERVED RESULT only
2. Predictive track          OBSERVED RESULT only
3. Both tracks compared      OBSERVED RESULT only (correlation, Case counts, best methods, sensitivity)
4. Discussion                POSSIBLE INTERPRETATION only. Each is conditional on what was observed, hedged,
                             and says what would be needed to test it. None is stated as a finding.
5. Limitations               fixed text (FR-7.3), plus the untested guess from Member 1 (FR-7.4)

Every finding is labelled OBSERVED RESULT or POSSIBLE INTERPRETATION (FR-7.2). Results are about the conditions
tested only: one dataset, synthetic missingness, our own strength setting. ``check_claims`` scans the finished
text for over-claims the team agreed not to make (a best method in general, MNAR solved, novelty, other datasets).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.m3_common import EXPERIMENT_RESULTS_DIR, PROJECT_ROOT, TABLES_DIR

REPORT_DIR = PROJECT_ROOT / "results" / "report"
REPORT_MD_PATH = REPORT_DIR / "results_discussion.md"
FINDINGS_CSV_PATH = REPORT_DIR / "findings.csv"

OBSERVED = "OBSERVED RESULT"
INTERPRETATION = "POSSIBLE INTERPRETATION"
LABELS = (OBSERVED, INTERPRETATION)

COMPLETE_CASE = "complete_case"
MECH_ORDER = ["MCAR", "MAR", "MNAR"]
METHOD_ORDER = [COMPLETE_CASE, "mean", "knn", "regression", "random_forest"]

# Phrases the team agreed never to claim (SRS 1.4). The generator must not produce them.
FORBIDDEN_PATTERNS = [
    r"universally",
    r"always the best",
    r"best method (overall|in general)",
    r"(mnar|missing not at random) (is|has been) (solved|fixed)",
    r"solves? (the )?mnar",
    r"(no ?one|nobody) has (ever )?(studied|examined|looked)",
    r"first (study|to study)",
    r"generali[sz]es? to (all|other|any) (other )?(data|datasets)",
    r"\bproves?\b",
]


@dataclass
class Finding:
    id: str
    section: str
    label: str
    text: str
    source: str


@dataclass
class Inputs:
    """Saved tables the report is built from. Anything may be None (that part is then PENDING)."""

    stat_runs: pd.DataFrame | None = None
    reference_ols: pd.DataFrame | None = None
    pred_runs: pd.DataFrame | None = None
    pred_reference: pd.DataFrame | None = None
    merged: pd.DataFrame | None = None
    table6: pd.DataFrame | None = None
    cases: pd.DataFrame | None = None
    sensitivity: pd.DataFrame | None = None
    best: pd.DataFrame | None = None
    git_commit: str | None = None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _f(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}"


def _pct(level: float) -> str:
    return f"{int(round(float(level) * 100))}%"


def _levels(df: pd.DataFrame) -> list[float]:
    return sorted(df["level"].astype(float).round(4).unique())


def _mechs(df: pd.DataFrame) -> list[str]:
    return [m for m in MECH_ORDER if m in set(df["mechanism"])]


def _methods(df: pd.DataFrame) -> list[str]:
    present = list(dict.fromkeys(df["method"]))
    return [m for m in METHOD_ORDER if m in present] + sorted(m for m in present if m not in METHOD_ORDER)


def _cond_means(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Repeat-averaged metric by method x mechanism x level."""
    d = df.assign(level=df["level"].astype(float).round(4))
    return d.groupby(["method", "mechanism", "level"], sort=False)[metric].mean().reset_index()


def _rank_corr(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank()))


def _range_text(series_by_level: pd.Series) -> str:
    """'0.53 (10%) to 3.60 (50%)' for a series indexed by level."""
    s = series_by_level.sort_index()
    return f"{_f(s.iloc[0])} ({_pct(s.index[0])}) to {_f(s.iloc[-1])} ({_pct(s.index[-1])})"


# ---------------------------------------------------------------------------
# 1. Statistical track
# ---------------------------------------------------------------------------
def statistical_findings(stat_runs: pd.DataFrame, reference_ols: pd.DataFrame | None = None) -> list[Finding]:
    out: list[Finding] = []
    src_runs = "stat_recovery_runs.csv"
    n = 0

    def add(text: str, source: str) -> None:
        nonlocal n
        n += 1
        out.append(Finding(f"S{n}", "Statistical track", OBSERVED, text, source))

    if reference_ols is not None:
        body = reference_ols[reference_ols["variable"] != "const"]
        add(f"The complete-data reference regression (log_price on {len(body)} predictors) has "
            f"{int(body['significant'].sum())} of {len(body)} coefficients significant at the 5% level.",
            "reference_ols.csv")

    # complete-case baseline
    cc = stat_runs[stat_runs["method"] == COMPLETE_CASE]
    if not cc.empty:
        cm = _cond_means(cc, "stat_mean_abs_z_affected")
        parts = [f"{mech} {_range_text(cm[cm['mechanism'] == mech].set_index('level')['stat_mean_abs_z_affected'])}"
                 for mech in _mechs(cm)]
        add("Complete-case analysis, mean |z difference| on the four affected coefficients (reference-SE units, "
            "mean of the repeats), from the lowest to the highest missingness level: " + "; ".join(parts) + ".", src_runs)

        pivot = cm.pivot(index="level", columns="mechanism", values="stat_mean_abs_z_affected")
        for mech in ("MAR", "MNAR"):
            if mech in pivot and "MCAR" in pivot:
                k = int((pivot[mech] > pivot["MCAR"]).sum())
                add(f"Complete-case |z| is larger under {mech} than under MCAR at {k} of {len(pivot)} missingness levels.", src_runs)

        cov = _cond_means(cc, "stat_ci_coverage_affected")
        parts = [f"{mech} {_f(g['stat_ci_coverage_affected'].min())} to {_f(g['stat_ci_coverage_affected'].max())}"
                 for mech, g in cov.groupby("mechanism", sort=False) if mech in MECH_ORDER]
        add("Complete-case share of the 95% intervals (affected coefficients) that contain the reference coefficient, "
            "range over the levels: " + "; ".join(parts) + ".", src_runs)

        flips = _cond_means(cc, "stat_n_sig_flips")
        parts = [f"{mech} {_f(flips[flips['mechanism'] == mech]['stat_n_sig_flips'].min(), 1)} to "
                 f"{_f(flips[flips['mechanism'] == mech]['stat_n_sig_flips'].max(), 1)}" for mech in _mechs(flips)]
        add("Complete-case significance flips (coefficients whose significance at 5% differs from the reference, out of "
            "15), mean per run, range over the levels: " + "; ".join(parts) + ".", src_runs)

        if "n_obs" in cc:
            nobs = cc.assign(level=cc["level"].astype(float).round(4)).groupby("level")["n_obs"].mean()
            add(f"Complete-case analysis keeps on average {int(round(nobs.iloc[0]))} rows at {_pct(nobs.index[0])} missingness "
                f"and {int(round(nobs.iloc[-1]))} rows at {_pct(nobs.index[-1])} (averaged over mechanisms and repeats).", src_runs)

    # imputers
    imp = stat_runs[stat_runs["method"] != COMPLETE_CASE]
    if imp.empty:
        return out

    cm_all = _cond_means(stat_runs, "stat_mean_abs_z_affected")
    for mech in _mechs(cm_all):
        sub = cm_all[cm_all["mechanism"] == mech].groupby("method")["stat_mean_abs_z_affected"].mean()
        sub = sub.reindex([m for m in METHOD_ORDER if m in sub.index]).sort_values()
        order = ", ".join(f"{m} {_f(v)}" for m, v in sub.items())
        add(f"Under {mech}, mean |z| on the affected coefficients averaged over all levels, from lowest to highest: {order}.", src_runs)

    cm_imp = _cond_means(imp, "stat_mean_abs_z_affected")
    inc = tot = 0
    for _, g in cm_imp.groupby(["method", "mechanism"]):
        vals = g.sort_values("level")["stat_mean_abs_z_affected"].to_numpy()
        tot += 1
        inc += int(np.all(np.diff(vals) > 0))
    add(f"For {inc} of {tot} imputer x mechanism combinations, mean |z| increases at every step up in missingness level.", src_runs)

    bias = _cond_means(imp, "stat_mean_signed_z_affected").groupby(["method", "mechanism"])["stat_mean_signed_z_affected"].mean()
    parts = [f"{m}/{k} {bias[(m, k)]:+.2f}" for m in _methods(imp) for k in _mechs(imp) if (m, k) in bias.index]
    add("Signed z difference (bias), mean over the four affected coefficients, levels and repeats; negative means the imputed "
        "coefficient is below the reference. Averaging signs can hide opposite shifts in different coefficients. "
        + "; ".join(parts) + ".", src_runs)

    cov = imp.groupby("method")["stat_ci_coverage_affected"].mean().reindex([m for m in METHOD_ORDER if m in set(imp["method"])])
    add("Share of imputed 95% intervals (affected coefficients) that contain the reference coefficient, by method: "
        + ", ".join(f"{m} {_f(v)}" for m, v in cov.items()) + ".", src_runs)

    flips = imp.groupby("method")["stat_n_sig_flips"].agg(["sum", "size"])
    add("Significance flips over all imputed runs, by method (flips / runs): "
        + ", ".join(f"{m} {int(r['sum'])}/{int(r['size'])}" for m, r in flips.reindex(
            [m for m in METHOD_ORDER if m in flips.index]).iterrows()) + ".", src_runs)

    grouped = imp.assign(level=imp["level"].astype(float).round(4)).groupby(["method", "mechanism", "level"])["stat_mean_abs_z_affected"]
    sd = grouped.std(ddof=1)
    add(f"Repeat-to-repeat spread of mean |z| (sd over the repeats), median over conditions: {_f(float(sd.median()))}; "
        f"largest: {_f(float(sd.max()))}.", src_runs)
    return out


# ---------------------------------------------------------------------------
# 2. Predictive track
# ---------------------------------------------------------------------------
def predictive_findings(pred_runs: pd.DataFrame, ref: pd.DataFrame) -> list[Finding]:
    out: list[Finding] = []
    n = 0

    def add(text: str, source: str) -> None:
        nonlocal n
        n += 1
        out.append(Finding(f"P{n}", "Predictive track", OBSERVED, text, source))

    primary = ref[ref["is_primary"].astype(bool)].iloc[0] if "is_primary" in ref else ref.iloc[0]
    r_rmse = float(primary["rmse"])
    text = (f"The complete-data Random Forest (upper bound) scores RMSE {_f(r_rmse, 4)}, MAE {_f(float(primary['mae']), 4)} "
            f"and R2 {_f(float(primary['r2']), 4)} on the test rows (log-price units).")
    seed_sd = None
    if len(ref) > 1:
        seed_sd = float(ref["rmse"].std(ddof=1))
        text += f" Across {len(ref)} forest seeds the RMSE has sd {_f(seed_sd, 5)}."
    add(text, "predictive_reference.csv")

    d = pred_runs.assign(level=pred_runs["level"].astype(float).round(4), gap_rel=(pred_runs["rmse"] - r_rmse) / r_rmse)
    cond = d.groupby(["method", "mechanism", "level"], sort=False)["gap_rel"].mean().reset_index()
    for mech in _mechs(cond):
        parts = []
        for m in _methods(cond):
            g = cond[(cond["method"] == m) & (cond["mechanism"] == mech)].set_index("level")["gap_rel"]
            if not g.empty:
                parts.append(f"{m} {_f(100 * g.iloc[0], 1)}% ({_pct(g.index[0])}) to {_f(100 * g.iloc[-1], 1)}% ({_pct(g.index[-1])})")
        add(f"Under {mech}, RMSE of the model trained on imputed data relative to the complete-data model (gap in % of the "
            "complete-data RMSE, mean of the repeats), lowest to highest level: " + "; ".join(parts) + ".", "predictive_results.csv")

    add(f"Largest relative RMSE gap in any condition: {_f(100 * cond['gap_rel'].max(), 1)}%; smallest: "
        f"{_f(100 * cond['gap_rel'].min(), 1)}%. Lowest test R2 in any single run: {_f(float(pred_runs['r2'].min()), 3)}.",
        "predictive_results.csv")

    # are method differences within repeat noise?
    stats = d.groupby(["mechanism", "level", "method"])["rmse"].agg(["mean", "std"]).reset_index()
    within = total = 0
    for _, g in stats.groupby(["mechanism", "level"]):
        if len(g) < 2:
            continue
        g = g.sort_values("mean")
        margin = float(g["mean"].iloc[1] - g["mean"].iloc[0])
        noise = float(np.nanmax(g["std"].iloc[:2].to_numpy()))
        total += 1
        within += int(margin < noise)
    if total:
        add(f"In {within} of {total} mechanism x level cells, the RMSE difference between the best and second-best method is "
            "smaller than the repeat-to-repeat sd of the two.", "predictive_results.csv")
    if seed_sd is None:
        add("The seed-to-seed spread of the forest was not measured (run scripts/m3_03_predictive.py --reference-seeds 5).",
            "predictive_reference.csv")

    if {"rmse_truth_test", "r2_truth_test"} <= set(pred_runs.columns):
        parts = []
        for m in _methods(pred_runs):
            g = pred_runs[pred_runs["method"] == m]
            inputs = float((g["rmse"] - g["rmse_truth_test"]).mean())
            training = float((g["rmse_truth_test"] - r_rmse).mean())
            parts.append(f"{m}: imputed test inputs {inputs:+.4f}, training on imputed data {training:+.4f}")
        add("Diagnostic (scored on the complete test rows, truth used only to score). Mean change in RMSE attributable to "
            "(a) imputed test inputs = RMSE on imputed test rows minus RMSE on complete test rows, and (b) training on imputed "
            "data = RMSE on complete test rows minus the complete-data model's RMSE. " + "; ".join(parts) + ".",
            "predictive_results.csv")
    return out


# ---------------------------------------------------------------------------
# 3. Comparison
# ---------------------------------------------------------------------------
def comparison_findings(
    table6: pd.DataFrame,
    cases: pd.DataFrame | None,
    sensitivity: pd.DataFrame | None,
    best: pd.DataFrame | None,
) -> list[Finding]:
    out: list[Finding] = []
    n = 0

    def add(text: str, source: str) -> None:
        nonlocal n
        n += 1
        out.append(Finding(f"C{n}", "Both tracks compared", OBSERVED, text, source))

    s_col, p_col = "stat_mean_abs_z_affected_mean", "rmse_gap_rel_mean"
    if {s_col, p_col} <= set(table6.columns) and len(table6) >= 3:
        rho = _rank_corr(table6[s_col], table6[p_col])
        add(f"Across the {len(table6)} method x mechanism x level conditions, the rank correlation between statistical error "
            f"(mean |z|) and predictive error (relative RMSE gap) is {rho:+.2f}.", "table6_overall_recovery.csv")
        for mech, g in table6.groupby("mechanism", sort=False):
            if mech in MECH_ORDER and len(g) >= 3:
                add(f"Within {mech} ({len(g)} conditions) the same rank correlation is {_rank_corr(g[s_col], g[p_col]):+.2f}.",
                    "table6_overall_recovery.csv")

    if cases is not None and not cases.empty:
        status = str(cases["thresholds_status"].iloc[0]) if "thresholds_status" in cases else "UNKNOWN"
        vc = cases["case"].value_counts()
        counts = ", ".join(f"Case {c} {int(vc.get(c, 0))}" for c in "ABCD")
        thr = (f"statistical |z| <= {cases['stat_threshold'].iloc[0]:g}, predictive relative RMSE gap <= "
               f"{cases['pred_threshold'].iloc[0]:g}") if "stat_threshold" in cases else "thresholds not recorded"
        add(f"Case labels for {len(cases)} conditions ({thr}; thresholds {status}): {counts}.", "cases_by_condition.csv")
        b = cases[cases["case"] == "B"]
        if len(b):
            names = [f"{r.method}/{r.mechanism}/{_pct(r.level)}" for r in b.itertuples()]
            shown = ", ".join(names[:12]) + (f" and {len(names) - 12} more" if len(names) > 12 else "")
            add(f"Conditions in Case B (predicts well, statistically distorted): {shown}.", "cases_by_condition.csv")
        else:
            add("No condition is in Case B.", "cases_by_condition.csv")

    if best is not None and not best.empty:
        k = int(best["same_method"].sum())
        add(f"The same method has the lowest statistical error and the lowest predictive error in {k} of {len(best)} "
            "mechanism x level cells.", "table6b_best_methods.csv")

    if sensitivity is not None and not sensitivity.empty:
        parts = []
        for which, name in (("stat", "statistical"), ("pred", "predictive")):
            for f in (0.75, 1.25):
                row = sensitivity[(sensitivity["varied"] == which) & np.isclose(sensitivity["factor"], f)]
                if not row.empty:
                    parts.append(f"{name} threshold x{f:g}: {int(row['n_changed'].iloc[0])} labels change")
        if parts:
            add("Case sensitivity (one threshold moved at a time): " + "; ".join(parts) + ".", "case_sensitivity.csv")
    return out


# ---------------------------------------------------------------------------
# 4. Discussion: possible interpretations, conditional on what was observed
# ---------------------------------------------------------------------------
def interpretation_findings(inp: Inputs) -> list[Finding]:
    out: list[Finding] = []
    n = 0

    def add(text: str, source: str) -> None:
        nonlocal n
        n += 1
        out.append(Finding(f"D{n}", "Discussion", INTERPRETATION, text, source))

    cases, t6 = inp.cases, inp.table6
    if cases is not None and not cases.empty:
        n_b = int((cases["case"] == "B").sum())
        if n_b:
            add("Case B conditions (see the Case B finding) may arise because a Random Forest predicts from many columns at "
                "once, and unaffected columns such as lat, long and sqft_living15 carry much of the price signal, whereas each "
                "OLS coefficient depends on how the one affected column was filled. This was not tested here; it could be "
                "examined by checking how prediction changes when only one affected column is imputed at a time.",
                "cases_by_condition.csv")
        else:
            add("Because no condition is in Case B, these results do not show good prediction coexisting with poor statistical "
                "recovery for the conditions and thresholds used. That does not show the two cannot diverge elsewhere.",
                "cases_by_condition.csv")

    if inp.pred_runs is not None and inp.pred_reference is not None and inp.stat_runs is not None:
        add("If predictive differences between imputers are small relative to repeat noise, one possible reason is a ceiling "
            "effect: the always-observed columns (lat, long) and correlated columns (sqft_living15 with sqft_living) may let "
            "the forest recover most of the signal whatever is imputed. This was not tested; dropping those columns in a "
            "side experiment would test it.", "predictive_results.csv")

    cc = None if inp.stat_runs is None else inp.stat_runs[inp.stat_runs["method"] == COMPLETE_CASE]
    if cc is not None and not cc.empty:
        cm = _cond_means(cc, "stat_mean_abs_z_affected").pivot(index="level", columns="mechanism", values="stat_mean_abs_z_affected")
        flips = _cond_means(cc, "stat_n_sig_flips")["stat_n_sig_flips"].max()
        if flips >= 1.0:
            add("Significance flips were expected to be rare because n stays near 21,000, but for complete-case analysis they "
                "occur. A possible reason is that deleting rows lowers n and widens the intervals, so borderline coefficients "
                "change status; the number of rows kept is reported in the statistical findings. This was not tested.",
                "stat_recovery_runs.csv")
        if "MAR" in cm and "MCAR" in cm and (cm["MAR"] > cm["MCAR"]).sum() > len(cm) / 2:
            add("Complete-case error is larger under MAR than MCAR at most levels. Member 1 suggested this may partly reflect an "
                "imperfect linear model (the dropped rows depend on long, which the linear model captures only partly). That "
                "guess is untested and is not a finding of this study.", "stat_recovery_runs.csv")
    return out


LIMITATIONS = [
    "One dataset (King County house sales, 21,436 rows). Nothing here is claimed for other datasets.",
    "Missingness is synthetic. MAR depends on long and MNAR on the value itself, with a strength of 1.0 that is our own documented choice; other strengths or drivers may behave differently.",
    "Location is only partly captured by the predictors (lat and long enter as plain numbers), so the OLS reference is itself an imperfect model.",
    "Conclusions are about the conditions tested: 3 mechanisms, 5 levels (10% to 50%), 4 imputers, 5 repeats each. No method is claimed to be best in general, and nothing here says MNAR is handled.",
    "Missingness is injected into the test rows as well, so test error includes imputed test inputs. The diagnostic on complete test rows separates the two sources.",
    "|z difference| is measured in reference-standard-error units and mixes bias and noise; the signed average is the bias, and averaging signs over coefficients can hide opposite shifts.",
    "Five repeats give a rough estimate of variability only; small differences between methods should not be over-read.",
    "The Random Forest is fixed and untuned for every condition; tuning per condition could change the predictive results.",
]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def _pending(section: str, needs: str, command: str) -> str:
    return f"> **PENDING.** {section} cannot be written yet: {needs}. Run `{command}`, then regenerate this report."


def build_findings(inp: Inputs) -> tuple[list[Finding], dict[str, str]]:
    """All findings plus a {section: pending message} for sections that cannot be written yet."""
    findings: list[Finding] = []
    pending: dict[str, str] = {}

    if inp.stat_runs is not None and not inp.stat_runs.empty:
        findings += statistical_findings(inp.stat_runs, inp.reference_ols)
        if (inp.stat_runs["method"] == COMPLETE_CASE).all():
            pending["Statistical track (imputers)"] = _pending(
                "The comparison of the four imputers", "only the complete-case baseline has been computed so far",
                "python scripts/m3_01_validate_imputed.py && python scripts/m3_02_stat_track.py")
    else:
        pending["Statistical track"] = _pending("The statistical track", "stat_recovery_runs.csv is missing",
                                                "python scripts/m3_02_stat_track.py")

    if inp.pred_runs is not None and inp.pred_reference is not None and not inp.pred_runs.empty:
        findings += predictive_findings(inp.pred_runs, inp.pred_reference)
    else:
        pending["Predictive track"] = _pending("The predictive track", "predictive_results.csv or predictive_reference.csv is missing",
                                               "python scripts/m3_03_predictive.py")

    if inp.table6 is not None and not inp.table6.empty:
        findings += comparison_findings(inp.table6, inp.cases, inp.sensitivity, inp.best)
    else:
        pending["Both tracks compared"] = _pending("The comparison", "Table 6 is missing",
                                                   "python scripts/m3_04_merge.py && python scripts/m3_05_cases.py")

    findings += interpretation_findings(inp)
    return findings, pending


def check_claims(text: str) -> list[str]:
    """Phrases in ``text`` that the team agreed not to claim (SRS 1.4). Empty list = clean."""
    return [p for p in FORBIDDEN_PATTERNS if re.search(p, text, flags=re.IGNORECASE)]


SECTION_ORDER = ["Statistical track", "Predictive track", "Both tracks compared", "Discussion"]


def render_markdown(inp: Inputs) -> tuple[str, list[Finding]]:
    findings, pending = build_findings(inp)
    provisional = inp.cases is not None and "thresholds_status" in inp.cases and (inp.cases["thresholds_status"] != "LOCKED").any()

    lines = [
        "# Results and discussion (Member 3)",
        "",
        "_Generated by `scripts/m3_07_report.py` from the saved result tables. Every number comes from the table named after it; "
        "nothing is typed by hand. Do not edit this file; change the tables or the generator and regenerate._",
        "",
        "**How to read this.** `OBSERVED RESULT` is something computed from our tables. `POSSIBLE INTERPRETATION` is an explanation "
        "we have not tested. Results describe the conditions tested only (one dataset, synthetic missingness, strength 1.0).",
        "",
    ]
    if provisional:
        lines += ["> **Case thresholds are PROVISIONAL** (not yet locked by the team, decision D3). Read the Case counts as a "
                  "demonstration of the method, not as results.", ""]
    section_no = 0
    for section in SECTION_ORDER:
        section_no += 1
        lines += [f"## {section_no}. {section}", ""]
        items = [f for f in findings if f.section == section]
        for f in items:
            lines.append(f"- **[{f.id}] {f.label}.** {f.text} _(source: {f.source})_")
        for key, msg in pending.items():
            if key.startswith(section):
                lines += ["", msg]
        if not items and not any(k.startswith(section) for k in pending):
            lines.append("_Nothing to report for this section yet._")
        lines.append("")

    section_no += 1
    lines += [f"## {section_no}. Limitations", ""]
    lines += [f"- {text}" for text in LIMITATIONS]
    lines.append("")

    text = "\n".join(lines)
    problems = check_claims(text)
    if problems:
        raise ValueError(f"the generated text contains claims the team agreed not to make: {problems}")
    return text, findings


# ---------------------------------------------------------------------------
# Loading and saving
# ---------------------------------------------------------------------------
def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def load_inputs(exp_dir: Path = EXPERIMENT_RESULTS_DIR, tab_dir: Path = TABLES_DIR) -> Inputs:
    from src.m3_reproduce import git_commit

    return Inputs(
        stat_runs=_read(exp_dir / "stat_recovery_runs.csv"),
        reference_ols=_read(tab_dir / "reference_ols.csv"),
        pred_runs=_read(exp_dir / "predictive_results.csv"),
        pred_reference=_read(exp_dir / "predictive_reference.csv"),
        merged=_read(exp_dir / "merged_results.csv"),
        table6=_read(tab_dir / "table6_overall_recovery.csv"),
        cases=_read(tab_dir / "cases_by_condition.csv"),
        sensitivity=_read(tab_dir / "case_sensitivity.csv"),
        best=_read(tab_dir / "table6b_best_methods.csv"),
        git_commit=git_commit(),
    )


def save_report(text: str, findings: list[Finding], md_path: Path = REPORT_MD_PATH, csv_path: Path = FINDINGS_CSV_PATH) -> dict[str, Path]:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(text, encoding="utf-8")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(f) for f in findings]).to_csv(csv_path, index=False)
    return {"report": md_path, "findings": csv_path}
