"""Tests for Member 3 / Module 7 (results and discussion generator). Hand-built tables with known answers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.m3_report as report  # noqa: E402
from src.m3_report import (  # noqa: E402
    INTERPRETATION,
    LABELS,
    OBSERVED,
    Inputs,
    build_findings,
    check_claims,
    comparison_findings,
    predictive_findings,
    render_markdown,
    save_report,
    statistical_findings,
)

METHOD_FACTOR = {"complete_case": 1.0, "mean": 2.0, "knn": 1.5}
MECH_FACTOR = {"MCAR": 1.0, "MAR": 2.0}


def make_stat(methods=("complete_case", "mean", "knn")):
    rows = []
    for m in methods:
        for mech in MECH_FACTOR:
            for lv in (0.1, 0.5):
                for rep, eps in ((0, -0.1), (1, 0.1)):
                    z = METHOD_FACTOR[m] * MECH_FACTOR[mech] * lv * 10
                    rows.append(dict(method=m, mechanism=mech, level=lv, repeat=rep, n_obs=1000 if lv == 0.1 else 500,
                                     stat_mean_abs_z_affected=z + eps, stat_mean_signed_z_affected=-z,
                                     stat_ci_coverage_affected=0.5, stat_n_sig_flips=1))
    return pd.DataFrame(rows)


REF = pd.DataFrame([
    {"model": "complete_data", "seed": 42, "is_primary": True, "rmse": 0.20, "mae": 0.15, "r2": 0.9},
    {"model": "complete_data", "seed": 43, "is_primary": False, "rmse": 0.202, "mae": 0.15, "r2": 0.9},
])
GAP = {("mean", 0.1): 0.10, ("mean", 0.5): 0.30, ("knn", 0.1): 0.05, ("knn", 0.5): 0.20}


def make_pred():
    rows = []
    for (m, lv), gap in GAP.items():
        for mech in MECH_FACTOR:
            for rep in (0, 1):
                rmse = 0.2 * (1 + gap)
                rows.append(dict(method=m, mechanism=mech, level=lv, repeat=rep, rmse=rmse, mae=0.1, r2=0.8,
                                 rmse_truth_test=rmse - 0.01, r2_truth_test=0.82))
    return pd.DataFrame(rows)


def make_table6(case_pattern=None):
    rows = []
    for m in ("mean", "knn"):
        for mech in MECH_FACTOR:
            for lv in (0.1, 0.5):
                z = METHOD_FACTOR[m] * MECH_FACTOR[mech] * lv * 10
                rows.append(dict(method=m, mechanism=mech, level=lv, n_repeats=2,
                                 stat_mean_abs_z_affected_mean=z, rmse_gap_rel_mean=GAP[(m, lv)] * (1 + 0.01 * MECH_FACTOR[mech])))
    return pd.DataFrame(rows)


def make_cases(t6, status="PROVISIONAL", stat_thr=3.0, pred_thr=0.15):
    c = t6.copy()
    c["stat_good"] = c["stat_mean_abs_z_affected_mean"] <= stat_thr
    c["pred_good"] = c["rmse_gap_rel_mean"] <= pred_thr
    c["case"] = np.select([c.stat_good & c.pred_good, ~c.stat_good & c.pred_good, c.stat_good & ~c.pred_good], list("ABC"), "D")
    c["stat_threshold"], c["pred_threshold"], c["thresholds_status"] = stat_thr, pred_thr, status
    return c


def texts(findings):
    return " ".join(f.text for f in findings)


# --- statistical -------------------------------------------------------------
def test_statistical_findings_match_hand_computed_numbers():
    f = statistical_findings(make_stat())
    t = texts(f)
    assert "MCAR 1.00 (10%) to 5.00 (50%); MAR 2.00 (10%) to 10.00 (50%)" in t           # complete-case ranges
    assert "larger under MAR than under MCAR at 2 of 2" in t
    assert "complete_case 3.00, knn 4.50, mean 6.00" in t                                # MCAR average over levels
    assert "6 of 6 imputer x mechanism" not in t and "4 of 4 imputer x mechanism" in t   # 2 imputers x 2 mechanisms
    assert "keeps on average 1000 rows at 10% missingness and 500 rows at 50%" in t
    assert "mean/MCAR -6.00" in t and "mean/MAR -12.00" in t                             # signed bias, mean of levels 10% and 50%
    assert "flips over all imputed runs, by method (flips / runs): mean 8/8, knn 8/8" in t


def test_every_finding_is_labelled_and_has_a_source():
    findings, _ = build_findings(Inputs(stat_runs=make_stat(), pred_runs=make_pred(), pred_reference=REF,
                                        table6=make_table6(), cases=make_cases(make_table6())))
    assert findings and all(f.label in LABELS and f.source and f.text for f in findings)
    assert len({f.id for f in findings}) == len(findings)


def test_reference_ols_finding_counts_significant_coefficients():
    ols = pd.DataFrame({"variable": ["const", "a", "b", "c"], "significant": [True, True, True, False]})
    f = statistical_findings(make_stat(), ols)
    assert "2 of 3 coefficients significant" in f[0].text


def test_only_complete_case_marks_imputer_section_pending():
    inp = Inputs(stat_runs=make_stat(methods=("complete_case",)))
    _, pending = build_findings(inp)
    assert "Statistical track (imputers)" in pending and "PENDING" in pending["Statistical track (imputers)"]
    assert "Predictive track" in pending and "Both tracks compared" in pending


# --- predictive --------------------------------------------------------------
def test_predictive_findings_match_hand_computed_numbers():
    t = texts(predictive_findings(make_pred(), REF))
    assert "RMSE 0.2000, MAE 0.1500 and R2 0.9000" in t
    assert "Across 2 forest seeds the RMSE has sd 0.00141" in t
    assert "mean 10.0% (10%) to 30.0% (50%); knn 5.0% (10%) to 20.0% (50%)" in t
    assert "Largest relative RMSE gap in any condition: 30.0%; smallest: 5.0%" in t
    # decomposition against the PRIMARY reference (0.20): mean rmse 0.22/0.26 -> complete test 0.21/0.25 -> +0.03
    assert "mean: imputed test inputs +0.0100, training on imputed data +0.0300" in t
    assert "knn: imputed test inputs +0.0100, training on imputed data +0.0150" in t


def test_predictive_noise_comparison_counts_cells():
    t = texts(predictive_findings(make_pred(), REF))
    assert "In 0 of 4 mechanism x level cells" in t        # zero repeat sd, so no margin is below noise


def test_single_seed_reference_says_spread_not_measured():
    t = texts(predictive_findings(make_pred(), REF.iloc[:1]))
    assert "seed-to-seed spread of the forest was not measured" in t


# --- comparison --------------------------------------------------------------
def test_comparison_correlation_and_cases():
    t6 = make_table6()
    f = comparison_findings(t6, make_cases(t6), None, None)
    t = texts(f)
    assert "rank correlation between statistical error (mean |z|) and predictive error (relative RMSE gap) is +" in t
    assert "Case A" in t and "thresholds PROVISIONAL" in t
    assert any("Case B" in x.text and "Conditions in" in x.text for x in f) or "No condition is in Case B" in t


def test_case_b_listing_and_none_case():
    t6 = make_table6()
    cases = make_cases(t6, stat_thr=0.5, pred_thr=1.0)       # everything statistically bad, predictively good -> all B
    t = texts(comparison_findings(t6, cases, None, None))
    assert "Conditions in Case B" in t and "mean/MCAR/10%" in t
    none_b = make_cases(t6, stat_thr=100.0, pred_thr=1.0)     # everything good -> all A
    assert "No condition is in Case B" in texts(comparison_findings(t6, none_b, None, None))


def test_sensitivity_and_best_findings():
    sens = pd.DataFrame([{"varied": "stat", "factor": 0.75, "n_changed": 2}, {"varied": "stat", "factor": 1.25, "n_changed": 1},
                         {"varied": "pred", "factor": 0.75, "n_changed": 0}, {"varied": "pred", "factor": 1.25, "n_changed": 3}])
    best = pd.DataFrame({"same_method": [True, False, True, True]})
    t = texts(comparison_findings(make_table6(), None, sens, best))
    assert "statistical threshold x0.75: 2 labels change" in t and "predictive threshold x1.25: 3 labels change" in t
    assert "in 3 of 4 mechanism x level cells" in t


# --- interpretations are conditional, hedged, and never findings ---------------
def interp(inp):
    return [f for f in build_findings(inp)[0] if f.label == INTERPRETATION]


def test_interpretations_depend_on_observations():
    t6 = make_table6()
    all_b = interp(Inputs(stat_runs=make_stat(), pred_runs=make_pred(), pred_reference=REF, table6=t6,
                          cases=make_cases(t6, stat_thr=0.5, pred_thr=1.0)))
    none_b = interp(Inputs(stat_runs=make_stat(), pred_runs=make_pred(), pred_reference=REF, table6=t6,
                           cases=make_cases(t6, stat_thr=100.0, pred_thr=1.0)))
    assert any("Case B conditions" in f.text for f in all_b) and not any("Case B conditions" in f.text for f in none_b)
    assert any("no condition is in Case B" in f.text for f in none_b)
    for f in all_b + none_b:
        assert "not tested" in f.text or "untested" in f.text or "does not show" in f.text or "do not show" in f.text


def test_member1_guess_is_noted_as_untested_not_a_finding():
    f = interp(Inputs(stat_runs=make_stat(methods=("complete_case",))))
    guess = [x for x in f if "Member 1" in x.text]
    assert len(guess) == 1 and "untested" in guess[0].text and "not a finding" in guess[0].text and guess[0].label == INTERPRETATION


def test_no_interpretation_when_there_is_nothing_to_interpret():
    assert interp(Inputs()) == []


# --- claims guard, rendering, saving ------------------------------------------
@pytest.mark.parametrize("bad", ["KNN is universally best", "MNAR is solved", "Nobody has studied this", "This generalizes to other datasets",
                                 "the best method overall", "This proves it"])
def test_check_claims_flags_overclaims(bad):
    assert check_claims(bad)


def test_generated_text_and_limitations_are_free_of_overclaims():
    for t in report.LIMITATIONS:
        assert not check_claims(t), t
    t6 = make_table6()
    text, _ = render_markdown(Inputs(stat_runs=make_stat(), pred_runs=make_pred(), pred_reference=REF, table6=t6, cases=make_cases(t6)))
    assert not check_claims(text)


def test_render_refuses_text_with_a_forbidden_claim(monkeypatch):
    monkeypatch.setattr(report, "LIMITATIONS", report.LIMITATIONS + ["KNN is universally best."])
    with pytest.raises(ValueError, match="agreed not to make"):
        render_markdown(Inputs(stat_runs=make_stat()))


def test_render_marks_provisional_thresholds_and_limits():
    t6 = make_table6()
    text, _ = render_markdown(Inputs(stat_runs=make_stat(), table6=t6, cases=make_cases(t6, status="PROVISIONAL")))
    assert "Case thresholds are PROVISIONAL" in text and "## 5. Limitations" in text
    locked, _ = render_markdown(Inputs(stat_runs=make_stat(), table6=t6, cases=make_cases(t6, status="LOCKED")))
    assert "Case thresholds are PROVISIONAL" not in locked


def test_missing_inputs_never_produce_numbers():
    text, findings = render_markdown(Inputs(stat_runs=make_stat(methods=("complete_case",))))
    assert text.count("PENDING.") == 3
    assert not [f for f in findings if f.section in ("Predictive track", "Both tracks compared")]


def test_render_is_deterministic_and_saves(tmp_path):
    inp = Inputs(stat_runs=make_stat(), pred_runs=make_pred(), pred_reference=REF)
    a, fa = render_markdown(inp)
    b, _ = render_markdown(inp)
    assert a == b
    written = save_report(a, fa, tmp_path / "r.md", tmp_path / "f.csv")
    saved = pd.read_csv(written["findings"])
    assert list(saved.columns) == ["id", "section", "label", "text", "source"] and len(saved) == len(fa)
    assert set(saved["label"]) <= set(LABELS)
