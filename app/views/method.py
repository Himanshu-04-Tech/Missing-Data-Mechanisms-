import streamlit as st

import data as D
import ui
from ui import pill

cfg = D.config()
clean = D.clean_data()
n_mech, n_lev, n_rep = len(cfg["mechanisms"]), len(cfg["levels"]), int(cfg["n_repeats"])
n_meth = len(cfg["imputation_methods"])
n_datasets = n_mech * n_lev * n_rep

ui.section(
    "Method",
    "How the study works",
    f"One clean dataset, {n_datasets} damaged copies, {n_datasets * n_meth} repaired copies, two ways of judging each repair.",
)

steps = [
    ("01", "Clean", f"King County house sales: {len(clean):,} houses, {clean.shape[1]} columns. Log price is the target. Fixed 80/20 train/test split."),
    ("02", "Remove", f"Delete values from 4 columns ({', '.join(cfg['missingness_columns'])}) by MCAR, MAR (driven by {cfg['mar_driver']}) or MNAR. "
                     f"{n_mech} mechanisms × {n_lev} levels × {n_rep} repeats = {n_datasets} datasets."),
    ("03", "Fill", f"Each damaged dataset is filled {n_meth} ways: mean, KNN (k = 5, scaled), iterative regression, iterative random forest. "
                   f"That makes {n_datasets * n_meth} imputed datasets."),
    ("04", "Measure twice", "Statistics track: refit the OLS regression and compare every coefficient with the complete-data one. "
                            "Prediction track: a fixed random forest, scored against the complete-data model."),
    ("05", "Classify", f"Case A-D from two thresholds: mean |z shift| ≤ {cfg['cases']['stat_threshold']} SE and RMSE gap ≤ "
                       f"{cfg['cases']['pred_threshold']:.0%}. Then a sensitivity check moves both thresholds."),
]
st.markdown('<div class="pipe">' + "".join(ui.card(n, t, b) for n, t, b in steps) + "</div>", unsafe_allow_html=True)

ui.section("No-cheating rules", "Three rules every step follows")
r1, r2, r3 = st.columns(3)
r1.markdown(ui.card("RULE 1", "Fit on train only", "Every imputer and every model learns from the training rows only, then fills or predicts the test rows."), unsafe_allow_html=True)
r2.markdown(ui.card("RULE 2", "The target never fills a gap", "Price and log price are never inputs to the prediction-track imputers, so the answer cannot leak into the questions."), unsafe_allow_html=True)
r3.markdown(ui.card("RULE 3", "Truth is for scoring only", "The real values that were deleted are only used at the end, to measure how far off the guesses were."), unsafe_allow_html=True)

ui.section("Key formulas", "How the two dials are computed")
f1, f2 = st.columns(2)
with f1.container(key="glass_f1"):
    st.markdown("**Statistical error** · how far a coefficient moved, in units of its own uncertainty")
    st.latex(r"z_j = \frac{\hat\beta_j^{\,\text{imputed}} - \hat\beta_j^{\,\text{complete}}}{SE_j^{\,\text{complete}}}")
    st.caption("Averaged over the 4 coefficients with missing values. |z| ≤ 1 means the shift is within normal sampling noise.")
with f2.container(key="glass_f2"):
    st.markdown("**Predictive error** · how much worse the model predicts than with complete data")
    st.latex(r"\text{gap} = \frac{RMSE_{\text{imputed}} - RMSE_{\text{complete}}}{RMSE_{\text{complete}}}")
    st.caption("A gap ≤ 5% counts as predicting well.")

st.markdown(
    f'<div class="card" style="margin-top:1rem">{ui.threshold_pill(D.cases()["thresholds_status"].iloc[0])}<h4>A known limitation, shown openly</h4><p>The Case thresholds '
    "(1 SE and 5%) were chosen by the team from the problem definition and locked before looking at the imputed results; they are still a judgement call. "
    "That is why the Results Explorer lets you drag them: the main pattern (many Case B, almost no Case C) holds over a wide range.</p></div>",
    unsafe_allow_html=True,
)

# ── about ─────────────────────────────────────────────────────────────────────
ui.section("About", "Team, course fit and data")
t1, t2, t3 = st.columns(3)
t1.markdown(ui.card("MEMBER 1", "Missingness", "Data cleaning, train/test split, and the MCAR / MAR / MNAR generator (<code>src/missingness.py</code>)."), unsafe_allow_html=True)
t2.markdown(ui.card("MEMBER 2", "Imputation", "The four imputers, the leakage rules and imputation scoring (<code>src/imputation.py</code>)."), unsafe_allow_html=True)
t3.markdown(ui.card("MEMBER 3", "Evaluation", "Statistical and predictive tracks, the Case A-D flags and the sensitivity check (<code>src/m3_*.py</code>)."), unsafe_allow_html=True)

with st.container(key="glass_syllabus"):
    st.markdown("**Where the course topics appear**")
    st.markdown(
        "| Topic | Where it appears in this project |\n|---|---|\n"
        "| Data cleaning, scaling | `data_processing.py`, scaled KNN imputer |\n"
        "| Functions, loops, modules | the `src/` package and the sweep over 300 conditions |\n"
        "| Pandas and NumPy | every table on this site |\n"
        "| Regression | OLS with statsmodels (statistics track) |\n"
        "| Visualisation | the charts on this site and in `results/figures/` |"
    )

ui.section("Downloads", "Take the numbers with you", "The exact tables this site is built from.")
files = {
    "Case labels per condition": D.TABLES / "cases_by_condition.csv",
    "Overall recovery (table 6)": D.TABLES / "table6_overall_recovery.csv",
    "Best methods (table 6b)": D.TABLES / "table6b_best_methods.csv",
    "Statistical recovery (table 4)": D.TABLES / "table4_statistical_recovery.csv",
    "Predictive recovery (table 5)": D.TABLES / "table5_predictive_recovery.csv",
    "Case sensitivity": D.TABLES / "case_sensitivity.csv",
}
cols = st.columns(3)
for i, (label, path) in enumerate(files.items()):
    cols[i % 3].download_button(f"↓ {label}", path.read_bytes(), file_name=path.name, mime="text/csv", width="stretch")

ui.footer()
