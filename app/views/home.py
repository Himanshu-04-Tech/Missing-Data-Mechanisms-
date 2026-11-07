import streamlit as st

import data as D
import ui
from ui import esc, pill

cases = D.cases()
best = D.best_methods()
ref = D.predictive_reference()
cfg = D.config()
stat_thr = float(cfg["cases"]["stat_threshold"])
pred_thr = float(cfg["cases"]["pred_threshold"])

n_cond = len(cases)
counts = cases["case"].value_counts()
n_b, n_c = int(counts.get("B", 0)), int(counts.get("C", 0))
n_same = int(best["same_method"].sum())
cov_by_mech = cases.groupby("mechanism")["stat_ci_coverage_affected_mean"].mean()

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<div class="hero">
  <div class="blob b1"></div><div class="blob b2"></div><div class="blob b3"></div>
  <div class="hero-badge"><span class="live-dot"></span>Live demo · {n_cond} conditions · 4 imputation methods · 21,436 houses</div>
  <div class="hero-title">Good prediction <span style="color:#8888a8">≠</span><br><span class="grad-text">good statistics</span></div>
  <div class="hero-sub">A missing-data study on King County house prices</div>
  <div class="hero-tagline">We deleted data three ways, refilled it four ways, and then checked two things:
  does the model still <b>predict</b> well, and does it still tell the <b>truth</b> about each variable?
  Often the answer to the first is yes while the answer to the second is no.</div>
  <div class="btn-row">
    <a class="btn-primary" href="lab" target="_self">Try the Live Lab →</a>
    <a class="btn-secondary" href="explorer" target="_self">Explore the results</a>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── TWO DIALS ─────────────────────────────────────────────────────────────────
ui.section(
    "The two dials",
    "One condition, two very different verdicts",
    "Pick any of the 60 conditions. The left dial asks how much of the complete-data model's "
    "predictive accuracy survives. The right dial asks how many confidence intervals still contain the "
    "true coefficient. A healthy analysis would keep both near the white marker.",
)

methods = ["mean", "knn", "regression", "random_forest"]
c1, c2, c3 = st.columns([1.2, 1.3, 1.5])
method = c1.selectbox("Imputation method", methods, index=3, format_func=ui.method_label)
mech = c2.segmented_control("Mechanism", ["MCAR", "MAR", "MNAR"], default="MAR") or "MAR"
level = c3.select_slider("Share of values removed", [0.1, 0.2, 0.3, 0.4, 0.5], value=0.2, format_func=lambda v: f"{v:.0%}")

row = cases[(cases.method == method) & (cases.mechanism == mech) & (cases.level == level)].iloc[0]
r2_ref = float(ref["r2"])
r2_imp = float(row["r2_mean"])
pred_kept = 100 * r2_imp / r2_ref
coverage = 100 * float(row["stat_ci_coverage_affected_mean"])
pred_good, stat_good = bool(row["pred_good"]), bool(row["stat_good"])

d1, d2 = st.columns(2)
with d1.container(key="glass_dial_pred"):
    st.plotly_chart(ui.gauge(pred_kept, "PREDICTION · R² kept vs complete-data model", pred_good, 100), width="stretch", config={"displayModeBar": False})
    st.markdown(
        f'<div class="statline" style="text-align:center">R² <b>{r2_imp:.3f}</b> vs <b>{r2_ref:.3f}</b> · RMSE gap <b>{row["pred_value"]:+.1%}</b> '
        f'{pill("good", "within 5%") if pred_good else pill("bad", "more than 5% worse")}</div>',
        unsafe_allow_html=True,
    )
with d2.container(key="glass_dial_stat"):
    st.plotly_chart(ui.gauge(coverage, "STATISTICS · 95% intervals that still hold the truth", stat_good, 95), width="stretch", config={"displayModeBar": False})
    st.markdown(
        f'<div class="statline" style="text-align:center">mean |z shift| <b>{row["stat_value"]:.2f}</b> reference SEs · '
        f'{pill("good", "within 1 SE") if stat_good else pill("bad", "more than 1 SE")}</div>',
        unsafe_allow_html=True,
    )

case = row["case"]
st.markdown(
    f'<div style="text-align:center;margin:1.2rem 0 .3rem">{pill("observed")} &nbsp;'
    f'<span class="statline">{esc(ui.method_label(method))} · {mech} · {level:.0%} missing → '
    f'R² gap <b>{float(row["r2_gap_mean"]):.4f}</b>, yet <b>{coverage:.0f}%</b> of the 4 affected coefficients\' intervals hold the true value.</span>'
    f'&nbsp; {pill("case", f"Case {case}")} <span class="statline">{esc(ui.CASE_MEANINGS[case])}</span> {pill("provisional", "thresholds provisional")}</div>',
    unsafe_allow_html=True,
)

# ── THE PROBLEM ───────────────────────────────────────────────────────────────
ui.section("The problem", "Why should anyone care?")
p1, p2, p3 = st.columns(3)
p1.markdown(ui.card("01", "Values go missing", "Real data has holes. Sometimes at random (MCAR), sometimes because of another column such as location (MAR), sometimes because of the value itself: big houses hide their size (MNAR)."), unsafe_allow_html=True)
p2.markdown(ui.card("02", "Imputation fills them", "We fill the holes with a guess: the column mean, similar houses (KNN), a regression, or a random forest. The data looks complete again."), unsafe_allow_html=True)
p3.markdown(ui.card("03", "Does it still tell the truth?", "A filled dataset can still predict prices well. But are the regression coefficients, standard errors and confidence intervals still right? That is what we measured."), unsafe_allow_html=True)

# ── HEADLINE FINDINGS ─────────────────────────────────────────────────────────
ui.section("Headline findings", "Three things the data shows", "Every number below is computed live from <code>results/tables/</code>.")
f1, f2, f3 = st.columns(3)
f1.markdown(
    ui.card("FINDING 01", "Prediction fine, statistics distorted",
            f"{n_b} of {n_cond} conditions are Case B: within 5% of the complete-data RMSE but more than one standard error off on the affected coefficients. "
            f"{n_c} conditions show the opposite pattern (Case C).",
            extra=f'{pill("observed")} {pill("provisional")}<div class="big grad-text">{n_b}/{n_cond}</div>'),
    unsafe_allow_html=True,
)
f2.markdown(
    ui.card("FINDING 02", "The best predictor is rarely the most truthful",
            f"Across the {len(best)} mechanism × level settings, the method that predicts best is also the one that recovers the statistics best in only {n_same}.",
            extra=f'{pill("observed")}<div class="big grad-text">{n_same}/{len(best)}</div>'),
    unsafe_allow_html=True,
)
f3.markdown(
    ui.card("FINDING 03", "MNAR is the hardest",
            f"Averaged over methods and levels, only {cov_by_mech['MNAR']:.0%} of the affected 95% intervals contain the true coefficient under MNAR, "
            f"vs {cov_by_mech['MCAR']:.0%} under MCAR and {cov_by_mech['MAR']:.0%} under MAR.",
            extra=f'{pill("observed")}<div class="big grad-text">{cov_by_mech["MNAR"]:.0%}</div>'),
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="card" style="margin-top:1rem">{pill("untested")}<h4>A likely reason</h4><p>Imputed values sit closer to the centre than the real ones '
    "(their spread shrinks). A prediction model averages over many houses, so it barely notices. A regression coefficient depends on exactly that spread, "
    "so it shifts and its interval becomes too confident. We did not test this mechanism directly.</p></div>",
    unsafe_allow_html=True,
)

ui.footer()
