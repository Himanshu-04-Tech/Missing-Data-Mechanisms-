import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data as D
import lab_engine as L
import ui
from ui import pill

COLS = list(D.config()["missingness_columns"])
COL_LABELS = {"sqft_living": "Living area (sq ft)", "grade": "Grade", "bathrooms": "Bathrooms", "yr_built": "Year built"}

st.markdown(
    f'<span class="section-label" style="margin-top:.5rem">Live Lab {pill("live")}</span>'
    '<div class="section-title">Break the data. Fix it. Check the damage.</div>'
    '<div class="section-sub">This page runs the project\'s own Python code, <code>generate_missingness</code>, <code>impute</code> '
    "and <code>fit_ols</code>, on a random sample of real houses. Pick how values go missing, watch them disappear, "
    "then fill the gaps and see what survives.</div>",
    unsafe_allow_html=True,
)

# ── STEP 1: controls ──────────────────────────────────────────────────────────
with st.container(key="glass_controls"):
    a, b, c, d = st.columns([1.4, 1.5, 1, 1])
    mech = a.segmented_control("1 · How values go missing", ["MCAR", "MAR", "MNAR"], default="MNAR", key="lab_mech") or "MNAR"
    level = b.select_slider("2 · Share removed", [0.1, 0.2, 0.3, 0.4, 0.5], value=0.3, format_func=lambda v: f"{v:.0%}", key="lab_level")
    n_rows = c.selectbox("Sample size", [1000, 2000, 4000], index=1, format_func=lambda n: f"{n:,} houses")
    seed = int(d.number_input("Sample seed", 1, 999, 7, help="Change it to draw a different sample of houses."))
    st.markdown(
        {
            "MCAR": "**MCAR** · every house has the same chance of losing a value, like a coffee spill on random rows.",
            "MAR": "**MAR** · houses further **east** (higher longitude) lose values more often. The cause is a column we can see.",
            "MNAR": "**MNAR** · **bigger values hide themselves**: large houses lose their size, high grades lose their grade. The cause is the missing value itself.",
        }[mech]
    )

truth, train_idx, test_idx = L.sample(n_rows, seed)
corrupted, mask = L.corrupt(n_rows, seed, mech, level)

# ── STEP 2: watch values dissolve ─────────────────────────────────────────────
ui.section("Step 1 · Remove", "Watch the values disappear")
col_focus = st.segmented_control("Column to watch", COLS, default="sqft_living", format_func=lambda c: COL_LABELS[c], key="lab_col") or "sqft_living"

m = mask[col_focus]
color = ui.MECH_COLORS[mech]
g1, g2 = st.columns(2)

with g1.container(key="glass_map"):
    fig = go.Figure()
    fig.add_scattergl(x=truth.loc[~m, "long"], y=truth.loc[~m, "lat"], mode="markers", name="kept",
                      marker=dict(size=4, color="rgba(245,245,247,0.22)"))
    fig.add_scattergl(x=truth.loc[m, "long"], y=truth.loc[m, "lat"], mode="markers", name="removed",
                      marker=dict(size=5, color=color, line=dict(width=0)))
    fig.update_layout(title=dict(text=f"Where {COL_LABELS[col_focus].lower()} went missing", font=dict(size=14)))
    fig.update_xaxes(title="longitude (west → east)")
    fig.update_yaxes(title="latitude", scaleanchor="x", scaleratio=1.5)
    st.plotly_chart(ui.style_fig(fig, 380), width="stretch", config={"displayModeBar": False})

with g2.container(key="glass_hist"):
    vals = truth[col_focus]
    bins = dict(start=float(vals.min()), end=float(vals.max()), size=float((vals.max() - vals.min()) / 40) or 1)
    fig = go.Figure()
    fig.add_histogram(x=vals[~m], name="kept", xbins=bins, marker_color="rgba(245,245,247,0.28)")
    fig.add_histogram(x=vals[m], name="removed", xbins=bins, marker_color=color)
    fig.update_layout(barmode="stack", title=dict(text=f"Which {COL_LABELS[col_focus].lower()} values went missing", font=dict(size=14)))
    fig.update_xaxes(title=COL_LABELS[col_focus])
    fig.update_yaxes(title="houses")
    st.plotly_chart(ui.style_fig(fig, 380), width="stretch", config={"displayModeBar": False})

k1, k2, k3, k4 = st.columns(4)
complete_rows = (~mask.any(axis=1)).mean()
k1.markdown(f'<div class="kpi">{int(mask.values.sum()):,}</div><div class="kpi-label">cells removed</div>', unsafe_allow_html=True)
k2.markdown(f'<div class="kpi">{complete_rows:.0%}</div><div class="kpi-label">houses still complete</div>', unsafe_allow_html=True)
k3.markdown(f'<div class="kpi">{vals[~m].mean():,.0f}</div><div class="kpi-label">mean of what is left</div>', unsafe_allow_html=True)
k4.markdown(f'<div class="kpi">{vals.mean():,.0f}</div><div class="kpi-label">true mean</div><div class="kpi-delta">'
            f'left-over data is {(vals[~m].mean() / vals.mean() - 1):+.1%} off</div>', unsafe_allow_html=True)

with st.expander("See the holes in the table (first 12 houses with a missing value)"):
    show = corrupted.loc[mask.any(axis=1), ["price"] + COLS + ["lat", "long"]].head(12)
    st.dataframe(
        show.style.format(precision=2, na_rep="— missing —").map(
            lambda v: "background-color: rgba(248,113,113,.18); color:#fca5a5" if pd.isna(v) else ""
        ),
        width="stretch",
    )

# ── STEP 3: fill ──────────────────────────────────────────────────────────────
ui.section("Step 2 · Fill", "Fill the gaps with a real imputer")
f1, f2 = st.columns([2, 1])
method = f1.segmented_control("3 · Imputation method", ["mean", "knn", "regression", "random_forest"], default="random_forest",
                              format_func=ui.method_label, key="lab_method") or "random_forest"
params = (n_rows, seed, mech, level, method)
f2.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
if f2.button("Fill the gaps →", type="primary", width="stretch"):
    st.session_state["lab_params"] = params

if st.session_state.get("lab_params") != params:
    st.info("Choose a method and press **Fill the gaps**. Random forest takes about 5 seconds; the others take about 1 second.")
    ui.footer()
    st.stop()

with st.spinner(f"Running src.imputation.impute(method='{method}') and refitting both models…"):
    res = L.fill(*params)
filled, scores, stat = res["filled"], res["scores"], res["stat"]

# guess vs truth
s1, s2 = st.columns([1.35, 1])
with s1.container(key="glass_gvt"):
    cells = mask.index[mask[col_focus]]
    t, g = truth.loc[cells, col_focus], filled.loc[cells, col_focus]
    lo, hi = float(min(t.min(), g.min())), float(max(t.max(), g.max()))
    fig = go.Figure()
    fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect guess", line=dict(color="rgba(245,245,247,.45)", dash="dot"))
    fig.add_scattergl(x=t, y=g, mode="markers", name=f"{ui.method_label(method)} guess",
                      marker=dict(size=6, color=color, opacity=0.75),
                      hovertemplate="true %{x:,.1f}<br>guess %{y:,.1f}<extra></extra>")
    fig.update_layout(title=dict(text=f"Guess vs true value · {COL_LABELS[col_focus].lower()}", font=dict(size=14)))
    fig.update_xaxes(title="true value (hidden from the imputer)")
    fig.update_yaxes(title="filled-in guess")
    st.plotly_chart(ui.style_fig(fig, 400), width="stretch", config={"displayModeBar": False})

with s2.container(key="glass_scores"):
    sc = scores[scores["split"] == "all"].set_index("column").loc[COLS]
    st.markdown('<div class="kpi-label">How good were the guesses?</div>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame(
            {
                "column": [COL_LABELS[c] for c in COLS],
                "NRMSE": sc["nrmse"].to_numpy(),
                "bias": sc["bias"].to_numpy(),
                "spread kept": sc["sd_ratio"].to_numpy(),
            }
        ).style.format({"NRMSE": "{:.2f}", "bias": "{:+,.2f}", "spread kept": "{:.0%}"}),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "NRMSE 1.0 = no better than guessing the mean. Bias below 0 = guesses too low. "
        "Spread kept below 100% = filled values are squeezed towards the middle."
    )

# ── STEP 4: the two dials, live ───────────────────────────────────────────────
ui.section("Step 3 · Check", "The two dials, live on your sample")
pred_kept = 100 * res["r2_imp"] / res["r2_ref"]
cov = 100 * res["coverage"]
pred_good = res["pred_value"] <= float(D.config()["cases"]["pred_threshold"])
stat_good = res["stat_value"] <= float(D.config()["cases"]["stat_threshold"])
d1, d2 = st.columns(2)
with d1.container(key="glass_ldial1"):
    st.plotly_chart(ui.gauge(pred_kept, "PREDICTION · R² kept (random forest)", pred_good, 100), width="stretch", config={"displayModeBar": False})
    st.markdown(f'<div class="statline" style="text-align:center">R² <b>{res["r2_imp"]:.3f}</b> vs <b>{res["r2_ref"]:.3f}</b> on true data · RMSE gap <b>{res["pred_value"]:+.1%}</b></div>', unsafe_allow_html=True)
with d2.container(key="glass_ldial2"):
    st.plotly_chart(ui.gauge(cov, "STATISTICS · affected 95% CIs holding the truth", stat_good, 95), width="stretch", config={"displayModeBar": False})
    st.markdown(f'<div class="statline" style="text-align:center"><b>{res["n_cover"]} of {res["n_aff"]}</b> intervals · mean |z shift| <b>{res["stat_value"]:.2f}</b> SEs</div>', unsafe_allow_html=True)

with st.container(key="glass_forest"):
    st_ = stat.sort_values("z_diff")
    fig = go.Figure()
    fig.add_vrect(x0=-1, x1=1, fillcolor="rgba(255,255,255,0.05)", line_width=0, annotation_text="within 1 SE", annotation_position="top left",
                  annotation_font=dict(color=ui.MUTED, size=11))
    fig.add_bar(y=st_["term"], x=st_["z_diff"], orientation="h",
                marker_color=[color if a else "rgba(245,245,247,.3)" for a in st_["is_affected"]],
                hovertemplate="%{y}: %{x:+.2f} SE<extra></extra>", showlegend=False)
    fig.update_layout(title=dict(text="How far each coefficient moved, in reference standard errors (coloured = had missing values)", font=dict(size=14)))
    fig.update_xaxes(title="z shift = (β imputed − β true) / SE true")
    st.plotly_chart(ui.style_fig(fig, 460, legend=False), width="stretch", config={"displayModeBar": False})

st.markdown(
    f'{pill("untested", "Demo, not the paper")} <span class="statline">This runs on {n_rows:,} houses with a {L.LAB_RF_TREES}-tree forest. '
    "Smaller samples give wider intervals, which are easier to cover, so the statistics dial looks kinder here than in the full study. "
    "The paper's numbers (21,436 houses × 5 repeats) are on the Home and Results Explorer pages.</span>",
    unsafe_allow_html=True,
)

ui.footer()
