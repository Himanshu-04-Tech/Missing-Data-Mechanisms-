import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data as D
import ui
from ui import esc, pill

cases = D.cases().copy()
cfg = D.config()
BASE_STAT = float(cfg["cases"]["stat_threshold"])
BASE_PRED = float(cfg["cases"]["pred_threshold"])
METHODS = ["mean", "knn", "regression", "random_forest"]
MECHS = ["MCAR", "MAR", "MNAR"]
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5]

ui.section(
    "Results Explorer",
    "Every condition, on one map",
    "Each point is one condition (method × mechanism × level), averaged over 5 repeats. "
    "<b>Right</b> = statistics drift further from the truth. <b>Up</b> = predictions get worse. "
    "The dashed lines are the Case thresholds the team committed to. Drag the sliders to move them and see whether the story changes. "
    "Click a point to open its details.",
)

# ── filters + thresholds ──────────────────────────────────────────────────────
with st.container(key="glass_filters"):
    a, b = st.columns([1, 1.6])
    c, d = st.columns(2)
    mechs = a.pills("Mechanism", MECHS, selection_mode="multi", default=MECHS) or MECHS
    methods = b.pills("Method", METHODS, selection_mode="multi", default=METHODS, format_func=ui.method_label) or METHODS
    stat_thr = c.select_slider("Statistics threshold (mean |z|, SEs)", [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0], value=BASE_STAT)
    pred_thr = d.select_slider("Prediction threshold (RMSE gap)", [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.075, 0.1, 0.15],
                               value=BASE_PRED, format_func=lambda v: f"{v:.0%}".replace(".0%", "%") if v >= 0.01 else f"{v:.1%}")

cases["case_live"] = D.classify(cases["stat_value"], cases["pred_value"], stat_thr, pred_thr)
cases["changed"] = cases["case_live"] != cases["case"]
view = cases[cases.mechanism.isin(mechs) & cases.method.isin(methods)]

# ── case counts ───────────────────────────────────────────────────────────────
cnt = view["case_live"].value_counts()
base_cnt = view["case"].value_counts()
k = st.columns(5)
for col, case in zip(k[:4], "ABCD"):
    delta = int(cnt.get(case, 0)) - int(base_cnt.get(case, 0))
    col.markdown(
        f'<div class="kpi">{int(cnt.get(case, 0))}</div><div class="kpi-label">Case {case} · {esc(ui.CASE_MEANINGS[case])}</div>'
        f'<div class="kpi-delta">{"" if delta == 0 else f"{delta:+d} vs committed thresholds"}</div>',
        unsafe_allow_html=True,
    )
n_changed = int(view["changed"].sum())
k[4].markdown(
    f'<div class="kpi">{n_changed}</div><div class="kpi-label">labels changed</div>'
    f'<div class="kpi-delta">of {len(view)} shown<br>{ui.threshold_pill(cases["thresholds_status"].iloc[0])}</div>',
    unsafe_allow_html=True,
)

# ── central scatter (log10 x is computed here so the case zones line up exactly) ─
x_all = np.log10(cases["stat_value"])
x_lo, x_hi = float(x_all.min()) - 0.15, float(x_all.max()) + 0.15
y_lo, y_hi = min(-0.005, float(cases["pred_value"].min()) - 0.005), float(cases["pred_value"].max()) * 1.08
xt, yt = np.log10(stat_thr), pred_thr

fig = go.Figure()
zones = [  # (x0, x1, y0, y1, label, fill)
    (x_lo, xt, y_lo, yt, "A · good on both", "rgba(255,255,255,0.06)"),
    (xt, x_hi, y_lo, yt, "B · predicts well, stats distorted", "rgba(255,255,255,0.025)"),
    (x_lo, xt, yt, y_hi, "C · stats fine, predicts poorly", "rgba(255,255,255,0.025)"),
    (xt, x_hi, yt, y_hi, "D · poor on both", "rgba(255,255,255,0.0)"),
]
for x0, x1, y0, y1, label, fill in zones:
    fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=fill, line_width=0, layer="below")
    fig.add_annotation(x=x0 + 0.03, y=y1, text=label, showarrow=False, xanchor="left", yanchor="top",
                       font=dict(size=11, color=ui.MUTED))
fig.add_vline(x=xt, line=dict(color="rgba(245,245,247,.6)", dash="dash", width=1.5))
fig.add_hline(y=yt, line=dict(color="rgba(245,245,247,.6)", dash="dash", width=1.5))

for mech in mechs:
    for method in methods:
        sub = view[(view.mechanism == mech) & (view.method == method)]
        if sub.empty:
            continue
        fig.add_scatter(
            x=np.log10(sub["stat_value"]),
            y=sub["pred_value"],
            mode="markers",
            name=f"{mech} · {ui.method_label(method)}",
            legendgroup=mech,
            marker=dict(
                symbol=ui.METHOD_SYMBOLS[method],
                size=8 + 22 * sub["level"],
                color=ui.MECH_COLORS[mech],
                line=dict(width=[2.5 if c else 0.6 for c in sub["changed"]], color=["#f5f5f7" if c else "rgba(8,8,15,.8)" for c in sub["changed"]]),
                opacity=0.9,
            ),
            customdata=np.stack([sub["method"], sub["mechanism"], sub["level"], sub["stat_value"], sub["case_live"],
                                 sub["stat_ci_coverage_affected_mean"]], axis=-1),
            hovertemplate=(
                "<b>%{customdata[1]} · %{customdata[0]} · %{customdata[2]:.0%}</b><br>"
                "mean |z| %{customdata[3]:.2f} SE<br>RMSE gap %{y:+.2%}<br>CI coverage %{customdata[5]:.0%}<br>"
                "Case %{customdata[4]}<extra></extra>"
            ),
        )

ticks = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
fig.update_xaxes(title="Statistical error · mean |z shift| on affected coefficients (log scale) →", range=[x_lo, x_hi],
                 tickvals=np.log10(ticks), ticktext=[f"{t:g}" for t in ticks], showgrid=False)
fig.update_yaxes(title="Predictive error · RMSE gap vs complete data →", range=[y_lo, y_hi], tickformat=".0%")
ui.style_fig(fig, 560)
fig.update_layout(legend=dict(orientation="v", x=1.01, y=1, yanchor="top", font=dict(size=11)), margin=dict(r=170, t=20), clickmode="event+select")

with st.container(key="glass_scatter"):
    st.markdown(ui.mech_legend() + '<span class="chip">size = share missing · shape = method · white ring = label changed</span>', unsafe_allow_html=True)
    event = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points", key="scatter",
                            config={"displayModeBar": False})

if event and event.selection and event.selection.points:
    cd = event.selection.points[0]["customdata"]
    st.session_state["cond"] = (cd[0], cd[1], float(cd[2]))
cond = st.session_state.get("cond", ("random_forest", "MAR", 0.2))

# ── condition drawer ──────────────────────────────────────────────────────────
c_method, c_mech, c_level = cond
row = cases[(cases.method == c_method) & (cases.mechanism == c_mech) & (cases.level == c_level)].iloc[0]
ui.section("Condition detail", f'{esc(ui.method_label(c_method))} · <span style="color:{ui.MECH_COLORS[c_mech]}">{c_mech}</span> · {c_level:.0%} missing',
           f'{pill("case", "Case " + row["case_live"])} {esc(ui.CASE_MEANINGS[row["case_live"]])} · click any point above to switch.')

m = st.columns(4)
tiles = [
    (f'{row["stat_value"]:.2f}', "mean |z shift| (SEs)"),
    (f'{row["stat_ci_coverage_affected_mean"]:.0%}', "affected 95% CIs holding truth"),
    (f'{row["pred_value"]:+.1%}', "RMSE gap vs complete data"),
    (f'{row["r2_mean"]:.3f}', "R² on imputed test set"),
]
for col, (v, lab) in zip(m, tiles):
    col.markdown(f'<div class="kpi">{v}</div><div class="kpi-label">{lab}</div>', unsafe_allow_html=True)

coef = D.coefficients()
cc = coef[(coef.method == c_method) & (coef.mechanism == c_mech) & (coef.level == c_level) & (coef.term != "const")]
per = cc.groupby("term").agg(z=("z_diff", "mean"), sd=("z_diff", "std"), cover=("ci_covers_ref", "mean"), affected=("is_affected", "first")).sort_values("z")

e1, e2 = st.columns([1.5, 1])
with e1.container(key="glass_forest"):
    f2 = go.Figure()
    f2.add_vrect(x0=-1, x1=1, fillcolor="rgba(255,255,255,0.05)", line_width=0)
    f2.add_scatter(
        x=per["z"], y=per.index, mode="markers", error_x=dict(type="data", array=per["sd"], color="rgba(245,245,247,.5)", thickness=1.2),
        marker=dict(size=10, color=[ui.MECH_COLORS[c_mech] if a else "rgba(245,245,247,.45)" for a in per["affected"]]),
        hovertemplate="%{y}: %{x:+.2f} SE<extra></extra>", showlegend=False,
    )
    f2.update_layout(title=dict(text="Coefficient shift per variable, mean ± sd over 5 repeats (coloured = had missing values)", font=dict(size=13)))
    f2.update_xaxes(title="z shift (reference SEs) · grey band = within 1 SE")
    st.plotly_chart(ui.style_fig(f2, 470, legend=False), width="stretch", config={"displayModeBar": False})
with e2.container(key="glass_cover"):
    st.markdown('<div class="kpi-label">Does each interval still contain the true coefficient?</div>', unsafe_allow_html=True)
    tbl = per.reset_index().sort_values(["affected", "z"], ascending=[False, True])
    st.dataframe(
        pd.DataFrame({"variable": tbl["term"], "had missing": tbl["affected"].map({True: "yes", False: ""}),
                      "z shift": tbl["z"], "CI covers truth": tbl["cover"]})
        .style.format({"z shift": "{:+.2f}", "CI covers truth": "{:.0%}"}),
        hide_index=True, width="stretch", height=470,
    )

# ── disagreement strip ────────────────────────────────────────────────────────
best = D.best_methods()
n_same = int(best["same_method"].sum())
ui.section("Disagreement strip", f"Best for statistics vs best for prediction: they agree in {n_same} of {len(best)}",
           "For each mechanism × level, <b>S</b> is the method with the smallest statistical error and <b>P</b> the one with the smallest RMSE gap. "
           "A gradient border marks the settings where they match.")
cells = ['<div class="dgrid"><div></div>' + "".join(f'<div class="hd">{lv:.0%}</div>' for lv in LEVELS)]
for mech in MECHS:
    cells.append(f'<div class="rowlab"><i style="width:9px;height:9px;border-radius:50%;background:{ui.MECH_COLORS[mech]};display:inline-block"></i>{mech}</div>')
    for lv in LEVELS:
        r = best[(best.mechanism == mech) & (np.isclose(best.level, lv))].iloc[0]
        cells.append(
            f'<div class="dcell{" match" if r["same_method"] else ""}">'
            f'<div class="dchip s"><span>S</span>{esc(ui.method_label(r["best_statistical_method"]))}</div>'
            f'<div class="dchip p"><span>P</span>{esc(ui.method_label(r["best_predictive_method"]))}</div></div>'
        )
cells.append("</div>")
st.markdown("".join(cells), unsafe_allow_html=True)
st.markdown(f'<div style="margin-top:.8rem">{pill("observed")}</div>', unsafe_allow_html=True)

# ── all 60 at a glance ────────────────────────────────────────────────────────
ui.section("All 60 at a glance", "Case labels under the current thresholds",
           "Rows are methods, columns are mechanism × level. A ringed cell changed label compared with the committed thresholds.")
hdr = "".join(f'<div class="hd" style="color:{ui.MECH_COLORS[me]}">{me[:4]} {lv:.0%}</div>' for me in MECHS for lv in LEVELS)
grid = [f'<div style="display:grid;grid-template-columns:110px repeat(15,minmax(0,1fr));gap:4px;overflow-x:auto"><div></div>{hdr}']
shade = {"A": "rgba(255,255,255,.22)", "B": "rgba(255,255,255,.12)", "C": "rgba(255,255,255,.07)", "D": "rgba(255,255,255,.03)"}
for method in METHODS:
    grid.append(f'<div class="rowlab" style="font-size:.8rem">{esc(ui.method_label(method))}</div>')
    for me in MECHS:
        for lv in LEVELS:
            r = cases[(cases.method == method) & (cases.mechanism == me) & (np.isclose(cases.level, lv))].iloc[0]
            ring = "box-shadow:0 0 0 2px #f5f5f7 inset;" if r["changed"] else ""
            hatch = "background-image:repeating-linear-gradient(135deg,rgba(255,255,255,.10) 0 3px,transparent 3px 7px);" if r["case_live"] == "B" else ""
            grid.append(f'<div title="{esc(ui.CASE_MEANINGS[r["case_live"]])}" style="background-color:{shade[r["case_live"]]};{hatch}{ring}border-radius:6px;'
                        f'text-align:center;padding:.45rem 0;font-family:Space Grotesk;font-weight:700">{r["case_live"]}</div>')
grid.append("</div>")
st.markdown("".join(grid), unsafe_allow_html=True)

with st.expander("Robustness check from the paper (results/tables/case_sensitivity.csv)"):
    st.dataframe(D.case_sensitivity().drop(columns=["thresholds_status"]), hide_index=True, width="stretch")

ui.footer()
