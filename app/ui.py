"""Shared look and feel for the Recovery Lab app.

Visual language: dark "night lab" theme, blue -> sky -> indigo gradient accents,
Space Grotesk headings, Inter body text, glass cards, pill buttons, soft glow blobs
and a faint grain texture. Colour inside charts is reserved for data:
MCAR / MAR / MNAR always use the same three colourblind-safe colours.
"""

import html

import plotly.graph_objects as go
import streamlit as st

# ── Data colours (Okabe-Ito family, readable on the dark background) ─────────
MECH_COLORS = {"MCAR": "#56B4E9", "MAR": "#E69F00", "MNAR": "#009E73"}
METHOD_LABELS = {
    "mean": "Mean",
    "knn": "KNN",
    "regression": "Regression",
    "random_forest": "Random forest",
    "complete_case": "Complete case",
}
METHOD_SYMBOLS = {"mean": "circle", "knn": "diamond", "regression": "square", "random_forest": "triangle-up"}
CASE_MEANINGS = {
    "A": "good on both",
    "B": "predicts well, statistically distorted",
    "C": "statistically fine, predicts poorly",
    "D": "poor on both",
}
GOOD = "#38bdf8"
BAD = "#f87171"
MUTED = "#8888a8"
GRID = "rgba(255,255,255,0.06)"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --bg: #08080f; --bg2: #10101c; --bg3: #16162a;
  --text: #f5f5f7; --body: rgba(245,245,247,0.75); --muted: #8888a8;
  --border: rgba(255,255,255,0.08); --card: rgba(22,22,42,0.72);
  --grad: linear-gradient(135deg, #2563eb 0%, #38bdf8 50%, #6366f1 100%);
  --grad2: linear-gradient(135deg, #2563eb, #38bdf8);
  --radius: 16px;
}

/* ── canvas: glow blobs + grain ── */
.stApp {
  background:
    radial-gradient(620px circle at -4% -6%, rgba(37,99,235,0.20), transparent 62%),
    radial-gradient(520px circle at 104% 12%, rgba(56,189,248,0.13), transparent 62%),
    radial-gradient(480px circle at 40% 108%, rgba(168,85,247,0.12), transparent 62%),
    var(--bg);
  background-attachment: fixed;
  color: var(--text);
  font-family: 'Inter', sans-serif;
}
.stApp::before {
  content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 0; opacity: 0.45;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E");
}
.block-container { max-width: 1180px; padding-top: 4.5rem; padding-bottom: 4rem; }

/* ── top bar ── */
header[data-testid="stHeader"] {
  background: rgba(8,8,15,0.78); backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--border);
}
header[data-testid="stHeader"] a { font-family: 'Inter', sans-serif; font-weight: 500; }

/* ── type ── */
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.02em; }
p, li, label, .stMarkdown { font-family: 'Inter', sans-serif; }
.mono, code { font-family: 'JetBrains Mono', monospace !important; font-variant-numeric: tabular-nums; }

/* ── hero ── */
.hero { position: relative; text-align: center; padding: 4.5rem 1rem 3.5rem; }
.hero .blob { position: fixed; z-index: 0; border-radius: 50%; filter: blur(80px); opacity: .22; pointer-events: none; }
.hero .b1 { width: 420px; height: 420px; background: radial-gradient(circle,#2563eb,transparent 70%); top:-160px; left:-120px; animation: float1 8s ease-in-out infinite; }
.hero .b2 { width: 360px; height: 360px; background: radial-gradient(circle,#38bdf8,transparent 70%); top: 60px; right:-120px; animation: float2 10s ease-in-out infinite; }
.hero .b3 { width: 300px; height: 300px; background: radial-gradient(circle,#a855f7,transparent 70%); top: 55vh; left: 35%; animation: float1 12s ease-in-out infinite reverse; }
@keyframes float1 { 0%,100% { transform: translate(0,0) scale(1);} 50% { transform: translate(30px,-30px) scale(1.05);} }
@keyframes float2 { 0%,100% { transform: translate(0,0) scale(1);} 50% { transform: translate(-25px,25px) scale(1.06);} }
.hero > *:not(.blob) { position: relative; z-index: 1; }
.hero-badge {
  display: inline-flex; align-items: center; gap: .5rem; padding: .4rem 1rem; border-radius: 50px;
  border: 1px solid rgba(56,189,248,.28); background: rgba(56,189,248,.07);
  font-size: .78rem; font-weight: 600; letter-spacing: .04em; color: #bae6fd; margin-bottom: 1.6rem;
}
.live-dot { width: 7px; height: 7px; border-radius: 50%; background: #4ade80; box-shadow: 0 0 10px #4ade80; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100% { opacity: 1;} 50% { opacity: .35;} }
.hero-title {
  font-family: 'Space Grotesk', sans-serif; font-size: clamp(2.6rem, 6.4vw, 5rem); font-weight: 700;
  line-height: 1.02; letter-spacing: -0.035em; margin: 0 0 1.1rem; color: var(--text);
}
.grad-text { background: var(--grad); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.hero-sub { font-size: clamp(1.05rem, 2.3vw, 1.3rem); color: var(--muted); margin-bottom: 1.2rem; letter-spacing: .01em; }
.hero-tagline { font-size: 1.05rem; color: var(--body); max-width: 620px; margin: 0 auto 2.3rem; line-height: 1.7; }
.btn-row { display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; }
.btn-primary, .btn-secondary {
  display: inline-block; padding: .85rem 2rem; border-radius: 50px; font-weight: 700; font-size: .95rem;
  text-decoration: none !important; transition: transform .2s, box-shadow .2s, background .2s, border-color .2s;
}
.btn-primary { background: var(--grad2); color: #08080f !important; }
.btn-primary:hover { transform: translateY(-3px); box-shadow: 0 12px 30px rgba(37,99,235,.4); }
.btn-secondary { border: 1px solid rgba(255,255,255,.14); color: var(--text) !important; font-weight: 600; }
.btn-secondary:hover { border-color: rgba(255,255,255,.3); background: rgba(255,255,255,.05); transform: translateY(-2px); }

/* ── section headers ── */
.section-label {
  display: block; font-size: .72rem; font-weight: 700; letter-spacing: .18em; text-transform: uppercase;
  background: var(--grad2); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
  margin: 2.8rem 0 .5rem;
}
.section-title { font-family: 'Space Grotesk', sans-serif; font-size: clamp(1.7rem, 3.4vw, 2.5rem); font-weight: 700; line-height: 1.15; margin: 0 0 .6rem; color: var(--text); }
.section-sub { color: var(--body); max-width: 760px; line-height: 1.7; margin-bottom: 1.4rem; }

/* ── cards ── */
.card {
  background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 1.4rem 1.5rem; height: 100%; backdrop-filter: blur(12px);
  transition: transform .3s, box-shadow .3s, border-color .3s;
}
.card:hover { transform: translateY(-4px); box-shadow: 0 25px 50px rgba(0,0,0,.4); border-color: rgba(56,189,248,.3); }
.card .num { font-family: 'JetBrains Mono', monospace; font-size: .8rem; color: var(--muted); letter-spacing: .1em; }
.card h4 { margin: .5rem 0 .4rem; font-size: 1.15rem; color: var(--text); }
.card p { color: var(--body); font-size: .93rem; line-height: 1.6; margin: 0; }
.card .big { font-family: 'Space Grotesk', sans-serif; font-size: 2.6rem; font-weight: 700; line-height: 1; margin: .8rem 0 .5rem; }
/* st.container(key="glass_...") renders as a glass card */
[class*="st-key-glass"] {
  background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.1rem 1.2rem;
  backdrop-filter: blur(12px);
}

/* ── pills (honesty labels) ── */
.pill { display: inline-block; padding: .18rem .65rem; border-radius: 50px; font-size: .68rem; font-weight: 700;
        letter-spacing: .1em; text-transform: uppercase; vertical-align: middle; }
.pill-observed { background: #f5f5f7; color: #08080f; }
.pill-untested { border: 1px dashed rgba(245,245,247,.6); color: var(--text); }
.pill-provisional { color: var(--text); border: 1px solid rgba(245,245,247,.35);
  background: repeating-linear-gradient(135deg, rgba(255,255,255,.14) 0 4px, transparent 4px 8px); }
.pill-live { border: 1px solid rgba(74,222,128,.5); color: #4ade80; }
.pill-locked { border: 1px solid rgba(245,245,247,.6); color: var(--text); }
.pill-case { background: var(--grad2); color: #08080f; }
.pill-good { border: 1px solid rgba(56,189,248,.55); color: #7dd3fc; }
.pill-bad { border: 1px solid rgba(248,113,113,.55); color: #fca5a5; }

/* ── mechanism chips ── */
.chip { display: inline-flex; align-items: center; gap: .4rem; font-size: .82rem; font-weight: 600; margin-right: 1rem; color: var(--body); }
.chip i { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }

/* ── stat strip ── */
.statline { font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums; color: var(--body); font-size: .92rem; }
.statline b { color: var(--text); }
.kpi { font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 700; color: var(--text); line-height: 1.1; }
.kpi-label { font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
.kpi-delta { font-family: 'JetBrains Mono', monospace; font-size: .8rem; color: var(--muted); }

/* ── disagreement strip ── */
.dgrid { display: grid; grid-template-columns: 70px repeat(5, minmax(0,1fr)); gap: .5rem; align-items: stretch; }
.dgrid .hd { font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); font-weight: 700; text-align: center; padding-bottom: .2rem; }
.dgrid .rowlab { font-weight: 700; font-size: .85rem; display: flex; align-items: center; gap: .4rem; }
.dcell { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: .55rem .5rem; display: flex; flex-direction: column; gap: .3rem; }
.dcell.match { border: 1px solid transparent; background: linear-gradient(var(--bg3), var(--bg3)) padding-box, var(--grad) border-box; }
.dchip { font-size: .74rem; border-radius: 8px; padding: .22rem .45rem; display: flex; justify-content: space-between; gap: .3rem; }
.dchip span { color: var(--muted); font-family: 'JetBrains Mono', monospace; }
.dchip.s { background: rgba(255,255,255,.05); }
.dchip.p { background: rgba(56,189,248,.09); }

/* ── pipeline ── */
.pipe { display: grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap: .8rem; }
.pipe .card { position: relative; }
@media (max-width: 900px) { .pipe { grid-template-columns: 1fr; } .dgrid { grid-template-columns: 52px repeat(5, minmax(0,1fr)); } }

/* ── widgets ── */
.stButton > button, .stDownloadButton > button {
  border-radius: 50px !important; font-weight: 700 !important; padding: .55rem 1.6rem !important;
  transition: transform .2s, box-shadow .2s !important;
}
.stButton > button[kind="primary"] { background: var(--grad2) !important; color: #08080f !important; border: none !important; }
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-2px); box-shadow: 0 10px 26px rgba(37,99,235,.35); }
div[data-baseweb="select"] > div, .stNumberInput input { background: rgba(22,22,42,.85) !important; border-color: var(--border) !important; border-radius: 12px !important; }
[data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
hr { border-color: var(--border) !important; }

.footer { margin-top: 4rem; padding-top: 1.5rem; border-top: 1px solid var(--border); color: var(--muted); font-size: .85rem; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 1rem; }
.footer a { color: #7dd3fc !important; text-decoration: none; }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(text) -> str:
    return html.escape(str(text))


def section(label: str, title: str, sub: str = "") -> None:
    """Two-line section header: gradient eyebrow label, then a big title."""
    sub_html = f'<div class="section-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<span class="section-label">{esc(label)}</span>'
        f'<div class="section-title">{title}</div>{sub_html}',
        unsafe_allow_html=True,
    )


def pill(kind: str, text: str = "") -> str:
    """Honesty label. kind: observed | untested | provisional | live | case | good | bad."""
    default = {
        "observed": "Observed",
        "untested": "Interpretation · untested",
        "provisional": "Provisional",
        "live": "● Live",
    }
    return f'<span class="pill pill-{kind}">{esc(text or default.get(kind, kind))}</span>'


def threshold_pill(status: str) -> str:
    """Status of the Case thresholds, as written by src/m3_cases.py (LOCKED or PROVISIONAL)."""
    if str(status).upper() == "LOCKED":
        return pill("locked", "🔒 Thresholds locked")
    return pill("provisional", "Thresholds provisional")


def card(num: str, title: str, body: str, extra: str = "") -> str:
    return f'<div class="card"><div class="num">{esc(num)}</div>{extra}<h4>{title}</h4><p>{body}</p></div>'


def mech_legend() -> str:
    return "".join(f'<span class="chip"><i style="background:{c}"></i>{m}</span>' for m, c in MECH_COLORS.items())


def method_label(m: str) -> str:
    return METHOD_LABELS.get(m, m)


def style_fig(fig: go.Figure, height: int = 420, legend: bool = True) -> go.Figure:
    """Apply the dark, quiet chart style: transparent canvas, faint grid, Inter text."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#d6d6e0", size=13),
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=legend,
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(bgcolor="#16162a", bordercolor="#38bdf8", font=dict(family="JetBrains Mono, monospace", size=12)),
    )
    if fig.layout.title.text:
        fig.update_layout(margin=dict(t=78), title=dict(x=0, xanchor="left", y=0.985, yanchor="top"))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor="rgba(255,255,255,0.12)", linecolor="rgba(255,255,255,0.12)")
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="rgba(255,255,255,0.12)", linecolor="rgba(255,255,255,0.12)")
    return fig


def gauge(value: float, title: str, good: bool, threshold: float, suffix: str = "%") -> go.Figure:
    """Half-circle dial, 0-100, with a white marker at the target."""
    color = GOOD if good else BAD
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number=dict(suffix=suffix, font=dict(family="Space Grotesk, sans-serif", size=46, color="#f5f5f7"), valueformat=".1f"),
            title=dict(text=title, font=dict(family="Inter, sans-serif", size=13, color=MUTED)),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor=MUTED, tickfont=dict(color=MUTED, size=11), ticksuffix="%"),
                bar=dict(color=color, thickness=0.32),
                bgcolor="rgba(255,255,255,0.04)",
                borderwidth=0,
                steps=[dict(range=[0, 100], color="rgba(255,255,255,0.03)")],
                threshold=dict(line=dict(color="#f5f5f7", width=3), thickness=0.9, value=threshold),
            ),
        )
    )
    fig.update_layout(height=270, margin=dict(l=35, r=35, t=50, b=22), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f5f5f7"))
    return fig


def footer() -> None:
    st.markdown(
        '<div class="footer"><span>Recovery Lab · Missing-data mechanisms study · Python for Data Science</span>'
        '<span>Every number on this site is read from <code>results/</code>. Nothing is typed by hand. · '
        '<a href="https://github.com/Himanshu-04-Tech/Missing-Data-Mechanisms-" target="_blank">GitHub ↗</a></span></div>',
        unsafe_allow_html=True,
    )
