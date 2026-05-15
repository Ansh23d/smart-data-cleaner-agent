"""AI Data Cleaner — single-file Streamlit application with linear navigation."""
from __future__ import annotations

import sys
import io
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AI Data Cleaner",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from config import CLEANING_OPERATION_ORDER
from models.cleaning_plan import CleaningPlan
from models.kpi import KPI
from pipeline.ingestion import ingest, IngestionError
from pipeline.profiler import profile as run_profile
from pipeline.executor import execute, check_code, BlockedCodeError, _strip_redundant_imports
from pipeline.validator import validate
from pipeline.domain_detector import detect as detect_domain
from agents.cleaning_agent import CleaningAgent, CleaningAgentError
from agents.kpi_agent import KPIAgent, KPIAgentError
from services.llm_client import LLMClient
from utils.cost_tracker import CostTracker


# ═══════════════════════════════════════════════════════════════════════════
# CSS / THEME
# ═══════════════════════════════════════════════════════════════════════════

def _css() -> None:
    st.markdown(
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">',
        unsafe_allow_html=True,
    )
    st.markdown("""<style>
/* ── Hide Streamlit chrome ─────────────────────────────────────────────── */
#MainMenu, footer, header { display: none !important; }
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
.stDeployButton { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

/* ── CSS variables ──────────────────────────────────────────────────────── */
:root {
    --bg:      #0F1117;
    --surface: #161922;
    --raised:  #1E2130;
    --border:  #2A2D3D;
    --accent:  #6366F1;
    --alt:     #818CF8;
    --green:   #10B981;
    --yellow:  #F59E0B;
    --red:     #EF4444;
    --text:    #F1F5F9;
    --muted:   #8B92A5;
    --dim:     #515870;
    --r:       12px;
}

/* ── Base ───────────────────────────────────────────────────────────────── */
.stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], body { background: var(--bg) !important; }
[data-testid="stMain"] > div { padding: 2.5rem 4rem !important; }
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    color: var(--text);
}
h1 { color: #F8FAFC !important; font-weight: 700 !important; }
h2, h3 { color: #E2E8F0 !important; font-weight: 600 !important; }
p, li { color: var(--muted); line-height: 1.65; }

/* ── Primary buttons ────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.75rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 10px rgba(99,102,241,0.3) !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important;
}
.stButton > button:active { transform: translateY(0) !important; }
.stButton > button:disabled {
    background: #2A2D3D !important;
    color: var(--dim) !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
}

/* ── Metrics ────────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    padding: 1.25rem 1.5rem !important;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] *,
[data-testid="stMetricLabel"] div,
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] label {
    color: #CBD5E1 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] *,
[data-testid="stMetricValue"] div {
    color: #FFFFFF !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] svg { display: none; }
[data-testid="stMetricDelta"] { font-size: 0.82rem !important; }

/* ── File uploader ──────────────────────────────────────────────────────── */
[data-testid="stFileUploadDropzone"] {
    background: var(--surface) !important;
    border: 2px dashed var(--accent) !important;
    border-radius: var(--r) !important;
    padding: 3rem 2rem !important;
    text-align: center !important;
    transition: all 0.2s !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: var(--alt) !important;
    background: var(--raised) !important;
}
[data-testid="stFileUploaderFileName"] { color: var(--text) !important; }
[data-testid="stFileUploadDropzone"] small { color: var(--muted) !important; }

/* ── Text inputs ────────────────────────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}

/* ── Checkboxes ─────────────────────────────────────────────────────────── */
[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] span { color: var(--text) !important; }
[data-testid="stCheckbox"] [data-testid="stWidgetLabel"] { color: var(--text) !important; }

/* ── Dataframes ─────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    overflow: hidden !important;
}

/* ── Expanders ──────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
}
[data-testid="stExpander"] summary { color: var(--text) !important; }

/* ── Progress bar ───────────────────────────────────────────────────────── */
[data-testid="stProgressBar"] > div > div > div {
    background: linear-gradient(90deg, #6366F1, #8B5CF6) !important;
}
[data-testid="stProgressBar"] > div > div {
    background: var(--raised) !important;
    border-radius: 99px !important;
}

/* ── Alerts ─────────────────────────────────────────────────────────────── */
[data-testid="stAlertContainer"] { border-radius: var(--r) !important; }

/* ── Dividers ───────────────────────────────────────────────────────────── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Labels ─────────────────────────────────────────────────────────────── */
[data-testid="stWidgetLabel"] { color: var(--muted) !important; font-size: 0.85rem !important; }

/* ── Download button ────────────────────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.5rem !important;
    box-shadow: none !important;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: var(--accent) !important;
    transform: translateY(-1px) !important;
    background: var(--raised) !important;
}

/* ── Step indicator ─────────────────────────────────────────────────────── */
.steps {
    display: flex;
    align-items: flex-start;
    margin-bottom: 2.5rem;
    padding: 0.5rem 0;
    overflow-x: auto;
}
.step-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    flex: 1;
    min-width: 70px;
}
.step-circle {
    width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 700;
    border: 2px solid #4B5268;
    background: var(--bg);
    color: #8B92A5;
    z-index: 1;
    transition: all 0.25s;
    flex-shrink: 0;
}
.step-circle.active {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
    box-shadow: 0 0 0 5px rgba(99,102,241,0.18);
}
.step-circle.done {
    background: rgba(16,185,129,0.15);
    border-color: var(--green);
    color: var(--green);
}
.step-lbl { font-size: 0.7rem; color: #8B92A5; font-weight: 500; white-space: nowrap; font-family: 'Inter', sans-serif; }
.step-lbl.active { color: var(--alt); }
.step-lbl.done   { color: var(--green); }
.step-connector {
    height: 2px; flex: 1;
    background: #4B5268;
    margin-top: 17px; flex-shrink: 0;
    min-width: 16px;
}
.step-connector.done { background: var(--green); }

/* ── Hero ───────────────────────────────────────────────────────────────── */
.hero { text-align: center; padding: 3rem 1rem 2rem; }
.hero-icon { font-size: 3.5rem; margin-bottom: 1rem; display: block; }
.hero h1 {
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #818CF8 0%, #C084FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
    letter-spacing: -0.5px;
}
.hero p { font-size: 1.15rem; color: var(--muted); max-width: 560px; margin: 0 auto 2.5rem; }

/* ── Feature cards ──────────────────────────────────────────────────────── */
.feat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.25rem;
    max-width: 860px;
    margin: 0 auto 2.5rem;
}
.feat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1.5rem 1.25rem;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s;
}
.feat-card:hover { border-color: var(--accent); transform: translateY(-2px); }
.feat-icon { font-size: 1.8rem; margin-bottom: 0.65rem; display: block; }
.feat-icon-wrap {
    width: 60px; height: 60px;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 1rem;
    box-shadow: 0 0 20px rgba(99,102,241,0.12);
}
.feat-title { color: var(--text); font-weight: 600; font-size: 0.95rem; margin-bottom: 0.4rem; }
.feat-desc  { color: var(--muted); font-size: 0.82rem; line-height: 1.55; }

/* ── Page header ────────────────────────────────────────────────────────── */
.pg-header { margin-bottom: 2rem; }
.pg-header h1 { font-size: 1.75rem !important; margin-bottom: 0.25rem !important; }
.pg-header p  { color: var(--muted); font-size: 0.92rem; margin: 0; }

/* ── Generic card ───────────────────────────────────────────────────────── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* ── Badges ─────────────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.b-high { background: rgba(16,185,129,.12); color: #10B981; border: 1px solid rgba(16,185,129,.3); }
.b-med  { background: rgba(245,158,11,.12); color: #F59E0B; border: 1px solid rgba(245,158,11,.3); }
.b-low  { background: rgba(239,68,68,.12);  color: #EF4444; border: 1px solid rgba(239,68,68,.3); }
.b-type { background: rgba(99,102,241,.12); color: #818CF8; border: 1px solid rgba(99,102,241,.3); }
.b-info { background: rgba(99,102,241,.12); color: #818CF8; border: 1px solid rgba(99,102,241,.3); }

/* ── Op card ────────────────────────────────────────────────────────────── */
.op-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.15s;
}
.op-wrap:hover { border-color: #374151; }

/* ── Flag chips ─────────────────────────────────────────────────────────── */
.flag-chip {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(245,158,11,.1);
    border: 1px solid rgba(245,158,11,.3);
    border-radius: 20px;
    padding: 3px 11px;
    font-size: 0.78rem; color: #F59E0B;
    margin: 3px 2px;
}
.flag-chip.err {
    background: rgba(239,68,68,.1);
    border-color: rgba(239,68,68,.3);
    color: #EF4444;
}
.flag-chip.inf {
    background: rgba(99,102,241,.1);
    border-color: rgba(99,102,241,.3);
    color: #818CF8;
}

/* ── KPI card ───────────────────────────────────────────────────────────── */
.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1.4rem;
    height: 100%;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: rgba(99,102,241,.45); }
.kpi-name  { color: var(--text); font-weight: 600; font-size: 0.95rem; }
.kpi-val   { color: #818CF8; font-size: 2rem; font-weight: 700; margin: 0.6rem 0; line-height: 1; }
.kpi-cat   { color: var(--dim); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.5px; }
.kpi-form  { color: var(--muted); font-size: 0.8rem; margin: 0.4rem 0; }
.kpi-why   { color: var(--dim); font-size: 0.8rem; line-height: 1.45; }

/* ── Result row ─────────────────────────────────────────────────────────── */
.res-row { display:flex; gap:8px; align-items:center; padding:6px 0; border-bottom:1px solid #1a1d27; }
.res-row:last-child { border-bottom: none; }

/* ── Nav bar (step indicator top row) ──────────────────────────────────── */
.nav-brand {
    color: #818CF8;
    font-weight: 700;
    font-size: 1rem;
    margin: 0;
    padding: 0.5rem 0;
    letter-spacing: -0.01em;
}

/* Ghost home/back button — targets the button inside the column
   that contains the .ghost-marker sentinel span */
[data-testid="stColumn"]:has(.ghost-marker) .stButton > button {
    background: transparent !important;
    color: var(--alt) !important;
    -webkit-text-fill-color: var(--alt) !important;
    border: 1px solid rgba(129,140,248,0.45) !important;
    box-shadow: none !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 0.45rem 1rem !important;
    letter-spacing: 0;
}
[data-testid="stColumn"]:has(.ghost-marker) .stButton > button:hover {
    background: rgba(99,102,241,0.1) !important;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    border-color: var(--alt) !important;
    transform: none !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}

/* ── Responsive ─────────────────────────────────────────────────────────── */
@media (max-width: 860px) {
    [data-testid="stMain"] > div { padding: 1.5rem 1.25rem !important; }
    .feat-grid { grid-template-columns: 1fr; }
    .hero h1 { font-size: 2rem !important; }
}
</style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# STATE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def S(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)


def go_to(page: str) -> None:
    st.session_state.page = page
    st.rerun()


def get_tracker() -> CostTracker:
    if "tracker" not in st.session_state:
        st.session_state.tracker = CostTracker()
    return st.session_state.tracker


def get_llm() -> LLMClient:
    if "llm" not in st.session_state:
        st.session_state.llm = LLMClient(
            tracker=get_tracker(),
            status_callback=st.toast,
        )
    return st.session_state.llm


def compute_kpi_value(code: str, df: pd.DataFrame) -> Any:
    """Safely evaluate KPI code. Tries eval() then exec()."""
    try:
        check_code(code)
    except BlockedCodeError:
        return None
    clean = _strip_redundant_imports(code).strip()
    ns: dict[str, Any] = {
        "__builtins__": {
            "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
            "round": round, "float": float, "int": int, "str": str,
            "list": list, "dict": dict, "bool": bool, "print": print,
            "True": True, "False": False, "None": None,
            "range": range, "enumerate": enumerate, "zip": zip,
        },
        "pd": pd, "np": np, "df": df.copy(),
    }
    try:
        return eval(clean, ns)
    except SyntaxError:
        pass
    except Exception:
        return None
    try:
        exec(clean, ns)
        for var in ("result", "value", "output", "kpi_value"):
            if var in ns and ns[var] is not ns.get("df"):
                return ns[var]
    except Exception:
        pass
    return None


def fmt_value(v: Any, unit: str = "") -> str:
    """Format a KPI scalar for display, applying unit prefix/suffix."""
    if v is None:
        return "N/A"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        is_pct = unit == "%"
        is_money = unit == "$"
        # Percentages: just show the number with % suffix
        if is_pct:
            return f"{v:.1f}%"
        # Build the numeric part
        if abs(v) >= 1_000_000_000:
            num = f"{v/1_000_000_000:.2f}B"
        elif abs(v) >= 1_000_000:
            num = f"{v/1_000_000:.2f}M"
        elif abs(v) >= 10_000:
            num = f"{v:,.0f}"
        elif abs(v) >= 100:
            num = f"{v:,.1f}"
        else:
            num = f"{v:.3g}"
        if is_money:
            return f"${num}"
        if unit and unit not in ("$", "%"):
            return f"{num} {unit}"
        return num
    if isinstance(v, (list, np.ndarray)):
        return f"{len(v)} values"
    if isinstance(v, dict):
        return f"{len(v)} entries"
    try:
        import pandas as _pd
        if isinstance(v, _pd.Series):
            return f"{len(v)} values"
        if isinstance(v, _pd.DataFrame):
            return f"{v.shape[0]}×{v.shape[1]}"
    except Exception:
        pass
    return str(v)[:40]


# ═══════════════════════════════════════════════════════════════════════════
# STEP INDICATOR
# ═══════════════════════════════════════════════════════════════════════════

_STEPS = ["Upload", "Profile", "Clean", "Results", "KPIs", "Dashboard"]
_PAGE_IDX = {"upload": 0, "profile": 1, "cleaning": 2, "results": 3, "kpis": 4, "dashboard": 5}


def step_indicator(page: str) -> None:
    col_logo, col_nav = st.columns([8, 2])
    with col_logo:
        st.markdown('<p class="nav-brand">✦ AI Data Cleaner</p>', unsafe_allow_html=True)
    with col_nav:
        st.markdown('<span class="ghost-marker"></span>', unsafe_allow_html=True)
        if st.button("← Home", key=f"home_btn_{page}", use_container_width=True):
            go_to("home")
    st.markdown('<hr style="border-color:var(--border);margin:0.5rem 0 1.75rem;">', unsafe_allow_html=True)
    current = _PAGE_IDX.get(page, 0)
    parts: list[str] = []
    for i, label in enumerate(_STEPS):
        if i < current:
            cls, lbl_cls, icon = "done", "done", "✓"
        elif i == current:
            cls, lbl_cls, icon = "active", "active", str(i + 1)
        else:
            cls, lbl_cls, icon = "", "", str(i + 1)
        parts.append(
            f'<div class="step-wrap">'
            f'<div class="step-circle {cls}">{icon}</div>'
            f'<div class="step-lbl {lbl_cls}">{label}</div>'
            f'</div>'
        )
        if i < len(_STEPS) - 1:
            conn_cls = "done" if i < current else ""
            parts.append(f'<div class="step-connector {conn_cls}"></div>')
    st.markdown(f'<div class="steps">{"".join(parts)}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# CHART RENDERER
# ═══════════════════════════════════════════════════════════════════════════

_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#8B92A5", size=12),
    title_font=dict(color="#E2E8F0", size=14),
    xaxis=dict(gridcolor="#2A2D3D", linecolor="#2A2D3D", tickfont=dict(color="#8B92A5")),
    yaxis=dict(gridcolor="#2A2D3D", linecolor="#2A2D3D", tickfont=dict(color="#8B92A5")),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8B92A5")),
    margin=dict(t=40, b=30, l=10, r=10),
)

_PALETTE = ["#6366F1", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4", "#F97316"]


def _base_layout(height: int) -> dict:
    return {**_DARK_LAYOUT, "height": height, "margin": dict(t=50, b=65, l=55, r=20)}


def _resample_period(df: pd.DataFrame, date_col: str) -> str:
    """Pick monthly/weekly/daily based on date range."""
    try:
        span = (df[date_col].max() - df[date_col].min()).days
    except Exception:
        return "ME"
    if span > 180:
        return "ME"
    if span > 30:
        return "W"
    return "D"


# ── Smart chart builders (pure pandas, no LLM) ─────────────────────────────

def _chart_timeseries(df: pd.DataFrame, date_col: str, num_col: str,
                      agg: str = "sum", height: int = 400) -> None:
    period = _resample_period(df, date_col)
    period_label = {"ME": "Monthly", "W": "Weekly", "D": "Daily"}[period]
    label = num_col.replace("_", " ").title()
    agg_fn = {"sum": "sum", "mean": "mean"}[agg]
    plot_df = (
        df.set_index(date_col)[num_col]
        .resample(period)
        .agg(agg_fn)
        .reset_index()
        .dropna()
    )
    if plot_df.empty:
        return
    fig = px.line(
        plot_df, x=date_col, y=num_col,
        title=f"{period_label} {label} Over Time",
        color_discrete_sequence=_PALETTE, markers=True,
    )
    fig.update_traces(line_width=2.5, marker_size=6)
    fig.update_layout(**_base_layout(height))
    st.plotly_chart(fig, use_container_width=True)


def _chart_bar_category(df: pd.DataFrame, cat_col: str, num_col: str,
                         agg: str = "mean", top_n: int = 12, height: int = 380) -> None:
    label_num = num_col.replace("_", " ").title()
    label_cat = cat_col.replace("_", " ").title()
    agg_fn = {"sum": "sum", "mean": "mean"}[agg]
    plot_df = (
        df.groupby(cat_col)[num_col]
        .agg(agg_fn)
        .reset_index()
        .sort_values(num_col, ascending=False)
        .head(top_n)
    )
    title = f"Top {top_n} {label_cat} by {agg.title()} {label_num}"
    fig = px.bar(
        plot_df, x=cat_col, y=num_col, title=title,
        text=num_col, color_discrete_sequence=_PALETTE,
    )
    fig.update_traces(texttemplate="%{text:.3s}", textposition="outside",
                      marker_line_width=0)
    fig.update_layout(**_base_layout(height),
                      uniformtext_minsize=9, uniformtext_mode="hide")
    fig.update_xaxes(tickangle=-35, title=label_cat)
    fig.update_yaxes(title=f"{agg.title()} {label_num}")
    st.plotly_chart(fig, use_container_width=True)



def _chart_histogram(df: pd.DataFrame, num_col: str, height: int = 360) -> None:
    label = num_col.replace("_", " ").title()
    fig = px.histogram(
        df, x=num_col, title=f"Distribution of {label}",
        color_discrete_sequence=_PALETTE, nbins=30,
    )
    fig.update_traces(marker_line_width=0.4, marker_line_color="#0F1117")
    fig.update_layout(**_base_layout(height))
    fig.update_xaxes(title=label)
    fig.update_yaxes(title="Count")
    st.plotly_chart(fig, use_container_width=True)


def _chart_pie(df: pd.DataFrame, cat_col: str, height: int = 380) -> None:
    label = cat_col.replace("_", " ").title()
    vc = df[cat_col].value_counts().head(8)
    fig = px.pie(
        values=vc.values, names=vc.index.astype(str),
        title=f"{label} Breakdown",
        color_discrete_sequence=_PALETTE, hole=0.4,
    )
    fig.update_traces(textinfo="label+percent", textposition="outside",
                      pull=[0.03] * len(vc))
    fig.update_layout(**_base_layout(height))
    st.plotly_chart(fig, use_container_width=True)


# ── Smart chart selector ────────────────────────────────────────────────────

def generate_smart_charts(df: pd.DataFrame, max_charts: int = 6) -> None:
    """Auto-select up to max_charts most informative charts for any DataFrame.

    Priority order:
      1. Time-series trend (1 chart, full-width)
      2. Category × numeric bar charts (up to 2, side-by-side)
      3. Pie chart for the most meaningful low-cardinality categorical (1 chart)
      4. Numeric distributions for the highest-variance columns (up to 2, side-by-side)
    """
    _MONEY_KEYWORDS = ("amount", "revenue", "sales", "spend", "cost", "price", "total", "salary", "profit", "fee")

    date_cols   = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    num_cols    = [c for c in df.select_dtypes("number").columns
                   if df[c].nunique() > 5 and c.lower() not in ("index", "id", "row", "unnamed")]
    cat_cols    = [c for c in df.select_dtypes("object").columns
                   if 2 <= df[c].nunique() <= 25]
    pie_cols    = [c for c in df.select_dtypes("object").columns
                   if 2 <= df[c].nunique() <= 8]

    # Score categorical columns — prefer those that pair well with a numeric column
    # (skip pure-id columns)
    cat_cols = [c for c in cat_cols if df[c].nunique() / len(df) < 0.8]

    slots: list[tuple[str, Any]] = []  # (chart_type, args)

    # 1. Best time-series: date × the most "money-like" numeric column
    if date_cols and num_cols:
        dc = date_cols[0]
        nc = next((c for c in num_cols if any(k in c.lower() for k in _MONEY_KEYWORDS)), num_cols[0])
        agg = "sum" if any(k in nc.lower() for k in _MONEY_KEYWORDS) else "mean"
        slots.append(("timeseries", (dc, nc, agg)))

    # 2. Best category × numeric pairs — pick 2 most varied categoricals
    cat_by_cardinality = sorted(cat_cols, key=lambda c: df[c].nunique(), reverse=True)
    for cc in cat_by_cardinality[:2]:
        if not num_cols:
            break
        nc = next((c for c in num_cols if any(k in c.lower() for k in _MONEY_KEYWORDS)), num_cols[0])
        agg = "sum" if any(k in nc.lower() for k in _MONEY_KEYWORDS) else "mean"
        slots.append(("bar_cat", (cc, nc, agg)))

    # 3. One pie chart — most informative low-cardinality categorical
    if pie_cols:
        best_pie = sorted(pie_cols, key=lambda c: df[c].nunique(), reverse=True)[0]
        slots.append(("pie", (best_pie,)))

    # 4. Numeric distributions — top 2 by std (most spread = most interesting)
    interesting_num = sorted(num_cols, key=lambda c: df[c].std(), reverse=True)[:2]
    for nc in interesting_num:
        slots.append(("histogram", (nc,)))

    # Enforce cap
    slots = slots[:max_charts]

    # ── Render ────────────────────────────────────────────────────────────────
    i = 0
    while i < len(slots):
        kind, args = slots[i]

        if kind == "timeseries":
            # Full-width
            st.markdown('<div class="card" style="padding:0.5rem 1rem 0;">', unsafe_allow_html=True)
            _chart_timeseries(df, args[0], args[1], agg=args[2], height=400)
            st.markdown('</div><br>', unsafe_allow_html=True)
            i += 1

        elif kind in ("bar_cat", "pie", "histogram") and i + 1 < len(slots) and slots[i + 1][0] != "timeseries":
            # Pair with next chart side-by-side
            next_kind, next_args = slots[i + 1]
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown('<div class="card" style="padding:0.5rem 1rem 0;">', unsafe_allow_html=True)
                _render_slot(df, kind, args)
                st.markdown('</div>', unsafe_allow_html=True)
            with col_r:
                st.markdown('<div class="card" style="padding:0.5rem 1rem 0;">', unsafe_allow_html=True)
                _render_slot(df, next_kind, next_args)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            i += 2

        else:
            # Odd chart — full width
            st.markdown('<div class="card" style="padding:0.5rem 1rem 0;">', unsafe_allow_html=True)
            _render_slot(df, kind, args)
            st.markdown('</div><br>', unsafe_allow_html=True)
            i += 1

    if not slots:
        st.info("Not enough column variety to generate charts automatically.")


def _render_slot(df: pd.DataFrame, kind: str, args: tuple) -> None:
    if kind == "bar_cat":
        _chart_bar_category(df, args[0], args[1], agg=args[2], height=380)
    elif kind == "pie":
        _chart_pie(df, args[0], height=380)
    elif kind == "histogram":
        _chart_histogram(df, args[0], height=380)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════

def page_home() -> None:
    st.markdown("""
    <div class="hero">
        <span class="hero-icon">✦</span>
        <h1>AI Data Cleaner</h1>
        <p>Upload messy data and let the AI agent profile it, propose intelligent
        cleaning steps, compute business KPIs, and build a live dashboard in minutes.</p>
    </div>
    <div class="feat-grid">
        <div class="feat-card">
            <div class="feat-icon-wrap">
                <svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="12" cy="12" r="8.5" stroke="#6366F1" stroke-width="2"/>
                    <line x1="18.5" y1="18.5" x2="25" y2="25" stroke="#818CF8" stroke-width="2" stroke-linecap="round"/>
                    <line x1="8" y1="10" x2="16" y2="10" stroke="#818CF8" stroke-width="1.5" stroke-linecap="round"/>
                    <line x1="8" y1="13" x2="14" y2="13" stroke="#818CF8" stroke-width="1.5" stroke-linecap="round"/>
                    <line x1="8" y1="16" x2="15" y2="16" stroke="#818CF8" stroke-width="1.5" stroke-linecap="round"/>
                </svg>
            </div>
            <div class="feat-title">Smart Profiling</div>
            <div class="feat-desc">Detects data types, null rates, duplicates
            and quality issues across every column automatically.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon-wrap">
                <svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M5 23L19 9" stroke="#6366F1" stroke-width="2" stroke-linecap="round"/>
                    <path d="M19 9L23 5" stroke="#818CF8" stroke-width="2" stroke-linecap="round"/>
                    <circle cx="24" cy="4" r="2" fill="#C084FC"/>
                    <path d="M11 5L12 2L13 5L16 6L13 7L12 10L11 7L8 6Z" fill="#6366F1"/>
                    <path d="M21 16L21.8 14L22.6 16L24.6 16.8L22.6 17.6L21.8 19.6L21 17.6L19 16.8Z" fill="#818CF8"/>
                </svg>
            </div>
            <div class="feat-title">AI-Powered Cleaning</div>
            <div class="feat-desc">The agent proposes targeted cleaning operations.
            You review and approve — nothing changes without your consent.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon-wrap">
                <svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="2" y="17" width="6" height="9" rx="1.5" fill="#6366F1" opacity="0.6"/>
                    <rect x="11" y="11" width="6" height="15" rx="1.5" fill="#6366F1"/>
                    <rect x="20" y="5" width="6" height="21" rx="1.5" fill="#818CF8"/>
                    <path d="M3 20L11 14L17 17L25 8" stroke="#C084FC" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="2 2"/>
                </svg>
            </div>
            <div class="feat-title">KPIs & Dashboard</div>
            <div class="feat-desc">Auto-generated business KPIs and interactive
            charts tailored to your dataset's domain.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if st.button("Get Started →", use_container_width=True):
            go_to("upload")

    tracker = get_tracker()
    if tracker.records:
        st.markdown(
            f'<p style="text-align:center;color:var(--dim);font-size:0.78rem;margin-top:1rem;">'
            f'Session cost: ${tracker.total_cost_usd:.4f} · {len(tracker.records)} LLM calls</p>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: UPLOAD
# ═══════════════════════════════════════════════════════════════════════════

def page_upload() -> None:
    step_indicator("upload")

    st.markdown("""
    <div class="pg-header">
        <h1>Upload Your Data</h1>
        <p>Supports CSV, Excel (.xlsx / .xls), JSON, and Parquet · Max 500 MB</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop your file here or click to browse",
        type=["csv", "xlsx", "xls", "json", "parquet"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.markdown(
            '<p style="text-align:center;color:var(--dim);margin-top:0.75rem;font-size:0.85rem;">'
            "No file selected yet.</p>",
            unsafe_allow_html=True,
        )
        return

    raw_bytes = uploaded.read()
    try:
        df = ingest(raw_bytes, uploaded.name)
    except IngestionError as exc:
        st.error(str(exc))
        return

    st.session_state.df_raw = df
    st.session_state.filename = uploaded.name
    st.session_state.file_size = len(raw_bytes)
    st.session_state.file_format = Path(uploaded.name).suffix.lstrip(".").upper()

    # Clear downstream state when a new file is uploaded
    for key in ("profile_raw", "cleaning_plan", "selected_op_ids", "df_clean",
                 "results_log", "validation_report", "profile_clean", "kpis", "domain_context"):
        st.session_state.pop(key, None)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns):,}")
    c3.metric("Size", f"{len(raw_bytes)/1_048_576:.1f} MB")
    c4.metric("Format", S("file_format", "?"))

    if len(df) == 0:
        st.error("The file has 0 data rows (header only). Please upload a file that contains data.")
        return
    if len(df) == 1:
        st.warning("Only 1 data row detected — analysis may be limited.")
    if len(df.columns) > 500:
        st.warning(f"{len(df.columns)} columns detected — the cleaning plan may be truncated.")

    st.markdown("<br>", unsafe_allow_html=True)
    _, col_c, _ = st.columns([4, 2, 4])
    with col_c:
        if st.button("Analyse Data →", use_container_width=True, key="analyse_btn"):
            go_to("profile")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: PROFILE
# ═══════════════════════════════════════════════════════════════════════════

def page_profile() -> None:
    step_indicator("profile")

    df: pd.DataFrame | None = S("df_raw")
    if df is None:
        st.error("No data loaded. Please upload a file first.")
        if st.button("← Go to Upload"):
            go_to("upload")
        return

    if S("profile_raw") is None:
        with st.spinner("Profiling your dataset…"):
            st.session_state.profile_raw = run_profile(df)

    prof = S("profile_raw")
    fn = S("filename", "dataset")

    st.markdown(f"""
    <div class="pg-header">
        <h1>Dataset Profile</h1>
        <p>{fn} — analysed {prof.row_count:,} rows × {prof.column_count} columns</p>
    </div>
    """, unsafe_allow_html=True)

    # Summary metrics
    null_total = sum(n.null_count for n in prof.null_analysis)
    total_cells = prof.row_count * prof.column_count
    completeness = (1 - null_total / max(total_cells, 1)) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{prof.row_count:,}")
    c2.metric("Columns", f"{prof.column_count}")
    c3.metric("Duplicates", f"{prof.duplicate_count:,}")
    c4.metric("Completeness", f"{completeness:.1f}%")
    c5.metric("Quality Flags", f"{len(prof.quality_flags)}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Domain detection
    domain_result = detect_domain(list(df.columns))
    if domain_result.top and "domain_context" not in st.session_state:
        st.session_state.domain_context = domain_result.context_string
    if domain_result.top:
        dom = domain_result.top
        conf_cls = {"high": "b-high", "medium": "b-med", "low": "b-low"}.get(dom.confidence, "b-info")
        st.markdown(
            f'<div class="card" style="border-left:3px solid var(--accent);">'
            f'<span style="color:var(--muted);font-size:0.75rem;text-transform:uppercase;letter-spacing:.5px;">Detected Domain</span><br>'
            f'<span style="color:var(--text);font-weight:600;font-size:1.05rem;">{dom.domain}</span>'
            f'&nbsp;&nbsp;<span class="badge {conf_cls}">{dom.confidence}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Quality flags
    if prof.quality_flags:
        st.markdown("### Quality Issues")
        _flag_icons = {
            "high_nulls": "🔴", "mixed_types": "🟡", "looks_numeric": "🔵",
            "looks_datetime": "🔵", "whitespace_issues": "🟡",
            "potential_id": "⚪", "constant_column": "🔴", "high_cardinality": "🟡",
        }
        chips = ""
        for f in prof.quality_flags[:15]:
            icon = _flag_icons.get(f.flag_type.value, "⚠")
            chips += (
                f'<span class="flag-chip">{icon} '
                f'<strong>{f.column}</strong>: {f.flag_type.value.replace("_"," ")}'
                f'</span>'
            )
        remaining = len(prof.quality_flags) - 15
        if remaining > 0:
            chips += f'<span class="flag-chip inf">+{remaining} more</span>'
        st.markdown(f'<div style="margin-bottom:1.5rem;">{chips}</div>', unsafe_allow_html=True)

    # Column table
    st.markdown("### Column Breakdown")
    null_map = {n.column: n for n in prof.null_analysis}
    rows = []
    for s in prof.schema_report:
        nr = null_map.get(s.name)
        rows.append({
            "Column": s.name,
            "Dtype": s.dtype,
            "Semantic Type": s.semantic_type.value,
            "Null %": f"{nr.null_pct:.1%}" if nr else "0%",
            "Flagged": "⚠" if (nr and nr.flagged) else "",
        })
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        height=min(420, 45 + len(rows) * 36),
    )

    st.markdown("<br>", unsafe_allow_html=True)
    _, col_c, _ = st.columns([4, 2, 4])
    with col_c:
        if st.button("Generate Cleaning Plan →", use_container_width=True):
            go_to("cleaning")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: CLEANING PLAN
# ═══════════════════════════════════════════════════════════════════════════

def page_cleaning() -> None:
    step_indicator("cleaning")

    prof = S("profile_raw")
    if prof is None:
        st.error("No profile found. Please start from the beginning.")
        if st.button("← Go to Upload"):
            go_to("upload")
        return

    st.markdown("""
    <div class="pg-header">
        <h1>Cleaning Plan</h1>
        <p>Your data has been analysed and transformations have been proposed below.
        Tick the ones you want to apply.</p>
    </div>
    """, unsafe_allow_html=True)

    # Generate plan if needed
    if S("cleaning_plan") is None:
        with st.spinner("Generating a cleaning plan…"):
            try:
                plan = CleaningAgent(get_llm()).plan(prof)
                st.session_state.cleaning_plan = plan
                st.session_state.selected_op_ids = [op.id for op in plan.operations]
            except CleaningAgentError as exc:
                st.error(f"Failed to generate cleaning plan: {exc}")
                return

    plan: CleaningPlan = S("cleaning_plan")

    if not plan.operations:
        st.success("Your data is already clean — no operations needed.")
        if st.button("Generate KPIs →"):
            st.session_state.df_clean = S("df_raw").copy()
            st.session_state.profile_clean = prof
            go_to("kpis")
        return

    if "selected_op_ids" not in st.session_state:
        st.session_state.selected_op_ids = [op.id for op in plan.operations]

    selected_set: set[str] = set(S("selected_op_ids", []))

    # Select / Deselect all controls
    cA, cB, cC = st.columns([1.5, 1.5, 7])
    with cA:
        st.markdown('<span class="ghost-marker"></span>', unsafe_allow_html=True)
        if st.button("Select All", use_container_width=True, key="sel_all"):
            st.session_state.selected_op_ids = [op.id for op in plan.operations]
            for op in plan.operations:
                st.session_state[f"chk_{op.id}"] = True
    with cB:
        st.markdown('<span class="ghost-marker"></span>', unsafe_allow_html=True)
        if st.button("Deselect All", use_container_width=True, key="desel_all"):
            st.session_state.selected_op_ids = []
            for op in plan.operations:
                st.session_state[f"chk_{op.id}"] = False

    st.markdown("<br>", unsafe_allow_html=True)

    new_selected: list[str] = []
    _conf_cls = {"high": "b-high", "medium": "b-med", "low": "b-low"}

    for op in plan.operations:
        is_on = op.id in selected_set
        conf_badge = f'<span class="badge {_conf_cls.get(op.confidence, "b-info")}">{op.confidence.upper()}</span>'
        type_badge = f'<span class="badge b-type" style="margin-left:5px;">{op.type.value.replace("_"," ")}</span>'

        st.markdown(f'<div class="op-wrap">', unsafe_allow_html=True)
        colL, colR = st.columns([0.5, 11.5])
        with colL:
            checked = st.checkbox("", value=is_on, key=f"chk_{op.id}", label_visibility="collapsed")
        with colR:
            st.markdown(
                f'{conf_badge}{type_badge}'
                f'<div style="color:var(--text);font-weight:600;font-size:0.93rem;margin-top:5px;">{op.description}</div>'
                f'<div style="color:var(--dim);font-size:0.8rem;margin-top:3px;">{op.impact}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        if checked:
            new_selected.append(op.id)

    st.session_state.selected_op_ids = new_selected
    n = len(new_selected)

    st.markdown("<br>", unsafe_allow_html=True)
    _, col_c, _ = st.columns([4, 2, 4])
    with col_c:
        label = f"Apply {n} Transformation{'s' if n != 1 else ''} →" if n else "No transformations selected"
        if st.button(label, use_container_width=True, disabled=(n == 0), key="apply_btn"):
            st.session_state.pop("df_clean", None)
            st.session_state.pop("results_log", None)
            st.session_state.pop("validation_report", None)
            st.session_state.pop("profile_clean", None)
            go_to("results")

    tracker = get_tracker()
    if tracker.records:
        st.markdown(
            f'<p style="color:var(--dim);font-size:0.76rem;margin-top:.5rem;">'
            f'LLM cost so far: ${tracker.total_cost_usd:.4f}</p>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: CLEANING RESULTS
# ═══════════════════════════════════════════════════════════════════════════

def page_results() -> None:
    step_indicator("results")

    df_raw: pd.DataFrame | None = S("df_raw")
    plan: CleaningPlan | None = S("cleaning_plan")
    selected_ids: list[str] = S("selected_op_ids", [])

    if df_raw is None or plan is None:
        st.error("Missing data. Please start from the beginning.")
        if st.button("← Go to Upload"):
            go_to("upload")
        return

    st.markdown("""
    <div class="pg-header">
        <h1>Cleaning Results</h1>
        <p>Applying your selected transformations to the dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    if S("df_clean") is None:
        selected_plan = plan.filter_by_ids(selected_ids)
        df_work = df_raw.copy()
        log: list[tuple[str, str]] = []
        agent = CleaningAgent(get_llm())
        total = len(selected_plan.operations)

        if total == 0:
            st.session_state.df_clean = df_work
            st.session_state.results_log = []
            st.session_state.profile_clean = S("profile_raw")
            st.session_state.validation_report = None
        else:
            bar = st.progress(0, text="Starting…")
            for i, op in enumerate(selected_plan.operations):
                bar.progress(i / total, text=f"[{i+1}/{total}] {op.description[:55]}…")
                result = execute(op.code, df_work)
                if result.success:
                    df_work = result.df
                    log.append(("ok", op.description))
                else:
                    fixed = False
                    last_err = result.error or ""
                    for _ in range(2):
                        try:
                            fixed_code = agent.fix_operation(op, last_err)
                            fix_res = execute(fixed_code, df_work)
                            if fix_res.success:
                                df_work = fix_res.df
                                log.append(("fixed", f"{op.description} (auto-fixed)"))
                                fixed = True
                                break
                            last_err = fix_res.error or ""
                        except CleaningAgentError:
                            break
                    if not fixed:
                        log.append(("skip", f"{op.description} — skipped"))
            bar.progress(1.0, text="Done!")

            st.session_state.df_clean = df_work
            st.session_state.results_log = log
            validation = validate(df_raw, df_work, selected_plan)
            st.session_state.validation_report = validation
            with st.spinner("Re-profiling cleaned data…"):
                st.session_state.profile_clean = run_profile(df_work)

    df_clean: pd.DataFrame = S("df_clean")
    log = S("results_log", [])
    validation = S("validation_report")

    # Before / after comparison
    rows_removed = len(df_raw) - len(df_clean)
    cols_removed = len(df_raw.columns) - len(df_clean.columns)
    null_before = int(df_raw.isna().sum().sum())
    null_after = int(df_clean.isna().sum().sum())
    applied = sum(1 for s, _ in log if s in ("ok", "fixed"))

    def _delta_html(val: int, good_direction: str = "down") -> str:
        if val == 0:
            return '<span style="color:var(--dim);font-size:0.78rem;">No change</span>'
        color = "var(--green)" if (good_direction == "down" and val > 0) or (good_direction == "up" and val > 0) else "var(--yellow)"
        arrow = "↓" if good_direction == "down" else "↑"
        return f'<span style="color:{color};font-size:0.78rem;font-weight:600;">{arrow} {abs(val):,}</span>'

    st.markdown("""
    <h3 style="color:#E2E8F0;font-size:1.1rem;font-weight:600;margin-bottom:1rem;">Before vs After</h3>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, before, after, delta, direction in [
        (c1, "Rows",    f"{len(df_raw):,}",          f"{len(df_clean):,}",          rows_removed,                "down"),
        (c2, "Columns", f"{len(df_raw.columns):,}",  f"{len(df_clean.columns):,}",  cols_removed,                "down"),
        (c3, "Nulls",   f"{null_before:,}",           f"{null_after:,}",             null_before - null_after,    "down"),
        (c4, "Ops Applied", "—",                      f"{applied}/{len(log)}",       0,                           "up"),
    ]:
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center;padding:1.2rem 1rem;">
                <div style="color:var(--muted);font-size:0.72rem;text-transform:uppercase;letter-spacing:0.6px;font-weight:600;margin-bottom:0.6rem;">{label}</div>
                <div style="display:flex;align-items:center;justify-content:center;gap:0.5rem;margin-bottom:0.4rem;">
                    <span style="color:var(--dim);font-size:1rem;">{before}</span>
                    <span style="color:var(--dim);font-size:0.8rem;">→</span>
                    <span style="color:#FFFFFF;font-size:1.3rem;font-weight:700;">{after}</span>
                </div>
                {_delta_html(delta, direction)}
            </div>
            """, unsafe_allow_html=True)

    if log:
        st.markdown("""
        <h3 style="color:#E2E8F0;font-size:1.1rem;font-weight:600;margin:1.5rem 0 0.75rem;">Applied Operations</h3>
        """, unsafe_allow_html=True)
        icons   = {"ok": "✅", "fixed": "🔧", "skip": "⏭"}
        colours = {"ok": "var(--green)", "fixed": "var(--yellow)", "skip": "var(--dim)"}
        rows_html = "".join(
            f'<div class="res-row"><span>{icons[s]}</span>'
            f'<span style="color:{colours[s]};font-size:.87rem;">{desc}</span></div>'
            for s, desc in log
        )
        st.markdown(f'<div style="margin-bottom:1rem;">{rows_html}</div>', unsafe_allow_html=True)

    if validation:
        with st.expander("Validation Report", expanded=False):
            for chk in validation.checks:
                icon = {"PASS": "✅", "WARNING": "⚠️", "FAIL": "❌"}[chk.status.value]
                st.markdown(f"{icon} **{chk.name}**: {chk.message}")

    st.markdown("<br>", unsafe_allow_html=True)
    _, col_c, _ = st.columns([4, 2, 4])
    with col_c:
        if st.button("Generate KPIs →", use_container_width=True, key="gen_kpis_btn"):
            st.session_state.pop("kpis", None)
            go_to("kpis")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: KPIs
# ═══════════════════════════════════════════════════════════════════════════

def page_kpis() -> None:
    step_indicator("kpis")

    df_clean: pd.DataFrame | None = S("df_clean") if S("df_clean") is not None else S("df_raw")
    prof_clean = S("profile_clean") if S("profile_clean") is not None else S("profile_raw")

    if df_clean is None or prof_clean is None:
        st.error("No cleaned data available. Please complete the cleaning step.")
        if st.button("← Back"):
            go_to("results")
        return

    st.markdown("""
    <div class="pg-header">
        <h1>Business KPIs</h1>
        <p>Key performance indicators will be generated tailored to your dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    # Auto-build a rich domain context from columns if not already set
    if not S("domain_context") or S("domain_context") == "General business dataset":
        col_names = [c.name for c in prof_clean.schema_report]
        domain_result = detect_domain(col_names)

        numeric_cols = [c.name for c in prof_clean.schema_report if c.dtype in ("int64", "float64")]
        categorical_cols = [c.name for c in prof_clean.schema_report if c.dtype == "object"]
        datetime_cols = [c.name for c in prof_clean.schema_report if "datetime" in c.dtype]

        domain_label = domain_result.top.domain if domain_result.top else "General"
        parts = [f"{domain_label} dataset with {prof_clean.row_count:,} rows and {prof_clean.column_count} columns."]
        if numeric_cols:
            parts.append(f"Numeric columns: {', '.join(numeric_cols[:8])}{'...' if len(numeric_cols) > 8 else ''}.")
        if categorical_cols:
            parts.append(f"Categorical columns: {', '.join(categorical_cols[:8])}{'...' if len(categorical_cols) > 8 else ''}.")
        if datetime_cols:
            parts.append(f"Date/time columns: {', '.join(datetime_cols)}.")

        st.session_state.domain_context = " ".join(parts)

    domain_ctx = st.text_input(
        "Dataset context (auto-detected — edit to refine)",
        value=S("domain_context"),
        placeholder="Describe what this data represents, e.g. 'E-commerce transactions for an online store'",
        help="This helps suggest relevant KPIs for your dataset.",
    )
    st.session_state.domain_context = domain_ctx

    st.markdown("<br>", unsafe_allow_html=True)

    if S("kpis") is None:
        with st.spinner("Generating KPIs for your data…"):
            try:
                raw_kpis = KPIAgent(get_llm()).suggest(prof_clean, domain_ctx)
                computed: list[KPI] = []
                for kpi in raw_kpis:
                    val = compute_kpi_value(kpi.code, df_clean)
                    computed.append(kpi.model_copy(update={"computed_value": val}))
                st.session_state.kpis = computed
            except KPIAgentError as exc:
                st.error(f"KPI generation failed: {exc}")
                return

    kpis: list[KPI] = S("kpis", [])

    if not kpis:
        st.warning("No KPIs were generated. Try updating the domain context and clicking Regenerate.")
        if st.button("Regenerate KPIs"):
            st.session_state.pop("kpis", None)
            st.rerun()
        return

    st.markdown(f"### {len(kpis)} KPIs Generated")

    _pri_cls = {"high": "b-high", "medium": "b-med", "low": "b-low"}

    for i in range(0, len(kpis), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(kpis):
                break
            kpi = kpis[idx]
            val_str = fmt_value(kpi.computed_value, kpi.unit)
            pri_cls = _pri_cls.get(kpi.priority, "b-info")
            with col:
                st.markdown(
                    f'<div class="kpi-card">'
                    f'<div style="display:flex;justify-content:space-between;align-items:flex-start;">'
                    f'<span class="kpi-name">{kpi.name}</span>'
                    f'<span class="badge {pri_cls}">{kpi.priority.upper()}</span>'
                    f'</div>'
                    f'<div class="kpi-cat">{kpi.category}</div>'
                    f'<div class="kpi-val">{val_str}</div>'
                    f'<div class="kpi-form">Formula: {kpi.formula_description}</div>'
                    f'<div class="kpi-why">{kpi.business_value}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)
    _, col_c, _ = st.columns([4, 2, 4])
    with col_c:
        if st.button("View Dashboard →", use_container_width=True, key="view_dashboard_btn"):
            go_to("dashboard")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def page_dashboard() -> None:
    step_indicator("dashboard")

    df_clean: pd.DataFrame | None = S("df_clean") if S("df_clean") is not None else S("df_raw")
    kpis: list[KPI] = S("kpis", [])
    prof = S("profile_clean") if S("profile_clean") is not None else S("profile_raw")
    fn = S("filename", "dataset")

    if not kpis or df_clean is None:
        st.error("No KPIs or data found. Please complete the KPI step first.")
        if st.button("← Back to KPIs"):
            go_to("kpis")
        return

    rows_str = f"{prof.row_count:,}" if prof else f"{len(df_clean):,}"
    cols_str = str(prof.column_count) if prof else str(len(df_clean.columns))
    stem = Path(fn).stem

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:2rem;">
        <div>
            <h1 style="font-size:1.75rem;font-weight:700;color:#F8FAFC;margin:0 0 0.25rem;">{stem}</h1>
            <p style="color:var(--muted);font-size:0.88rem;margin:0;">
                {rows_str} rows &nbsp;·&nbsp; {cols_str} columns &nbsp;·&nbsp; {len(kpis)} KPIs generated
            </p>
        </div>
        <span class="badge b-info" style="margin-top:0.4rem;">Auto-generated</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Scorecard row ────────────────────────────────────────────────────────
    scalar_kpis = [k for k in kpis if isinstance(k.computed_value, (int, float)) and k.computed_value is not None]
    high_kpis   = [k for k in scalar_kpis if k.priority == "high"]
    other_kpis  = [k for k in scalar_kpis if k.priority != "high"]
    scorecards  = (high_kpis + other_kpis)[:6]

    if scorecards:
        st.markdown('<h3 style="color:#E2E8F0;font-size:1.1rem;font-weight:600;margin-bottom:1rem;">Key Metrics</h3>',
                    unsafe_allow_html=True)
        cols = st.columns(len(scorecards))
        for col, kpi in zip(cols, scorecards):
            with col:
                pri_color = {"high": "#10B981", "medium": "#F59E0B", "low": "#8B92A5"}.get(kpi.priority, "#8B92A5")
                st.markdown(f"""
                <div class="card" style="text-align:center;padding:1.25rem 0.75rem;">
                    <div style="color:var(--muted);font-size:0.7rem;text-transform:uppercase;
                                letter-spacing:0.6px;font-weight:600;margin-bottom:0.5rem;">{kpi.name}</div>
                    <div style="color:#FFFFFF;font-size:1.6rem;font-weight:700;line-height:1.1;
                                margin-bottom:0.4rem;">{fmt_value(kpi.computed_value, kpi.unit)}</div>
                    <div style="color:{pri_color};font-size:0.68rem;font-weight:600;
                                text-transform:uppercase;letter-spacing:0.5px;">{kpi.priority}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ───────────────────────────────────────────────────────────────
    st.markdown('<h3 style="color:#E2E8F0;font-size:1.1rem;font-weight:600;margin-bottom:1.25rem;">Charts</h3>',
                unsafe_allow_html=True)

    # Charts with many categories get full width; simpler ones share a row
    generate_smart_charts(df_clean)

    # ── Export ───────────────────────────────────────────────────────────────
    st.markdown('<hr style="border-color:var(--border);margin:1rem 0 1.5rem;">', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#E2E8F0;font-size:1.1rem;font-weight:600;margin-bottom:1rem;">Export</h3>',
                unsafe_allow_html=True)

    col_a, col_b, col_c, _ = st.columns([2, 2, 2, 2])

    with col_a:
        csv_bytes = df_clean.to_csv(index=False).encode()
        st.download_button("⬇ Download CSV", csv_bytes, f"{stem}_cleaned.csv",
                           "text/csv", use_container_width=True)
    with col_b:
        buf = io.BytesIO()
        try:
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                df_clean.to_excel(writer, index=False, sheet_name="Cleaned Data")
            st.download_button("⬇ Download Excel", buf.getvalue(),
                               f"{stem}_cleaned.xlsx", use_container_width=True)
        except Exception:
            st.caption("Excel export unavailable.")
    with col_c:
        if st.button("↩ Start Over", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            go_to("home")

    tracker = get_tracker()
    if tracker.records:
        st.markdown(
            f'<p style="color:var(--dim);font-size:0.76rem;margin-top:1.25rem;">'
            f'Session cost: ${tracker.total_cost_usd:.4f} &nbsp;·&nbsp; '
            f'{tracker.total_tokens:,} tokens &nbsp;·&nbsp; {len(tracker.records)} LLM calls</p>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════════════

_PAGES = {
    "home":      page_home,
    "upload":    page_upload,
    "profile":   page_profile,
    "cleaning":  page_cleaning,
    "results":   page_results,
    "kpis":      page_kpis,
    "dashboard": page_dashboard,
}

_css()

current_page = st.session_state.get("page", "home")
handler = _PAGES.get(current_page, page_home)
handler()
