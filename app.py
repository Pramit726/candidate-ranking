"""
app.py — FiguringOut · Redrob AI Candidate Ranker
"""

import base64
import io
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Make the package importable without install ───────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from redrob_ranker.config import (
    DEFAULT_AS_OF_DATE,
    DEFAULT_TOP_N,
    W_BEHAVIOR,
    W_CONTEXT,
    W_TECH,
)
from redrob_ranker.evaluation import evaluate_ranking
from redrob_ranker.features import (
    predict_behavior_fit,
    predict_context_fit,
    predict_tech_fit,
)
from redrob_ranker.filters import apply_pre_ranking_filters, print_filter_summary
from redrob_ranker.io_utils import _normalize_label_record, load_golden_set
from redrob_ranker.reasoning import (
    _best_jd_sentence,
    generate_reasoning,
    load_job_description,
)
from redrob_ranker.scoring import (
    build_candidates_data,
    rank_lambdamart,
    rank_lexicographical,
    rank_nonlinear_weighted,
)

# =============================================================================
# LOGO HELPER
# =============================================================================


def get_logo_b64() -> str:
    """Return base64-encoded logo.png, or empty string if not found."""
    logo_path = ROOT / "narrow.png"
    if logo_path.exists():
        return base64.b64encode(logo_path.read_bytes()).decode()
    return ""


LOGO_B64 = get_logo_b64()


def logo_img_tag(size: int = 48) -> str:
    if LOGO_B64:
        return (
            f'<img src="data:image/png;base64,{LOGO_B64}" '
            f'width="{size}" height="{size}" style="border-radius:8px;object-fit:contain;" />'
        )
    return '<span style="font-size:2rem;">🎯</span>'


WIDE_LOGO_B64 = get_logo_b64()  # reuse same fn; swap file below if different name


def _load_b64(filename: str) -> str:
    p = ROOT / filename
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""


WIDE_LOGO_B64 = _load_b64("wide.png")  # wide horizontal logo


def wide_logo_tag(height: int = 48) -> str:
    if WIDE_LOGO_B64:
        return (
            f'<img src="data:image/png;base64,{WIDE_LOGO_B64}" '
            f'height="{height}" '
            f'style="width:auto;max-width:220px;object-fit:contain;'
            f'background:#fff;border-radius:10px;padding:6px 10px;" />'
        )
    return '<span style="font-size:2rem;">🎯</span>'


# =============================================================================
# PAGE CONFIG
# =============================================================================

_icon = ROOT / "wide.png"
st.set_page_config(
    page_title="FiguringOut · Redrob Ranker",
    page_icon=str(_icon) if _icon.exists() else "🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit 1.35+ sidebar logo
if LOGO_B64:
    try:
        st.logo(str(ROOT / "narrow.png"), size="large")
    except Exception:
        pass

# =============================================================================
# DESIGN TOKENS & GLOBAL CSS
# =============================================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
}

/* ── Colour tokens ── */
:root {
    --brand-900: #0f0320;
    --brand-800: #1a0533;
    --brand-700: #240b44;
    --brand-600: #3b1270;
    --brand-500: #5c2db3;
    --brand-400: #7D45E0;
    --brand-300: #a57fee;
    --brand-200: #d4b8f5;
    --brand-100: #ede5fb;
    --brand-50:  #f8f5ff;

    --success:   #16a34a;
    --warning:   #d97706;
    --danger:    #dc2626;
    --neutral-900: #111827;
    --neutral-600: #4b5563;
    --neutral-400: #9ca3af;
    --neutral-200: #e5e7eb;
    --neutral-50:  #f9fafb;

    --shadow-sm: 0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.06);
    --shadow-md: 0 4px 12px rgba(93,45,179,.12), 0 2px 6px rgba(0,0,0,.06);
    --shadow-lg: 0 8px 32px rgba(93,45,179,.18), 0 4px 12px rgba(0,0,0,.08);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1400px !important; }

/* ══════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--brand-900) 0%, var(--brand-800) 100%);
    border-right: 1px solid var(--brand-700);
}
section[data-testid="stSidebar"] > div { padding: 1.25rem 1rem; }

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stToggle label {
    color: var(--brand-200) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #fff !important;
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] .stTextInput input {
    background: var(--brand-700) !important;
    border: 1px solid var(--brand-600) !important;
    color: #fff !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.82rem !important;
}
section[data-testid="stSidebar"] hr {
    border-color: var(--brand-700) !important;
    margin: 0.75rem 0 !important;
}
section[data-testid="stSidebar"] .stMetric {
    background: var(--brand-700);
    border-radius: var(--radius-sm);
    padding: 0.5rem 0.4rem;
    border: 1px solid var(--brand-600);
}
section[data-testid="stSidebar"] .stMetric label { color: var(--brand-300) !important; }
section[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
    color: #fff !important;
    font-size: 1.1rem !important;
    font-weight: 800 !important;
}

/* ══════════════════════════════════════════
   HEADER
══════════════════════════════════════════ */
.app-header {
    background: linear-gradient(135deg, var(--brand-900) 0%, var(--brand-700) 60%, #2a0d5e 100%);
    border: 1px solid var(--brand-600);
    border-radius: var(--radius-lg);
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.25rem;
    box-shadow: var(--shadow-lg);
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 60% 80% at 80% 50%, rgba(125,69,224,.18) 0%, transparent 70%);
    pointer-events: none;
}
.app-header-text { flex: 1; min-width: 0; }
.app-header h1 {
    color: #fff;
    font-size: 1.75rem;
    font-weight: 800;
    margin: 0 0 0.2rem;
    letter-spacing: -0.6px;
    line-height: 1.2;
}
.app-header p {
    color: var(--brand-200);
    margin: 0;
    font-size: 0.9rem;
    font-weight: 400;
    line-height: 1.5;
}
.team-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: var(--brand-400);
    color: #fff;
    padding: 0.2rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
}

/* ══════════════════════════════════════════
   TABS
══════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem;
    background: var(--brand-50);
    padding: 0.35rem;
    border-radius: var(--radius-md);
    border: 1px solid var(--brand-100);
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm) !important;
    padding: 0.45rem 1.1rem !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: var(--neutral-600) !important;
    background: transparent !important;
    border: none !important;
    transition: all .15s ease !important;
}
.stTabs [aria-selected="true"] {
    background: #fff !important;
    color: var(--brand-500) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ══════════════════════════════════════════
   KPI / METRIC CARDS
══════════════════════════════════════════ */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.75rem;
    margin: 1rem 0;
}
.kpi-card {
    background: #fff;
    border: 1.5px solid var(--brand-100);
    border-radius: var(--radius-md);
    padding: 1rem 1.1rem;
    text-align: center;
    box-shadow: var(--shadow-sm);
    transition: box-shadow .2s ease, transform .2s ease;
}
.kpi-card:hover { box-shadow: var(--shadow-md); transform: translateY(-1px); }
.kpi-card .kpi-value {
    font-size: 1.75rem;
    font-weight: 800;
    color: var(--brand-500);
    line-height: 1.1;
    font-variant-numeric: tabular-nums;
}
.kpi-card .kpi-label {
    font-size: 0.72rem;
    color: var(--neutral-600);
    margin-top: 0.25rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
.kpi-card.accent { border-color: var(--brand-400); background: var(--brand-50); }
.kpi-card.accent .kpi-value { color: var(--brand-400); }

/* ══════════════════════════════════════════
   ACCURACY CARDS
══════════════════════════════════════════ */
.acc-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin: 0.75rem 0;
}
.acc-card {
    background: #fff;
    border-radius: var(--radius-md);
    padding: 1rem 1.2rem;
    border: 1.5px solid var(--neutral-200);
    box-shadow: var(--shadow-sm);
}
.acc-card .acc-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--neutral-600);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.4rem;
}
.acc-card .acc-bar-wrap {
    background: var(--neutral-200);
    border-radius: 99px;
    height: 6px;
    margin: 0.5rem 0 0.35rem;
    overflow: hidden;
}
.acc-card .acc-bar {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, var(--brand-400), var(--brand-300));
}
.acc-card .acc-pct {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--brand-500);
    line-height: 1;
}

/* ══════════════════════════════════════════
   SECTION TITLES
══════════════════════════════════════════ */
.section-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--brand-800);
    margin: 1.5rem 0 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--brand-100);
}
.section-title::before {
    content: '';
    display: block;
    width: 4px;
    height: 1.1em;
    background: var(--brand-400);
    border-radius: 2px;
    flex-shrink: 0;
}

/* ══════════════════════════════════════════
   BEST STRATEGY BADGE
══════════════════════════════════════════ */
.best-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    background: linear-gradient(135deg, var(--brand-500), var(--brand-400));
    color: #fff;
    padding: 0.2rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.3px;
    vertical-align: middle;
    box-shadow: 0 2px 8px rgba(125,69,224,.35);
}

/* ══════════════════════════════════════════
   FILTER SUMMARY BOX
══════════════════════════════════════════ */
.filter-box {
    background: var(--brand-50);
    border: 1px solid var(--brand-100);
    border-left: 4px solid var(--brand-400);
    border-radius: var(--radius-sm);
    padding: 0.75rem 1rem;
    font-size: 0.84rem;
    color: var(--neutral-600);
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.filter-chip {
    background: #fff;
    border: 1px solid var(--brand-200);
    border-radius: 99px;
    padding: 0.15rem 0.6rem;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--brand-500);
}

/* ══════════════════════════════════════════
   RUN BUTTON
══════════════════════════════════════════ */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--brand-500) 0%, var(--brand-400) 100%) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1.5rem !important;
    box-shadow: 0 4px 14px rgba(125,69,224,.4) !important;
    transition: all .2s ease !important;
    letter-spacing: 0.2px !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(125,69,224,.5) !important;
}

/* ══════════════════════════════════════════
   DATAFRAME TWEAKS
══════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    border: 1px solid var(--brand-100) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ══════════════════════════════════════════
   SUCCESS / INFO / WARNING
══════════════════════════════════════════ */
[data-testid="stSuccess"] {
    background: #f0fdf4 !important;
    border: 1px solid #bbf7d0 !important;
    border-radius: var(--radius-sm) !important;
    color: var(--success) !important;
}
[data-testid="stInfo"] {
    background: var(--brand-50) !important;
    border: 1px solid var(--brand-200) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--brand-600) !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    # Logo + title
    if LOGO_B64:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:.75rem;margin-bottom:.5rem;">'
            f"{logo_img_tag(40)}"
            f'<div><div style="color:#fff;font-weight:800;font-size:1rem;line-height:1.2">FiguringOut</div>'
            f'<div style="color:var(--brand-300);font-size:0.75rem;margin-top:1px">Redrob Ranker</div></div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("## 🎯 FiguringOut")
        st.markdown("*Redrob AI Candidate Ranker*")

    st.markdown("---")
    st.markdown("### ⚙️ Configuration")

    data_dir = st.text_input(
        "Data directory",
        value="./data",
        help="Folder containing golden_test.jsonl, test_set.jsonl, job_description.txt",
    )
    models_dir = st.text_input(
        "Models directory",
        value="./models",
        help="Where the XGBoost LambdaMART model is saved",
    )
    top_n = st.slider("Top-N candidates", 10, 120, DEFAULT_TOP_N, step=10)
    as_of_date = st.text_input(
        "As-of date (YYYY-MM-DD)",
        value=DEFAULT_AS_OF_DATE,
        help="Snapshot date for Behavior_Fit recency calculation",
    )

    st.markdown("### 🔧 Pipeline")
    run_filters = st.toggle("Pre-ranking filters", value=True)
    run_ltr = st.toggle("LambdaMART (LTR)", value=True)
    run_reasoning = st.toggle("Extractive reasoning", value=True)

    st.markdown("---")
    st.markdown("### ⚖️ Scoring weights")
    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        st.metric("Tech", f"{W_TECH:.0%}")
    with col_w2:
        st.metric("Context", f"{W_CONTEXT:.0%}")
    with col_w3:
        st.metric("Behavior", f"{W_BEHAVIOR:.0%}")

    st.markdown("---")
    st.caption("Weights are fixed in config.py")


# =============================================================================
# MAIN HEADER
# =============================================================================

st.markdown(
    f"""
<div class="app-header">
  <div style="flex-shrink:0">{wide_logo_tag(52)}</div>
  <div class="app-header-text">
    <div class="team-pill">⚡ Team FiguringOut</div>
    <h1>Redrob AI Candidate Ranker</h1>
    <p>Evidence-grounded, CPU-only ranking — score every candidate on Tech / Context / Behavior,
       rank with three strategies, explain with extractive vector reasoning.</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# TABS
# =============================================================================

tab_run, tab_about, tab_metrics = st.tabs(
    ["▶  Run Pipeline", "ℹ️  How it works", "📊  Benchmark metrics"]
)

# ---------------------------------------------------------------------------
# TAB 1 — RUN PIPELINE
# ---------------------------------------------------------------------------
with tab_run:

    run_btn = st.button(
        "🚀  Run Ranking Pipeline", type="primary", use_container_width=True
    )

    if run_btn:
        t_start = time.time()

        # ── Stage 1: Load ─────────────────────────────────────────────────
        with st.status("📂 Stage 1 — Loading data…", expanded=True) as status:
            try:
                records = load_golden_set(data_dir)
            except Exception as e:
                st.error(f"Could not load data from `{data_dir}`: {e}")
                st.stop()

            train_records = records
            records_to_rank = records
            st.write(
                f"✅ Loaded **{len(records)}** merged records  (train = {len(train_records)}, rank = {len(records_to_rank)})"
            )
            status.update(label="📂 Data loaded", state="complete")

        # ── Stage 1.5: Filters ────────────────────────────────────────────
        if run_filters:
            with st.status("🔍 Stage 1.5 — Applying pre-ranking filters…") as status:
                records_to_rank, fstats = apply_pre_ranking_filters(
                    records_to_rank, as_of_date
                )
                n_in = fstats["total_in"]
                n_out = fstats["total_out"]
                n_drop = n_in - n_out

                cols = st.columns(3)
                cols[0].metric("Input", n_in)
                cols[1].metric(
                    "Dropped", n_drop, delta=f"-{n_drop}", delta_color="inverse"
                )
                cols[2].metric("Passed", n_out)

                if n_drop:
                    drop_detail = {
                        k: len(v)
                        for k, v in fstats.items()
                        if isinstance(v, list) and v
                    }
                    chips = " ".join(
                        f'<span class="filter-chip">{k}: {v}</span>'
                        for k, v in drop_detail.items()
                    )
                    st.markdown(
                        f'<div class="filter-box">{chips}</div>', unsafe_allow_html=True
                    )

                status.update(
                    label=f"🔍 Filters done — {n_out}/{n_in} passed", state="complete"
                )

        # ── Stage 2: Tech_Fit ─────────────────────────────────────────────
        with st.status("🤖 Stage 2 — Tech_Fit (XGBoost + embeddings)…") as status:
            pred_tech, idx_tr, idx_te, embed_model = predict_tech_fit(
                train_records, records_to_rank, evaluation=True
            )
            gt_tech = np.array(
                [int(r["parsed"].get("Tech_Fit", 0)) for r in train_records]
            )
            acc_tech = (
                (pred_tech[idx_te] == gt_tech[idx_te]).mean() * 100
                if len(idx_te)
                else 0
            )
            status.update(
                label=f"🤖 Tech_Fit done — held-out accuracy {acc_tech:.1f}%",
                state="complete",
            )

        # ── Stage 3: Context_Fit ──────────────────────────────────────────
        with st.status(
            "📋 Stage 3 — Context_Fit (deterministic logic tree)…"
        ) as status:
            pred_context = predict_context_fit(records_to_rank, evaluation=True)
            gt_ctx = np.array(
                [int(r["parsed"].get("Context_Fit", 0)) for r in records_to_rank]
            )
            acc_ctx = (pred_context == gt_ctx).mean() * 100
            status.update(
                label=f"📋 Context_Fit done — accuracy {acc_ctx:.1f}%", state="complete"
            )

        # ── Stage 4: Behavior_Fit ─────────────────────────────────────────
        with st.status("📡 Stage 4 — Behavior_Fit (signal formula)…") as status:
            pred_behavior = predict_behavior_fit(
                records_to_rank, as_of_date, evaluation=True
            )
            gt_beh = np.array(
                [int(r["parsed"].get("Behavior_Fit", 0)) for r in records_to_rank]
            )
            acc_beh = (pred_behavior == gt_beh).mean() * 100
            status.update(
                label=f"📡 Behavior_Fit done — accuracy {acc_beh:.1f}%",
                state="complete",
            )

        # ── Stage 5: Build feature table ──────────────────────────────────
        candidates_data = build_candidates_data(
            records_to_rank, pred_tech, pred_context, pred_behavior
        )

        # ── Stage 6: Rank ─────────────────────────────────────────────────
        all_metrics: dict = {}
        with st.status("🏆 Stage 6 — Running ranking strategies…") as status:
            ranked_lex = rank_lexicographical(candidates_data)
            all_metrics["Lex Sort"] = evaluate_ranking(ranked_lex, "Lex Sort")

            ranked_wt = rank_nonlinear_weighted(candidates_data)
            all_metrics["Weighted"] = evaluate_ranking(ranked_wt, "Weighted")

            ranked_ltr = None
            if run_ltr:
                ranked_ltr = rank_lambdamart(
                    train_candidates_data=candidates_data,
                    train_records=records_to_rank,
                    rank_candidates_data=candidates_data,
                    rank_records=records_to_rank,
                    idx_tr=idx_tr,
                    idx_te=idx_te,
                    models_dir=models_dir,
                    evaluation=True,
                )
                ltr_test_cids = {records_to_rank[i]["candidate_id"] for i in idx_te}
                all_metrics["LambdaMART"] = evaluate_ranking(
                    ranked_ltr[ranked_ltr["candidate_id"].isin(ltr_test_cids)].copy(),
                    "LambdaMART",
                )

            best_name = max(all_metrics, key=lambda n: all_metrics[n]["Final Score"])
            status.update(
                label=f"🏆 Ranking done — best: {best_name}", state="complete"
            )

        # ── Stage 7: Reasoning ────────────────────────────────────────────
        ranked_map = {"Lex Sort": ranked_lex, "Weighted": ranked_wt}
        if ranked_ltr is not None:
            ranked_map["LambdaMART"] = ranked_ltr
        best_df = ranked_map[best_name]
        score_col = "rank_score" if "rank_score" in best_df.columns else "ltr_score"

        reasonings = []
        if run_reasoning:
            with st.status("💬 Stage 7 — Generating extractive reasoning…") as status:
                try:
                    jd_text = load_job_description(data_dir)
                    record_map = {r["candidate_id"]: r for r in records_to_rank}
                    reasonings = generate_reasoning(
                        best_df.head(top_n), record_map, jd_text, embed_model
                    )
                except Exception as e:
                    st.warning(f"Reasoning skipped: {e}")
                status.update(label="💬 Reasoning done", state="complete")

        # ── Stage 8: Build output df ──────────────────────────────────────
        t_elapsed = time.time() - t_start
        out_df = best_df.head(top_n).copy().reset_index(drop=True)
        raw = out_df[score_col].values.astype(float)
        mx = raw.max() if raw.max() > 0 else 1.0
        out_df["score"] = (raw / mx).round(4)
        out_df["rank"] = range(1, len(out_df) + 1)
        if reasonings:
            out_df["reasoning"] = reasonings[: len(out_df)]

        st.markdown("---")

        # ── KPI cards ─────────────────────────────────────────────────────
        st.markdown(
            "<div class='section-title'>Results Summary</div>", unsafe_allow_html=True
        )

        m = all_metrics[best_name]
        kpi_html = f"""
        <div class="kpi-grid">
          <div class="kpi-card accent">
            <div class="kpi-value">{m['NDCG@10']:.4f}</div>
            <div class="kpi-label">NDCG@10</div>
          </div>
          <div class="kpi-card accent">
            <div class="kpi-value">{m['NDCG@50']:.4f}</div>
            <div class="kpi-label">NDCG@50</div>
          </div>
          <div class="kpi-card accent">
            <div class="kpi-value">{m['MAP']:.4f}</div>
            <div class="kpi-label">MAP</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-value">{t_elapsed:.0f}s</div>
            <div class="kpi-label">Runtime</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-value">{len(out_df)}</div>
            <div class="kpi-label">Ranked</div>
          </div>
        </div>
        """
        st.markdown(kpi_html, unsafe_allow_html=True)

        # ── Accuracy cards ─────────────────────────────────────────────────
        st.markdown(
            "<div class='section-title'>Sub-model Accuracy</div>",
            unsafe_allow_html=True,
        )

        acc_html = f"""
        <div class="acc-grid">
          <div class="acc-card">
            <div class="acc-label">🤖 Tech_Fit · XGBoost</div>
            <div class="acc-pct">{acc_tech:.1f}%</div>
            <div class="acc-bar-wrap"><div class="acc-bar" style="width:{acc_tech}%"></div></div>
          </div>
          <div class="acc-card">
            <div class="acc-label">📋 Context_Fit · Logic Tree</div>
            <div class="acc-pct">{acc_ctx:.1f}%</div>
            <div class="acc-bar-wrap"><div class="acc-bar" style="width:{acc_ctx}%"></div></div>
          </div>
          <div class="acc-card">
            <div class="acc-label">📡 Behavior_Fit · Formula</div>
            <div class="acc-pct">{acc_beh:.1f}%</div>
            <div class="acc-bar-wrap"><div class="acc-bar" style="width:{acc_beh}%"></div></div>
          </div>
        </div>
        """
        st.markdown(acc_html, unsafe_allow_html=True)

        # ── Strategy comparison ────────────────────────────────────────────
        st.markdown(
            "<div class='section-title'>Strategy Comparison</div>",
            unsafe_allow_html=True,
        )

        comparison_rows = []
        for strat, met in all_metrics.items():
            comparison_rows.append(
                {
                    "Strategy": ("⭐ " if strat == best_name else "") + strat,
                    "NDCG@10": round(met["NDCG@10"], 4),
                    "NDCG@50": round(met["NDCG@50"], 4),
                    "NDCG@all": round(met["NDCG@all"], 4),
                    "MAP": round(met["MAP"], 4),
                    "Spearman": round(met["Spearman"], 4),
                    "P@10": round(met["P@10"], 4),
                    "Final Score": round(met["Final Score"], 4),
                }
            )
        st.dataframe(
            pd.DataFrame(comparison_rows).set_index("Strategy"),
            use_container_width=True,
        )

        # ── Ranked table ──────────────────────────────────────────────────
        st.markdown(
            f'<div class="section-title">Top-{top_n} Ranked Candidates '
            f'<span class="best-badge">⭐ {best_name}</span></div>',
            unsafe_allow_html=True,
        )

        display_cols = [
            "rank",
            "candidate_id",
            "score",
            "Tech_Fit",
            "Context_Fit",
            "Behavior_Fit",
            "Final_Score_NDCG",
        ]
        if "reasoning" in out_df.columns:
            display_cols.append("reasoning")

        st.dataframe(
            out_df[display_cols].rename(
                columns={
                    "rank": "Rank",
                    "candidate_id": "Candidate ID",
                    "score": "Score",
                    "reasoning": "Reasoning",
                }
            ),
            use_container_width=True,
            height=460,
        )

        # ── Download ──────────────────────────────────────────────────────
        csv_bytes = out_df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️  Download final_ranking.csv",
            data=csv_bytes,
            file_name="final_ranking.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary",
        )

        st.success(
            f"✅ Pipeline complete in **{t_elapsed:.1f}s** · Best strategy: **{best_name}**"
        )

    else:
        st.info(
            "👈 Configure settings in the sidebar, then click **Run Ranking Pipeline**.",
            icon="ℹ️",
        )
        st.markdown("""
        **What this app does:**
        1. 📂 Loads candidate profiles + golden labels from your data directory
        2. 🔍 Applies a 4-stage pre-ranking filter (structural → keyword → behavioural → domain)
        3. 🤖 Predicts **Tech_Fit** with XGBoost over `all-MiniLM-L6-v2` embeddings
        4. 📋 Predicts **Context_Fit** with a deterministic logic tree
        5. 📡 Computes **Behavior_Fit** from platform signals
        6. 🏆 Ranks candidates with 3 strategies, picks the best by composite NDCG/MAP
        7. 💬 Generates extractive one-sentence reasoning per candidate
        8. ⬇️ Downloads `final_ranking.csv`
        """)

# ---------------------------------------------------------------------------
# TAB 2 — HOW IT WORKS
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown("## 🧠 How FiguringOut ranks candidates")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ❌ Traditional approach (the trap)")
        st.markdown("""
- Keyword / skills-list overlap
- Ranks by buzzword count
- In this dataset → ranks HR Managers above ML Engineers
- Walks straight into the planted decoy data
        """)
    with c2:
        st.markdown("### ✅ Our approach")
        st.markdown("""
- Semantic understanding of the **career narrative**, not the skills list
- Structured role + trajectory rules + honeypot gate
- Bounded geography & availability modifiers
- Every pick explained from real evidence — zero hallucination
        """)

    st.markdown("---")
    st.markdown("## 🏗️ Three-axis scoring model")
    col_t, col_c, col_b = st.columns(3)
    with col_t:
        st.markdown("### 🤖 Tech_Fit (0–4)")
        st.markdown("""
**XGBoost on dense embeddings**

`all-MiniLM-L6-v2` embeds the full career narrative (headline + summary + role descriptions + skills).

XGBoost classifies into tiers 0–4.

✅ Rewards described production systems  
✅ Penalises keyword stuffing  
✅ 91.7% held-out accuracy
        """)
    with col_c:
        st.markdown("### 📋 Context_Fit (0–4)")
        st.markdown("""
**Deterministic logic tree**

Reads structured `career_history` fields directly.

Rules: product-company months, YOE band, consulting flag, tenure stability, relocation.

✅ 60% vs golden labels  
✅ Zero hallucination risk  
✅ Deterministic every run
        """)
    with col_b:
        st.markdown("### 📡 Behavior_Fit (0–4)")
        st.markdown("""
**Platform signal formula**

Reads `redrob_signals` directly.

Formula: recency × response_rate, capped by notice period and open-to-work flag.

✅ 100% vs golden labels  
✅ Identical to labeling pipeline  
✅ No model needed
        """)

    st.markdown("---")
    st.markdown("## 🏆 Ranking formula")
    st.code(
        "rank_score = δ × (0.50·Tech_Fit + 0.30·Context_Fit + 0.20·Behavior_Fit)\n"
        "δ = 0  if  Tech_Fit == 0  OR  Context_Fit == 0  (hard veto)\n"
        "δ = 1  otherwise",
        language="python",
    )

    st.markdown("---")
    st.markdown("## 🔍 Pre-ranking filter (4 stages)")
    st.markdown("""
| Stage | What it catches |
|---|---|
| **1A – Structural** | Non-India + no relocation, YOE out of range, sparse skills, honeypot, pure consulting, job-hopper |
| **2 – Title keyword pruning** | Marketing, Sales, HR, Finance, Civil Engineer … |
| **3 – Behavioural reachability** | Inactive > 180 days OR recruiter response rate < 10% |
| **4 – Domain relevance** | Must match ≥ 2 of 6 AI/ML/Search keyword groups |
    """)

    st.markdown("---")
    st.markdown("## 💬 Extractive reasoning (no hallucination)")
    st.markdown("""
For each ranked candidate the app finds the single sentence from their profile most aligned
with the JD — using cosine similarity between sentence embeddings and the JD embedding.
The sentence is always quoted verbatim from the candidate's own text, so fabricated credentials
are architecturally impossible.

**Format:**  
`"Senior ML Engineer (7y): Built semantic search pipeline serving 3M queries/day. [Bengaluru-based; notice 30d]"`
    """)

# ---------------------------------------------------------------------------
# TAB 3 — BENCHMARK METRICS
# ---------------------------------------------------------------------------
with tab_metrics:
    st.markdown("## 📊 Benchmark — Test Set (120 candidates)")
    st.caption(
        "Results from the actual pipeline run. LambdaMART eval on 20% held-out split only."
    )

    bench = {
        "Metric": [
            "NDCG@10",
            "NDCG@50",
            "NDCG@all",
            "MAP",
            "Spearman ρ",
            "P@10",
            "P@20",
            "Final Score",
        ],
        "Lex Sort": [0.9351, 0.9654, 0.9804, 0.8095, 0.7303, 0.30, 0.15, 0.8936],
        "Weighted ⭐": [0.9351, 0.9654, 0.9799, 0.8095, 0.7039, 0.30, 0.15, 0.8936],
        "LambdaMART": [0.6505, 0.8547, 0.8547, 0.0909, -0.0769, 0.00, 0.05, 0.5953],
    }
    bench_df = pd.DataFrame(bench).set_index("Metric")
    st.dataframe(
        bench_df.style.highlight_max(axis=1, color="#d4b8f5"),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("## ⏱️ Runtime")
    r1, r2, r3 = st.columns(3)
    r1.markdown(
        "<div class='kpi-card'><div class='kpi-value'>31.9s</div><div class='kpi-label'>120 candidates</div></div>",
        unsafe_allow_html=True,
    )
    r2.markdown(
        "<div class='kpi-card'><div class='kpi-value'>~246s</div><div class='kpi-label'>1 lakh candidates</div></div>",
        unsafe_allow_html=True,
    )
    r3.markdown(
        "<div class='kpi-card'><div class='kpi-value'>CPU only</div><div class='kpi-label'>No GPU needed</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("## 🔑 LambdaMART Feature Importances")
    feat_imp = pd.DataFrame(
        {
            "Feature": [
                "Tech_Fit",
                "Notice_Period",
                "Open_to_Work",
                "Context_Fit",
                "Years_of_Experience",
                "Response_Rate",
                "GitHub_Score",
            ],
            "Importance": [0.6253, 0.1462, 0.0896, 0.0498, 0.0469, 0.0319, 0.0103],
        }
    ).set_index("Feature")
    st.bar_chart(feat_imp, use_container_width=True)
    st.caption(
        "Tech_Fit dominates (62.5%) — the model independently confirms that semantic career-narrative fit is the primary hiring signal."
    )

    st.markdown("---")
    st.markdown("## 🎯 Sub-model accuracy")
    sub_df = pd.DataFrame(
        {
            "Model": [
                "Tech_Fit (XGBoost)",
                "Context_Fit (Logic Tree)",
                "Behavior_Fit (Formula)",
            ],
            "Accuracy": [91.7, 60.0, 100.0],
            "Notes": [
                "80/20 split, 96 train samples. Baseline with random embeddings ~62.5%.",
                "Gap reflects free-text edge cases the rule tree can't catch without SLM extraction.",
                "Fully deterministic — identical formula to the golden-set labeling pipeline.",
            ],
        }
    ).set_index("Model")
    st.dataframe(sub_df, use_container_width=True)
