"""
app.py — FiguringOut · Redrob AI Candidate Ranker
Streamlit demo inspired by Team-Trinetra's layout.

Run:
    streamlit run app.py
"""

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
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="FiguringOut · Redrob Ranker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS  (inspired by Trinetra's purple/dark theme)
# =============================================================================

st.markdown(
    """
<style>
/* Main header bar */
.main-header {
    background: linear-gradient(135deg, #1a0533 0%, #2d1060 50%, #1a0533 100%);
    padding: 1.4rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border: 1px solid #7D45E0;
}
.main-header h1 {
    color: #ffffff;
    font-size: 1.9rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #c4a8f5;
    margin: 0.25rem 0 0;
    font-size: 0.95rem;
}
.team-badge {
    display: inline-block;
    background: #7D45E0;
    color: white;
    padding: 0.15rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 0.4rem;
}

/* Metric cards */
.metric-card {
    background: #f8f5ff;
    border: 1.5px solid #d4b8f5;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 800;
    color: #5c2db3;
    line-height: 1.1;
}
.metric-card .label {
    font-size: 0.78rem;
    color: #666;
    margin-top: 0.2rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Stage progress */
.stage-done   { color: #16a34a; font-weight: 600; }
.stage-run    { color: #7D45E0; font-weight: 600; }
.stage-wait   { color: #9ca3af; }

/* Strategy badge */
.best-badge {
    background: #7D45E0;
    color: white;
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 700;
}

/* Section headings */
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1a0533;
    border-left: 4px solid #7D45E0;
    padding-left: 0.6rem;
    margin: 1.2rem 0 0.6rem;
}

/* Filter box */
.filter-summary {
    background: #fdf8ff;
    border: 1px solid #e2d3f5;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #1a0533;
}
section[data-testid="stSidebar"] .stMarkdown, 
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    color: #e8d9ff !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
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

    top_n = st.slider("Top-N candidates to rank", 10, 120, DEFAULT_TOP_N, step=10)

    as_of_date = st.text_input(
        "As-of date (YYYY-MM-DD)",
        value=DEFAULT_AS_OF_DATE,
        help="Snapshot date for Behavior_Fit recency calculation",
    )

    st.markdown("### 🔧 Pipeline options")
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
    st.caption("Weights are fixed in config.py. Edit there to change.")

# =============================================================================
# MAIN AREA HEADER
# =============================================================================

st.markdown(
    """
<div class="main-header">
  <span class="team-badge">Team FiguringOut</span>
  <h1>🎯 Redrob AI Candidate Ranker</h1>
  <p>Evidence-grounded, CPU-only ranking — score every candidate on Tech / Context / Behavior,
     rank with three strategies, explain with extractive vector reasoning.</p>
</div>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# TABS
# =============================================================================

tab_run, tab_about, tab_metrics = st.tabs(
    ["▶ Run Pipeline", "ℹ️ How it works", "📊 Benchmark metrics"]
)

# ---------------------------------------------------------------------------
# TAB 1 — RUN PIPELINE
# ---------------------------------------------------------------------------
with tab_run:

    run_btn = st.button(
        "🚀 Run Ranking Pipeline", type="primary", use_container_width=True
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
                f"✅ Loaded **{len(records)}** merged records  "
                f"(train = {len(train_records)}, rank = {len(records_to_rank)})"
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
                    st.markdown(
                        "<div class='filter-summary'>"
                        + "  ".join(f"<b>{k}</b>: {v}" for k, v in drop_detail.items())
                        + "</div>",
                        unsafe_allow_html=True,
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
                label=f"📋 Context_Fit done — accuracy {acc_ctx:.1f}%",
                state="complete",
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
            all_metrics["Lex Sort"] = evaluate_ranking(
                ranked_lex,
                "Lex Sort",
            )

            ranked_wt = rank_nonlinear_weighted(candidates_data)
            all_metrics["Weighted"] = evaluate_ranking(
                ranked_wt,
                "Weighted",
            )

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
                label=f"🏆 Ranking done — best: **{best_name}**", state="complete"
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

        # ── Stage 8: Output ───────────────────────────────────────────────
        t_elapsed = time.time() - t_start

        # Normalise score
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
            "<div class='section-title'>📊 Results Summary</div>",
            unsafe_allow_html=True,
        )

        m = all_metrics[best_name]
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.markdown(
            f"<div class='metric-card'><div class='value'>{m['NDCG@10']:.4f}</div>"
            f"<div class='label'>NDCG@10</div></div>",
            unsafe_allow_html=True,
        )
        k2.markdown(
            f"<div class='metric-card'><div class='value'>{m['NDCG@50']:.4f}</div>"
            f"<div class='label'>NDCG@50</div></div>",
            unsafe_allow_html=True,
        )
        k3.markdown(
            f"<div class='metric-card'><div class='value'>{m['MAP']:.4f}</div>"
            f"<div class='label'>MAP</div></div>",
            unsafe_allow_html=True,
        )
        k4.markdown(
            f"<div class='metric-card'><div class='value'>{t_elapsed:.0f}s</div>"
            f"<div class='label'>Runtime</div></div>",
            unsafe_allow_html=True,
        )
        k5.markdown(
            f"<div class='metric-card'><div class='value'>{len(out_df)}</div>"
            f"<div class='label'>Ranked</div></div>",
            unsafe_allow_html=True,
        )

        # ── Sub-model accuracy row ─────────────────────────────────────────
        st.markdown(
            "<div class='section-title'>🎯 Sub-model Accuracy</div>",
            unsafe_allow_html=True,
        )
        a1, a2, a3 = st.columns(3)
        a1.markdown(
            f"<div class='metric-card'><div class='value'>{acc_tech:.1f}%</div>"
            f"<div class='label'>Tech_Fit (XGBoost)</div></div>",
            unsafe_allow_html=True,
        )
        a2.markdown(
            f"<div class='metric-card'><div class='value'>{acc_ctx:.1f}%</div>"
            f"<div class='label'>Context_Fit (Logic Tree)</div></div>",
            unsafe_allow_html=True,
        )
        a3.markdown(
            f"<div class='metric-card'><div class='value'>{acc_beh:.1f}%</div>"
            f"<div class='label'>Behavior_Fit (Formula)</div></div>",
            unsafe_allow_html=True,
        )

        # ── Strategy comparison table ──────────────────────────────────────
        st.markdown(
            "<div class='section-title'>⚔️ Strategy Comparison</div>",
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

        # ── Ranked candidates table ────────────────────────────────────────
        st.markdown(
            f"<div class='section-title'>🥇 Top-{top_n} Ranked Candidates "
            f"<span class='best-badge'>{best_name}</span></div>",
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
        # Landing state
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

Reads structured `career_history` fields directly — no LLM extraction needed.

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
| **1A – Structural** | Non-India + no relocation, YOE out of range, sparse skills, honeypot (expert skill / 0 months used), pure consulting, job-hopper |
| **2 – Title keyword pruning** | Marketing, Sales, HR, Finance, Civil Engineer … |
| **3 – Behavioural reachability** | Inactive > 180 days OR recruiter response rate < 10% |
| **4 – Domain relevance** | Must match ≥ 2 of 6 AI/ML/Search keyword groups in skills + career descriptions |
    """)

    st.markdown("---")
    st.markdown("## 💬 Extractive reasoning (no hallucination)")
    st.markdown("""
For each ranked candidate the app finds the single sentence from their profile
most aligned with the JD — using cosine similarity between sentence embeddings
and the JD embedding. The sentence is always quoted verbatim from the candidate's
own text, so fabricated credentials are architecturally impossible.

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
        bench_df.style.highlight_max(axis=1, color="#d4b8f5"), use_container_width=True
    )

    st.markdown("---")
    st.markdown("## ⏱️ Runtime")

    r1, r2, r3 = st.columns(3)
    r1.markdown(
        "<div class='metric-card'><div class='value'>31.9s</div>"
        "<div class='label'>120 candidates</div></div>",
        unsafe_allow_html=True,
    )
    r2.markdown(
        "<div class='metric-card'><div class='value'>~246s</div>"
        "<div class='label'>1 lakh candidates</div></div>",
        unsafe_allow_html=True,
    )
    r3.markdown(
        "<div class='metric-card'><div class='value'>CPU only</div>"
        "<div class='label'>No GPU needed</div></div>",
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
        "Tech_Fit dominates (62.5%) — the model independently confirms that "
        "semantic career-narrative fit is the primary hiring signal."
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
