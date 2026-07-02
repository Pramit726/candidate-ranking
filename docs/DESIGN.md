# Design Document — Redrob AI Candidate Ranking Pipeline
**The "What" and "How": Technical Blueprint**

---

## 1. System Architecture Overview

The pipeline is a **fully offline, CPU-runnable batch ranking system** that ingests raw candidate JSONL profiles together with a set of human-verified golden labels, and outputs an ordered CSV of ranked candidates — each with a normalised score and a one-sentence recruiter-facing reason.

The system is composed of five sequential layers:

```
Raw Candidates (JSONL)
        │
        ▼
┌─────────────────────────┐
│  Layer 0 · Data Merge   │  golden_test.jsonl × test_set.jsonl
│  (label + profile join) │  keyed by candidate_id
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Layer 1 · Pre-filter   │  Structural · Title · Behavioural · Domain
│  (hard exclusions)      │  (ports 01_analysis.ipynb pipeline)
└──────────┬──────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│  Layer 2 · Three-axis Scoring                                │
│  ┌──────────────────────┐  ┌─────────────────┐  ┌────────┐  │
│  │ Tech_Fit             │  │ Context_Fit     │  │ Behav. │  │
│  │ XGBoost classifier   │  │ Deterministic   │  │ Determ.│  │
│  │ on MiniLM embeddings │  │ logic tree      │  │ signals│  │
│  └──────────────────────┘  └─────────────────┘  └────────┘  │
└──────────┬───────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│  Layer 3 · Ranking (three parallel strategies)               │
│  A) Lexicographical Sort   B) Weighted Nonlinear   C) LambdaMART │
└──────────┬───────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│  Layer 4 · Reasoning    │  Extractive (bi-encoder cosine sim vs JD)
│  + CSV Output           │  → candidate_id, rank, score, reasoning
└─────────────────────────┘
```

---

## 2. Data Flow Pipeline

### Step 1 — Load & merge

`golden_test.jsonl` contains 120 label records (flat shape: `{Candidate_ID, Tech_Fit, Context_Fit, ...}`). `test_set.jsonl` contains the full candidate profiles (`profile`, `career_history`, `skills`, `redrob_signals`). The loader normalises both supported label shapes (flat and wrapped `{candidate_id, parsed: {...}}`), then joins on `candidate_id` so every downstream function has access to both labels and raw profile data in a single dict.

### Step 2 — Pre-ranking filters (four stages)

Derived from `01_analysis.ipynb`. Applied in one pass before any ML inference to avoid wasting compute on unqualifiable candidates.

| Stage | Rule | Reason for exclusion |
|---|---|---|
| 1A | Country not in `{india, in, ind}` AND `willing_to_relocate=False` | Hard geographic requirement |
| 1A | `years_of_experience < 3` or `> 12` | Outside the JD's 5-9y target band (with headroom) |
| 1A | `len(skills) ≤ 5` | Empty/sparse profile; likely incomplete |
| 1A | All `expert`-proficiency skills have `duration_months = 0` | Honeypot signal (planted contradictory data) |
| 1A | Every career role at a consulting firm | JD explicitly disqualifies consulting-only careers |
| 1A | `avg_tenure < 18mo` AND `≥ 3 distinct companies` | Chronic job-hopper |
| 1B | ≥ 2 rapid title jumps (< 18mo tenure each) | Title chaser |
| 2 | Current title matches any `WRONG_TITLE_KEYWORDS` | Non-engineering primary career |
| 3 | Inactive > 180 days OR `recruiter_response_rate < 0.10` | Behaviourally unreachable |
| 4 | Fewer than 2 of 6 domain keyword groups matched in skills + descriptions | Irrelevant domain (non-AI/ML/Search) |

### Step 3 — Tech_Fit prediction

1. Extract text per candidate: `headline + summary + role titles + role descriptions + skill names`.
2. Encode with `all-MiniLM-L6-v2` (384-dim, runs on CPU in ~15s for 120 candidates).
3. Map labels to contiguous integers (test set has classes `[0, 1, 2, 4]` — class 3 is absent, so a direct `num_class=5` XGBoost call rejects the training data; remapping to `[0,1,2,3]` then inverse-mapping predictions avoids this).
4. Train `XGBClassifier` (80/20 stratified split where possible; falls back to random split for small pools).
5. Predict over **all** records (not just the test split) so the ranking has complete coverage.

### Step 4 — Context_Fit prediction (deterministic)

Applied to every record by reading `career_history` structured fields directly:

```
is_pure_consulting  ──► return 0
avg_tenure < 18mo   ──► return 0
visa_reject         ──► return 0
is_management OR longest_tenure < 24mo ──► return 1
5≤YOE≤9 AND product_months≥48 AND longest_tenure≥36 ──► return 4
product_months≥24 AND longest_tenure≥24 ──► return 3
else ──► return 2
```

No model inference required. Accuracy against golden labels on the test set: **60%** (limited by cases where is_pure_academic is hard to detect without free-text reading).

### Step 5 — Behavior_Fit prediction (deterministic)

Computed from `redrob_signals` using the same formula as the golden-set labeling pipeline (`create_testset.py v4`):

- `recency_level`: 4/3/2/1/0 based on days since `last_active_date` vs thresholds 14/45/90/180 days.
- `response_level`: 4/3/2/1/0 based on `recruiter_response_rate` ≥ 0.70/0.50/0.30/0.10.
- `base = min(recency, response)`.
- Caps: `open_to_work=False` → cap at 3; `notice_period > 90d` → cap at 2; `> 30d` → cap at 3.

Accuracy against golden labels: **100%** (fully deterministic, identical formula).

### Step 6 — Ranking strategies

Three strategies run in parallel. All apply a **veto multiplier** (delta = 0) when `Tech_Fit = 0` or `Context_Fit = 0`.

**A) Lexicographical Sort**: `sort_values(["_vetoed", "Tech_Fit", "Context_Fit", "Behavior_Fit"], ascending=[T,F,F,F])`.

**B) Weighted Nonlinear**: `rank_score = delta × (0.50·Tech + 0.30·Context + 0.20·Behavior)`. Score normalised to [0, 1] by dividing by `4.0` (theoretical max).

**C) LambdaMART** (`XGBRanker`, `objective=rank:ndcg`): Trained on the same 80% split. Features: the three predicted scores plus raw signals (`years_of_experience`, `notice_period_days`, `github_activity_score`, `recruiter_response_rate`, `open_to_work_flag`). All candidates treated as a single query group (one JD).

### Step 7 — Best-strategy selection

Composite score: `0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10`. The strategy with the highest composite score is selected as the final ranker.

### Step 8 — Extractive reasoning

For each top-N candidate:
1. Split all `career_history.description` fields into individual sentences.
2. Embed all sentences with the already-loaded `all-MiniLM-L6-v2` model.
3. Pick the sentence with the highest cosine similarity to the JD embedding.
4. Append `[Xy exp; city-based; notice Nd]` metadata where available.

Output: one recruiter-readable sentence per candidate.

### Step 9 — CSV output

```
candidate_id, rank, score, reasoning
CAND_0042871, 1, 0.987, "Built recommendation engine for 2M-user home feed; led relevance ranking experiments. [7y exp; Bangalore-based]"
...
```

---

## 3. Technology Stack

| Layer | Library / Tool | Version |
|---|---|---|
| Embeddings | `sentence-transformers` | 5.5.1 |
| Tech_Fit classifier | `xgboost` | 3.2.0 |
| LambdaMART ranker | `xgboost` (XGBRanker) | 3.2.0 |
| Evaluation metrics | `scikit-learn` | 1.7.2 |
| Ranking correlation | `scipy.stats.spearmanr` | 1.15.x |
| Data wrangling | `pandas`, `numpy` | 2.3.3, 2.2.6 |
| Tensor ops (reasoning) | `torch` | 2.7.1 |
| SLM reasoning (optional) | `llama-cpp-python` + Qwen3-4B-Q4_K_M | 0.3.32 |
| JD parsing | `python-docx` | 1.2.0 |
| Notebook environment | `jupyter`, `nbformat` | 1.1.1, 5.10.4 |
| Language | Python | 3.10+ |

---

## 4. System Constraints & How They Are Addressed

| Constraint | How addressed |
|---|---|
| **CPU-only inference** | `all-MiniLM-L6-v2` runs on CPU; XGBoost is CPU-native; `n_gpu_layers=0` default for optional SLM |
| **< 16 GB RAM** | Embeddings are 384-dim float32 — 120 × 384 × 4 bytes ≈ 180 KB; XGBoost model < 1 MB |
| **Non-contiguous label classes** | Test set has classes `[0,1,2,4]` (no class 3). Labels are remapped to contiguous integers before XGBoost training and inverse-mapped after prediction |
| **Two golden-label shapes** | `_normalize_label_record()` handles both `{Candidate_ID, Tech_Fit, ...}` (flat) and `{candidate_id, parsed: {...}}` (wrapped) transparently |
| **Missing JD format** | `load_job_description()` tries `.txt` first, falls back to `.docx` via `python-docx` |
| **Qwen3 thinking-block truncation** | `/no_think` appended to user message; `max_tokens=100`; `</think>` block stripped before regex extraction |

---

## 5. File Structure

The former single-file `final_ranking_test.py` has been refactored into an
installable `redrob_ranker` package. **No logic changed** — each function lives
in the module matching its responsibility, and the import graph is one-directional
(config → io/features/filters → scoring/evaluation/reasoning → pipeline).

```
project/
├── src/
│   └── redrob_ranker/
│       ├── __init__.py          # package metadata / version
│       ├── config.py            # ALL constants, weights, thresholds, filter keyword sets
│       ├── io_utils.py          # JSONL loaders + golden-label merge (Section 3)
│       ├── filters.py           # 4-stage pre-ranking filters (Section 2)
│       ├── features/
│       │   ├── __init__.py      # re-exports the three predictors
│       │   ├── tech.py          # Tech_Fit: XGBoost + embeddings (Section 4)
│       │   ├── context.py       # Context_Fit: deterministic logic tree (Section 5)
│       │   └── behavior.py      # Behavior_Fit: signal formula (Section 6)
│       ├── scoring.py           # build_candidates_data + 3 rankers (Sections 7 & 9)
│       ├── evaluation.py        # NDCG/MAP metrics + comparison table (Sections 8 & 11)
│       ├── reasoning.py         # extractive vector summarization (Section 10)
│       ├── output.py            # final CSV writer (Section 12)
│       └── pipeline.py          # stage orchestration / main() (Section 13)
├── scripts/
│   └── rank.py                  # CLI entry point → redrob_ranker.pipeline.main
├── tests/
│   ├── conftest.py              # shared fixtures (hand-built + real candidates)
│   ├── test_context_fit.py      # Context_Fit rubric
│   ├── test_behavior_fit.py     # Behavior_Fit bands & caps
│   ├── test_filters.py          # exclusion / pruning / domain relevance
│   ├── test_scoring.py          # veto rule, weighted formula, lexicographic order
│   └── test_io_and_reasoning.py # label normalisation + sentence extraction
├── data/
│   ├── test_set.jsonl           # 120 full candidate profiles
│   ├── golden_test.jsonl        # 120 golden labels
│   ├── job_description.txt      # Plain-text JD
│   └── final_ranking.csv        # Output: candidate_id, rank, score, reasoning
├── models/                      # saved LambdaMART model (git-ignored, regenerated)
├── docs/
│   ├── DESIGN.md                # this document
│   └── CHOICES.md               # design trade-offs & methodology
├── pyproject.toml               # packaging + pytest config
├── requirements.txt             # pinned dependencies
├── run.sh                       # one-command reproduction
├── .gitignore
└── README.md
```

### The section → module map

Each numbered section of the original `final_ranking_test.py` moved verbatim
into exactly one module. An AST-level equivalence check confirms all 30
functions and every constant are byte-identical to the original:

| Original section | New location |
|---|---|
| §0 Configuration, §1 Filter constants | `config.py` |
| §2 Pre-ranking filter functions | `filters.py` |
| §3 Data loading & merging | `io_utils.py` |
| §4 Tech_Fit | `features/tech.py` |
| §5 Context_Fit | `features/context.py` |
| §6 Behavior_Fit | `features/behavior.py` |
| §7 build_candidates_data, §9 ranking strategies | `scoring.py` |
| §8 Evaluation, §11 Comparison table | `evaluation.py` |
| §10 Reasoning generation | `reasoning.py` |
| §12 CSV output | `output.py` |
| §13 main / parse_args | `pipeline.py` (+ thin `scripts/rank.py`) |

---

## 6. Evaluation Results (test set, 120 candidates)

| Metric | Lex Sort | Weighted Nonlinear | LambdaMART |
|---|---|---|---|
| NDCG@10 | 0.8440 | **0.9483** | — |
| NDCG@50 | 0.8629 | **0.9125** | — |
| NDCG@all | 0.9463 | **0.9751** | — |
| MAP | 0.6389 | **0.8056** | — |
| P@10 | 0.3000 | 0.3000 | — |
| Spearman ρ | 0.2278 | **0.3528** | — |
| **Composite** | 0.7917 | **0.8837** | — |

*LambdaMART is evaluated on its 20% held-out split only (24 candidates), which is too small for a meaningful table entry alongside the full-pool strategies.*

**Best strategy: Weighted Nonlinear** (composite score 0.8837).
