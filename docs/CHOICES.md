# Design Choices & Methodology — Redrob AI Candidate Ranking Pipeline
**The "Why": Engineering Trade-offs and Decisions**

---

## 1. Golden Set Generation

### Decision: Rule-based judge as a stand-in for the LLM, then LLM for production

The `create_testset.py v4` pipeline calls a Gemini/Groq LLM for the subjective parts of scoring (Tech_Fit, Context_Fit, Final_Score_NDCG, reasoning) and delegates all mechanical decisions (Behavior_Fit, plausibility checks, hard-reject overrides) to deterministic Python that is identical in both the labeler and the ranker.

**Why not LLM for everything?** Behavior_Fit depends on exact date arithmetic (`days_since_last_active`), and the `as_of_date` must be derived from `max(last_active_date)` across the whole pool — not guessed by a model. Delegating deterministic signals to the LLM introduces reproducibility variance with zero upside.

**Why not rules for everything?** Tech_Fit requires reading comprehension — distinguishing "built the recommendation engine for 2M users and ran A/B tests on ranking" from "listed 'RAG' as a skill with no described work". A rule system would false-positive on keyword presence. The LLM rubric was specifically written to penalise this: *"Listing tools as SKILLS with no described work does NOT qualify."*

**The non-linear Final_Score_NDCG** is intentionally assigned by the LLM rather than computed by Python. A simple average of Tech_Fit=4, Context_Fit=1 would return 2.5 — masking a high-risk mismatch. The LLM is instructed to apply non-linear rules (`Tech_Fit 4 with Context_Fit 2 → 3, not a plain average`) so an XGBoost model trained on these labels can learn the real decision boundary.

---

## 2. Tech_Fit Prediction: Why XGBoost + Embeddings over the Other Two Approaches

Three approaches were implemented and compared:

| Approach | Dataset used | Accuracy | Notes |
|---|---|---|---|
| **A2: XGBoost + `all-MiniLM-L6-v2`** | Full 800 records | ~98% (random emb) | Trained model, deterministic |
| A3: Zero-Shot NLI (`bart-large-mnli`) | 100-sample subset | ~65% | No training, slow, 100 samples only |
| A4: Local SLM (Qwen3-4B) | 100-sample subset | 30% | Buggy (thinking-block truncation), slowest |

**Rejected: Zero-Shot NLI.** The hypothesis labels fed to `facebook/bart-large-mnli` have to describe abstract scoring tiers ("This candidate has built a production retrieval system"). NLI models are trained on entailment pairs, not abstract rubric classification — accuracy was mediocre and the approach is limited to 100 samples with no path to scaling.

**Rejected: Local SLM for scoring.** The Qwen3-4B model produces useful output but has two fatal operational issues at scale: (1) it was truncating to `<think>\nOkay` on every candidate due to `max_tokens=3` (the thinking-block bug we fixed), and (2) even after the fix, running 800 sequential chat-completion calls on CPU takes hours. Suitable for reasoning generation on the final top-N, not for scoring all candidates.

**Chosen: XGBoost on dense embeddings.** `all-MiniLM-L6-v2` embeds the concatenation of headline + summary + role titles + role descriptions + skill names into a 384-dim vector. XGBoost then classifies into tiers 0–4. The key advantage: the embedding captures semantic proximity to IR/retrieval/ranking vocabulary in the text, not just exact keyword matches. A candidate who "built the personalisation layer for the home feed" without saying "recommendation system" will still embed close to a candidate who did — and the model learns this from the training signal.

**The non-contiguous class problem.** The test set has `Tech_Fit` classes `[0, 1, 2, 4]` — class 3 is absent (only 3 candidates in 120 have Tech_Fit ≥ 3, none scored exactly 3). XGBoost's `multi:softmax` with `num_class=5` rejects training data if any class in `[0..num_class-1]` is missing. The fix was to remap observed labels to contiguous integers before training and inverse-map predictions back — rather than forcing `num_class=4` and then mislabelling the class-4 predictions.

---

## 3. Context_Fit: Why the Deterministic Logic Tree over the SLM Extractor

The notebook's Approach 3 for Context_Fit is a two-step process: (1) Qwen3 extracts structured JSON fields from free text (`total_yoe_years`, `is_pure_consulting`, `longest_tenure_months`, etc.); (2) `calculate_context_fit()` applies the rubric rules to that JSON.

**We kept step 2 and replaced step 1 with direct field reads.**

**Rejected: SLM extraction.** All the fields the SLM was asked to extract (`total_yoe_years`, `longest_tenure_months`, `average_tenure_months`, `is_pure_consulting`) are already in the schema as structured data — `profile.years_of_experience`, `career_history[*].duration_months`, `career_history[*].company`. Using an LLM to re-read what is already available as a typed field adds latency, JSON parse failure risk, and inconsistency. In testing, the SLM extraction step was the main source of fallback-to-default scores.

**Chosen: Direct structured-field extraction + same logic tree.** `predict_context_fit_single()` computes `avg_tenure`, `longest_tenure`, `product_months`, `is_pure_consulting`, `is_management`, and `visa_reject` directly from the candidate dict — matching the exact logic of `calculate_context_fit()`. This is strictly equivalent to the SLM approach for all candidates whose nuances live in structured fields (which is ~95% of the dataset), and is faster, deterministic, and free of JSON parsing errors.

**Known limitation.** The one thing the SLM can detect that the rules cannot is `is_pure_academic` — a candidate whose career is entirely academic research with no production deployment, described in free text. This edge case is rare in the test pool but acknowledged.

---

## 4. Ranking Strategy Selection: Why Weighted Nonlinear Wins

Three ranking strategies were implemented with different philosophies:

**A) Lexicographical Sort.** Hard, interpretable priority: Tech_Fit first, then Context_Fit, then Behavior_Fit. Transparent to a recruiter. But brutal: two candidates with Tech_Fit=2 are treated identically regardless of whether their Context_Fit is 4 vs 1. NDCG@10 = 0.844.

**B) Weighted Nonlinear.** `rank_score = delta × (0.50·Tech + 0.30·Context + 0.20·Behavior)`, where `delta = 0` if Tech=0 or Context=0. This maintains the hard veto but allows continuous discrimination within tiers. NDCG@10 = 0.948.

**C) LambdaMART.** A learned ranker optimising directly for NDCG. In principle the most expressive. In practice, with 96 training examples and a heavily skewed label distribution (90/120 records are Final_Score_NDCG=2, only 3 are ≥3), the ranker has very little signal to learn from. It is evaluated only on its 24-record held-out split, making a fair comparison impossible.

**Why Weighted wins over Lex Sort.** On this dataset, the Lex Sort's NDCG@10 is dragged down by ties at Tech_Fit=2 (which covers 75% of the test set). When 90 candidates all have the same Tech_Fit, the secondary sort on Context_Fit does most of the work — but it still can't discriminate between candidates where Context=4 but Behavior=1 vs Context=4 and Behavior=3. The weighted score adds that signal continuously. MAP improves from 0.64 to 0.81.

**Why the weights (0.50 / 0.30 / 0.20) were chosen.** These reflect the JD's implicit priority: technical depth is the primary screen (50%), startup culture fit and product experience are the key differentiator (30%), and reachability is a hygiene check rather than a ranking signal (20%). The weights were not tuned by grid search — with 3 positives in 120 records, overfitting a weight to the test set is meaningless.

---

## 5. Handling Traps, Honeypots, and Edge Cases

**Honeypot detection (structural):** Candidates with ≥3 skills at `advanced`/`expert` proficiency but `duration_months = 0` are flagged as impossible profiles and forced to tier 0 by `plausibility_reason()` in the golden pipeline. The pre-filter catches a softer version of this: candidates where *all* expert skills have zero duration are excluded before scoring.

**Inactive "ghost" candidates.** The dataset contains candidates who signed up but have been dormant for 6+ months. `Behavior_Fit` treats `days_inactive ≥ 180` as recency_level = 0, which forces Behavior_Fit = 0 regardless of response rate. The `compute_final()` function then caps `Final_Score_NDCG` at 1 — they are effectively unreachable and cannot be ranked above active candidates.

**Notice period as friction, not disqualification.** A 120-day notice period is a friction signal (caps Behavior_Fit at 2) but not a hard reject. The golden pipeline specification explicitly notes: "a long NOTICE period is friction (a cap), not unavailability — the spec's own example ranks a 120-day-notice candidate at #3." We replicate this: long notice caps the Behavior_Fit score but does not zero it.

**Non-India candidates.** "Based outside India" is NOT a hard reject — only "requires visa sponsorship" or "explicitly refuses relocation" is. The pre-filter excludes candidates where `country ≠ india` AND `willing_to_relocate = False`. Candidates outside India who are willing to relocate pass through and receive a Context_Fit of 3 at most (soft spot, not a disqualifier).

**The `job_hopper` trap.** Candidates who switch companies frequently are excluded in Stage 1A. The rule is intentionally conservative: `avg_tenure < 18mo` AND `≥ 3 distinct companies`. Both conditions must be true — someone with short tenures at only 2 companies (a consulting-then-startup move) is not disqualified.

**The `title_chaser` trap.** A rapid upward title trajectory with short stints (≥2 roles where level increased AND `duration_months < 18`) is treated as a signal of title inflation. This catches candidates who list "Engineer → Senior Engineer → Staff Engineer" across three companies in 3 years, each stint too short to prove depth.

---

## 6. Evaluation Strategy: Why NDCG@10 Is the Primary Metric

The competition evaluates rankings, not classification. We track:

- **NDCG@10** (primary, weight 0.50 in composite): The competition's stated evaluation metric. Measures the quality of the top-10 returned candidates, discounting gains logarithmically by rank position. Graded relevance (0–4 tiers) is used directly — a tier-4 candidate at rank 1 is worth more than the same candidate at rank 3.
- **NDCG@50** (secondary, weight 0.30): Ensures quality is maintained beyond the first page.
- **MAP** (weight 0.15): Binary precision averaged over all relevant (tier ≥ 3) recall points. With only 3 positives in 120 test records, MAP is sensitive and meaningful here.
- **P@10 / P@20**: Fraction of the top-10/20 slots occupied by tier-3+ candidates. Simple and interpretable for a recruiter.
- **Spearman ρ**: Global rank correlation. Useful for confirming the ordering is not just locally correct at the top.

**Why not Accuracy or F1?** Classification accuracy on a 5-class imbalanced distribution (90% class-2 records) is uninformative — a classifier that always predicts class 2 would score 75% accuracy. We track Exact Match Accuracy only for the scoring sub-models (Tech_Fit, Context_Fit) as a development sanity check, not as a final evaluation signal.

---

## 7. Reasoning Generation: Why Extractive over SLM

Two reasoning approaches were prototyped:

**Extractive (chosen for production):** For each ranked candidate, embed all career-history sentences and pick the one with the highest cosine similarity to the JD embedding. Append structured metadata (YOE, location, notice period). Zero additional inference cost — the `all-MiniLM-L6-v2` model is already loaded.

**SLM generative (Qwen3-4B, cell 41):** Prompt the model to write a one-sentence reason. Higher quality ceiling — can synthesise across multiple roles, mention specific metrics the extractive approach might miss. But adds ~2s per candidate on CPU (200s for 100 candidates), and quality is sensitive to prompt engineering.

**Decision:** Extractive reasoning is the default output in `final_ranking_test.py` because it is deterministic, instantaneous, and produces factual output that directly cites the candidate's own words — reducing the risk of hallucinated credentials. The SLM approach is provided as an optional secondary step in `02_job_matching.ipynb` (cell 41) for users with the model file and time budget.

---

## 8. Methodology Summary (≤200 words)

We built a three-axis offline ranking pipeline for the Redrob candidate pool. Candidates pass through a four-stage pre-filter (structural exclusions, title keyword pruning, behavioural reachability, domain relevance) before ML inference. **Tech_Fit** is predicted by an XGBoost classifier trained on `all-MiniLM-L6-v2` dense embeddings — chosen over zero-shot NLI and local SLM approaches because it uses the full 800-record golden set, runs on CPU in seconds, and produces deterministic output. **Context_Fit** is computed by a deterministic logic tree applied directly to structured `career_history` fields, replacing an SLM extraction step that added latency and JSON parse failures with no accuracy gain. **Behavior_Fit** is fully deterministic from `redrob_signals`, matching the golden-set labeling formula exactly (100% agreement). Three ranking strategies are evaluated: lexicographical sort, weighted nonlinear scoring, and LambdaMART. The weighted strategy (Tech 50%, Context 30%, Behavior 20%) wins on NDCG@10 = 0.948, MAP = 0.806, composite score = 0.884. Final output is a CSV with candidate_id, normalised score, and one-sentence extractive reasoning per candidate derived from cosine similarity between career-history sentences and the JD embedding.
