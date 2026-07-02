# Redrob — Evidence-Grounded Candidate Ranker

Top-100 candidate ranking for the Redrob **Intelligent Candidate Discovery &
Ranking** challenge (Senior AI Engineer, Founding Team JD). A transparent,
reproducible, **CPU-only** ranking system that scores every candidate on three
independent axes, ranks them with interchangeable strategies, and emits a
grounded one-sentence reason per candidate.

> **The one insight that drives everything:** in this dataset the `skills` array
> is noisy and the real fit signal lives in the free-text career narrative. So
> Tech_Fit is scored on *semantic embeddings of the career history*, not on
> keyword matches against the skills list.

---

## How it works

The pipeline runs in stages, each isolated in its own module:

```
load + merge  →  pre-filter  →  score (3 axes)  →  rank (3 strategies)  →  evaluate  →  reason  →  CSV
```

- **Tech_Fit** — an XGBoost classifier over `all-MiniLM-L6-v2` embeddings of the
  candidate narrative (headline + summary + role descriptions + skills).
- **Context_Fit** — a deterministic logic tree over structured `career_history`
  fields (product-company months, tenure, consulting flag, relocation).
- **Behavior_Fit** — a deterministic formula over `redrob_signals` (recency,
  recruiter response rate, notice period, open-to-work).
- **Ranking** — three strategies run in parallel: Lexicographical Sort,
  Weighted Nonlinear scoring (`0.5·Tech + 0.3·Context + 0.2·Behavior` with a
  hard veto when Tech or Context is 0), and LambdaMART (`XGBRanker`). The best
  strategy is chosen by a composite NDCG/MAP score.
- **Reasoning** — Extractive Vector Summarization: the candidate profile
  sentence most similar to the JD embedding, plus a structured metadata tag.
  Fully extractive, so no hallucinated credentials are possible.

Full technical detail is in **[docs/DESIGN.md](docs/DESIGN.md)**; the trade-offs
and rejected approaches are in **[docs/CHOICES.md](docs/CHOICES.md)**.

---

## Repository layout

```
src/redrob_ranker/
  config.py        single source of truth for weights / paths / thresholds
  io_utils.py      streaming JSONL loader + golden-label merge
  filters.py       4-stage pre-ranking exclusion / pruning / domain relevance
  features/
    tech.py        Tech_Fit  — XGBoost + all-MiniLM-L6-v2 embeddings
    context.py     Context_Fit — deterministic logic tree
    behavior.py    Behavior_Fit — deterministic signal formula
  scoring.py       build candidate rows + the 3 ranking strategies
  evaluation.py    NDCG@10/50/all, MAP, Spearman, P@10/20 + comparison table
  reasoning.py     evidence-grounded extractive one-liner per candidate
  output.py        final CSV writer
  pipeline.py      stage orchestration (main)
scripts/rank.py    CLI entry point
tests/             deterministic unit tests (pytest)
data/              candidate pool, golden labels, JD, output CSV
docs/              DESIGN.md, CHOICES.md
```

This is a straight refactor of a former single-file script into a package.
**No scoring logic changed** — every function and constant is byte-identical to
the original (verified by an AST-level equivalence check). See the
section → module map in [docs/DESIGN.md](docs/DESIGN.md).

---

## Setup

Requires Python 3.11–3.12, CPU, ~2 GB RAM for the 120-candidate test set.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or install the package itself (exposes the `redrob-rank` command):

```bash
pip install -e .
```

---

## Run it

**One command** (wraps `scripts/rank.py`):

```bash
./run.sh                 # full run
./run.sh --no-ltr        # skip LambdaMART (faster)
```

**Or directly:**

```bash
python scripts/rank.py --data-dir ./data --output ./data/final_ranking.csv --top-n 100
```

**Or via the installed console script:**

```bash
redrob-rank --data-dir ./data --output ./data/final_ranking.csv
```

### Command-line options

| Flag | Default | Meaning |
|---|---|---|
| `--data-dir` | `../data` | Folder with `test_set.jsonl`, `golden_test.jsonl`, `job_description.txt` |
| `--models-dir` | `../models` | Where the LambdaMART model is saved |
| `--output` | `../data/final_ranking.csv` | Output CSV path |
| `--top-n` | `100` | Number of ranked candidates to write |
| `--as-of-date` | `2026-05-27` | Snapshot date for Behavior_Fit recency |
| `--no-ltr` | off | Skip the LambdaMART strategy |
| `--no-filter` | off | Skip the pre-ranking exclusion filters |

> The first run downloads the `all-MiniLM-L6-v2` embedding model (needs network
> once); every later run uses the local cache and is CPU-only / offline.

### Output

`final_ranking.csv` with one row per ranked candidate:

```
candidate_id,rank,score,reasoning
CAND_0042871,1,0.987,"Senior ML Engineer (7y): Built semantic search serving 3M queries/day. [Bengaluru-based; notice 30d]"
...
```

---

## Tests

The unit tests cover the deterministic logic (Context_Fit, Behavior_Fit,
filters, ranking strategies, IO normalisation, reasoning extraction). They need
no ML models or network and run in a couple of seconds:

```bash
pytest -q
```

Expected: **43 passed**.

---

## Data files

Place these in `data/` (bundled in this repo):

| File | Description |
|---|---|
| `test_set.jsonl` | 120 full candidate profiles |
| `golden_test.jsonl` | 120 golden labels (Tech / Context / Behavior / Final) |
| `job_description.txt` | Senior AI Engineer JD (plain text) |

---

## License

MIT.
