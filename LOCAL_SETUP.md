# 🚀 Running Redrob AI Candidate Ranker Locally

## Prerequisites

- Python **3.11 or higher** ([download here](https://www.python.org/downloads/))
- Git ([download here](https://git-scm.com/downloads))
- ~5 GB free disk space (for PyTorch + sentence-transformers models)

---

## Step 1 — Clone the repository

```bash
git clone https://huggingface.co/spaces/pramitde726/figuring-out-candidate_ranking
cd figuring-out-candidate_ranking
```

---

## Step 2 — Create a virtual environment

**macOS / Linux:**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` appear at the start of your terminal prompt.

---

## Step 3 — Install dependencies

> ⚠️ Skip `llama-cpp-python` — it requires a C++ compiler and is not used by the Streamlit app.

```bash
pip install --upgrade pip
pip install streamlit>=1.35.0
pip install requests==2.34.2 joblib==1.5.3 numpy==2.2.6 pandas==2.3.3
pip install scikit-learn==1.7.2 xgboost==3.2.0
pip install sentence-transformers==5.5.1 transformers==4.57.6 torch==2.7.1
pip install python-docx==1.2.0
```

Or install everything at once from `requirements.txt` after removing the `llama-cpp-python` line:

```bash
# On macOS/Linux:
grep -v "llama-cpp-python" requirements.txt > requirements_local.txt
pip install -r requirements_local.txt
pip install streamlit>=1.35.0

# On Windows (PowerShell):
Get-Content requirements.txt | Where-Object { $_ -notmatch "llama-cpp-python" } | Set-Content requirements_local.txt
pip install -r requirements_local.txt
pip install streamlit>=1.35.0
```

---

## Step 4 — Add your data files

The app needs these files in a `data/` folder at the project root:

```
figuring-out-candidate_ranking/
├── data/
│   ├── test_set.jsonl          ← candidate profiles
│   ├── golden_test.jsonl       ← ground-truth labels
│   └── job_description.txt     ← (or .docx) job description
├── models/
│   └── xgboost_ranker.json     ← pre-trained model (already in repo)
├── app.py
└── ...
```

If the `data/` folder is missing from the cloned repo, create it and add your files:

```bash
mkdir data
# then copy your .jsonl and job description files into it
```

---

## Step 5 — Run the app

```bash
streamlit run app.py
```

You should see:

```
  You can now view your Streamlit app in your browser.
  Local URL:  http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open **http://localhost:8501** in your browser.

---

## Step 6 — Use the app

1. The sidebar shows default paths `./data` and `./models` — leave them as-is if you followed Step 4
2. Adjust **Top-N candidates** and **As-of date** if needed
3. Toggle **Pre-ranking filters**, **LambdaMART**, and **Extractive reasoning** on/off
4. Click **🚀 Run Ranking Pipeline**
5. Wait ~1 minute for the first run (sentence-transformers downloads a 90MB model on first use)
6. Download results as `final_ranking.csv`

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: streamlit` | Run `pip install streamlit` |
| `FileNotFoundError: data/test_set.jsonl` | Add your data files to the `data/` folder (Step 4) |
| First run is very slow (~1–2 min) | Normal — `all-MiniLM-L6-v2` model is downloading (~90MB). Subsequent runs are fast. |
| `torch` install fails on Python 3.13 | Use Python 3.11 exactly — `torch==2.7.1` requires it |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |
| `venv` not activating on Windows | Run PowerShell as Administrator, then `Set-ExecutionPolicy RemoteSigned` |

---

## Running tests

```bash
pip install pytest==9.1.1
pytest
```

---

## Project structure

```
figuring-out-candidate_ranking/
├── app.py                        ← Streamlit UI entry point
├── requirements.txt              ← Python dependencies
├── Dockerfile                    ← HuggingFace Spaces deployment
├── wide.png                      ← Logo (wide)
├── narrow.png                    ← Logo (sidebar icon)
├── src/
│   └── redrob_ranker/
│       ├── config.py             ← All weights and thresholds
│       ├── io_utils.py           ← Data loading
│       ├── filters.py            ← Pre-ranking filters
│       ├── scoring.py            ← Ranking strategies
│       ├── evaluation.py         ← NDCG/MAP metrics
│       ├── reasoning.py          ← Extractive reasoning
│       ├── pipeline.py           ← CLI pipeline
│       └── features/
│           ├── tech.py           ← Tech_Fit (XGBoost + embeddings)
│           ├── context.py        ← Context_Fit (logic tree)
│           └── behavior.py       ← Behavior_Fit (signal formula)
├── models/
│   └── xgboost_ranker.json       ← Pre-trained LambdaMART model
├── data/                         ← ⚠️ Add your own data files here
├── docs/
│   ├── DESIGN.md
│   └── CHOICES.md
└── tests/
```

---

## System requirements

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB |
| CPU cores | 2 | 4+ |
| Disk space | 5 GB | 8 GB |
| Python | 3.11 | 3.11 |
| OS | Windows 10 / macOS 12 / Ubuntu 20.04 | Any modern OS |

> **No GPU required** — the entire pipeline runs on CPU only.
