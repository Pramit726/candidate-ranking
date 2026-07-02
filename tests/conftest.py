"""Shared pytest fixtures for the Redrob ranker test suite.

Fixtures build small, hand-written candidate records that exercise the
deterministic scoring / filter logic without needing the ML models. This keeps
the whole suite runnable on CPU in a couple of seconds with no network.
"""

import json
from datetime import date
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
AS_OF_DATE = "2026-05-27"


@pytest.fixture
def as_of_date():
    return AS_OF_DATE


@pytest.fixture
def as_of_dt():
    return date.fromisoformat(AS_OF_DATE)


def _candidate(**overrides):
    """A plausible, fully-formed base candidate; override any field per test."""
    base = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "current_title": "Senior ML Engineer",
            "years_of_experience": 6,
            "location": "Bengaluru, Karnataka",
            "country": "India",
            "headline": "ML engineer focused on retrieval and ranking",
            "summary": "Six years building production search and ranking systems.",
        },
        "career_history": [
            {
                "title": "Senior ML Engineer",
                "company": "ProductCo",
                "industry": "SaaS",
                "duration_months": 40,
                "start_date": "2021-01-01",
                "end_date": None,
                "is_current": True,
                "description": (
                    "Built and owned the semantic search and ranking pipeline "
                    "serving 2M queries per day with monthly NDCG evaluation."
                ),
            },
            {
                "title": "ML Engineer",
                "company": "StartupX",
                "industry": "SaaS",
                "duration_months": 30,
                "start_date": "2018-06-01",
                "end_date": "2020-12-01",
                "is_current": False,
                "description": "Developed recommendation models and A/B tested ranking.",
            },
        ],
        "skills": [
            {"name": "Python", "proficiency": "expert", "duration_months": 60},
            {"name": "PyTorch", "proficiency": "advanced", "duration_months": 40},
            {"name": "Retrieval", "proficiency": "advanced", "duration_months": 36},
            {"name": "FAISS", "proficiency": "intermediate", "duration_months": 24},
            {"name": "Ranking", "proficiency": "advanced", "duration_months": 30},
            {"name": "Embeddings", "proficiency": "advanced", "duration_months": 28},
        ],
        "redrob_signals": {
            "last_active_date": "2026-05-20",
            "recruiter_response_rate": 0.8,
            "notice_period_days": 30,
            "open_to_work_flag": True,
            "willing_to_relocate": True,
            "github_activity_score": 55,
        },
    }
    # shallow-merge overrides at the top level
    for k, v in overrides.items():
        base[k] = v
    return base


@pytest.fixture
def make_candidate():
    return _candidate


@pytest.fixture
def strong_candidate():
    return _candidate()


@pytest.fixture
def real_records():
    """First 40 real candidates from the bundled test set (if present)."""
    path = DATA_DIR / "test_set.jsonl"
    if not path.exists():
        pytest.skip(f"{path} not available")
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= 40:
                break
            if line.strip():
                out.append(json.loads(line))
    return out
