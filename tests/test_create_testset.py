"""
Automated tests for the Redrob AI golden-set scoring pipeline.

All candidate profiles conform to the Redrob Candidate Profile Schema
(candidate_id: CAND_XXXXXXX, profile, career_history, education,
skills, redrob_signals).

Reference date for activity calculations: 2026-06-14

Usage:
    export GEMINI_API_KEY=...   # or GROQ_API_KEY
    pytest tests/create_testset.py --provider gemini -v
    pytest tests/create_testset.py --provider groq -v
"""

import json
import os

import pytest

from testset.create_testset import (
    PROVIDERS,
    REQUIRED_FIELDS,
    extract_json,
    score_candidate,
    validate_output,
)


@pytest.fixture(scope="session")
def provider(request):
    provider_name = request.config.getoption("--provider")
    model = request.config.getoption("--model")
    provider_cls = PROVIDERS[provider_name]
    key_env = "GEMINI_API_KEY" if provider_name == "gemini" else "GROQ_API_KEY"
    if not os.environ.get(key_env):
        pytest.skip(
            f"{key_env} not set — skipping live LLM tests for provider={provider_name}"
        )
    kwargs = {}
    if model:
        kwargs["model"] = model
    return provider_cls(**kwargs)


# ---------------------------------------------------------------------------
# Shared helper: build a minimal valid redrob_signals block
# ---------------------------------------------------------------------------


def make_signals(
    notice_period_days,
    last_active_date,
    recruiter_response_rate,
    open_to_work_flag=True,
    profile_views_received_30d=10,
    applications_submitted_30d=2,
    avg_response_time_hours=6.0,
    connection_count=200,
    endorsements_received=30,
    interview_completion_rate=0.8,
    offer_acceptance_rate=0.5,
    preferred_work_mode="hybrid",
    willing_to_relocate=True,
    github_activity_score=50,
    search_appearance_30d=5,
    saved_by_recruiters_30d=2,
    verified_email=True,
    verified_phone=True,
    linkedin_connected=True,
    skill_assessment_scores=None,
    signup_date="2022-01-01",
    salary_min=30,
    salary_max=60,
):
    return {
        "profile_completeness_score": 85,
        "signup_date": signup_date,
        "last_active_date": last_active_date,
        "open_to_work_flag": open_to_work_flag,
        "profile_views_received_30d": profile_views_received_30d,
        "applications_submitted_30d": applications_submitted_30d,
        "recruiter_response_rate": recruiter_response_rate,
        "avg_response_time_hours": avg_response_time_hours,
        "skill_assessment_scores": skill_assessment_scores or {"Python": 88},
        "connection_count": connection_count,
        "endorsements_received": endorsements_received,
        "notice_period_days": notice_period_days,
        "expected_salary_range_inr_lpa": {"min": salary_min, "max": salary_max},
        "preferred_work_mode": preferred_work_mode,
        "willing_to_relocate": willing_to_relocate,
        "github_activity_score": github_activity_score,
        "search_appearance_30d": search_appearance_30d,
        "saved_by_recruiters_30d": saved_by_recruiters_30d,
        "interview_completion_rate": interview_completion_rate,
        "offer_acceptance_rate": offer_acceptance_rate,
        "verified_email": verified_email,
        "verified_phone": verified_phone,
        "linkedin_connected": linkedin_connected,
    }


def make_education(institution, degree, field, start_year, end_year, tier="tier_2"):
    return [
        {
            "institution": institution,
            "degree": degree,
            "field_of_study": field,
            "start_year": start_year,
            "end_year": end_year,
            "grade": None,
            "tier": tier,
        }
    ]


def make_skill(name, proficiency="advanced", endorsements=10, duration_months=36):
    return {
        "name": name,
        "proficiency": proficiency,
        "endorsements": endorsements,
        "duration_months": duration_months,
    }


# ---------------------------------------------------------------------------
# TC-01 — The Unicorn
# Expected: Tech=4, Context=4, Behavior=4, Final=4, MAP=1
# ---------------------------------------------------------------------------

TC01 = {
    "candidate_id": "CAND_0000001",
    "profile": {
        "anonymized_name": "Candidate TC01",
        "headline": "Senior ML Engineer — Search & Ranking Systems",
        "summary": (
            "7 years building and owning production retrieval and ranking systems "
            "at product-focused companies. Expertise in embedding-based search, "
            "NDCG-driven evaluation, and vector index lifecycle management."
        ),
        "location": "Pune, Maharashtra",
        "country": "India",
        "years_of_experience": 7,
        "current_title": "Senior ML Engineer",
        "current_company": "SearchStartup",
        "current_company_size": "51-200",
        "current_industry": "Technology",
    },
    "career_history": [
        {
            "company": "SearchStartup",
            "title": "Senior ML Engineer",
            "start_date": "2021-03-01",
            "end_date": None,
            "duration_months": 40,
            "is_current": True,
            "industry": "Technology",
            "company_size": "51-200",
            "description": (
                "Built and owned the end-to-end semantic search system serving 1M monthly "
                "active users. Designed embedding-based retrieval pipeline backed by Pinecone "
                "in production. Owned embedding drift monitoring and scheduled monthly index "
                "refresh cycles to keep the vector index accurate as the product catalog grew. "
                "Designed and ran NDCG and MAP-based offline evaluation pipelines. Ran A/B "
                "tests to measure ranking quality improvements before each release."
            ),
        },
        {
            "company": "ProductLabs",
            "title": "Machine Learning Engineer",
            "start_date": "2018-01-01",
            "end_date": "2021-02-28",
            "duration_months": 38,
            "is_current": False,
            "industry": "Technology",
            "company_size": "201-500",
            "description": (
                "Built recommendation and candidate matching models in Python deployed to "
                "production serving real users on a consumer platform. Collaborated with the "
                "search team on retrieval quality improvements and hybrid search experiments."
            ),
        },
    ],
    "education": make_education(
        "IIT Bombay", "B.Tech", "Computer Science", 2014, 2018, tier="tier_1"
    ),
    "skills": [
        make_skill("Python", "expert", 40, 84),
        make_skill("Pinecone", "advanced", 20, 40),
        make_skill("NDCG Evaluation", "advanced", 15, 40),
        make_skill("A/B Testing", "advanced", 12, 40),
        make_skill("Embeddings", "expert", 25, 60),
        make_skill("XGBoost", "advanced", 10, 36),
        make_skill("Learning to Rank", "intermediate", 8, 24),
        make_skill("Docker", "advanced", 20, 60),
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "professional"}],
    "redrob_signals": make_signals(
        notice_period_days=15,
        last_active_date="2026-06-10",  # active within 14 days ✓
        recruiter_response_rate=0.92,  # > 0.80 ✓
        open_to_work_flag=True,  # ✓
        profile_views_received_30d=25,
        interview_completion_rate=0.95,
        offer_acceptance_rate=0.80,
    ),
}

TC01_EXPECTED = {
    "Tech_Fit": 4,
    "Context_Fit": 4,
    "Behavior_Fit": 4,
    "Final_Score_NDCG": 4,
    "Binary_Label_MAP": 1,
}

# ---------------------------------------------------------------------------
# TC-02 — Strong but Not Perfect
# Expected: Tech=3, Context=3, Behavior=3, Final=3, MAP=1
# No evaluation frameworks (NDCG/MAP/A-B) in profile → caps Tech at 3
# ---------------------------------------------------------------------------

TC02 = {
    "candidate_id": "CAND_0000002",
    "profile": {
        "anonymized_name": "Candidate TC02",
        "headline": "ML Engineer — Semantic Search & Embeddings",
        "summary": (
            "5 years in production ML at product companies. Shipped semantic "
            "search and embedding-based retrieval to real users. Strong Python "
            "engineering background."
        ),
        "location": "Hyderabad, Telangana",
        "country": "India",
        "years_of_experience": 5,
        "current_title": "ML Engineer",
        "current_company": "FinTechProduct",
        "current_company_size": "201-500",
        "current_industry": "Financial Technology",
    },
    "career_history": [
        {
            "company": "FinTechProduct",
            "title": "ML Engineer",
            "start_date": "2021-07-01",
            "end_date": None,
            "duration_months": 30,
            "is_current": True,
            "industry": "Financial Technology",
            "company_size": "201-500",
            "description": (
                "Built and deployed a semantic search feature for the company's product "
                "using vector embeddings stored in a vector database, serving real users "
                "in production. Wrote Python services for embedding generation and retrieval "
                "query handling."
            ),
        },
        {
            "company": "DataApp Inc",
            "title": "Software Engineer (ML)",
            "start_date": "2019-01-01",
            "end_date": "2021-06-30",
            "duration_months": 30,
            "is_current": False,
            "industry": "Technology",
            "company_size": "51-200",
            "description": (
                "Built backend ML services in Python including model serving infrastructure "
                "and data pipelines for a consumer-facing product."
            ),
        },
    ],
    "education": make_education(
        "BITS Pilani", "B.E.", "Computer Science", 2015, 2019, tier="tier_1"
    ),
    "skills": [
        make_skill("Python", "expert", 30, 60),
        make_skill("Vector Databases", "advanced", 15, 30),
        make_skill("Embeddings", "advanced", 15, 30),
        make_skill("FastAPI", "advanced", 20, 48),
        make_skill("Docker", "advanced", 20, 48),
        make_skill("PostgreSQL", "intermediate", 10, 36),
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "professional"}],
    "redrob_signals": make_signals(
        notice_period_days=45,  # 31–60 ✓
        last_active_date="2026-05-23",  # within 45 days ✓
        recruiter_response_rate=0.65,  # > 0.60 ✓
        open_to_work_flag=True,
        interview_completion_rate=0.70,
        offer_acceptance_rate=0.40,
    ),
}

TC02_EXPECTED = {
    "Tech_Fit": 3,
    "Context_Fit": 3,
    "Behavior_Fit": 3,
    "Final_Score_NDCG": 3,
    "Binary_Label_MAP": 1,
}

# ---------------------------------------------------------------------------
# TC-03 — Tech Unicorn Killed by Behavior (Inactive 8 months)
# Expected: Tech=4, Behavior=0 → Veto fires → Final=1, MAP=0
# ---------------------------------------------------------------------------

TC03 = {
    "candidate_id": "CAND_0000003",
    "profile": {
        "anonymized_name": "Candidate TC03",
        "headline": "Principal ML Engineer — Ranking & Retrieval",
        "summary": "Deep expertise in production search and ranking systems.",
        "location": "Bengaluru, Karnataka",
        "country": "India",
        "years_of_experience": 8,
        "current_title": "Principal ML Engineer",
        "current_company": "RankingCo",
        "current_company_size": "201-500",
        "current_industry": "Technology",
    },
    "career_history": [
        {
            "company": "RankingCo",
            "title": "Principal ML Engineer",
            "start_date": "2020-01-01",
            "end_date": None,
            "duration_months": 42,
            "is_current": True,
            "industry": "Technology",
            "company_size": "201-500",
            "description": (
                "Owned the end-to-end product ranking system for an e-commerce platform "
                "with 5M daily active users. Built hybrid retrieval pipelines combining "
                "BM25 and dense embeddings. Designed NDCG and MRR evaluation frameworks "
                "and managed embedding drift monitoring with weekly index refresh cycles."
            ),
        },
    ],
    "education": make_education(
        "NIT Trichy", "B.Tech", "Computer Science", 2012, 2016, tier="tier_2"
    ),
    "skills": [
        make_skill("Python", "expert", 40, 96),
        make_skill("BM25", "advanced", 20, 42),
        make_skill("NDCG", "advanced", 18, 42),
        make_skill("Embeddings", "expert", 30, 60),
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "professional"}],
    "redrob_signals": make_signals(
        notice_period_days=30,
        last_active_date="2025-10-01",  # 8+ months ago → Behavior=0 ✓
        recruiter_response_rate=0.05,  # < 0.10 → Behavior=0 ✓
        open_to_work_flag=False,
    ),
}

TC03_EXPECTED = {
    "Tech_Fit": 4,
    "Context_Fit": 4,
    "Behavior_Fit": 0,
    "Final_Score_NDCG": 1,  # veto fires
    "Binary_Label_MAP": 0,
}

# ---------------------------------------------------------------------------
# TC-04 — Tech Unicorn Killed by Context (IT services only career)
# Expected: Tech=4, Context=0 → Veto fires → Final=1, MAP=0
# ---------------------------------------------------------------------------

TC04 = {
    "candidate_id": "CAND_0000004",
    "profile": {
        "anonymized_name": "Candidate TC04",
        "headline": "AI Engineer — Search and Recommendation Systems",
        "summary": (
            "Built production search and ranking systems throughout career. "
            "Strong retrieval and evaluation background."
        ),
        "location": "Chennai, Tamil Nadu",
        "country": "India",
        "years_of_experience": 7,
        "current_title": "AI Engineer",
        "current_company": "Infosys",
        "current_company_size": "10001+",
        "current_industry": "IT Services",
    },
    "career_history": [
        {
            "company": "Infosys",
            "title": "AI Engineer",
            "start_date": "2021-01-01",
            "end_date": None,
            "duration_months": 30,
            "is_current": True,
            "industry": "IT Services",
            "company_size": "10001+",
            "description": (
                "Built semantic search system for a client project using Qdrant and "
                "embedding-based retrieval. Implemented NDCG-based evaluation pipeline "
                "and managed vector index refresh for the client's product catalog."
            ),
        },
        {
            "company": "TCS",
            "title": "Senior Software Engineer",
            "start_date": "2017-06-01",
            "end_date": "2020-12-31",
            "duration_months": 43,
            "is_current": False,
            "industry": "IT Services",
            "company_size": "10001+",
            "description": (
                "Delivered recommendation module for a banking client using Python "
                "and ML models. Worked on retrieval pipelines as part of a client engagement."
            ),
        },
    ],
    "education": make_education(
        "Anna University", "B.E.", "Computer Science", 2013, 2017, tier="tier_3"
    ),
    "skills": [
        make_skill("Python", "expert", 35, 84),
        make_skill("Qdrant", "advanced", 15, 30),
        make_skill("NDCG", "advanced", 10, 30),
        make_skill("Embeddings", "advanced", 20, 42),
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "professional"}],
    "redrob_signals": make_signals(
        notice_period_days=20,
        last_active_date="2026-06-08",
        recruiter_response_rate=0.85,
        open_to_work_flag=True,
    ),
}

TC04_EXPECTED = {
    "Tech_Fit": 4,
    "Context_Fit": 0,  # entire career at TCS + Infosys only → hard reject
    "Behavior_Fit": 4,
    "Final_Score_NDCG": 0,  # ctx==0 → tier 0 (hard reject, not veto-to-1)
    "Binary_Label_MAP": 0,
}

# ---------------------------------------------------------------------------
# TC-05 — Veto via Context=1 (Architecture role, US-based, willing to relocate)
# Expected: Context=1 → Veto fires → Final=1, MAP=0
# ---------------------------------------------------------------------------

TC05 = {
    "candidate_id": "CAND_0000005",
    "profile": {
        "anonymized_name": "Candidate TC05",
        "headline": "AI Solutions Architect",
        "summary": (
            "Solutions architect with ML background. Moved into architecture "
            "and consulting role 2 years ago. Based in the US, open to relocating to India."
        ),
        "location": "San Francisco, CA",
        "country": "USA",
        "years_of_experience": 9,
        "current_title": "AI Solutions Architect",
        "current_company": "BigConsultingFirm",
        "current_company_size": "5001-10000",
        "current_industry": "Consulting",
    },
    "career_history": [
        {
            "company": "BigConsultingFirm",
            "title": "AI Solutions Architect",
            "start_date": "2024-01-01",
            "end_date": None,
            "duration_months": 29,
            "is_current": True,
            "industry": "Consulting",
            "company_size": "5001-10000",
            "description": (
                "Lead architecture reviews and design technical AI strategy for enterprise "
                "clients. No hands-on implementation or production code ownership."
            ),
        },
        {
            "company": "ProductCo",
            "title": "Senior ML Engineer",
            "start_date": "2019-06-01",
            "end_date": "2023-12-31",
            "duration_months": 54,
            "is_current": False,
            "industry": "Technology",
            "company_size": "201-500",
            "description": (
                "Built ML pipelines and backend services in Python for a "
                "recommendation product serving real users."
            ),
        },
    ],
    "education": make_education(
        "University of Michigan", "M.S.", "Computer Science", 2013, 2015
    ),
    "skills": [
        make_skill("Python", "advanced", 20, 60),
        make_skill("System Design", "expert", 15, 30),
        make_skill("Machine Learning", "advanced", 20, 60),
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "native"}],
    "redrob_signals": make_signals(
        notice_period_days=120,  # > 90 → Behavior=1
        last_active_date="2026-02-01",  # 4+ months ago → Behavior=1
        recruiter_response_rate=0.25,  # 0.10-0.30 → Behavior=1
        open_to_work_flag=True,
        willing_to_relocate=True,
    ),
}

TC05_EXPECTED = {
    "Tech_Fit": 3,
    "Context_Fit": 1,  # architecture role, no production code 18+ months, outside India
    "Behavior_Fit": 1,  # notice >90, inactive 4+ months, low response rate
    "Final_Score_NDCG": 2,  # round(3×0.5+1×0.3+1×0.2)=round(2.0)=2; no hard reject
    "Binary_Label_MAP": 0,
}

# ---------------------------------------------------------------------------
# TC-06 — Veto via Behavior=1 only (good tech and context, unavailable)
# Expected: Tech=3, Context=3, Behavior=1 → Veto → Final=1, MAP=0
# ---------------------------------------------------------------------------

TC06 = {
    "candidate_id": "CAND_0000006",
    "profile": {
        "anonymized_name": "Candidate TC06",
        "headline": "ML Engineer — Retrieval Systems",
        "summary": "Strong ML engineering background at product companies. Currently locked in.",
        "location": "Mumbai, Maharashtra",
        "country": "India",
        "years_of_experience": 6,
        "current_title": "ML Engineer",
        "current_company": "ProductStartup",
        "current_company_size": "51-200",
        "current_industry": "Technology",
    },
    "career_history": [
        {
            "company": "ProductStartup",
            "title": "ML Engineer",
            "start_date": "2021-01-01",
            "end_date": None,
            "duration_months": 41,
            "is_current": True,
            "industry": "Technology",
            "company_size": "51-200",
            "description": (
                "Built semantic search and embedding-based retrieval features in Python, "
                "deployed to production serving real users on the company's platform."
            ),
        },
        {
            "company": "AnotherProduct",
            "title": "Software Engineer (ML)",
            "start_date": "2018-06-01",
            "end_date": "2020-12-31",
            "duration_months": 30,
            "is_current": False,
            "industry": "Technology",
            "company_size": "201-500",
            "description": (
                "Developed ML-powered backend services in Python for a B2B SaaS product."
            ),
        },
    ],
    "education": make_education(
        "NSIT Delhi", "B.E.", "Computer Science", 2014, 2018, tier="tier_2"
    ),
    "skills": [
        make_skill("Python", "expert", 30, 72),
        make_skill("Embeddings", "advanced", 20, 41),
        make_skill("Vector Databases", "advanced", 15, 30),
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "professional"}],
    "redrob_signals": make_signals(
        notice_period_days=110,  # > 90 → Behavior=1 trigger
        last_active_date="2026-01-20",  # ~5 months ago → Behavior=1
        recruiter_response_rate=0.20,  # 0.10–0.30 → Behavior=1
        open_to_work_flag=False,
    ),
}

TC06_EXPECTED = {
    "Tech_Fit": 3,
    "Context_Fit": 3,
    "Behavior_Fit": 1,  # notice >90, inactive ~5 months, low response rate
    "Final_Score_NDCG": 3,  # round(3×0.5+3×0.3+1×0.2)=round(2.6)=3; beh=1 not 0 so no cap
    "Binary_Label_MAP": 1,
}

# ---------------------------------------------------------------------------
# TC-07 — LangChain Keyword Trap
# Expected: Tech=2 (not 3 or 4) — keywords only, no production evidence
# ---------------------------------------------------------------------------

TC07 = {
    "candidate_id": "CAND_0000007",
    "profile": {
        "anonymized_name": "Candidate TC07",
        "headline": "AI Engineer — RAG & LangChain Specialist",
        "summary": "Building AI chatbots and RAG systems using LangChain, Pinecone, and OpenAI.",
        "location": "Bengaluru, Karnataka",
        "country": "India",
        "years_of_experience": 3,
        "current_title": "AI Engineer",
        "current_company": "ChatbotStartup",
        "current_company_size": "11-50",
        "current_industry": "Technology",
    },
    "career_history": [
        {
            "company": "ChatbotStartup",
            "title": "AI Engineer",
            "start_date": "2023-04-01",
            "end_date": None,
            "duration_months": 14,
            "is_current": True,
            "industry": "Technology",
            "company_size": "11-50",
            "description": (
                "Built an AI chatbot for customer support using LangChain and OpenAI API "
                "with Pinecone for context storage. Developed RAG pipeline to retrieve "
                "relevant documents and feed them into the LLM prompt."
            ),
        },
        {
            "company": "AgencyXYZ",
            "title": "Python Developer",
            "start_date": "2021-01-01",
            "end_date": "2023-03-31",
            "duration_months": 26,
            "is_current": False,
            "industry": "Technology",
            "company_size": "11-50",
            "description": (
                "Built backend APIs in Python and Django for client web applications."
            ),
        },
    ],
    "education": make_education(
        "Pune University", "B.E.", "Information Technology", 2017, 2021, tier="tier_3"
    ),
    "skills": [
        make_skill("LangChain", "advanced", 12, 14),
        make_skill("Pinecone", "intermediate", 8, 14),
        make_skill("OpenAI API", "advanced", 12, 14),
        make_skill("RAG", "advanced", 10, 14),
        make_skill("Python", "advanced", 20, 40),
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "conversational"}],
    "redrob_signals": make_signals(
        notice_period_days=30,
        last_active_date="2026-06-05",
        recruiter_response_rate=0.75,
        open_to_work_flag=True,
    ),
}

TC07_EXPECTED = {
    "Tech_Fit": 2,  # LangChain + Pinecone mentions without retrieval ownership
    "Context_Fit": 2,
    "Behavior_Fit": 4,
    "Final_Score_NDCG": 2,
    "Binary_Label_MAP": 0,
}

# ---------------------------------------------------------------------------
# TC-08 — Keyword-Stuffed Skills Section (career history is vague)
# Expected: Tech=2 — skills list has NDCG/BM25/LTR but work history is generic
# ---------------------------------------------------------------------------

TC08 = {
    "candidate_id": "CAND_0000008",
    "profile": {
        "anonymized_name": "Candidate TC08",
        "headline": "ML Engineer — Search, Ranking, Retrieval",
        "summary": "Experienced in information retrieval, learning-to-rank, and XGBoost.",
        "location": "Hyderabad, Telangana",
        "country": "India",
        "years_of_experience": 5,
        "current_title": "ML Engineer",
        "current_company": "DataCorp",
        "current_company_size": "501-1000",
        "current_industry": "Technology",
    },
    "career_history": [
        {
            "company": "DataCorp",
            "title": "ML Engineer",
            "start_date": "2022-01-01",
            "end_date": None,
            "duration_months": 29,
            "is_current": True,
            "industry": "Technology",
            "company_size": "501-1000",
            "description": (
                "Worked on ML models for data processing tasks. Used Python for "
                "data analysis and model training. Supported team with various AI tasks."
            ),
        },
        {
            "company": "SoftwareHouse",
            "title": "Data Scientist",
            "start_date": "2019-06-01",
            "end_date": "2021-12-31",
            "duration_months": 30,
            "is_current": False,
            "industry": "Technology",
            "company_size": "201-500",
            "description": (
                "Built machine learning models using Python. Worked on data pipelines "
                "and feature engineering for various internal projects."
            ),
        },
    ],
    "education": make_education(
        "JNTU Hyderabad", "B.Tech", "Computer Science", 2015, 2019, tier="tier_3"
    ),
    "skills": [
        # Skills list looks impressive — history does NOT support it
        make_skill("NDCG", "intermediate", 5, 6),
        make_skill("BM25", "intermediate", 4, 6),
        make_skill("Learning to Rank", "intermediate", 3, 6),
        make_skill("XGBoost", "advanced", 10, 30),
        make_skill("Python", "advanced", 25, 59),
        make_skill("MAP", "beginner", 2, 3),
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "professional"}],
    "redrob_signals": make_signals(
        notice_period_days=60,
        last_active_date="2026-04-10",
        recruiter_response_rate=0.50,
        open_to_work_flag=True,
    ),
}

TC08_EXPECTED = {
    "Tech_Fit": 2,  # career descriptions are generic; skills cannot override
    "Context_Fit": 2,
    "Behavior_Fit": 2,
    "Final_Score_NDCG": 2,
    "Binary_Label_MAP": 0,
}

# ---------------------------------------------------------------------------
# TC-09 — Impressive Title, No Technical Evidence in Description
# Expected: Tech=2 max — "Lead AI Engineer - Search & Ranking" but desc is managerial
# ---------------------------------------------------------------------------

TC09 = {
    "candidate_id": "CAND_0000009",
    "profile": {
        "anonymized_name": "Candidate TC09",
        "headline": "Lead AI Engineer — Search and Ranking at BigTechCo",
        "summary": "Led a team of 5 engineers to improve search quality at BigTechCo.",
        "location": "Delhi, India",
        "country": "India",
        "years_of_experience": 8,
        "current_title": "Lead AI Engineer - Search and Ranking",
        "current_company": "BigTechCo",
        "current_company_size": "10001+",
        "current_industry": "Technology",
    },
    "career_history": [
        {
            "company": "BigTechCo",
            "title": "Lead AI Engineer - Search and Ranking",
            "start_date": "2021-03-01",
            "end_date": None,
            "duration_months": 39,
            "is_current": True,
            "industry": "Technology",
            "company_size": "10001+",
            "description": (
                "Led a team of 5 engineers on search quality improvements for the "
                "platform. Conducted design reviews, managed sprint planning, "
                "and coordinated with product managers on search feature roadmap."
            ),
        },
        {
            "company": "MidSizeProduct",
            "title": "ML Engineer",
            "start_date": "2017-07-01",
            "end_date": "2021-02-28",
            "duration_months": 43,
            "is_current": False,
            "industry": "Technology",
            "company_size": "201-500",
            "description": (
                "Built ML pipelines and backend services in Python. Worked on "
                "various ML-driven features for the product."
            ),
        },
    ],
    "education": make_education(
        "Delhi Technological University",
        "B.Tech",
        "Computer Science",
        2013,
        2017,
        tier="tier_2",
    ),
    "skills": [
        make_skill("Search Systems", "expert", 20, 39),
        make_skill("Python", "expert", 30, 80),
        make_skill("Team Leadership", "expert", 15, 39),
        make_skill("Machine Learning", "advanced", 20, 60),
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "professional"}],
    "redrob_signals": make_signals(
        notice_period_days=90,
        last_active_date="2026-03-20",
        recruiter_response_rate=0.55,
        open_to_work_flag=True,
    ),
}

TC09_EXPECTED = {
    "Tech_Fit": 2,  # title is impressive; description shows management, not technical work
    "Context_Fit": 2,
    "Behavior_Fit": 2,
    "Final_Score_NDCG": 2,
    "Binary_Label_MAP": 0,
}

# ---------------------------------------------------------------------------
# TC-10 — Weighted Formula: High Tech, Low Behavior
# Tech=4, Context=3, Behavior=2
# Weighted = (4×0.5)+(3×0.3)+(2×0.2) = 2.0+0.9+0.4 = 3.3 → rounds to 3
# Veto does NOT fire (both Context and Behavior > 1)
# Expected: Final=3, MAP=1
# ---------------------------------------------------------------------------

TC10 = {
    "candidate_id": "CAND_0000010",
    "profile": {
        "anonymized_name": "Candidate TC10",
        "headline": "Senior ML Engineer — Search Systems",
        "summary": "Strong search and ranking background. Currently serving a long notice period.",
        "location": "Bengaluru, Karnataka",
        "country": "India",
        "years_of_experience": 6,
        "current_title": "Senior ML Engineer",
        "current_company": "SearchProduct",
        "current_company_size": "201-500",
        "current_industry": "Technology",
    },
    "career_history": [
        {
            "company": "SearchProduct",
            "title": "Senior ML Engineer",
            "start_date": "2020-06-01",
            "end_date": None,
            "duration_months": 36,
            "is_current": True,
            "industry": "Technology",
            "company_size": "201-500",
            "description": (
                "Built and owned the ranking and retrieval system for the company's "
                "product search feature serving 500K users. Managed embedding-based "
                "retrieval backed by Qdrant with monthly index refresh cycles. "
                "Evaluated ranking quality using NDCG and MAP metrics offline."
            ),
        },
        {
            "company": "MidCo",
            "title": "ML Engineer",
            "start_date": "2018-04-01",
            "end_date": "2020-05-31",
            "duration_months": 26,
            "is_current": False,
            "industry": "Technology",
            "company_size": "51-200",
            "description": (
                "Built Python ML pipelines and deployed models to production for "
                "a B2C product."
            ),
        },
    ],
    "education": make_education(
        "VIT Vellore", "B.Tech", "Computer Science", 2014, 2018, tier="tier_2"
    ),
    "skills": [
        make_skill("Python", "expert", 30, 72),
        make_skill("Qdrant", "advanced", 18, 36),
        make_skill("NDCG", "advanced", 15, 36),
        make_skill("Embeddings", "expert", 25, 60),
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "professional"}],
    "redrob_signals": make_signals(
        notice_period_days=85,  # 61–90 → Behavior=2
        last_active_date="2026-04-01",  # within 90 days → Behavior=2
        recruiter_response_rate=0.50,  # 0.40–0.60 → Behavior=2
        open_to_work_flag=True,
    ),
}

TC10_EXPECTED = {
    "Tech_Fit": 4,
    "Context_Fit": 3,
    "Behavior_Fit": 2,
    "Final_Score_NDCG": 3,  # (4×0.5+3×0.3+2×0.2)=3.3 → 3; no veto
    "Binary_Label_MAP": 1,
}

# ---------------------------------------------------------------------------
# TC-13 — Job Hopper (Context=0 via average tenure < 18 months)
# Expected: Context=0 → Veto → Final=1, MAP=0
# ---------------------------------------------------------------------------

TC13 = {
    "candidate_id": "CAND_0000013",
    "profile": {
        "anonymized_name": "Candidate TC13",
        "headline": "ML Engineer — AI Systems",
        "summary": "Passionate ML engineer with diverse startup experience.",
        "location": "Pune, Maharashtra",
        "country": "India",
        "years_of_experience": 5,
        "current_title": "ML Engineer",
        "current_company": "NewStartup",
        "current_company_size": "11-50",
        "current_industry": "Technology",
    },
    "career_history": [
        {
            "company": "NewStartup",
            "title": "ML Engineer",
            "start_date": "2025-09-01",
            "end_date": None,
            "duration_months": 9,
            "is_current": True,
            "industry": "Technology",
            "company_size": "11-50",
            "description": "Building ML features for a B2B SaaS product.",
        },
        {
            "company": "ShortCo2",
            "title": "AI Engineer",
            "start_date": "2024-06-01",
            "end_date": "2025-08-31",
            "duration_months": 14,
            "is_current": False,
            "industry": "Technology",
            "company_size": "51-200",
            "description": "Worked on LLM-based features using LangChain and OpenAI.",
        },
        {
            "company": "ShortCo1",
            "title": "Data Scientist",
            "start_date": "2023-03-01",
            "end_date": "2024-05-31",
            "duration_months": 14,
            "is_current": False,
            "industry": "Technology",
            "company_size": "51-200",
            "description": "Built ML models and data pipelines in Python.",
        },
        {
            "company": "EarlyJob",
            "title": "Software Engineer",
            "start_date": "2021-07-01",
            "end_date": "2023-02-28",
            "duration_months": 19,
            "is_current": False,
            "industry": "Technology",
            "company_size": "11-50",
            "description": "Backend development in Python and Django.",
        },
    ],
    "education": make_education(
        "Symbiosis Institute", "B.Tech", "Computer Science", 2017, 2021, tier="tier_3"
    ),
    "skills": [
        make_skill("Python", "advanced", 20, 60),
        make_skill("Machine Learning", "advanced", 15, 56),
        make_skill("LangChain", "intermediate", 8, 14),
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "professional"}],
    "redrob_signals": make_signals(
        notice_period_days=15,
        last_active_date="2026-06-10",
        recruiter_response_rate=0.82,
        open_to_work_flag=True,
    ),
}

TC13_EXPECTED = {
    "Tech_Fit": 2,
    "Context_Fit": 0,  # average tenure ~14 months across 4 jobs < 18 months → hard reject
    "Behavior_Fit": 4,
    "Final_Score_NDCG": 0,  # ctx==0 → tier 0 (hard reject)
    "Binary_Label_MAP": 0,
}

# ---------------------------------------------------------------------------
# TC-14 — Pure Academic (zero production, research only)
# Expected: Context=0 → Veto → Final=1, MAP=0
# Tech score should also be conservative (research ≠ production retrieval)
# ---------------------------------------------------------------------------

TC14 = {
    "candidate_id": "CAND_0000014",
    "profile": {
        "anonymized_name": "Candidate TC14",
        "headline": "Research Scientist — Neural Information Retrieval",
        "summary": (
            "Published 8 papers on neural ranking and dense retrieval at top venues "
            "including SIGIR and ECIR. PhD from IIT Delhi."
        ),
        "location": "Delhi, India",
        "country": "India",
        "years_of_experience": 7,
        "current_title": "Research Scientist",
        "current_company": "IIT Delhi AI Lab",
        "current_company_size": "1001-5000",
        "current_industry": "Academia",
    },
    "career_history": [
        {
            "company": "IIT Delhi AI Lab",
            "title": "Research Scientist",
            "start_date": "2020-01-01",
            "end_date": None,
            "duration_months": 41,
            "is_current": True,
            "industry": "Academia",
            "company_size": "1001-5000",
            "description": (
                "Conducting research on neural information retrieval and dense "
                "passage retrieval. Published papers on NDCG-optimized ranking models "
                "and bi-encoder architectures. All work is experimental and evaluated "
                "on academic benchmarks (BEIR, MS-MARCO). No production deployments."
            ),
        },
        {
            "company": "IIT Delhi",
            "title": "PhD Researcher",
            "start_date": "2017-07-01",
            "end_date": "2019-12-31",
            "duration_months": 29,
            "is_current": False,
            "industry": "Academia",
            "company_size": "1001-5000",
            "description": (
                "PhD research on learning-to-rank models. Published 5 papers "
                "at SIGIR and ECIR conferences. Academic research environment only."
            ),
        },
    ],
    "education": [
        {
            "institution": "IIT Delhi",
            "degree": "PhD",
            "field_of_study": "Information Retrieval",
            "start_year": 2017,
            "end_year": 2022,
            "grade": None,
            "tier": "tier_1",
        }
    ],
    "skills": [
        make_skill("PyTorch", "expert", 20, 48),
        make_skill("NDCG", "expert", 20, 48),
        make_skill("Dense Retrieval", "expert", 18, 41),
        make_skill("Learning to Rank", "expert", 15, 41),
        make_skill("Python", "expert", 25, 60),
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "professional"}],
    "redrob_signals": make_signals(
        notice_period_days=60,
        last_active_date="2026-05-01",
        recruiter_response_rate=0.70,
        open_to_work_flag=True,
    ),
}

TC14_EXPECTED = {
    "Tech_Fit": 2,  # research NDCG ≠ production NDCG evaluation; no deployment
    "Context_Fit": 0,  # pure academic career, zero production → hard reject
    "Behavior_Fit": 3,
    "Final_Score_NDCG": 0,  # ctx==0 → tier 0 (hard reject)
    "Binary_Label_MAP": 0,
}

# ---------------------------------------------------------------------------
# TC-15 — Accenture + Product Company Mix (Context must NOT be 0)
# Rule: Context=0 only if ENTIRE career is consulting. Not partial.
# Expected: Context=3 (product company history redeems the services stint)
# ---------------------------------------------------------------------------

TC15 = {
    "candidate_id": "CAND_0000015",
    "profile": {
        "anonymized_name": "Candidate TC15",
        "headline": "ML Engineer — Transitioned from Services to Product",
        "summary": (
            "Started career at Accenture, pivoted to product companies 4 years ago. "
            "Currently at a Series B startup building ML features."
        ),
        "location": "Bengaluru, Karnataka",
        "country": "India",
        "years_of_experience": 7,
        "current_title": "ML Engineer",
        "current_company": "GrowthStartup",
        "current_company_size": "51-200",
        "current_industry": "Technology",
    },
    "career_history": [
        {
            "company": "GrowthStartup",
            "title": "ML Engineer",
            "start_date": "2022-06-01",
            "end_date": None,
            "duration_months": 36,
            "is_current": True,
            "industry": "Technology",
            "company_size": "51-200",
            "description": (
                "Building ML-powered features in Python deployed to production. "
                "Implemented semantic search using vector embeddings for product search."
            ),
        },
        {
            "company": "ProductCo",
            "title": "Software Engineer (ML)",
            "start_date": "2020-03-01",
            "end_date": "2022-05-31",
            "duration_months": 26,
            "is_current": False,
            "industry": "Technology",
            "company_size": "201-500",
            "description": (
                "Built backend ML services and data pipelines for a consumer product."
            ),
        },
        {
            "company": "Accenture",
            "title": "Software Engineer",
            "start_date": "2017-07-01",
            "end_date": "2020-02-28",
            "duration_months": 31,
            "is_current": False,
            "industry": "IT Services",
            "company_size": "10001+",
            "description": (
                "Worked on enterprise Java applications for a banking client. "
                "General software engineering, no ML work."
            ),
        },
    ],
    "education": make_education(
        "RV College of Engineering",
        "B.E.",
        "Computer Science",
        2013,
        2017,
        tier="tier_2",
    ),
    "skills": [
        make_skill("Python", "expert", 30, 62),
        make_skill("Embeddings", "advanced", 12, 36),
        make_skill("Machine Learning", "advanced", 20, 62),
        make_skill("FastAPI", "advanced", 15, 36),
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "professional"}],
    "redrob_signals": make_signals(
        notice_period_days=45,
        last_active_date="2026-05-20",
        recruiter_response_rate=0.68,
        open_to_work_flag=True,
    ),
}

TC15_EXPECTED = {
    "Tech_Fit": 3,
    "Context_Fit": 3,  # NOT 0 — only started at services, moved to product companies
    "Behavior_Fit": 3,
    "Final_Score_NDCG": 3,
    "Binary_Label_MAP": 1,
}

# ---------------------------------------------------------------------------
# TC-18 — open_to_work=false blocks Behavior=4
# All other Behavior=4 conditions met, but open_to_work_flag=False
# Expected: Behavior=3 (not 4)
# ---------------------------------------------------------------------------

TC18 = {
    "candidate_id": "CAND_0000018",
    "profile": {
        "anonymized_name": "Candidate TC18",
        "headline": "ML Engineer",
        "summary": "Strong ML engineering background. Not actively looking.",
        "location": "Noida, Uttar Pradesh",
        "country": "India",
        "years_of_experience": 5,
        "current_title": "ML Engineer",
        "current_company": "ProductFirm",
        "current_company_size": "201-500",
        "current_industry": "Technology",
    },
    "career_history": [
        {
            "company": "ProductFirm",
            "title": "ML Engineer",
            "start_date": "2021-04-01",
            "end_date": None,
            "duration_months": 38,
            "is_current": True,
            "industry": "Technology",
            "company_size": "201-500",
            "description": (
                "Built and deployed semantic search features using embedding-based "
                "retrieval in Python. Serving real users on the company's product."
            ),
        },
        {
            "company": "SoftwareProduct",
            "title": "Software Engineer",
            "start_date": "2019-01-01",
            "end_date": "2021-03-31",
            "duration_months": 26,
            "is_current": False,
            "industry": "Technology",
            "company_size": "51-200",
            "description": "Backend engineering in Python for a SaaS product.",
        },
    ],
    "education": make_education(
        "Amity University", "B.Tech", "Computer Science", 2015, 2019, tier="tier_3"
    ),
    "skills": [
        make_skill("Python", "expert", 25, 64),
        make_skill("Embeddings", "advanced", 15, 38),
        make_skill("FastAPI", "advanced", 15, 38),
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "professional"}],
    "redrob_signals": make_signals(
        notice_period_days=20,  # <= 30 ✓
        last_active_date="2026-06-08",  # within 14 days ✓
        recruiter_response_rate=0.88,  # > 0.80 ✓
        open_to_work_flag=False,  # ✗ blocks Behavior=4
    ),
}

TC18_EXPECTED = {
    "Tech_Fit": 3,
    "Context_Fit": 3,
    "Behavior_Fit": 3,  # open_to_work=False prevents Behavior=4
    "Final_Score_NDCG": 3,
    "Binary_Label_MAP": 1,
}

# ---------------------------------------------------------------------------
# TC-19 — Missing Behavior Fields (conservative scoring)
# last_active_date and recruiter_response_rate absent
# Expected: Model must NOT assume best case
# ---------------------------------------------------------------------------

TC19 = {
    "candidate_id": "CAND_0000019",
    "profile": {
        "anonymized_name": "Candidate TC19",
        "headline": "ML Engineer",
        "summary": "ML engineer with product experience.",
        "location": "Pune, Maharashtra",
        "country": "India",
        "years_of_experience": 5,
        "current_title": "ML Engineer",
        "current_company": "SomeProduct",
        "current_company_size": "51-200",
        "current_industry": "Technology",
    },
    "career_history": [
        {
            "company": "SomeProduct",
            "title": "ML Engineer",
            "start_date": "2021-01-01",
            "end_date": None,
            "duration_months": 41,
            "is_current": True,
            "industry": "Technology",
            "company_size": "51-200",
            "description": (
                "Built ML-powered features in Python and deployed to production "
                "for a consumer product."
            ),
        },
    ],
    "education": make_education(
        "Pune University", "B.E.", "Computer Science", 2016, 2020, tier="tier_3"
    ),
    "skills": [
        make_skill("Python", "advanced", 20, 41),
        make_skill("Machine Learning", "advanced", 15, 41),
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "professional"}],
    "redrob_signals": {
        # Intentionally missing: last_active_date, recruiter_response_rate
        "profile_completeness_score": 40,
        "signup_date": "2024-01-01",
        "last_active_date": None,  # missing
        "open_to_work_flag": True,
        "profile_views_received_30d": 2,
        "applications_submitted_30d": 0,
        "recruiter_response_rate": None,  # missing
        "avg_response_time_hours": None,
        "skill_assessment_scores": {},
        "connection_count": 50,
        "endorsements_received": 5,
        "notice_period_days": 60,
        "expected_salary_range_inr_lpa": {"min": 20, "max": 40},
        "preferred_work_mode": "hybrid",
        "willing_to_relocate": True,
        "github_activity_score": -1,
        "search_appearance_30d": 0,
        "saved_by_recruiters_30d": 0,
        "interview_completion_rate": -1,
        "offer_acceptance_rate": -1,
        "verified_email": False,
        "verified_phone": False,
        "linkedin_connected": False,
    },
}

TC19_EXPECTED = {
    # None signals treated as worst-case: last_active=None → inactive 6+ months,
    # response_rate=None → 0.0; both produce Behavior_Fit=0.
    # beh==0 caps Final at min(weighted,1)=1.
    "Behavior_Fit": 0,
    "Final_Score_NDCG": 1,
    "Binary_Label_MAP": 0,
}

# ---------------------------------------------------------------------------
# Test case registry
# ---------------------------------------------------------------------------

ALL_CASES = [
    pytest.param(TC01, TC01_EXPECTED, id="TC-01-unicorn"),
    pytest.param(TC02, TC02_EXPECTED, id="TC-02-strong-not-perfect"),
    pytest.param(TC03, TC03_EXPECTED, id="TC-03-veto-behavior-zero"),
    pytest.param(TC04, TC04_EXPECTED, id="TC-04-veto-context-zero-services"),
    pytest.param(TC05, TC05_EXPECTED, id="TC-05-veto-context-one-architect"),
    pytest.param(TC06, TC06_EXPECTED, id="TC-06-veto-behavior-one-notice"),
    pytest.param(TC07, TC07_EXPECTED, id="TC-07-langchain-keyword-trap"),
    pytest.param(TC08, TC08_EXPECTED, id="TC-08-keyword-stuffed-skills"),
    pytest.param(TC09, TC09_EXPECTED, id="TC-09-impressive-title-no-evidence"),
    pytest.param(
        TC10, TC10_EXPECTED, id="TC-10-weighted-formula-high-tech-low-behavior"
    ),
    pytest.param(TC13, TC13_EXPECTED, id="TC-13-job-hopper"),
    pytest.param(TC14, TC14_EXPECTED, id="TC-14-pure-academic"),
    pytest.param(TC15, TC15_EXPECTED, id="TC-15-accenture-plus-product-not-zero"),
    pytest.param(TC18, TC18_EXPECTED, id="TC-18-open-to-work-false-caps-behavior"),
]

VETO_CASES = [
    pytest.param(TC03, TC03_EXPECTED, id="TC-03-behavior-zero"),
    pytest.param(TC04, TC04_EXPECTED, id="TC-04-context-zero"),
    pytest.param(TC05, TC05_EXPECTED, id="TC-05-context-one"),
    pytest.param(TC13, TC13_EXPECTED, id="TC-13-job-hopper"),
    pytest.param(TC14, TC14_EXPECTED, id="TC-14-pure-academic"),
]

KEYWORD_TRAP_CASES = [
    pytest.param(TC07, TC07_EXPECTED, id="TC-07-langchain"),
    pytest.param(TC08, TC08_EXPECTED, id="TC-08-stuffed-skills"),
    pytest.param(TC09, TC09_EXPECTED, id="TC-09-title-no-evidence"),
]

# ---------------------------------------------------------------------------
# Shared assertion helpers
# ---------------------------------------------------------------------------


def assert_structurally_valid(result):
    assert (
        result["parsed"] is not None
    ), f"No parsed JSON. Raw: {result.get('raw_response')}"
    parsed = result["parsed"]
    for field in REQUIRED_FIELDS:
        assert field in parsed, f"Missing field '{field}' in: {parsed}"
    assert result["validation_issues"] == [], (
        f"Validation issues: {result['validation_issues']}\n"
        f"Parsed: {json.dumps(parsed, indent=2)}"
    )


def assert_scores_match(parsed, expected, tolerance=0):
    for field in ["Tech_Fit", "Context_Fit", "Behavior_Fit"]:
        if field not in expected:
            continue
        diff = abs(parsed[field] - expected[field])
        assert diff <= tolerance, (
            f"{field}: expected {expected[field]}, got {parsed[field]} "
            f"(tolerance={tolerance}). "
            f"Reasoning: {parsed.get(field.replace('_Fit','_Reasoning'))}"
        )
    if "Final_Score_NDCG" in expected:
        assert parsed["Final_Score_NDCG"] == expected["Final_Score_NDCG"], (
            f"Final_Score_NDCG: expected {expected['Final_Score_NDCG']}, "
            f"got {parsed['Final_Score_NDCG']}"
        )
    if "Binary_Label_MAP" in expected:
        assert parsed["Binary_Label_MAP"] == expected["Binary_Label_MAP"], (
            f"Binary_Label_MAP: expected {expected['Binary_Label_MAP']}, "
            f"got {parsed['Binary_Label_MAP']}"
        )
    assert 0 <= parsed["Profile_Strength_Score"] <= 100


def assert_reasoning_cites_evidence(parsed, keywords, field):
    text = parsed.get(field, "").lower()
    assert any(
        kw.lower() in text for kw in keywords
    ), f"{field} does not reference any of {keywords}. Got: {parsed.get(field)!r}"


def assert_reasoning_does_not_mention(parsed, forbidden_terms, field):
    text = parsed.get(field, "").lower()
    for term in forbidden_terms:
        assert term.lower() not in text, (
            f"{field} incorrectly mentions '{term}' which is absent in profile. "
            f"Got: {parsed.get(field)!r}"
        )


# ---------------------------------------------------------------------------
# Category 1 — Structural validity (all cases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("candidate,expected", ALL_CASES)
def test_structural_validity(provider, candidate, expected):
    result = score_candidate(provider, candidate, max_retries=3)
    assert_structurally_valid(result)


# ---------------------------------------------------------------------------
# Category 2 — Score correctness (all cases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("candidate,expected", ALL_CASES)
def test_score_correctness(provider, candidate, expected):
    result = score_candidate(provider, candidate, max_retries=3)
    assert_structurally_valid(result)
    assert_scores_match(result["parsed"], expected, tolerance=0)


# ---------------------------------------------------------------------------
# Category 3 — Veto rule enforcement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("candidate,expected", VETO_CASES)
def test_veto_fires_correctly(provider, candidate, expected):
    """Final_Score_NDCG must match the new formula for low Context/Behavior cases:
    - ctx == 0 (hard reject)  → Final must be 0
    - beh == 0 (unreachable)  → Final must be capped at 1
    - ctx == 1 or beh == 1    → no hard rule; weighted formula applies normally
    """
    result = score_candidate(provider, candidate, max_retries=3)
    assert_structurally_valid(result)
    parsed = result["parsed"]

    if parsed["Context_Fit"] == 0:
        assert parsed["Final_Score_NDCG"] == 0, (
            f"Context hard-reject (ctx=0) must yield Final=0. "
            f"Got Final={parsed['Final_Score_NDCG']}"
        )
        assert parsed["Binary_Label_MAP"] == 0
    elif parsed["Behavior_Fit"] == 0:
        assert parsed["Final_Score_NDCG"] <= 1, (
            f"Behavioral unreachability (beh=0) must cap Final at 1. "
            f"Got Final={parsed['Final_Score_NDCG']}"
        )
        assert parsed["Binary_Label_MAP"] == 0
    else:
        # ctx=1 or beh=1 — weighted formula applies; just confirm expected final
        assert parsed["Final_Score_NDCG"] == expected["Final_Score_NDCG"], (
            f"Final_Score_NDCG: expected {expected['Final_Score_NDCG']}, "
            f"got {parsed['Final_Score_NDCG']} "
            f"(ctx={parsed['Context_Fit']}, beh={parsed['Behavior_Fit']})"
        )


# ---------------------------------------------------------------------------
# Category 4 — Keyword trap: Tech must not exceed 2 on these cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("candidate,expected", KEYWORD_TRAP_CASES)
def test_keyword_trap_tech_capped(provider, candidate, expected):
    result = score_candidate(provider, candidate, max_retries=3)
    assert_structurally_valid(result)
    parsed = result["parsed"]
    assert parsed["Tech_Fit"] <= 2, (
        f"Keyword trap candidate incorrectly scored Tech_Fit={parsed['Tech_Fit']}. "
        f"Reasoning: {parsed.get('Tech_Reasoning')}"
    )


# ---------------------------------------------------------------------------
# Category 5 — Specific reasoning evidence tests
# ---------------------------------------------------------------------------


def test_tc01_reasoning_cites_evaluation_frameworks(provider):
    """TC-01: Tech_Reasoning must reference explicit eval evidence in profile."""
    result = score_candidate(provider, TC01, max_retries=3)
    assert_structurally_valid(result)
    assert_reasoning_cites_evidence(
        result["parsed"],
        keywords=["ndcg", "map", "a/b", "index refresh", "embedding drift", "pinecone"],
        field="Tech_Reasoning",
    )


def test_tc02_tech_reasoning_no_eval_terms(provider):
    """TC-02: Tech_Reasoning must NOT reference NDCG/MAP/A-B — they are absent."""
    result = score_candidate(provider, TC02, max_retries=3)
    assert_structurally_valid(result)
    assert_reasoning_does_not_mention(
        result["parsed"],
        forbidden_terms=[
            "ndcg",
            "mrr",
            " map ",
            "a/b test",
            "index refresh",
            "embedding drift",
        ],
        field="Tech_Reasoning",
    )


def test_tc04_context_reasoning_cites_services_only(provider):
    """TC-04: Context_Reasoning must cite TCS/Infosys as disqualifying signal."""
    result = score_candidate(provider, TC04, max_retries=3)
    assert_structurally_valid(result)
    assert_reasoning_cites_evidence(
        result["parsed"],
        keywords=["tcs", "infosys", "consulting", "services", "entire career"],
        field="Context_Reasoning",
    )


def test_tc14_tech_reasoning_no_production_inference(provider):
    """TC-14: Model must not infer production experience from academic NDCG research."""
    result = score_candidate(provider, TC14, max_retries=3)
    assert_structurally_valid(result)
    parsed = result["parsed"]
    assert parsed["Tech_Fit"] <= 2, (
        f"Academic NDCG research should not yield Tech_Fit > 2. "
        f"Got {parsed['Tech_Fit']}. Reasoning: {parsed.get('Tech_Reasoning')}"
    )


def test_tc15_context_not_zero_for_mixed_career(provider):
    """TC-15: Partial services background must not trigger Context=0 hard reject."""
    result = score_candidate(provider, TC15, max_retries=3)
    # print(f"TC-15 result: {result}")
    assert_structurally_valid(result)
    parsed = result["parsed"]
    # print(f"TC-15 parsed: {parsed}")
    assert parsed["Context_Fit"] >= 2, (
        f"Candidate with 4 years at product companies post-Accenture "
        f"should not receive Context_Fit < 2. Got {parsed['Context_Fit']}. "
        f"Reasoning: {parsed.get('Context_Reasoning')}"
    )


def test_tc18_open_to_work_false_caps_behavior(provider):
    """TC-18: open_to_work=False must prevent Behavior=4 even with all other signals met."""
    result = score_candidate(provider, TC18, max_retries=3)
    assert_structurally_valid(result)
    assert result["parsed"]["Behavior_Fit"] < 4, (
        f"open_to_work_flag=False should cap Behavior below 4. "
        f"Got Behavior_Fit={result['parsed']['Behavior_Fit']}"
    )


def test_tc19_missing_fields_scored_conservatively(provider):
    """TC-19: None last_active_date and None response_rate must be treated as
    worst-case, producing Behavior_Fit=0 (inactive 6+ months equivalent)."""
    result = score_candidate(provider, TC19, max_retries=3)
    assert_structurally_valid(result)
    parsed = result["parsed"]
    assert parsed["Behavior_Fit"] == 0, (
        f"None behavior signals must produce Behavior_Fit=0 (worst-case). "
        f"Got {parsed['Behavior_Fit']}. Reasoning: {parsed.get('Behavior_Reasoning')}"
    )


# ---------------------------------------------------------------------------
# Category 6 — Weighted formula validation
# ---------------------------------------------------------------------------


def test_tc10_weighted_formula_final_score(provider):
    """
    TC-10: Tech=4, Context=3, Behavior=2
    Weighted = (4×0.5)+(3×0.3)+(2×0.2) = 3.3 → 3
    ctx > 0 and beh > 0, so weighted formula applies directly. Final must be 3, MAP=1.
    """
    result = score_candidate(provider, TC10, max_retries=3)
    # print(f"TC-10 result: {result}")
    assert_structurally_valid(result)
    parsed = result["parsed"]
    assert parsed["Final_Score_NDCG"] == 3, (
        f"Weighted formula (4×0.5+3×0.3+2×0.2=3.3→3) should yield Final=3. "
        f"Got {parsed['Final_Score_NDCG']}"
    )
    assert parsed["Binary_Label_MAP"] == 1


# ---------------------------------------------------------------------------
# Category 7 — Reasoning quality (all cases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("candidate,expected", ALL_CASES)
def test_reasoning_conciseness(provider, candidate, expected):
    """Each reasoning field must be at most 2 sentences per spec."""
    result = score_candidate(provider, candidate, max_retries=3)
    assert_structurally_valid(result)
    parsed = result["parsed"]
    for field in ["Tech_Reasoning", "Context_Reasoning", "Behavior_Reasoning"]:
        text = parsed[field].strip()
        # count sentence terminators (small buffer for decimal numbers like 0.85)
        sentence_count = sum(text.count(c) for c in ".!?")
        assert (
            sentence_count <= 3
        ), f"{field} exceeds 2 sentences (~{sentence_count} terminators): {text!r}"
        assert len(text) > 10, f"{field} is suspiciously short: {text!r}"


# ---------------------------------------------------------------------------
# Offline unit tests — no API required
# ---------------------------------------------------------------------------


def test_validate_output_accepts_consistent_record():
    parsed = {
        "Candidate_ID": "CAND_0000001",
        "Tech_Fit": 4,
        "Tech_Reasoning": "x",
        "Context_Fit": 4,
        "Context_Reasoning": "x",
        "Behavior_Fit": 4,
        "Behavior_Reasoning": "x",
        "Final_Score_NDCG": 4,
        "Binary_Label_MAP": 1,
        "Profile_Strength_Score": 92,
    }
    assert validate_output(parsed) == []


def test_validate_output_flags_veto_violation():
    parsed = {
        "Candidate_ID": "CAND_0000004",
        "Tech_Fit": 4,
        "Tech_Reasoning": "x",
        "Context_Fit": 0,
        "Context_Reasoning": "x",  # hard-reject trigger
        "Behavior_Fit": 4,
        "Behavior_Reasoning": "x",
        "Final_Score_NDCG": 3,  # wrong — ctx==0 requires Final=0
        "Binary_Label_MAP": 1,
        "Profile_Strength_Score": 50,
    }
    issues = validate_output(parsed)
    assert any("Final_Score_NDCG" in i for i in issues)


def test_validate_output_flags_binary_mismatch():
    parsed = {
        "Candidate_ID": "CAND_0000007",
        "Tech_Fit": 2,
        "Tech_Reasoning": "x",
        "Context_Fit": 2,
        "Context_Reasoning": "x",
        "Behavior_Fit": 2,
        "Behavior_Reasoning": "x",
        "Final_Score_NDCG": 2,
        "Binary_Label_MAP": 1,  # wrong — should be 0
        "Profile_Strength_Score": 40,
    }
    issues = validate_output(parsed)
    assert any("Binary_Label_MAP" in i for i in issues)


def test_validate_output_flags_profile_strength_out_of_range():
    # Profile_Strength_Score range is enforced at the LLM-output validation stage.
    # validate_llm_output (called inside score_candidate) catches values outside [0,100].
    from testset.create_testset import validate_llm_output

    llm_output = {
        "Tech_Fit": 3,
        "Tech_Reasoning": "x",
        "Context_Fit": 3,
        "Context_Reasoning": "x",
        "Profile_Strength_Score": 150,  # out of range
    }
    issues = validate_llm_output(llm_output)
    assert any("Profile_Strength_Score" in i for i in issues)


def test_extract_json_strips_markdown_fences():
    raw = '```json\n{"Candidate_ID": "CAND_0000001", "Tech_Fit": 4}\n```'
    parsed = extract_json(raw)
    assert parsed == {"Candidate_ID": "CAND_0000001", "Tech_Fit": 4}


def test_extract_json_handles_stray_preamble():
    raw = 'Here is the evaluation:\n{"Candidate_ID": "CAND_0000001", "Tech_Fit": 2}\nDone.'
    parsed = extract_json(raw)
    assert parsed == {"Candidate_ID": "CAND_0000001", "Tech_Fit": 2}


def test_candidate_ids_match_schema_pattern():
    """All test case candidate_ids must follow CAND_XXXXXXX format."""
    import re

    pattern = re.compile(r"^CAND_\d{7}$")
    all_candidates = [
        TC01,
        TC02,
        TC03,
        TC04,
        TC05,
        TC06,
        TC07,
        TC08,
        TC09,
        TC10,
        TC13,
        TC14,
        TC15,
        TC18,
        TC19,
    ]
    for c in all_candidates:
        assert pattern.match(
            c["candidate_id"]
        ), f"candidate_id '{c['candidate_id']}' does not match CAND_XXXXXXX"
