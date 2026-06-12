"""
Automated tests for the Redrob AI golden-set scoring pipeline.

These tests run real candidate profiles through an LLM provider
(Gemini or Groq) using golden_set_pipeline.score_candidate(), then
assert that the structural/range/consistency validation passes AND
that the returned scores match the expected golden values for each
test case.

Because LLM outputs can have minor variance, exact-score assertions
are marked with pytest.mark.xfail-friendly tolerances where noted —
but for the "happy path" cases the spec's scoring rules are tight
enough that we assert exact expected values.

Usage:
    export GEMINI_API_KEY=...   # or GROQ_API_KEY
    pytest test_golden_set_pipeline.py --provider gemini -v
    pytest test_golden_set_pipeline.py --provider groq -v

Add --live to actually call the API (otherwise tests are skipped
unless the relevant API key env var is present).
"""

import json
import os

import pytest
from golden_set_pipeline import PROVIDERS, REQUIRED_FIELDS, score_candidate

# ---------------------------------------------------------------------------
# pytest CLI options
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--provider",
        action="store",
        default="gemini",
        choices=list(PROVIDERS.keys()),
        help="Which LLM provider to use for live scoring tests.",
    )
    parser.addoption(
        "--model",
        action="store",
        default=None,
        help="Override model name for the chosen provider.",
    )


@pytest.fixture(scope="session")
def provider(request):
    provider_name = request.config.getoption("--provider")
    model = request.config.getoption("--model")

    provider_cls = PROVIDERS[provider_name]

    key_env = "GEMINI_API_KEY" if provider_name == "gemini" else "GROQ_API_KEY"
    if not os.environ.get(key_env):
        pytest.skip(
            f"{key_env} not set; skipping live LLM tests for provider={provider_name}"
        )

    kwargs = {}
    if model:
        kwargs["model"] = model

    return provider_cls(**kwargs)


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

TC01 = {
    "candidate_id": "TC-01",
    "name": "TC-01 The Unicorn",
    "current_location": "Pune, India",
    "willing_to_relocate": True,
    "total_experience_years": 7,
    "career_history": [
        {
            "company": "SearchStartup",
            "title": "Senior ML Engineer",
            "duration_months": 40,
            "description": (
                "Built and owned the end-to-end semantic search system for a product "
                "serving 1M monthly active users, using embedding-based retrieval backed "
                "by Pinecone in production. Designed and ran NDCG-based offline evaluation "
                "pipelines and A/B tests to measure ranking quality improvements. Owned "
                "embedding drift monitoring and scheduled index refresh cycles to keep the "
                "vector index up to date as the catalog changed."
            ),
        },
        {
            "company": "ProductLabs",
            "title": "Machine Learning Engineer",
            "duration_months": 38,
            "description": (
                "Developed recommendation and matching models in Python, deployed to "
                "production serving real users on a consumer product. Worked closely with "
                "the search team on retrieval quality and ranking experiments."
            ),
        },
    ],
    "skills": [
        "Python",
        "Pinecone",
        "NDCG",
        "A/B Testing",
        "Embeddings",
        "XGBoost",
        "Learning to Rank",
        "Docker",
    ],
    "notice_period_days": 15,
    "last_active_date": "2026-06-09",
    "recruiter_response_rate": 0.92,
    "open_to_work_flag": True,
    "profile_views_received_30d": 25,
    "interview_completion_rate": 0.95,
    "offer_acceptance_rate": 0.8,
    "preferred_work_mode": "hybrid",
}

TC01_EXPECTED = {
    "Tech_Fit": 4,
    "Context_Fit": 4,
    "Behavior_Fit": 4,
    "Final_Score_NDCG": 4,
    "Binary_Label_MAP": 1,
}


TC02 = {
    "candidate_id": "TC-02",
    "name": "TC-02 Strong But Not Perfect",
    "current_location": "Hyderabad, India",
    "willing_to_relocate": True,
    "total_experience_years": 5,
    "career_history": [
        {
            "company": "FinTechProduct",
            "title": "ML Engineer",
            "duration_months": 30,
            "description": (
                "Built and deployed a semantic search feature for the company's product "
                "using vector embeddings stored in a vector database, serving real users "
                "in production. Wrote production Python services for embedding generation "
                "and retrieval."
            ),
        },
        {
            "company": "DataApp Inc",
            "title": "Software Engineer (ML)",
            "duration_months": 30,
            "description": (
                "Worked on backend ML services in Python, including model serving "
                "infrastructure and data pipelines for a consumer-facing product."
            ),
        },
    ],
    "skills": [
        "Python",
        "Vector Databases",
        "Embeddings",
        "FastAPI",
        "Docker",
        "PostgreSQL",
    ],
    "notice_period_days": 45,
    "last_active_date": "2026-05-23",
    "recruiter_response_rate": 0.65,
    "open_to_work_flag": True,
    "profile_views_received_30d": 10,
    "interview_completion_rate": 0.7,
    "offer_acceptance_rate": 0.4,
    "preferred_work_mode": "hybrid",
}

TC02_EXPECTED = {
    "Tech_Fit": 3,
    "Context_Fit": 3,
    "Behavior_Fit": 3,
    "Final_Score_NDCG": 3,
    "Binary_Label_MAP": 1,
}


HAPPY_PATH_CASES = [
    pytest.param(TC01, TC01_EXPECTED, id="TC-01-unicorn"),
    pytest.param(TC02, TC02_EXPECTED, id="TC-02-strong-not-perfect"),
]


# ---------------------------------------------------------------------------
# Helper assertions
# ---------------------------------------------------------------------------


def assert_structurally_valid(result):
    """Assert the pipeline produced a parsed, structurally valid response."""
    assert (
        result["parsed"] is not None
    ), f"No parsed JSON returned. Raw: {result.get('raw_response')}"

    parsed = result["parsed"]
    for field in REQUIRED_FIELDS:
        assert field in parsed, f"Missing required field '{field}' in: {parsed}"

    assert (
        result["validation_issues"] == []
    ), f"Validation issues found: {result['validation_issues']}\nParsed: {json.dumps(parsed, indent=2)}"


def assert_scores_match(parsed, expected, score_tolerance=0):
    """
    Assert that the core scoring fields match expected values.

    score_tolerance allows +/- N drift on Tech/Context/Behavior fits
    for less deterministic models (default 0 = exact match required).
    """
    for field in ["Tech_Fit", "Context_Fit", "Behavior_Fit"]:
        diff = abs(parsed[field] - expected[field])
        assert diff <= score_tolerance, (
            f"{field} mismatch: expected {expected[field]}, got {parsed[field]} "
            f"(tolerance={score_tolerance}). Reasoning: "
            f"{parsed.get(field.replace('_Fit', '_Reasoning'))}"
        )

    assert parsed["Final_Score_NDCG"] == expected["Final_Score_NDCG"], (
        f"Final_Score_NDCG mismatch: expected {expected['Final_Score_NDCG']}, "
        f"got {parsed['Final_Score_NDCG']}"
    )

    assert parsed["Binary_Label_MAP"] == expected["Binary_Label_MAP"], (
        f"Binary_Label_MAP mismatch: expected {expected['Binary_Label_MAP']}, "
        f"got {parsed['Binary_Label_MAP']}"
    )

    assert 0 <= parsed["Profile_Strength_Score"] <= 100


def assert_reasoning_cites_evidence(parsed, keywords, field="Tech_Reasoning"):
    """
    Loose check that the reasoning field references at least one of the
    given evidence keywords (case-insensitive substring match).
    """
    text = parsed.get(field, "").lower()
    assert any(
        kw.lower() in text for kw in keywords
    ), f"{field} does not mention any of {keywords}. Got: {parsed.get(field)!r}"


# ---------------------------------------------------------------------------
# Category 1 — Happy Path tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("candidate,expected", HAPPY_PATH_CASES)
def test_happy_path_structural_validity(provider, candidate, expected):
    """The response must be valid, well-formed, and internally consistent."""
    result = score_candidate(provider, candidate, max_retries=3)
    assert_structurally_valid(result)


@pytest.mark.parametrize("candidate,expected", HAPPY_PATH_CASES)
def test_happy_path_scores(provider, candidate, expected):
    """Scores must match the expected golden values exactly."""
    result = score_candidate(provider, candidate, max_retries=3)
    assert_structurally_valid(result)
    assert_scores_match(result["parsed"], expected, score_tolerance=0)


def test_tc01_evidence_based_reasoning(provider):
    """TC-01: Tech_Reasoning should cite the explicit evaluation/retrieval evidence."""
    result = score_candidate(provider, TC01, max_retries=3)
    assert_structurally_valid(result)
    assert_reasoning_cites_evidence(
        result["parsed"],
        keywords=["NDCG", "Pinecone", "index refresh", "embedding drift", "A/B"],
        field="Tech_Reasoning",
    )


def test_tc02_no_eval_framework_caps_tech_fit(provider):
    """
    TC-02: Tech_Fit must be capped at 3 because the profile has production
    semantic search/embeddings but explicitly lacks NDCG/MAP/MRR/A-B-testing
    or retrieval-quality-ownership language.
    """
    result = score_candidate(provider, TC02, max_retries=3)
    assert_structurally_valid(result)
    parsed = result["parsed"]

    assert parsed["Tech_Fit"] == 3, (
        f"Expected Tech_Fit=3 (no eval framework evidence), got {parsed['Tech_Fit']}. "
        f"Reasoning: {parsed.get('Tech_Reasoning')}"
    )

    # Make sure the model isn't hallucinating eval-framework terms into the reasoning
    text = parsed.get("Tech_Reasoning", "").lower()
    for forbidden in ["ndcg", "mrr", " map ", "a/b test"]:
        assert forbidden not in text, (
            f"Tech_Reasoning incorrectly references '{forbidden}' which is not "
            f"present in TC-02's profile: {parsed.get('Tech_Reasoning')!r}"
        )


# ---------------------------------------------------------------------------
# Reasoning-rule compliance (applies to all happy-path cases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("candidate,expected", HAPPY_PATH_CASES)
def test_reasoning_fields_are_concise(provider, candidate, expected):
    """Reasoning fields should be short (<= 2 sentences per spec) -- loose check via sentence count."""
    result = score_candidate(provider, candidate, max_retries=3)
    assert_structurally_valid(result)
    parsed = result["parsed"]

    for field in ["Tech_Reasoning", "Context_Reasoning", "Behavior_Reasoning"]:
        text = parsed[field].strip()
        # crude sentence count via '.' / '!' / '?' terminators
        sentence_count = sum(text.count(c) for c in ".!?")
        assert (
            sentence_count <= 3
        ), f"{field} appears to exceed 2 sentences (found ~{sentence_count} terminators): {text!r}"  # small buffer for decimals like "0.85"
        assert len(text) > 0


# ---------------------------------------------------------------------------
# Offline (no-API) unit tests for validation/consistency logic itself
# ---------------------------------------------------------------------------

from golden_set_pipeline import extract_json, validate_output  # noqa: E402


def test_validate_output_accepts_consistent_record():
    parsed = {
        "Candidate_ID": "TC-01",
        "Tech_Fit": 4,
        "Tech_Reasoning": "x",
        "Context_Fit": 4,
        "Context_Reasoning": "x",
        "Behavior_Fit": 4,
        "Behavior_Reasoning": "x",
        "Final_Score_NDCG": 4,
        "Binary_Label_MAP": 1,
        "Profile_Strength_Score": 95,
    }
    assert validate_output(parsed) == []


def test_validate_output_flags_inconsistent_final_score():
    parsed = {
        "Candidate_ID": "TC-02",
        "Tech_Fit": 3,
        "Tech_Reasoning": "x",
        "Context_Fit": 3,
        "Context_Reasoning": "x",
        "Behavior_Fit": 3,
        "Behavior_Reasoning": "x",
        "Final_Score_NDCG": 4,  # wrong: should be 3
        "Binary_Label_MAP": 1,
        "Profile_Strength_Score": 50,
    }
    issues = validate_output(parsed)
    assert any("Final_Score_NDCG inconsistent" in i for i in issues)


def test_validate_output_flags_veto_violation():
    parsed = {
        "Candidate_ID": "X",
        "Tech_Fit": 4,
        "Tech_Reasoning": "x",
        "Context_Fit": 1,  # veto: Context_Fit <= 1
        "Context_Reasoning": "x",
        "Behavior_Fit": 4,
        "Behavior_Reasoning": "x",
        "Final_Score_NDCG": 3,  # wrong: should be 1 due to veto
        "Binary_Label_MAP": 1,
        "Profile_Strength_Score": 50,
    }
    issues = validate_output(parsed)
    assert any("Final_Score_NDCG inconsistent" in i for i in issues)


def test_validate_output_flags_binary_label_mismatch():
    parsed = {
        "Candidate_ID": "X",
        "Tech_Fit": 2,
        "Tech_Reasoning": "x",
        "Context_Fit": 2,
        "Context_Reasoning": "x",
        "Behavior_Fit": 2,
        "Behavior_Reasoning": "x",
        "Final_Score_NDCG": 2,
        "Binary_Label_MAP": 1,  # wrong: should be 0 since Final < 3
        "Profile_Strength_Score": 50,
    }
    issues = validate_output(parsed)
    assert any("Binary_Label_MAP inconsistent" in i for i in issues)


def test_extract_json_handles_markdown_fences():
    raw = '```json\n{"Candidate_ID": "X", "Tech_Fit": 1}\n```'
    parsed = extract_json(raw)
    assert parsed == {"Candidate_ID": "X", "Tech_Fit": 1}


def test_extract_json_handles_stray_text():
    raw = 'Sure, here is the result:\n{"Candidate_ID": "X", "Tech_Fit": 1}\nHope that helps!'
    parsed = extract_json(raw)
    assert parsed == {"Candidate_ID": "X", "Tech_Fit": 1}
