import logging
import os
from pathlib import Path

import pytest

from testset.create_testset import PROVIDERS

logger = logging.getLogger(__name__)


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


def pytest_configure(config):
    config.option.log_cli = True
    config.option.log_cli_level = "INFO"


@pytest.fixture(scope="session")
def job_description():
    return (
        Path(__file__).resolve().parents[1] / "data" / "job_description.txt"
    ).read_text(encoding="utf-8")


@pytest.fixture(scope="session", autouse=True)
def inject_job_description(job_description):
    import tests.test_create_testset as test_module

    original_score_candidate = test_module.score_candidate

    def score_candidate_with_job_description(provider, candidate, **kwargs):
        import json

        kwargs.setdefault("job_description", job_description)
        result = original_score_candidate(provider, candidate, **kwargs)
        candidate_id = (
            candidate.get("candidate_id")
            or candidate.get("Candidate_ID")
            or candidate.get("id")
            or "UNKNOWN"
        )

        # Print formatted result for every test case
        parsed = result.get("parsed", {})
        if parsed:
            logger.info(
                f"\n{'='*80}\n"
                f"TEST RESULT: {candidate_id}\n"
                f"{'='*80}\n"
                f"Provider: {result.get('provider')} | Model: {result.get('model')}\n"
                f"Tech_Fit: {parsed.get('Tech_Fit')} | "
                f"Context_Fit: {parsed.get('Context_Fit')} | "
                f"Behavior_Fit: {parsed.get('Behavior_Fit')}\n"
                f"Final_Score_NDCG: {parsed.get('Final_Score_NDCG')} | "
                f"Binary_Label_MAP: {parsed.get('Binary_Label_MAP')} | "
                f"Profile_Strength_Score: {parsed.get('Profile_Strength_Score')}\n"
                f"\nTech_Reasoning: {parsed.get('Tech_Reasoning')}\n"
                f"Context_Reasoning: {parsed.get('Context_Reasoning')}\n"
                f"Behavior_Reasoning: {parsed.get('Behavior_Reasoning')}\n"
                f"\nFull Result JSON:\n{json.dumps(result, indent=2)}\n"
                f"{'='*80}\n"
            )

        logger.info(
            "Scored %s: Tech=%s, Context=%s, Behavior=%s, Final=%s",
            candidate_id,
            parsed.get("Tech_Fit"),
            parsed.get("Context_Fit"),
            parsed.get("Behavior_Fit"),
            parsed.get("Final_Score_NDCG"),
        )
        return result

    test_module.score_candidate = score_candidate_with_job_description
    yield
    test_module.score_candidate = original_score_candidate
