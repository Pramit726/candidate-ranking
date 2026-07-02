"""Tests for the deterministic Context_Fit logic tree.

These pin the exact rubric behaviour so the modularization (or any future
refactor) cannot silently change how Context_Fit is scored.
"""

import pytest

from redrob_ranker.features.context import (
    _is_consulting,
    predict_context_fit_single,
)


def test_ideal_candidate_scores_four(strong_candidate):
    # 6 YOE, >=48 product months, longest tenure 40 -> tier 4
    assert predict_context_fit_single(strong_candidate) == 4


def test_pure_consulting_is_zero(make_candidate):
    c = make_candidate(
        career_history=[
            {
                "title": "Consultant",
                "company": "Infosys",
                "industry": "IT Services",
                "duration_months": 48,
                "start_date": "2019-01-01",
                "end_date": None,
                "is_current": True,
                "description": "Delivered client projects.",
            }
        ]
    )
    assert predict_context_fit_single(c) == 0


def test_job_hopper_avg_tenure_below_18_is_zero(make_candidate):
    c = make_candidate(
        career_history=[
            {"title": "ML Engineer", "company": "A", "industry": "SaaS",
             "duration_months": 10, "start_date": "2023-01-01", "end_date": "2023-11-01",
             "is_current": False, "description": "x"},
            {"title": "ML Engineer", "company": "B", "industry": "SaaS",
             "duration_months": 12, "start_date": "2022-01-01", "end_date": "2022-12-01",
             "is_current": False, "description": "y"},
        ]
    )
    # avg tenure 11 months < 18 -> hard reject
    assert predict_context_fit_single(c) == 0


def test_visa_reject_when_outside_india_and_no_relocation(make_candidate):
    prof = {
        "current_title": "ML Engineer", "years_of_experience": 6,
        "location": "Berlin", "country": "Germany",
        "headline": "", "summary": "",
    }
    c = make_candidate(profile=prof)
    c["redrob_signals"]["willing_to_relocate"] = False
    assert predict_context_fit_single(c) == 0


def test_outside_india_but_willing_to_relocate_not_zero(make_candidate):
    prof = {
        "current_title": "ML Engineer", "years_of_experience": 6,
        "location": "Berlin", "country": "Germany",
        "headline": "", "summary": "",
    }
    c = make_candidate(profile=prof)
    c["redrob_signals"]["willing_to_relocate"] = True
    assert predict_context_fit_single(c) > 0


def test_management_title_caps_at_one(make_candidate):
    c = make_candidate(
        career_history=[
            {"title": "Engineering Manager", "company": "ProductCo", "industry": "SaaS",
             "duration_months": 48, "start_date": "2020-01-01", "end_date": None,
             "is_current": True, "description": "Led a team."},
        ]
    )
    assert predict_context_fit_single(c) == 1


def test_short_longest_tenure_caps_at_one(make_candidate):
    c = make_candidate(
        career_history=[
            {"title": "ML Engineer", "company": "A", "industry": "SaaS",
             "duration_months": 20, "start_date": "2023-01-01", "end_date": None,
             "is_current": True, "description": "x"},
            {"title": "ML Engineer", "company": "B", "industry": "SaaS",
             "duration_months": 22, "start_date": "2021-01-01", "end_date": "2022-11-01",
             "is_current": False, "description": "y"},
        ]
    )
    # avg 21 (>=18, not hard reject) but longest tenure 22 < 24 -> tier 1
    assert predict_context_fit_single(c) == 1


def test_is_consulting_detects_firm_and_industry():
    assert _is_consulting({"company": "TCS", "industry": "SaaS"}) is True
    assert _is_consulting({"company": "ProductCo", "industry": "Consulting"}) is True
    assert _is_consulting({"company": "ProductCo", "industry": "SaaS"}) is False


@pytest.mark.parametrize("yoe,expected_at_least", [(6, 3), (12, 2)])
def test_yoe_band_affects_tier(make_candidate, yoe, expected_at_least):
    c = make_candidate()
    c["profile"]["years_of_experience"] = yoe
    # within-band (5-9) reaches tier 4; outside band drops out of the tier-4 rule
    assert predict_context_fit_single(c) >= expected_at_least
