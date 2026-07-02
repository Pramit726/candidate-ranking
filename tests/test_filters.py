"""Tests for the pre-ranking exclusion / pruning / domain-relevance filters."""

import pytest

from redrob_ranker.filters import (
    _is_job_hopper,
    _is_pure_consulting_career,
    _is_title_chaser,
    _title_level,
    apply_pre_ranking_filters,
)


# ── consulting detection ─────────────────────────────────────────────────────

def test_pure_consulting_career_true():
    career = [
        {"company": "TCS"}, {"company": "Infosys"}, {"company": "Wipro"},
    ]
    assert _is_pure_consulting_career(career) is True


def test_mixed_career_not_pure_consulting():
    career = [{"company": "Infosys"}, {"company": "ProductCo"}]
    assert _is_pure_consulting_career(career) is False


def test_empty_career_not_pure_consulting():
    assert _is_pure_consulting_career([]) is False


# ── job hopper ───────────────────────────────────────────────────────────────

def test_job_hopper_true():
    career = [
        {"company": "A", "duration_months": 10},
        {"company": "B", "duration_months": 12},
        {"company": "C", "duration_months": 8},
    ]
    hopped, _ = _is_job_hopper(career)
    assert hopped is True


def test_job_hopper_false_with_long_tenures():
    career = [
        {"company": "A", "duration_months": 40},
        {"company": "B", "duration_months": 36},
    ]
    hopped, _ = _is_job_hopper(career)
    assert hopped is False


def test_job_hopper_needs_three_companies():
    # short tenures but only 2 companies -> not flagged
    career = [
        {"company": "A", "duration_months": 10},
        {"company": "B", "duration_months": 12},
    ]
    hopped, _ = _is_job_hopper(career)
    assert hopped is False


# ── title level / chaser ─────────────────────────────────────────────────────

def test_title_level_ranks():
    assert _title_level("Junior Engineer") >= 1
    assert _title_level("Principal Engineer") == 5
    assert _title_level("Director of Engineering") == 6
    assert _title_level("nonsense role") == 0


def test_title_chaser_flags_rapid_jumps():
    career = [
        {"title": "Engineer", "duration_months": 10, "start_date": "2020-01-01"},
        {"title": "Senior Engineer", "duration_months": 10, "start_date": "2020-11-01"},
        {"title": "Principal Engineer", "duration_months": 10, "start_date": "2021-09-01"},
    ]
    chased, _ = _is_title_chaser(career)
    assert chased is True


# ── full filter pass ─────────────────────────────────────────────────────────

def test_apply_filters_keeps_strong_candidate(strong_candidate, as_of_date):
    passed, stats = apply_pre_ranking_filters([strong_candidate], as_of_date)
    assert len(passed) == 1
    assert stats["total_in"] == 1
    assert stats["total_out"] == 1


def test_apply_filters_excludes_wrong_title(make_candidate, as_of_date):
    c = make_candidate()
    c["profile"]["current_title"] = "Marketing Manager"
    passed, stats = apply_pre_ranking_filters([c], as_of_date)
    assert len(passed) == 0
    assert c["candidate_id"] in stats["wrong_title"]


def test_apply_filters_excludes_too_junior(make_candidate, as_of_date):
    c = make_candidate()
    c["profile"]["years_of_experience"] = 1
    passed, stats = apply_pre_ranking_filters([c], as_of_date)
    assert len(passed) == 0
    assert c["candidate_id"] in stats["too_junior"]


def test_apply_filters_excludes_inactive(make_candidate, as_of_date):
    c = make_candidate()
    c["redrob_signals"]["last_active_date"] = "2025-01-01"  # >180 days
    passed, stats = apply_pre_ranking_filters([c], as_of_date)
    assert len(passed) == 0
    assert c["candidate_id"] in stats["inactive"]


def test_apply_filters_excludes_honeypot_skill(make_candidate, as_of_date):
    c = make_candidate()
    c["skills"].append(
        {"name": "Ghost", "proficiency": "expert", "duration_months": 0}
    )
    passed, stats = apply_pre_ranking_filters([c], as_of_date)
    assert len(passed) == 0
    assert c["candidate_id"] in stats["honeypot_skill"]
