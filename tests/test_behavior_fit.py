"""Tests for the deterministic Behavior_Fit signal formula.

Pins recency bands, response-rate bands, the weakest-link rule, and the
notice-period / open-to-work caps.
"""

import pytest

from redrob_ranker.features.behavior import _behavior_fit_single


def _sig(**overrides):
    base = {
        "last_active_date": "2026-05-20",   # 7 days before as_of
        "recruiter_response_rate": 0.8,
        "notice_period_days": 0,
        "open_to_work_flag": True,
    }
    base.update(overrides)
    return {"redrob_signals": base}


def test_top_score(as_of_dt):
    # active 7 days ago, response 0.8, open, no notice -> 4
    assert _behavior_fit_single(_sig(), as_of_dt) == 4


def test_recency_bands(as_of_dt):
    # 30 days -> recency band 3 (<=45)
    assert _behavior_fit_single(_sig(last_active_date="2026-04-27"), as_of_dt) == 3
    # 60 days -> recency band 2 (<=90)
    assert _behavior_fit_single(_sig(last_active_date="2026-03-28"), as_of_dt) == 2


def test_response_bands_are_weakest_link(as_of_dt):
    # response 0.4 -> band 2, recency still 4 -> min = 2
    assert _behavior_fit_single(_sig(recruiter_response_rate=0.4), as_of_dt) == 2
    # response 0.2 -> band 1
    assert _behavior_fit_single(_sig(recruiter_response_rate=0.2), as_of_dt) == 1


def test_inactive_six_months_is_zero(as_of_dt):
    # active 200 days ago (>=180) -> recency 0 -> overall 0
    assert _behavior_fit_single(_sig(last_active_date="2025-11-01"), as_of_dt) == 0


def test_open_to_work_false_caps_at_three(as_of_dt):
    # everything strong but not open-to-work -> capped at 3
    assert _behavior_fit_single(_sig(open_to_work_flag=False), as_of_dt) == 3


def test_long_notice_caps_at_two(as_of_dt):
    # notice > 90 caps at 2
    assert _behavior_fit_single(_sig(notice_period_days=120), as_of_dt) == 2


def test_medium_notice_caps_at_three(as_of_dt):
    # notice > 30 caps at 3
    assert _behavior_fit_single(_sig(notice_period_days=60), as_of_dt) == 3


def test_missing_signals_treated_conservatively(as_of_dt):
    # no last_active_date and no response rate -> worst case -> 0
    result = _behavior_fit_single({"redrob_signals": {}}, as_of_dt)
    assert result == 0
