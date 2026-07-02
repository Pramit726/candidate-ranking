"""Tests for candidate-data assembly and the three ranking strategies.

These use tiny hand-built candidates_data lists (no ML models needed) to pin the
veto rule, the weighted formula, and the lexicographical ordering.
"""

import pandas as pd
import pytest

from redrob_ranker.config import W_BEHAVIOR, W_CONTEXT, W_TECH
from redrob_ranker.scoring import (
    build_candidates_data,
    rank_lexicographical,
    rank_nonlinear_weighted,
)


def _cd(cid, tech, ctx, beh, final=0):
    return {
        "candidate_id": cid,
        "Tech_Fit": tech,
        "Context_Fit": ctx,
        "Behavior_Fit": beh,
        "Final_Score_NDCG": final,
    }


# ── build_candidates_data ─────────────────────────────────────────────────────

def test_build_candidates_data_shape(strong_candidate):
    records = [dict(strong_candidate, parsed={"Final_Score_NDCG": 3})]
    cd = build_candidates_data(records, [3], [4], [2])
    assert len(cd) == 1
    row = cd[0]
    assert row["Tech_Fit"] == 3
    assert row["Context_Fit"] == 4
    assert row["Behavior_Fit"] == 2
    assert row["Final_Score_NDCG"] == 3
    assert row["candidate_id"] == strong_candidate["candidate_id"]


# ── weighted nonlinear ────────────────────────────────────────────────────────

def test_weighted_formula_matches_weights():
    cd = [_cd("A", 4, 4, 4)]
    ranked = rank_nonlinear_weighted(cd)
    expected = W_TECH * 4 + W_CONTEXT * 4 + W_BEHAVIOR * 4
    assert ranked.iloc[0]["rank_score"] == pytest.approx(expected)


def test_weighted_veto_zeroes_score_when_tech_zero():
    cd = [_cd("A", 0, 4, 4)]
    ranked = rank_nonlinear_weighted(cd)
    assert ranked.iloc[0]["rank_score"] == 0


def test_weighted_veto_zeroes_score_when_context_zero():
    cd = [_cd("A", 4, 0, 4)]
    ranked = rank_nonlinear_weighted(cd)
    assert ranked.iloc[0]["rank_score"] == 0


def test_weighted_orders_descending():
    cd = [_cd("low", 2, 2, 2), _cd("high", 4, 4, 4), _cd("mid", 3, 3, 3)]
    ranked = rank_nonlinear_weighted(cd)
    assert list(ranked["candidate_id"]) == ["high", "mid", "low"]


# ── lexicographical ───────────────────────────────────────────────────────────

def test_lexicographical_prioritises_tech_then_context():
    cd = [
        _cd("t3c4", 3, 4, 4),
        _cd("t4c1", 4, 1, 1),
        _cd("t4c3", 4, 3, 1),
    ]
    ranked = rank_lexicographical(cd)
    # tech 4 sorts above tech 3; within tech 4, context 3 above context 1
    assert list(ranked["candidate_id"]) == ["t4c3", "t4c1", "t3c4"]


def test_lexicographical_vetoed_sink_to_bottom():
    cd = [_cd("vetoed", 0, 4, 4), _cd("ok", 2, 2, 2)]
    ranked = rank_lexicographical(cd)
    assert list(ranked["candidate_id"]) == ["ok", "vetoed"]
