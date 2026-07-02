"""Tests for IO helpers (label normalisation, merging) and reasoning helpers."""

import json

import pytest

from redrob_ranker.io_utils import _normalize_label_record


# ── label record normalisation ───────────────────────────────────────────────

def test_normalize_wrapped_shape():
    rec = {"candidate_id": "CAND_1", "parsed": {"Tech_Fit": 3}}
    cid, parsed = _normalize_label_record(rec)
    assert cid == "CAND_1"
    assert parsed == {"Tech_Fit": 3}


def test_normalize_flat_shape():
    rec = {"Candidate_ID": "CAND_2", "Tech_Fit": 2, "Context_Fit": 3}
    cid, parsed = _normalize_label_record(rec)
    assert cid == "CAND_2"
    assert parsed["Tech_Fit"] == 2


def test_normalize_unrecognised_returns_none():
    cid, parsed = _normalize_label_record({"foo": "bar"})
    assert cid is None
    assert parsed is None


# ── reasoning sentence extraction ────────────────────────────────────────────

def test_best_jd_sentence_mines_all_sources(strong_candidate, monkeypatch):
    """_best_jd_sentence should collect from career_history, summary, and
    headline. We stub the embedder so no model is needed."""
    import numpy as np

    from redrob_ranker import reasoning

    # Stub embed_model.encode + util.cos_sim to deterministically pick index 0
    class _StubModel:
        def encode(self, sents, convert_to_tensor=False):
            # return one row per sentence
            n = len(sents) if isinstance(sents, list) else 1
            return np.zeros((n, 4))

    def _fake_cos_sim(a, b):
        # b has one row per candidate sentence; make the first the best
        import numpy as np
        n = b.shape[0]
        scores = np.zeros((1, n))
        scores[0, 0] = 1.0
        return scores

    monkeypatch.setattr(reasoning.util, "cos_sim", _fake_cos_sim)
    monkeypatch.setattr(reasoning.torch, "argmax", lambda x: type("T", (), {"item": lambda self: 0})())

    jd_emb = np.zeros((1, 4))
    sent = reasoning._best_jd_sentence(strong_candidate, jd_emb, _StubModel())
    # first career-history sentence is the best per our stub
    assert "semantic search" in sent.lower()


def test_best_jd_sentence_handles_empty_profile(monkeypatch):
    import numpy as np

    from redrob_ranker import reasoning

    class _StubModel:
        def encode(self, sents, convert_to_tensor=False):
            return np.zeros((1, 4))

    empty = {"profile": {}, "career_history": []}
    jd_emb = np.zeros((1, 4))
    sent = reasoning._best_jd_sentence(empty, jd_emb, _StubModel())
    assert "Limited" in sent
