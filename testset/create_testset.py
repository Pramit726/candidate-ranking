"""
Golden Set Generation Pipeline for Redrob AI Candidate Scoring  (v2)
====================================================================

Produces the HIDDEN GROUND-TRUTH labels (Tech_Fit, Context_Fit, Behavior_Fit,
Final_Score_NDCG, Binary_Label_MAP, Profile_Strength_Score) that the whole
competition is scored against. Because these labels decide who wins, the design
puts every mechanical, rule-based decision in deterministic Python and reserves
the LLM only for the parts that need reading comprehension.

WHAT CHANGED FROM v1
--------------------
1. Revised judging prompt:
   - Tech_Fit score 4 no longer requires buzzword vocabulary (a plain-language
     "built the recommendation engine for 2M users" now qualifies).
   - Context_Fit aligned to the JD (5-9 yrs; "outside India but willing to
     relocate" is NOT an auto-reject; only visa sponsorship / refusal is).
   - Explicit -1 sentinel handling (no GitHub / no offer history = absent,
     never a low score).
2. Behavior_Fit and the final formula moved OUT of the LLM into Python:
   - as_of_date is derived from the pool = max(last_active_date), so recency
     ("inactive >= 6 months", etc.) is exact and reproducible.
   - The old Veto (Context<=1 OR Behavior<=1 -> 1) is gone. New rule:
       * impossible profile / planted honeypot / Context hard-reject -> tier 0
         (tier 0 is now reachable; v1 floored everything at 1, which broke the
         spec's "honeypots forced to tier 0").
       * behavioral UNREACHABILITY (Behavior_Fit == 0) caps the final at 1.
       * a long NOTICE period is friction (a cap), not unavailability — the
         spec's own example ranks a 120-day-notice candidate at #3.
3. Deterministic plausibility/honeypot gate (validated against the pool so it
   only fires on genuine contradictions ~0.03% of rows) + a forced honeypot-ID
   list you control.
4. Candidate_ID is taken from the source record, never trusted from the model.
5. Optional --samples N self-consistency vote on the LLM judgment (temp is
   already 0).

Output JSON keys are UNCHANGED, so any downstream scoring code keeps working.

Usage:
    python create_testset.py --provider gemini \
        --input golden_100.jsonl \
        --job-description job_description.txt \
        --pool candidates.jsonl \
        --honeypots honeypot_ids.txt \
        --output golden_set.jsonl

    # as_of_date is auto-derived from --pool (max last_active_date). You can
    # override it with --as-of-date YYYY-MM-DD. If --pool is omitted it is
    # derived from --input (with a warning, since that may not be the true
    # snapshot).

Environment variables required:
    GEMINI_API_KEY   (for provider=gemini)
    GROQ_API_KEY     (for provider=groq)
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Deterministic-scoring configuration  (tune here, not in the prompt)
# ---------------------------------------------------------------------------

WITHIN_RECENT_DAYS = 14  # "active within ~14 days"
WITHIN_MILD_DAYS = 45  # "active within ~45 days"
WITHIN_REAL_DAYS = 90  # "active within ~90 days"
SIX_MONTHS_DAYS = 180  # ">= 6 months inactive" -> hard 0 on availability

# open_to_work == False -> reachable only passively. Cap the score here.
# Set to 1 to treat "not open to work" as near-unavailable instead.
OPEN_TO_WORK_CAP = 3

# Plausibility gate: a role whose duration_months disagrees with its own dates
# by more than this many months is treated as an impossible (honeypot) profile.
# Validated against the pool: legit rows are within a couple of months; the only
# rows exceeding 6 are off by 100+ months (the "8 years at a 3-year company" trap).
DURATION_SLACK_MONTHS = 6


# ---------------------------------------------------------------------------
# Prompt templates  (LLM judges Tech/Context + reasoning + strength only)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a sharp, skeptical technical recruiter and hiring manager at Redrob AI,
a fast-moving Series A startup. You identify candidates who have demonstrably
built production systems. You evaluate evidence, not buzzwords. You also check
whether a profile is internally consistent, and you refuse to score a profile
whose own data contradicts itself."""

USER_PROMPT_TEMPLATE = """[Instruction]
Review the Candidate JSON profile against the Job Description below and assign
Tech_Fit and Context_Fit, each with a short evidence-based reasoning. Also flag
whether the profile is internally impossible, and give an overall strength
score. You do NOT score Behavior_Fit or compute any final score — those are
derived separately from platform signals.

---

[Reference date]
Treat {as_of_date} as "today" for any judgment about recency or tenure length.

[Reading the data]
- Some numeric signals use -1 to mean "not provided", NOT a low score
  (e.g. github_activity_score = -1 means no GitHub linked; offer_acceptance_rate
  = -1 means no prior offers). Treat -1 as absent/neutral, never as a bad value.
- An empty list or missing field means "not provided", not "zero ability".

---

[Evaluation Principles]
1. Judge only on explicit evidence present in the candidate profile.
2. If evidence is missing, treat the capability as absent. Do not infer it.
3. Never reward keyword presence without evidence of real work.
4. Career history descriptions take precedence over the skills list.
5. Every reasoning statement must cite specific evidence from the profile.
6. Never infer production deployment, real users, retrieval/ranking/search
   ownership, evaluation-framework usage, or hands-on coding recency unless it
   is explicitly stated.
7. Evidence may be in PLAIN LANGUAGE — a candidate need not use terms like
   "RAG", "NDCG", "embedding drift", or "vector database" to get credit. What
   matters is whether the described work demonstrates the capability.

---

[Impossible-profile check]
Set "Impossible": true if the profile's own data contradicts itself or claims
something that cannot be true (e.g. a single role claiming more years than the
company has existed, or "expert" proficiency in many skills with 0 months used).
If true, put the specific contradiction in "Impossible_Reason". Otherwise set
"Impossible": false and "Impossible_Reason": "".

---

[Tech_Fit] — Measures: retrieval, ranking, search, recommendation, matching,
             evaluation, production ML engineering.

Score 4 — The candidate has clearly BUILT, DEPLOYED, or OWNED a search /
  retrieval / ranking / recommendation / matching system that reached real
  users, AND there is evidence of engineering rigor around it (evidence may be
  plain-language). Concretely, career_history shows BOTH:
    A) building / deploying / owning such a system used by real users; AND
    B) at least one form of rigor: measuring or improving result quality
       (offline metrics, A/B tests, relevance tuning — by any name), OR
       operating it at scale (index/freshness/refresh, latency, drift,
       retraining), OR owning iteration on it over time.
  "Built the recommendation engine powering the home feed for 2M users and
  improved engagement via ranking experiments" qualifies for 4 with zero
  framework names. Listing tools (RAG, LangChain, Pinecone, Qdrant, embeddings)
  as SKILLS with no described work does NOT qualify.

Score 3 — Production ML + Python with user-facing semantic search, embeddings,
  or vector retrieval, but without clear evidence of result-quality measurement
  OR ownership of retrieval quality/operations.

Score 2 — General ML, AI, backend, or data engineering. No evidence of
  production retrieval, search, ranking, recommendation, or evaluation systems.

Score 1 — Primary expertise is Computer Vision, Speech, Robotics, or Embedded
  AI with little IR/search/ranking exposure.

Score 0 — Primary career is not software engineering (Marketing, Sales, HR,
  Operations, etc.), regardless of which AI keywords appear in the skills list.

Preferred signals (LoRA, QLoRA, learning-to-rank, XGBoost ranking, distributed
systems) NEVER independently raise a tier; they only strengthen reasoning within
an already-assigned tier.

---

[Context_Fit] — Measures: startup suitability only.
                Technical skill depth must NOT influence this score.

Score 4 — 5-9 years total experience (the JD's band; a strong candidate slightly
  outside is acceptable), of which 4+ years are applied ML/AI at PRODUCT
  companies (not services/consulting). Hands-on builder, stable tenure (at least
  one tenure >= 36 months, or a consistent multi-year track record). Located in
  India, OR outside India but clearly willing to relocate to Pune/Noida and not
  requiring visa sponsorship.

Score 3 — Product-engineering background transitioning into AI/ML, OR strong
  overall with one soft spot (e.g. based outside India but willing to relocate
  without sponsorship; slightly junior/senior for the band). Mostly stable.

Score 2 — Standard engineering profile, no strong signal either way.

Score 1 — Architecture- or management-heavy role with limited recent hands-on
  coding, OR somewhat unstable tenure — but not an auto-reject.

Score 0 (Hard Reject) — Any of:
  - Pure academic or research-only career with no production deployment
  - Entire career at consulting/services firms only (TCS, Infosys, Wipro,
    Cognizant, Capgemini, Accenture, etc.)
  - Average tenure clearly below ~18 months (chronic job-hopping)
  - Requires visa sponsorship
  - Explicitly refuses relocation
  (Being based outside India is NOT by itself a hard reject — only visa
   sponsorship or refusal to relocate is.)

---

[Profile_Strength_Score] — integer 0-100.
Overall strength of this candidate as a hire for this role, using the FULL
0-100 range (do not cluster around round numbers). This is used only to order
candidates who end up sharing a final tier, so spread your scores.

---

[Reasoning Rules]
- Max 2 sentences per reasoning field. Cite specific evidence (named roles,
  dates, durations). Factual only — no assumptions, no praise, no generics.

Bad:  "Strong retrieval background."
Good: "Owns semantic search with monthly index refresh in production at a
       2M-user platform (Senior MLE, 2021-2024, 41 months)."

---

[Output]
Return ONLY valid JSON. No markdown. No code fences. No extra text.

{{
  "Tech_Fit": integer,
  "Tech_Reasoning": "string",
  "Context_Fit": integer,
  "Context_Reasoning": "string",
  "Impossible": boolean,
  "Impossible_Reason": "string",
  "Profile_Strength_Score": integer
}}

---

[Data]

{job_description}

### CANDIDATE JSON
{candidate_json}
"""


# ---------------------------------------------------------------------------
# Provider clients  (unchanged from v1)
# ---------------------------------------------------------------------------


class BaseProvider:
    """Common interface for LLM providers."""

    name = "base"

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        self.model = model
        self.api_key = api_key
        self.extra = kwargs

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class GeminiProvider(BaseProvider):
    """
    Google AI Studio (Gemini API).

    Default model: gemini-2.0-flash (fast & cheap, good for golden-set scale runs).
    Docs: https://ai.google.dev/api/generate-content
    """

    name = "gemini"

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self, model: str = "gemini-2.0-flash", api_key: Optional[str] = None, **kwargs
    ):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Get one from https://aistudio.google.com/apikey"
            )
        super().__init__(model, api_key, **kwargs)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.BASE_URL}/{self.model}:generateContent?key={self.api_key}"

        payload = {
            "systemInstruction": {
                "role": "system",
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generationConfig": {
                # Low temperature for consistent, deterministic-ish judging
                "temperature": 0.0,
                "responseMimeType": "application/json",
            },
        }

        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected Gemini response shape: {data}") from e


class GroqProvider(BaseProvider):
    """
    Groq Cloud (OpenAI-compatible chat completions API), for Llama models.

    Default model: llama-3.3-70b-versatile.
    Docs: https://console.groq.com/docs/quickstart
    """

    name = "groq"

    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Get one from https://console.groq.com/keys"
            )
        super().__init__(model, api_key, **kwargs)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

        resp = requests.post(self.URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected Groq response shape: {data}") from e


PROVIDERS = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
}


# ---------------------------------------------------------------------------
# Deterministic scoring  (Python is the authority for these)
# ---------------------------------------------------------------------------


def _date(s: str) -> date:
    return date.fromisoformat(s)


def _months_between(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def days_inactive(as_of_date: str, last_active_date: str) -> int:
    return (_date(as_of_date) - _date(last_active_date)).days


def behavior_fit(sig: Dict[str, Any], as_of_date: str) -> int:
    """Behavior_Fit 0-4, purely from platform signals. Availability is gated by
    the weakest of (recency, response rate); notice period is a friction cap
    that never produces a 0.

    Missing / None signals are treated as worst-case (conservative scoring):
    - last_active_date None  -> treated as inactive >= 6 months (recency level 0)
    - recruiter_response_rate None -> treated as 0 (response level 0)
    """
    la = sig.get("last_active_date")
    rr = sig.get("recruiter_response_rate")
    notice = sig.get("notice_period_days", 0)
    open_to_work = sig.get("open_to_work_flag", False)

    # Conservative defaults for missing signals
    d = days_inactive(as_of_date, la) if la else SIX_MONTHS_DAYS
    r = rr if rr is not None else 0.0

    recency_lvl = (
        4
        if d <= WITHIN_RECENT_DAYS
        else (
            3
            if d <= WITHIN_MILD_DAYS
            else 2 if d <= WITHIN_REAL_DAYS else 1 if d < SIX_MONTHS_DAYS else 0
        )
    )
    response_lvl = (
        4
        if r >= 0.70
        else 3 if r >= 0.50 else 2 if r >= 0.30 else 1 if r >= 0.10 else 0
    )

    base = min(recency_lvl, response_lvl)
    if not open_to_work:
        base = min(base, OPEN_TO_WORK_CAP)
    if notice > 90:
        base = min(base, 2)
    elif notice > 30:
        base = min(base, 3)
    return base


def make_behavior_reasoning(sig: Dict[str, Any], as_of_date: str, bf: int) -> str:
    la = sig.get("last_active_date")
    rr = sig.get("recruiter_response_rate")
    notice = sig.get("notice_period_days", 0)
    open_to_work = sig.get("open_to_work_flag", False)
    d = days_inactive(as_of_date, la) if la else SIX_MONTHS_DAYS
    r = rr if rr is not None else 0.0
    facts = (
        f"last active {d} days ago, recruiter_response_rate {r:.2f}, "
        f"notice_period {notice} days, open_to_work="
        f"{'true' if open_to_work else 'false'}"
    )
    if d >= SIX_MONTHS_DAYS:
        why = "inactive 6+ months, effectively unreachable"
    elif r < 0.10:
        why = "response rate below 0.10, effectively unreachable"
    elif notice > 90 and bf <= 2:
        why = "long notice period caps reachability"
    elif not open_to_work and bf <= OPEN_TO_WORK_CAP:
        why = "not flagged open-to-work"
    else:
        why = "reachable on the platform"
    return f"Behavior_Fit {bf} (computed from signals): {facts}; {why}."


def plausibility_reason(candidate: Dict[str, Any], as_of_date: str) -> Optional[str]:
    """Return a reason string if the profile is structurally impossible, else
    None. Only fires on genuine contradictions (validated ~0.03% of the pool)."""
    as_of = _date(as_of_date)
    for r in candidate.get("career_history", []):
        start = _date(r["start_date"])
        if r["is_current"] and r["end_date"] is not None:
            return (
                f"role '{r['title']}' marked current but has end_date {r['end_date']}"
            )
        if (not r["is_current"]) and r["end_date"] is None:
            return f"past role '{r['title']}' has null end_date"
        end = _date(r["end_date"]) if r["end_date"] else as_of
        if end < start:
            return (
                f"role '{r['title']}' end_date {r['end_date']} precedes "
                f"start_date {r['start_date']}"
            )
        actual = _months_between(start, end)
        if abs(r["duration_months"] - actual) > DURATION_SLACK_MONTHS:
            return (
                f"role '{r['title']}' claims {r['duration_months']} months "
                f"but its dates span ~{actual} months"
            )
    ez = [
        s["name"]
        for s in candidate.get("skills", [])
        if s["proficiency"] in ("advanced", "expert")
        and s.get("duration_months", 0) == 0
    ]
    if len(ez) >= 3:
        return (
            f"{len(ez)} skills at advanced/expert proficiency with 0 months "
            f"used ({', '.join(ez[:4])})"
        )
    for e in candidate.get("education", []):
        if e["end_year"] < e["start_year"]:
            return (
                f"education '{e.get('degree', '')}' end_year {e['end_year']} "
                f"precedes start_year {e['start_year']}"
            )
    return None


def compute_final(tech: int, ctx: int, beh: int, impossible: bool) -> (int, int):
    """Return (Final_Score_NDCG, Binary_Label_MAP) from the three axes.

    - impossible profile / honeypot       -> tier 0
    - Context hard-reject (ctx == 0)       -> tier 0
    - behaviorally unreachable (beh == 0)  -> final capped at 1
    - otherwise weighted average of the three axes
    """
    if impossible or ctx == 0:
        return 0, 0
    weighted = round(tech * 0.5 + ctx * 0.3 + beh * 0.2)
    final = min(weighted, 1) if beh == 0 else weighted
    binary = 1 if final >= 3 else 0
    return final, binary


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def load_candidates(path: str) -> List[Dict[str, Any]]:
    """Load candidates from a .jsonl file (one object per line) or a .json file
    (single object or a list)."""
    p = str(path)
    if p.endswith(".jsonl"):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
        return out
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("Input file must be .jsonl, a JSON object, or a list of objects.")


def load_job_description(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_honeypot_ids(path: Optional[str]) -> set:
    """Load planted honeypot IDs from a JSON list, {'honeypot_ids': [...]}, or a
    plain text file with one ID per line. These are forced to tier 0."""
    if not path:
        return set()
    txt = Path(path).read_text(encoding="utf-8").strip()
    if not txt:
        return set()
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            return set(data)
        if isinstance(data, dict):
            return set(data.get("honeypot_ids", []))
    except json.JSONDecodeError:
        pass
    return {ln.strip() for ln in txt.splitlines() if ln.strip()}


def compute_as_of_date(pool_path: str) -> str:
    """Snapshot date = max(last_active_date) over the pool. Streams .jsonl."""
    p = str(pool_path)
    mx = None
    if p.endswith(".jsonl"):
        with open(pool_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                la = json.loads(line)["redrob_signals"]["last_active_date"]
                if mx is None or la > mx:
                    mx = la
    else:
        for c in load_candidates(pool_path):
            la = c["redrob_signals"]["last_active_date"]
            if mx is None or la > mx:
                mx = la
    if mx is None:
        raise ValueError("Could not derive as_of_date: no candidates found.")
    return mx


# ---------------------------------------------------------------------------
# Model-output extraction / validation
# ---------------------------------------------------------------------------


def extract_json(text: str) -> Dict[str, Any]:
    """Robustly pull a JSON object out of a model response."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


# All keys that must appear in every final output record.
# Imported by the test suite via: from create_testset import REQUIRED_FIELDS
REQUIRED_FIELDS = [
    "Candidate_ID",
    "Tech_Fit",
    "Tech_Reasoning",
    "Context_Fit",
    "Context_Reasoning",
    "Behavior_Fit",
    "Behavior_Reasoning",
    "Final_Score_NDCG",
    "Binary_Label_MAP",
    "Profile_Strength_Score",
]

LLM_REQUIRED_FIELDS = [
    "Tech_Fit",
    "Tech_Reasoning",
    "Context_Fit",
    "Context_Reasoning",
    "Profile_Strength_Score",
]


def validate_llm_output(parsed: Dict[str, Any]) -> List[str]:
    """Validate the fields the LLM is responsible for. (Behavior/Final/Binary
    are produced by Python and validated separately.)"""
    issues = []
    for field in LLM_REQUIRED_FIELDS:
        if field not in parsed:
            issues.append(f"missing field: {field}")
    if issues:
        return issues
    for f in ["Tech_Fit", "Context_Fit"]:
        if not isinstance(parsed[f], int) or not (0 <= parsed[f] <= 4):
            issues.append(f"{f} out of range [0,4]: {parsed[f]!r}")
    ps = parsed["Profile_Strength_Score"]
    if not isinstance(ps, int) or not (0 <= ps <= 100):
        issues.append(f"Profile_Strength_Score out of range [0,100]: {ps!r}")
    if "Impossible" in parsed and not isinstance(parsed["Impossible"], bool):
        issues.append(f"Impossible not a boolean: {parsed['Impossible']!r}")
    return issues


def validate_label(label: Dict[str, Any]) -> List[str]:
    """Sanity-check the final merged label for internal consistency."""
    issues = []
    for f in ["Tech_Fit", "Context_Fit", "Behavior_Fit", "Final_Score_NDCG"]:
        if not (0 <= label[f] <= 4):
            issues.append(f"{f} out of range [0,4]: {label[f]}")
    exp_final, exp_binary = compute_final(
        label["Tech_Fit"],
        label["Context_Fit"],
        label["Behavior_Fit"],
        impossible=(
            label["Final_Score_NDCG"] == 0
            and label["Tech_Fit"] == 0
            and label["Context_Fit"] == 0
            and label["Behavior_Fit"] == 0
        ),
    )
    # Only flag a mismatch that isn't explained by the hard-reject/honeypot path.
    if label["Final_Score_NDCG"] != exp_final and not (
        label["Final_Score_NDCG"] == 0 and label["Context_Fit"] == 0
    ):
        issues.append(
            f"Final_Score_NDCG inconsistent: got {label['Final_Score_NDCG']}, "
            f"expected {exp_final}"
        )
    if label["Binary_Label_MAP"] != (1 if label["Final_Score_NDCG"] >= 3 else 0):
        issues.append("Binary_Label_MAP inconsistent with Final_Score_NDCG")
    return issues


# Public alias used by the test suite.
# Accepts a full 10-key label dict and validates it end to end.
validate_output = validate_label


# ---------------------------------------------------------------------------
# Scoring one candidate
# ---------------------------------------------------------------------------


def _aggregate_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Majority-vote the LLM judgment across N samples (self-consistency)."""
    if len(samples) == 1:
        return samples[0]
    tech = Counter(s["Tech_Fit"] for s in samples).most_common(1)[0][0]
    ctx = Counter(s["Context_Fit"] for s in samples).most_common(1)[0][0]
    impossible = sum(bool(s.get("Impossible")) for s in samples) > len(samples) / 2
    strength = int(round(median(s["Profile_Strength_Score"] for s in samples)))
    # take reasoning from a run that matches the voted Tech_Fit
    pick = next((s for s in samples if s["Tech_Fit"] == tech), samples[0])
    return {
        "Tech_Fit": tech,
        "Tech_Reasoning": pick["Tech_Reasoning"],
        "Context_Fit": ctx,
        "Context_Reasoning": pick["Context_Reasoning"],
        "Impossible": impossible,
        "Impossible_Reason": pick.get("Impossible_Reason", ""),
        "Profile_Strength_Score": strength,
    }


def score_candidate(
    provider: BaseProvider,
    candidate: Dict[str, Any],
    job_description: str = "",
    as_of_date: Optional[str] = None,
    honeypot_ids: Optional[set] = None,
    samples: int = 1,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> Dict[str, Any]:
    """LLM judges Tech/Context/strength; Python decides Behavior + final tier.

    job_description defaults to "" so test fixtures that don't supply a JD
    still run (the LLM will have no JD context, which is fine for unit tests).

    as_of_date defaults to None; when None it is derived from the candidate's
    own last_active_date (safe for single-candidate test calls).
    """
    honeypot_ids = honeypot_ids or set()
    cid = (
        candidate.get("candidate_id")
        or candidate.get("Candidate_ID")
        or candidate.get("id")
    )
    sig = candidate.get("redrob_signals", {})

    # Derive as_of_date from the candidate itself when not supplied.
    # This keeps test calls (provider, candidate) valid without needing a pool.
    if as_of_date is None:
        la = sig.get("last_active_date")
        as_of_date = la if la else date.today().isoformat()

    candidate_json_str = json.dumps(candidate, indent=2)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        as_of_date=as_of_date,
        job_description=job_description,
        candidate_json=candidate_json_str,
    )

    # --- get the LLM judgment (with retries + optional self-consistency) ------
    runs: List[Dict[str, Any]] = []
    raws: List[str] = []
    last_error = None
    needed = samples
    attempts = 0
    while len(runs) < needed and attempts < needed * max_retries:
        attempts += 1
        try:
            raw = provider.generate(SYSTEM_PROMPT, user_prompt)
            parsed = extract_json(raw)
            issues = validate_llm_output(parsed)
            if issues:
                raise ValueError("; ".join(issues))
            runs.append(parsed)
            raws.append(raw)
        except Exception as e:  # noqa: BLE001
            last_error = str(e)
            time.sleep(retry_delay)

    if not runs:
        return {
            "candidate_id": cid,
            "provider": provider.name,
            "model": provider.model,
            "raw_response": None,
            "parsed": None,
            "llm_raw": None,
            "flags": {"impossible": None, "honeypot": cid in honeypot_ids},
            "validation_issues": [f"failed after retries: {last_error}"],
            "attempt": attempts,
        }

    judged = _aggregate_samples(runs)

    # --- deterministic layer (Python authority) -------------------------------
    tech = judged["Tech_Fit"]
    ctx = judged["Context_Fit"]
    strength = judged["Profile_Strength_Score"]

    beh = behavior_fit(sig, as_of_date)
    beh_reasoning = make_behavior_reasoning(sig, as_of_date, beh)

    plaus = plausibility_reason(candidate, as_of_date)
    llm_impossible = bool(judged.get("Impossible"))
    is_honeypot = cid in honeypot_ids
    impossible = bool(plaus) or llm_impossible or is_honeypot

    final, binary = compute_final(tech, ctx, beh, impossible)

    if impossible:
        reason = (
            plaus
            or judged.get("Impossible_Reason")
            or ("planted honeypot" if is_honeypot else "impossible profile")
        )
        note = f"Forced to tier 0 — {reason}."
        label = {
            "Candidate_ID": cid,
            "Tech_Fit": 0,
            "Tech_Reasoning": note,
            "Context_Fit": 0,
            "Context_Reasoning": note,
            "Behavior_Fit": 0,
            "Behavior_Reasoning": note,
            "Final_Score_NDCG": 0,
            "Binary_Label_MAP": 0,
            "Profile_Strength_Score": 0,
        }
    else:
        label = {
            "Candidate_ID": cid,
            "Tech_Fit": tech,
            "Tech_Reasoning": judged["Tech_Reasoning"],
            "Context_Fit": ctx,
            "Context_Reasoning": judged["Context_Reasoning"],
            "Behavior_Fit": beh,
            "Behavior_Reasoning": beh_reasoning,
            "Final_Score_NDCG": final,
            "Binary_Label_MAP": binary,
            "Profile_Strength_Score": strength,
        }

    return {
        "candidate_id": cid,
        "provider": provider.name,
        "model": provider.model,
        "raw_response": raws[0],
        "parsed": label,  # canonical, authoritative (downstream reads this)
        "llm_raw": judged,  # raw LLM judgment, for audit
        "flags": {
            "impossible": impossible,
            "honeypot": is_honeypot,
            "plausibility_reason": plaus,
            "llm_flagged_impossible": llm_impossible,
            "deterministic_behavior_fit": beh,
        },
        "validation_issues": validate_label(label),
        "attempt": attempts,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Golden set generation pipeline (v2) for Redrob AI candidate scoring."
    )
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS.keys()),
        required=True,
        help="Which LLM backend to use as the judge.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model override. Defaults: gemini-2.0-flash / llama-3.3-70b-versatile.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Candidates to label: .jsonl, a JSON object, or a list.",
    )
    parser.add_argument(
        "--job-description",
        required=True,
        help="Path to text file containing the job description.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSONL file (one labeled record per line).",
    )
    parser.add_argument(
        "--pool",
        default=None,
        help="Full candidate pool (.jsonl) used to derive as_of_date. "
        "Recommended; if omitted, as_of_date is derived from --input.",
    )
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="Override snapshot date (YYYY-MM-DD). Default: max(last_active_date).",
    )
    parser.add_argument(
        "--honeypots",
        default=None,
        help="File of planted honeypot candidate_ids (forced to tier 0).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="LLM samples per candidate for a majority-vote (self-consistency). Default 1.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=3.0,
        help="Seconds to sleep between candidates. Default 3.0",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per LLM sample. Default 3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N candidates (quick test).",
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output)
    if output_path.suffix:
        output_path = output_path.with_name(
            f"{output_path.stem}_{timestamp}{output_path.suffix}"
        )
    else:
        output_path = output_path / f"golden_set_{timestamp}.jsonl"

    # Resolve as_of_date
    if args.as_of_date:
        as_of_date = args.as_of_date
        logging.info("as_of_date (override): %s", as_of_date)
    elif args.pool:
        as_of_date = compute_as_of_date(args.pool)
        logging.info("as_of_date (from pool max last_active_date): %s", as_of_date)
    else:
        as_of_date = compute_as_of_date(args.input)
        logging.warning(
            "as_of_date derived from --input (%s). Pass --pool for the "
            "true snapshot date.",
            as_of_date,
        )

    honeypot_ids = load_honeypot_ids(args.honeypots)
    if honeypot_ids:
        logging.info(
            "Loaded %d planted honeypot id(s) — will force to tier 0.",
            len(honeypot_ids),
        )

    provider_cls = PROVIDERS[args.provider]
    provider_kwargs = {}
    if args.model:
        provider_kwargs["model"] = args.model
    try:
        provider = provider_cls(**provider_kwargs)
    except ValueError as e:
        logging.error("%s", e)
        sys.exit(1)

    candidates = load_candidates(args.input)
    job_description = load_job_description(args.job_description)
    if args.limit:
        candidates = candidates[: args.limit]

    logging.info(
        "Provider: %s | Model: %s | samples/candidate: %d",
        provider.name,
        provider.model,
        args.samples,
    )
    logging.info("Loaded %d candidate(s) from %s", len(candidates), args.input)
    logging.info("Writing results to %s", output_path)

    n_ok, n_issues, n_failed, n_zero = 0, 0, 0, 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, candidate in enumerate(candidates, start=1):
            cid = (
                candidate.get("candidate_id")
                or candidate.get("Candidate_ID")
                or candidate.get("id")
                or f"row_{i}"
            )
            logging.info("[%d/%d] Scoring: %s", i, len(candidates), cid)

            result = score_candidate(
                provider,
                candidate,
                job_description,
                as_of_date,
                honeypot_ids=honeypot_ids,
                samples=args.samples,
                max_retries=args.max_retries,
            )

            if result["parsed"] is None:
                n_failed += 1
                logging.error("FAILED")
            else:
                if result["parsed"]["Final_Score_NDCG"] == 0:
                    n_zero += 1
                if result["validation_issues"]:
                    n_issues += 1
                    logging.warning(
                        "OK (with %d issue(s))", len(result["validation_issues"])
                    )
                else:
                    n_ok += 1
                    logging.info("OK | tier=%d", result["parsed"]["Final_Score_NDCG"])

            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

            if i < len(candidates):
                time.sleep(args.sleep)

    logging.info("--- Summary ---")
    logging.info("Clean:        %d", n_ok)
    logging.info("With issues:  %d", n_issues)
    logging.info("Failed:       %d", n_failed)
    logging.info("Tier-0 (incl. honeypots/impossible): %d", n_zero)
    logging.info("Total:        %d", len(candidates))


if __name__ == "__main__":
    main()
