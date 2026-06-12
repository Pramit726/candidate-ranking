"""
Golden Set Generation Pipeline for Redrob AI Candidate Scoring
================================================================

Generates "golden" labeled outputs (Tech_Fit, Context_Fit, Behavior_Fit,
Final_Score_NDCG, Binary_Label_MAP, Profile_Strength_Score) for a list of
candidate JSON profiles, using the recruiter-evaluation prompt.

Two backend options are supported:
  1. Gemini (Google AI Studio)  -> provider="gemini"
  2. Groq Cloud (Llama models)  -> provider="groq"

Usage:
    python golden_set_pipeline.py --provider gemini --input candidates.json --output golden_set.jsonl
    python golden_set_pipeline.py --provider groq   --input candidates.json --output golden_set.jsonl

Environment variables required:
    GEMINI_API_KEY   (for provider=gemini)
    GROQ_API_KEY     (for provider=groq)

Each entry in candidates.json should be a single candidate profile dict
(or a list of such dicts). The job description is baked into the prompt
template already (per Redrob AI's spec) and the candidate JSON is
substituted in.
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Prompt templates (verbatim from the spec)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a sharp, skeptical technical recruiter and hiring manager at Redrob AI,
a fast-moving Series A startup. You identify candidates who have demonstrably
built production systems. You evaluate evidence, not buzzwords."""

USER_PROMPT_TEMPLATE = """[Instruction]
Review the Candidate JSON profile against the Job Description below.
Assign scores for Tech_Fit, Context_Fit, and Behavior_Fit.
Then compute Final_Score_NDCG, Binary_Label_MAP, and Profile_Strength_Score.

---

[Evaluation Principles]
1. Judge only on explicit evidence present in the candidate profile.
2. If evidence is missing, treat the capability as absent. Do not infer it.
3. Never reward keyword presence without evidence of real work.
4. Career history descriptions take precedence over skills lists.
5. Every reasoning statement must cite specific evidence from the profile.
6. Never infer the following unless explicitly stated:
   - Production deployment or real users
   - Retrieval, ranking, or search ownership
   - Evaluation framework design or usage
   - Startup readiness or hands-on coding recency

---

[Tech_Fit] — Measures: retrieval, ranking, search, recommendation,
             matching, evaluation, production ML engineering.

Score 4 — Candidate shows ALL of:
  A) Explicitly built, deployed, or owned at least one of:
     search / retrieval / ranking / recommendation / matching system
  B) Explicit use of evaluation frameworks:
     NDCG, MAP, MRR, A/B testing, offline benchmarking, or search
     quality monitoring
  C) Explicit production retrieval operations:
     embedding drift handling, index refresh ownership, hybrid
     retrieval, or vector DB operations at scale
  D) Strong Python, production deployment, real users
  Note: Generic RAG, LangChain, Pinecone, Qdrant, or LLM mentions
        alone do NOT qualify for score 4.

Score 3 — Candidate shows production ML + Python + semantic search
          or vector DB or embeddings with user-facing deployment,
          BUT lacks evidence of ranking evaluation frameworks
          OR retrieval quality ownership.

Score 2 — General ML, AI, backend, or data engineering background.
          No evidence of production retrieval, search, ranking,
          recommendation, or evaluation systems.

Score 1 — Primary expertise is Computer Vision, Speech, Robotics,
          or Embedded AI with little IR/search/ranking exposure.

Score 0 — Primary career is not software engineering
          (Marketing, Sales, HR, Operations, etc.).

Preferred signals (LoRA, QLoRA, learning-to-rank, XGBoost ranking,
distributed systems) NEVER independently raise a tier. They only
strengthen reasoning within an already-assigned tier.

---

[Context_Fit] — Measures: startup suitability only.
                Tech skills must NOT influence this score.

Score 4 — 6-8 years total, 4+ years applied ML, product-company
          background, multiple tenures >= 36 months, hands-on
          builder profile, located in India or willing to relocate
          to Pune/Noida/Tier-1 city.

Score 3 — Product engineering background, transitioning into AI/ML,
          open to relocation, mostly stable tenures.

Score 2 — Standard engineering profile, no strong signals either way.

Score 1 — Architecture or management-heavy role, limited recent
          coding, or located outside India but willing to relocate.

Score 0 (Hard Reject) — Any of:
  - Pure academic or research-only career
  - Entire career at consulting/services firms only
    (TCS, Infosys, Wipro, Cognizant, Capgemini, Accenture, etc.)
  - Average tenure below 18 months
  - Requires visa sponsorship
  - Refuses relocation

---

[Behavior_Fit] — Measures: actual hireability right now.

Primary signals: notice_period_days, last_active_date,
                 recruiter_response_rate, open_to_work_flag

Secondary signals (tie-breakers only): profile_views_received_30d,
interview_completion_rate, offer_acceptance_rate, preferred_work_mode

Score 4 — Notice <= 30 days AND active within 14 days
          AND response rate > 0.80 AND open_to_work = true

Score 3 — Notice 31-60 days AND active within 45 days
          AND response rate > 0.60

Score 2 — Notice 61-90 days AND active within 90 days
          AND response rate 0.40-0.60

Score 1 — Notice > 90 days OR active 4-5 months ago
          OR response rate 0.10-0.30

Score 0 — Inactive >= 6 months OR response rate < 0.10

---

[Scoring Formula]

Step 1 — Compute weighted average:
  Weighted_Score = (Tech_Fit * 0.5) + (Context_Fit * 0.3) + (Behavior_Fit * 0.2)
  Round to nearest integer → this is your base score.

Step 2 — Apply Veto Rule:
  IF Context_Fit <= 1 OR Behavior_Fit <= 1:
    Final_Score_NDCG = 1
  ELSE:
    Final_Score_NDCG = Weighted_Score (from Step 1)

Step 3 — Binary label:
  Binary_Label_MAP = 1 if Final_Score_NDCG >= 3, else 0

Step 4 — Profile_Strength_Score (0-100):
  Score this candidate against the ideal version of a candidate
  at their Final_Score_NDCG tier.
  100 = strongest possible candidate at this score level.
  0   = barely qualifies for this score level.
  This is NOT an average. It is your calibrated judgment of where
  this candidate sits within their tier.
  Use the full 0-100 range. Do not cluster around round numbers.

---

[Reasoning Rules]
- Maximum 2 sentences per reasoning field.
- Must cite specific evidence from the candidate profile.
- Must be factual. No assumptions. No praise. No generic statements.

Bad:  "Strong retrieval background."
Good: "Candidate explicitly describes owning semantic search using
       Pinecone with monthly index refresh cycles in production
       at a 2M-user platform."

---

[Output]
Return ONLY valid JSON. No markdown. No code fences. No extra text.

{{
  "Candidate_ID": "string",
  "Tech_Fit": integer,
  "Tech_Reasoning": "string",
  "Context_Fit": integer,
  "Context_Reasoning": "string",
  "Behavior_Fit": integer,
  "Behavior_Reasoning": "string",
  "Final_Score_NDCG": integer,
  "Binary_Label_MAP": integer,
  "Profile_Strength_Score": integer
}}

---

[Data]

### JOB DESCRIPTION
Role: Senior AI Engineer — Founding Team
Company: Redrob AI (Series A startup)
Location: Pune/Noida, India (Hybrid). Open to Tier-1 India relocation.
          No visa sponsorship.
Experience: 5-9 years
Mandate: Own the search, ranking, and matching layer. Must have
         hands-on production experience with embedding-based
         retrieval, vector databases, Python, and evaluation
         frameworks (NDCG, MAP).
Dealbreakers: Pure academics, LangChain-only under 12 months,
              no production code in 18 months, job-hopping,
              IT services-only background, CV/Speech-only ML.
              Notice > 30 days is a concern. No recruiter response
              is an instant rejection.

### CANDIDATE JSON
{candidate_json}
"""


# ---------------------------------------------------------------------------
# Provider clients
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
    You can pass a different model name via --model, e.g. gemini-1.5-pro.

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

    Default model: llama-3.3-70b-versatile (strong reasoning, good for judging).
    You can pass a different model via --model, e.g. llama-3.1-8b-instant.

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
# Pipeline helpers
# ---------------------------------------------------------------------------


def extract_json(text: str) -> Dict[str, Any]:
    """
    Robustly pull a JSON object out of a model response, in case the model
    wraps it in markdown fences or adds stray whitespace/text despite
    instructions.
    """
    text = text.strip()

    # Strip ```json ... ``` or ``` ... ``` fences if present
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # If there's leading/trailing junk, try to grab the outermost {...}
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

    return json.loads(text)


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


def validate_output(parsed: Dict[str, Any]) -> List[str]:
    """Return a list of validation issues (empty list = OK)."""
    issues = []

    for field in REQUIRED_FIELDS:
        if field not in parsed:
            issues.append(f"missing field: {field}")

    if issues:
        return issues  # can't validate further without all fields

    # Range checks
    for f in ["Tech_Fit", "Context_Fit", "Behavior_Fit", "Final_Score_NDCG"]:
        if not (0 <= parsed[f] <= 4):
            issues.append(f"{f} out of range [0,4]: {parsed[f]}")

    if parsed["Binary_Label_MAP"] not in (0, 1):
        issues.append(f"Binary_Label_MAP not 0/1: {parsed['Binary_Label_MAP']}")

    if not (0 <= parsed["Profile_Strength_Score"] <= 100):
        issues.append(
            f"Profile_Strength_Score out of range [0,100]: {parsed['Profile_Strength_Score']}"
        )

    # Re-derive Final_Score_NDCG / Binary_Label_MAP to sanity-check internal consistency
    tech, ctx, beh = parsed["Tech_Fit"], parsed["Context_Fit"], parsed["Behavior_Fit"]
    weighted = round(tech * 0.5 + ctx * 0.3 + beh * 0.2)
    expected_final = 1 if (ctx <= 1 or beh <= 1) else weighted
    if parsed["Final_Score_NDCG"] != expected_final:
        issues.append(
            f"Final_Score_NDCG inconsistent: got {parsed['Final_Score_NDCG']}, "
            f"expected {expected_final} (weighted={weighted}, ctx={ctx}, beh={beh})"
        )

    expected_binary = 1 if parsed["Final_Score_NDCG"] >= 3 else 0
    if parsed["Binary_Label_MAP"] != expected_binary:
        issues.append(
            f"Binary_Label_MAP inconsistent: got {parsed['Binary_Label_MAP']}, "
            f"expected {expected_binary} based on Final_Score_NDCG={parsed['Final_Score_NDCG']}"
        )

    return issues


def load_candidates(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("Input file must be a JSON object or a list of JSON objects.")


def score_candidate(
    provider: BaseProvider,
    candidate: Dict[str, Any],
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> Dict[str, Any]:
    """Run one candidate through the model, with retries and validation."""

    candidate_json_str = json.dumps(candidate, indent=2)
    user_prompt = USER_PROMPT_TEMPLATE.format(candidate_json=candidate_json_str)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = provider.generate(SYSTEM_PROMPT, user_prompt)
            parsed = extract_json(raw)
            issues = validate_output(parsed)

            result = {
                "candidate_id": candidate.get("candidate_id")
                or candidate.get("Candidate_ID")
                or candidate.get("id"),
                "provider": provider.name,
                "model": provider.model,
                "raw_response": raw,
                "parsed": parsed,
                "validation_issues": issues,
                "attempt": attempt,
            }
            return result

        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
            continue

    # All retries failed
    return {
        "candidate_id": candidate.get("candidate_id")
        or candidate.get("Candidate_ID")
        or candidate.get("id"),
        "provider": provider.name,
        "model": provider.model,
        "raw_response": None,
        "parsed": None,
        "validation_issues": [f"failed after {max_retries} attempts: {last_error}"],
        "attempt": max_retries,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Golden set generation pipeline for Redrob AI candidate scoring."
    )
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS.keys()),
        required=True,
        help="Which LLM backend to use as the golden-set labeler.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override. Defaults: gemini-2.0-flash (gemini), llama-3.3-70b-versatile (groq).",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSON file: a single candidate object or a list of candidate objects.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSONL file (one labeled record per line).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between API calls (rate-limit friendliness). Default 1.0",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per candidate on failure/invalid JSON. Default 3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: only process the first N candidates (useful for a quick test run).",
    )

    args = parser.parse_args()

    provider_cls = PROVIDERS[args.provider]
    provider_kwargs = {}
    if args.model:
        provider_kwargs["model"] = args.model

    try:
        provider = provider_cls(**provider_kwargs)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    candidates = load_candidates(args.input)
    if args.limit:
        candidates = candidates[: args.limit]

    print(f"Provider: {provider.name} | Model: {provider.model}")
    print(f"Loaded {len(candidates)} candidate(s) from {args.input}")
    print(f"Writing results to {args.output}")

    n_ok, n_issues, n_failed = 0, 0, 0

    with open(args.output, "w") as out_f:
        for i, candidate in enumerate(candidates, start=1):
            cid = (
                candidate.get("candidate_id")
                or candidate.get("Candidate_ID")
                or candidate.get("id")
                or f"row_{i}"
            )
            print(
                f"[{i}/{len(candidates)}] Scoring candidate: {cid} ...",
                end=" ",
                flush=True,
            )

            result = score_candidate(provider, candidate, max_retries=args.max_retries)

            if result["parsed"] is None:
                n_failed += 1
                print("FAILED")
            elif result["validation_issues"]:
                n_issues += 1
                print(
                    f"OK (with {len(result['validation_issues'])} validation issue(s))"
                )
            else:
                n_ok += 1
                print("OK")

            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

            if i < len(candidates):
                time.sleep(args.sleep)

    print("\n--- Summary ---")
    print(f"Clean:        {n_ok}")
    print(f"With issues:  {n_issues}")
    print(f"Failed:       {n_failed}")
    print(f"Total:        {len(candidates)}")


if __name__ == "__main__":
    main()
