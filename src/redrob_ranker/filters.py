"""Pre-ranking exclusion, keyword-pruning, and domain-relevance filters.

Ported verbatim from 01_analysis.ipynb. Runs before any ML inference to drop
candidates that are structurally disqualified, off-domain, or unreachable.
"""

from datetime import date

from .config import (
    INDIA_VARIANTS,
    MAX_INACTIVE_DAYS,
    MAX_YOE,
    MIN_RESPONSE_RATE,
    MIN_YOE,
    PURE_CONSULTING_FIRMS,
    TITLE_LEVELS,
    _COMPILED_DOMAIN_GROUPS,
    _TITLE_PATTERN,
)

def _is_consulting_firm(company: str) -> bool:
    c = company.lower().strip()
    return any(firm in c for firm in PURE_CONSULTING_FIRMS)


def _is_pure_consulting_career(career: list) -> bool:
    if not career:
        return False
    return all(_is_consulting_firm(r.get("company", "")) for r in career)


def _is_job_hopper(career: list) -> tuple:
    if not career or len(career) < 2:
        return False, ""
    durations = [r.get("duration_months") or 0 for r in career]
    avg_tenure = sum(durations) / len(durations)
    companies = {r.get("company", "").lower().strip() for r in career}
    if avg_tenure < 18 and len(companies) >= 3:
        return True, f"avg_tenure={avg_tenure:.1f}mo, switches={len(companies)}"
    return False, ""


def _title_level(title: str) -> int:
    t = title.lower()
    matched = [lvl for kw, lvl in TITLE_LEVELS.items() if kw in t]
    return max(matched) if matched else 0


def _is_title_chaser(career: list) -> tuple:
    sorted_career = sorted(career, key=lambda r: r.get("start_date") or "0000-00-00")
    jumps, prev_level = 0, 0
    for role in sorted_career:
        level = _title_level(role.get("title", ""))
        duration = role.get("duration_months") or 0
        if level > prev_level and prev_level != 0 and duration < 18:
            jumps += 1
        prev_level = max(prev_level, level)
    if jumps >= 2:
        return True, f"quick_jumps={jumps}"
    return False, ""


def apply_pre_ranking_filters(records: list, as_of_date_str: str) -> tuple:
    """Single-pass filter combining all stages from 01_analysis.ipynb:
      Stage 1 – Structural exclusions (country, YOE, skills, consulting, hopper)
      Stage 2 – Title keyword pruning
      Stage 3 – Behavioural reachability (inactive, low response rate)
      Stage 4 – Domain relevance (AI/ML/Search)
    Returns (passed_records, stats_dict).
    """
    as_of = date.fromisoformat(as_of_date_str)
    passed = []
    stats = {
        k: []
        for k in [
            "non_india",
            "too_junior",
            "too_senior",
            "null_skills",
            "honeypot_skill",
            "pure_consulting",
            "job_hopper",
            "title_chaser",
            "wrong_title",
            "inactive",
            "low_response",
            "irrelevant_domain",
        ]
    }
    stats["total_in"] = len(records)

    for c in records:
        cid = c.get("candidate_id", "")
        profile = c.get("profile") or {}
        career = c.get("career_history") or []
        signals = c.get("redrob_signals") or {}
        skills = c.get("skills") or []

        country = profile.get("country", "").lower().strip()
        yoe = profile.get("years_of_experience", 0) or 0
        title = profile.get("current_title", "")
        exclude = []

        # ── Stage 1A: structural ─────────────────────────────────────────
        if country not in INDIA_VARIANTS and not signals.get(
            "willing_to_relocate", False
        ):
            stats["non_india"].append(cid)
            exclude.append("non_india")

        if yoe < MIN_YOE:
            stats["too_junior"].append(cid)
            exclude.append("too_junior")
        elif yoe > MAX_YOE:
            stats["too_senior"].append(cid)
            exclude.append("too_senior")

        if len(skills) <= 5:
            stats["null_skills"].append(cid)
            exclude.append("null_skills")

        expert_zero = [
            s
            for s in skills
            if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) == 0
        ]
        if expert_zero:
            stats["honeypot_skill"].append(cid)
            exclude.append("honeypot_skill")

        if _is_pure_consulting_career(career):
            stats["pure_consulting"].append(cid)
            exclude.append("pure_consulting")

        hopper, _ = _is_job_hopper(career)
        if hopper:
            stats["job_hopper"].append(cid)
            exclude.append("job_hopper")

        if not exclude:
            chaser, _ = _is_title_chaser(career)
            if chaser:
                stats["title_chaser"].append(cid)
                exclude.append("title_chaser")

        # ── Stage 2: title keyword pruning ───────────────────────────────
        if _TITLE_PATTERN.search(title):
            stats["wrong_title"].append(cid)
            exclude.append("wrong_title")

        if exclude:
            continue

        # ── Stage 3: behavioural reachability ────────────────────────────
        la_str = signals.get("last_active_date")
        if not la_str:
            stats["inactive"].append(cid)
            continue
        if (as_of - date.fromisoformat(la_str[:10])).days > MAX_INACTIVE_DAYS:
            stats["inactive"].append(cid)
            continue
        rr = signals.get("recruiter_response_rate")
        if rr is None or rr < MIN_RESPONSE_RATE:
            stats["low_response"].append(cid)
            continue

        # ── Stage 4: domain relevance ─────────────────────────────────────
        text = " ".join(s.get("name", "") for s in skills).lower()
        text += " " + " ".join(r.get("description", "").lower() for r in career)
        matched_groups = sum(
            1 for pat in _COMPILED_DOMAIN_GROUPS.values() if pat.search(text)
        )
        if matched_groups < 2:
            stats["irrelevant_domain"].append(cid)
            continue

        passed.append(c)

    stats["total_out"] = len(passed)
    return passed, stats


def print_filter_summary(stats: dict):
    total_in = stats["total_in"]
    total_out = stats["total_out"]
    print(f"\n{'─'*50}")
    print(f"  Pre-ranking filter summary")
    print(f"{'─'*50}")
    print(f"  Input : {total_in:>5}")
    for key in [
        "non_india",
        "too_junior",
        "too_senior",
        "null_skills",
        "honeypot_skill",
        "pure_consulting",
        "job_hopper",
        "title_chaser",
        "wrong_title",
        "inactive",
        "low_response",
        "irrelevant_domain",
    ]:
        n = len(stats[key])
        if n:
            print(f"  -{key:<22}: {n:>4}")
    print(f"  {'─'*20}")
    print(f"  Passed: {total_out:>5}")
    print(f"{'─'*50}\n")
