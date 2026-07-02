"""Context_Fit prediction — deterministic logic tree over structured fields."""

import numpy as np

from ..config import _CONSULTING_FIRMS, _CONSULTING_INDUSTRIES, _MGMT_TITLE_TOKENS


def _is_consulting(role: dict) -> bool:
    company = role.get("company", "").strip().lower()
    industry = role.get("industry", "").strip().lower()
    return company in _CONSULTING_FIRMS or industry in _CONSULTING_INDUSTRIES


def predict_context_fit_single(c: dict) -> int:
    profile = c.get("profile", {})
    roles = c.get("career_history", [])
    sig = c.get("redrob_signals", {})

    total_yoe = float(profile.get("years_of_experience", 0) or 0)
    durations = [r.get("duration_months", 0) for r in roles]
    avg_tenure = sum(durations) / len(durations) if durations else 0
    longest_tenure = max(durations) if durations else 0
    product_months = sum(
        r.get("duration_months", 0) for r in roles if not _is_consulting(r)
    )

    is_pure_consulting = bool(roles) and all(_is_consulting(r) for r in roles)
    is_management = any(
        any(t in r.get("title", "").lower() for t in _MGMT_TITLE_TOKENS) for r in roles
    )
    country = profile.get("country", "India").strip().lower()
    relocate = sig.get("willing_to_relocate", True)
    visa_reject = (country != "india") and (not relocate)

    if is_pure_consulting or avg_tenure < 18 or visa_reject:
        return 0
    if is_management or longest_tenure < 24:
        return 1
    if (5 <= total_yoe <= 9) and product_months >= 48 and longest_tenure >= 36:
        return 4
    if product_months >= 24 and longest_tenure >= 24:
        return 3
    return 2


def predict_context_fit(
    records: list,
    evaluation: bool = True,
) -> np.ndarray:
    print("[*] Computing Context_Fit (deterministic logic tree)...")

    pred = np.array([predict_context_fit_single(r) for r in records])

    if evaluation:
        gt = np.array([int(r.get("parsed", {}).get("Context_Fit", 0)) for r in records])

        acc = (pred == gt).mean() * 100
        print(f"    Context_Fit accuracy vs golden labels: {acc:.1f}%")

    return pred
