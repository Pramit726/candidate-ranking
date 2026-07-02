"""Behavior_Fit prediction — deterministic formula over redrob_signals."""

from datetime import date

import numpy as np

from ..config import (
    OPEN_TO_WORK_CAP,
    RECENCY_MILD,
    RECENCY_REAL,
    RECENCY_RECENT,
    SIX_MONTHS,
)


def _behavior_fit_single(c: dict, as_of: date) -> int:
    sig = c.get("redrob_signals", {})
    la = sig.get("last_active_date")
    rr = sig.get("recruiter_response_rate")
    notice = sig.get("notice_period_days", 0)
    otw = sig.get("open_to_work_flag", False)

    d = (as_of - date.fromisoformat(la)).days if la else SIX_MONTHS
    r = rr if rr is not None else 0.0

    recency = (
        4
        if d <= RECENCY_RECENT
        else (
            3
            if d <= RECENCY_MILD
            else 2 if d <= RECENCY_REAL else 1 if d < SIX_MONTHS else 0
        )
    )
    response = (
        4
        if r >= 0.70
        else 3 if r >= 0.50 else 2 if r >= 0.30 else 1 if r >= 0.10 else 0
    )

    base = min(recency, response)
    if not otw:
        base = min(base, OPEN_TO_WORK_CAP)
    if notice > 90:
        base = min(base, 2)
    elif notice > 30:
        base = min(base, 3)
    return base


def predict_behavior_fit(
    records: list,
    as_of_date_str: str,
    evaluation: bool = True,
) -> np.ndarray:
    print("[*] Computing Behavior_Fit (deterministic from redrob_signals)...")

    as_of = date.fromisoformat(as_of_date_str)

    pred = np.array([_behavior_fit_single(r, as_of) for r in records])

    if evaluation:
        gt = np.array(
            [int(r.get("parsed", {}).get("Behavior_Fit", 0)) for r in records]
        )

        acc = (pred == gt).mean() * 100
        print(f"    Behavior_Fit accuracy vs golden labels: {acc:.1f}%")

    return pred
