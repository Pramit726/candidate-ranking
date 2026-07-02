"""Extractive Vector Summarization — one-sentence grounded reasoning per candidate.

Selects the candidate profile sentence most similar to the JD embedding and
appends a structured metadata tag. Logic unchanged from the original monolith.
"""

import os
import re

import pandas as pd
import torch
from sentence_transformers import util

def load_job_description(data_dir: str) -> str:
    """Support both .txt and .docx JD files."""
    txt_path = os.path.join(data_dir, "job_description.txt")
    docx_path = os.path.join(data_dir, "job_description.docx")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    if os.path.exists(docx_path):
        try:
            import docx

            doc = docx.Document(docx_path)
            return "\n".join(p.text for p in doc.paragraphs).strip()
        except ImportError:
            print(
                "[!] python-docx not installed; cannot read .docx JD. "
                "Install with: pip install python-docx"
            )
    print("[!] No job_description.txt or job_description.docx found in data_dir.")
    return ""


def _best_jd_sentence(candidate: dict, jd_emb, embed_model) -> str:
    """Return the single sentence from the candidate's profile most aligned
    with the JD, by cosine similarity.

    Sources searched (in priority order):
      1. career_history role descriptions (split into sentences)
      2. profile.summary (split into sentences)
      3. profile.headline (used whole)
    This ensures candidates with rich summaries but sparse role descriptions
    still get a meaningful extractive reason.
    """
    sentences = []

    # Primary source: role descriptions (richest technical evidence)
    for role in candidate.get("career_history", []):
        desc = role.get("description", "")
        if desc:
            sentences.extend(re.split(r"(?<=[.!?])\s+", desc))

    # Secondary source: profile summary
    summary = candidate.get("profile", {}).get("summary", "")
    if summary:
        sentences.extend(re.split(r"(?<=[.!?])\s+", summary))

    # Tertiary source: headline (kept whole — already one sentence)
    headline = candidate.get("profile", {}).get("headline", "")
    if headline:
        sentences.append(headline)

    valid = [s.strip() for s in sentences if len(s.split()) > 5]
    if not valid:
        return "Limited descriptive profile text available."

    sent_embs = embed_model.encode(valid, convert_to_tensor=True)
    scores    = util.cos_sim(jd_emb, sent_embs)[0]
    best_idx  = int(torch.argmax(scores).item())
    return valid[best_idx]


def generate_reasoning(
    ranked_df: pd.DataFrame, record_map: dict, jd_text: str, embed_model
) -> list:
    """Return a list of reasoning strings (one per row, in rank order)."""
    if not jd_text:
        return [f"Ranked #{i+1} by scoring model." for i in range(len(ranked_df))]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    jd_emb = embed_model.encode(jd_text, convert_to_tensor=True, device=device)

    reasonings = []
    for row in ranked_df.itertuples():
        c       = record_map.get(row.candidate_id, {})
        profile = c.get("profile", {})
        sig     = c.get("redrob_signals", {})

        yoe    = profile.get("years_of_experience", "")
        loc    = profile.get("location", profile.get("country", ""))
        notice = sig.get("notice_period_days")
        title  = profile.get("current_title", "")

        best_sent = _best_jd_sentence(c, jd_emb, embed_model)

        # ── Metadata tag: [Xy exp; City-based; notice Nd] ────────────────
        meta_parts = []
        if yoe:
            meta_parts.append(f"{yoe}y exp")
        if loc:
            meta_parts.append(f"{loc}-based")
        if notice and isinstance(notice, int) and notice > 60:
            meta_parts.append(f"notice {notice}d")
        meta = "; ".join(meta_parts)

        # ── Full reasoning string ─────────────────────────────────────────
        # Format: "{Title} ({yoe}y): {best JD-aligned sentence} [metadata]"
        # If no title available, fall back to sentence + metadata only.
        if title and yoe:
            prefix = f"{title} ({yoe}y): "
        elif title:
            prefix = f"{title}: "
        else:
            prefix = ""

        reasoning = prefix + best_sent + (f" [{meta}]" if meta else "")
        reasonings.append(reasoning)

    return reasonings
