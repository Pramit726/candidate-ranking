"""Write the final ranking CSV: candidate_id, rank, score, reasoning."""

import csv
import os

import numpy as np
import pandas as pd


def save_csv_output(
    ranked_df: pd.DataFrame,
    reasonings: list,
    output_path: str,
    top_n: int,
    score_col: str = "rank_score",
):
    """Write final CSV with candidate_id, rank, score (0–1 normalised), reasoning."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out = ranked_df.copy().reset_index(drop=True)

    # Normalise score to [0, 1]
    if score_col in out.columns:
        raw = out[score_col].values.astype(float)
        max_val = raw.max() if raw.max() > 0 else 1.0
        out["score"] = (raw / max_val).round(4)
    elif "ltr_score" in out.columns:
        raw = out["ltr_score"].values.astype(float)
        rng = raw.max() - raw.min()
        out["score"] = ((raw - raw.min()) / (rng if rng > 0 else 1)).round(4)
    else:
        # Rank-based score: 1.0 for rank 1 → ~0.0 for last
        n = len(out)
        out["score"] = ((n - np.arange(n)) / n).round(4)

    # sort by score desc, then candidate_id asc
    if "candidate_id" in out.columns:
        out = out.sort_values(
            by=["score", "candidate_id"], ascending=[False, True]
        ).reset_index(drop=True)
    else:
        out = out.sort_values(by=["score"], ascending=[False]).reset_index(drop=True)

    out = out.head(top_n).copy().reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    out["reasoning"] = reasonings[: len(out)]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for row in out.itertuples():
            writer.writerow(
                [row.candidate_id, row.rank, f"{row.score:.4f}", row.reasoning]
            )

    print(f"[*] Final ranking CSV saved → {output_path}  ({len(out)} rows)")
