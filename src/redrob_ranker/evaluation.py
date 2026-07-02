"""Ranking evaluation metrics and the cross-strategy comparison table.

evaluate_ranking / _composite (Section 8) and print_comparison_table
(Section 11). Logic unchanged.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, ndcg_score

from .config import COMPOSITE_WEIGHTS

def _composite(m: dict) -> float:
    return sum(COMPOSITE_WEIGHTS[k] * m[k] for k in COMPOSITE_WEIGHTS)


def evaluate_ranking(ranked_df: pd.DataFrame, approach_name: str) -> dict:
    ranked_df = ranked_df.copy().reset_index(drop=True)
    n = len(ranked_df)
    true_rel = ranked_df["Final_Score_NDCG"].values
    pred_scores = (n - ranked_df.index).values.astype(float)

    ndcg_10 = ndcg_score([true_rel], [pred_scores], k=10)
    ndcg_50 = ndcg_score([true_rel], [pred_scores], k=min(50, n))
    ndcg_all = ndcg_score([true_rel], [pred_scores])
    rho, _ = spearmanr(true_rel, pred_scores)
    binary = (true_rel >= 3).astype(int)
    map_val = average_precision_score(binary, pred_scores)
    p10 = binary[:10].mean()
    p20 = binary[:20].mean()

    metrics = {
        "NDCG@10": ndcg_10,
        "NDCG@50": ndcg_50,
        "NDCG@all": ndcg_all,
        "MAP": map_val,
        "Spearman": rho,
        "P@10": p10,
        "P@20": p20,
    }
    metrics["Final Score"] = _composite(metrics)

    print("=" * 56)
    print(f"  {approach_name}")
    print("=" * 56)
    for k in ["NDCG@10", "NDCG@50", "NDCG@all", "MAP", "Spearman", "P@10", "P@20"]:
        print(f"  {k:<12}: {metrics[k]:.4f}")
    print(f"  {'─'*36}")
    print(f"  Final Score: {metrics['Final Score']:.4f}")
    print("=" * 56)

    # Top-10 preview
    print(
        f"\n  {'Rank':<5} {'Candidate ID':<16} {'Tech':>5} {'Ctx':>5} {'Beh':>5} {'GT':>5}"
    )
    print("  " + "─" * 40)
    for rank, row in enumerate(ranked_df.head(10).itertuples(), 1):
        print(
            f"  {rank:<5} {row.candidate_id:<16} "
            f"{int(row.Tech_Fit):>5} {int(row.Context_Fit):>5} "
            f"{int(row.Behavior_Fit):>5} {int(row.Final_Score_NDCG):>5}"
        )
    print()
    return metrics


def print_comparison_table(results: dict):
    names = list(results.keys())
    cw = 13
    header = f"  {'Metric':<14}" + "".join(f"{n:>{cw}}" for n in names)
    div = "  " + "─" * (14 + cw * len(names))
    width = 16 + cw * len(names)

    print("\n" + "=" * width)
    print("  FINAL RANKING STRATEGY COMPARISON")
    print("  (XGBoost Tech_Fit + Deterministic Context/Behavior)")
    print("=" * width)
    print(header)
    print(div)
    for m in [
        "NDCG@10",
        "NDCG@50",
        "NDCG@all",
        "MAP",
        "P@10",
        "P@20",
        "Spearman",
        "Final Score",
    ]:
        print(f"  {m:<14}" + "".join(f"{results[n].get(m,0):>{cw}.4f}" for n in names))
    print("=" * width)
    best = max(names, key=lambda n: results[n]["Final Score"])
    print(f"\n  ✓ Best strategy: {best}\n")
