"""Pipeline orchestration.

Wires the stages together: load -> filter -> score (tech/context/behavior) ->
rank (3 strategies) -> evaluate -> pick best -> reason -> write CSV. The stage
logic is unchanged from the original single-file script; only the imports now
come from the package modules.
"""

import argparse
import os
import sys

from .config import (
    DEFAULT_AS_OF_DATE,
    DEFAULT_DATA_DIR,
    DEFAULT_MODELS_DIR,
    DEFAULT_OUTPUT,
    DEFAULT_TOP_N,
)
from .evaluation import evaluate_ranking, print_comparison_table
from .features import predict_behavior_fit, predict_context_fit, predict_tech_fit
from .filters import apply_pre_ranking_filters, print_filter_summary
from .io_utils import load_candidate_pool, load_golden_set
from .output import save_csv_output
from .reasoning import generate_reasoning, load_job_description
from .scoring import (
    build_candidates_data,
    rank_lambdamart,
    rank_lexicographical,
    rank_nonlinear_weighted,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Redrob AI – Final Ranking Pipeline (test set)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--as-of-date", default=DEFAULT_AS_OF_DATE)
    p.add_argument("--no-ltr", action="store_true", help="Skip LambdaMART (faster).")
    p.add_argument(
        "--no-filter", action="store_true", help="Skip pre-ranking exclusion filters."
    )
    p.add_argument(
        "--mode",
        choices=["test", "candidate"],
        default="test",
        help="test = evaluate on test set, candidate = rank candidates.jsonl",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  Redrob AI – Final Ranking Pipeline")
    print("=" * 60)
    print(f"  data-dir  : {args.data_dir}")
    print(f"  output    : {args.output}")
    print(f"  top-n     : {args.top_n}")
    print(f"  as-of-date: {args.as_of_date}")
    print(f"  mode      : {args.mode}")
    print(f"  LTR       : {not args.no_ltr}")
    print(f"  filters   : {not args.no_filter}")
    print("=" * 60 + "\n")

    # ── Stage 1: Load ───────────────────────────────────────────────

    if args.mode == "test":
        print("[*] Loading labelled test set...")
        train_records = load_golden_set(args.data_dir)
        records_to_rank = train_records

    else:
        print("[*] Loading labelled training data...")
        train_records = load_golden_set(args.data_dir)

        print("[*] Loading candidate pool...")
        records_to_rank = load_candidate_pool(args.data_dir)

    if not train_records:
        print("[!] No labelled records loaded.")
        sys.exit(1)

    if not records_to_rank:
        print("[!] No records to rank.")
        sys.exit(1)

    print(f"    Training records : {len(train_records)}")
    print(f"    Records to rank  : {len(records_to_rank)}\n")

    # ── Stage 1.5: Pre-ranking filters ──────────────────────────────

    if not args.no_filter:
        print("[*] Stage 1.5: Applying pre-ranking filters...")

        records_to_rank, fstats = apply_pre_ranking_filters(
            records_to_rank,
            args.as_of_date,
        )

        print_filter_summary(fstats)

        if not records_to_rank:
            print("[!] All records filtered out.")
            sys.exit(1)

    # ── Stage 2: Tech Fit ────────────────────────────────────────────

    print("[*] Stage 2: Tech_Fit prediction...")

    if args.mode == "test":
        train_pred_tech, idx_tr, idx_te, embed_model = predict_tech_fit(
            train_records=train_records,
            records_to_rank=train_records,
            evaluation=True,
        )

        # Same dataset in test mode
        pred_tech = train_pred_tech

    else:
        # Predictions for the training dataset (needed to train LambdaMART)
        train_pred_tech, _, _, _ = predict_tech_fit(
            train_records=train_records,
            records_to_rank=train_records,
            evaluation=False,
        )

        # Predictions for the candidate pool
        pred_tech, idx_tr, idx_te, embed_model = predict_tech_fit(
            train_records=train_records,
            records_to_rank=records_to_rank,
            evaluation=False,
        )

    print()

    # ── Stage 3: Context Fit ─────────────────────────────────────────

    print("[*] Stage 3: Context_Fit prediction...")

    train_context = predict_context_fit(
        train_records,
        evaluation=True,
    )

    if args.mode == "test":
        rank_context = train_context
    else:
        rank_context = predict_context_fit(
            records_to_rank,
            evaluation=False,
        )

    print()

    # ── Stage 4: Behavior Fit ────────────────────────────────────────

    print("[*] Stage 4: Behavior_Fit computation...")

    train_behavior = predict_behavior_fit(
        train_records,
        args.as_of_date,
        evaluation=True,
    )

    if args.mode == "test":
        rank_behavior = train_behavior
    else:
        rank_behavior = predict_behavior_fit(
            records_to_rank,
            args.as_of_date,
            evaluation=False,
        )

    print()

    # ── Stage 5: Build feature tables ────────────────────────────────

    train_candidates_data = build_candidates_data(
        train_records,
        train_pred_tech,
        train_context,
        train_behavior,
        include_labels=True,
    )

    if args.mode == "test":
        rank_candidates_data = train_candidates_data
    else:
        rank_candidates_data = build_candidates_data(
            records_to_rank,
            pred_tech,
            rank_context,
            rank_behavior,
            include_labels=False,
        )

    print(f"[*] Training candidates : {len(train_candidates_data)}")
    print(f"[*] Ranking candidates  : {len(rank_candidates_data)}\n")

    # ── Stage 6: Rank + Evaluate ──────────────────────────────────────────────

    if args.mode == "test":

        all_metrics: dict = {}

        print("[*] Stage 6a: Lexicographical Sort\n")
        ranked_lex = rank_lexicographical(rank_candidates_data)
        all_metrics["Lex Sort"] = evaluate_ranking(
            ranked_lex,
            "APPROACH A: LEXICOGRAPHICAL SORT",
        )

        print("[*] Stage 6b: Weighted Nonlinear Scoring\n")
        ranked_wt = rank_nonlinear_weighted(rank_candidates_data)
        all_metrics["Weighted"] = evaluate_ranking(
            ranked_wt,
            "APPROACH B: WEIGHTED NONLINEAR",
        )

        ranked_ltr = None

        if not args.no_ltr:
            print("[*] Stage 6c: LambdaMART (LTR)\n")

            ranked_ltr = rank_lambdamart(
                train_candidates_data=train_candidates_data,
                train_records=train_records,
                rank_candidates_data=rank_candidates_data,
                rank_records=records_to_rank,
                idx_tr=idx_tr,
                idx_te=idx_te,
                models_dir=args.models_dir,
                evaluation=True,
            )

            ltr_test_cids = {train_records[i]["candidate_id"] for i in idx_te}

            all_metrics["LambdaMART"] = evaluate_ranking(
                ranked_ltr[ranked_ltr["candidate_id"].isin(ltr_test_cids)].copy(),
                "APPROACH C: LambdaMART — held-out test set",
            )

        # ── Stage 7: Compare ─────────────────────────────────────────────

        print_comparison_table(all_metrics)

        best_name = max(
            all_metrics,
            key=lambda n: all_metrics[n]["Final Score"],
        )

        ranked_map = {
            "Lex Sort": ranked_lex,
            "Weighted": ranked_wt,
        }

        if ranked_ltr is not None:
            ranked_map["LambdaMART"] = ranked_ltr

        best_df = ranked_map[best_name]

    else:
        print(
            "[*] Ranking complete candidate pool using Weighted Nonlinear Scoring...\n"
        )
        best_name = "Weighted"
        best_df = rank_nonlinear_weighted(rank_candidates_data)

    # Choose the score column
    score_col = "rank_score" if "rank_score" in best_df.columns else "ltr_score"

    # ── Stage 8: Generate reasoning ───────────────────────────────────────────

    print("[*] Stage 8: Generating extractive reasoning...")

    jd_text = load_job_description(args.data_dir)

    record_map = {r["candidate_id"]: r for r in records_to_rank}

    reasonings = generate_reasoning(
        best_df.head(args.top_n),
        record_map,
        jd_text,
        embed_model,
    )

    # ── Stage 9: Save CSV output ──────────────────────────────────────────────

    # Choose output filename based on mode
    output_path = args.output

    if args.mode == "candidate":
        root, ext = os.path.splitext(output_path)
        output_path = f"{root}_candidate{ext}"
    else:
        root, ext = os.path.splitext(output_path)
        output_path = f"{root}_test{ext}"

    save_csv_output(
        best_df,
        reasonings,
        output_path,
        args.top_n,
        score_col,
    )

    print(f"\n[*] Done. Ranking strategy: {best_name}\n")
