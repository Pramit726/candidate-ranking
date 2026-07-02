"""Assemble candidate feature rows and the three ranking strategies.

build_candidates_data (Section 7) plus the Lexicographical, Weighted Nonlinear,
and LambdaMART rankers (Section 9). Logic unchanged from the original monolith.
"""

import os

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import LTR_PARAMS, RANKER_MODEL_FILE, W_BEHAVIOR, W_CONTEXT, W_TECH


def build_candidates_data(
    records,
    pred_tech,
    pred_context,
    pred_behavior,
    include_labels: bool = True,
) -> list:
    candidates = []

    for r, tf, cf, bf in zip(
        records,
        pred_tech,
        pred_context,
        pred_behavior,
    ):
        candidate = {
            "candidate_id": r["candidate_id"],
            "Tech_Fit": int(tf),
            "Context_Fit": int(cf),
            "Behavior_Fit": int(bf),
            "_record": r,
        }

        if include_labels:
            candidate["Final_Score_NDCG"] = int(
                r.get("parsed", {}).get("Final_Score_NDCG", 0)
            )

        candidates.append(candidate)

    return candidates


def rank_lexicographical(candidates_data: list) -> pd.DataFrame:
    df = pd.DataFrame(
        [{k: v for k, v in d.items() if k != "_record"} for d in candidates_data]
    )
    df["_vetoed"] = (df["Tech_Fit"] == 0) | (df["Context_Fit"] == 0)
    return (
        df.sort_values(
            by=["_vetoed", "Tech_Fit", "Context_Fit", "Behavior_Fit"],
            ascending=[True, False, False, False],
        )
        .drop(columns=["_vetoed"])
        .reset_index(drop=True)
    )


def rank_nonlinear_weighted(
    candidates_data: list, wt=W_TECH, wc=W_CONTEXT, wb=W_BEHAVIOR
) -> pd.DataFrame:
    df = pd.DataFrame(
        [{k: v for k, v in d.items() if k != "_record"} for d in candidates_data]
    )
    delta = np.where((df["Tech_Fit"] == 0) | (df["Context_Fit"] == 0), 0, 1)
    df["rank_score"] = delta * (
        wt * df["Tech_Fit"] + wc * df["Context_Fit"] + wb * df["Behavior_Fit"]
    )
    return df.sort_values("rank_score", ascending=False).reset_index(drop=True)


def rank_lambdamart(
    train_candidates_data: list,
    train_records: list,
    rank_candidates_data: list,
    rank_records: list,
    idx_tr,
    idx_te,
    models_dir: str,
    evaluation: bool = True,
) -> pd.DataFrame:
    """
    Train LambdaMART on labelled candidate data and rank another dataset.

    Test mode:
        train_candidates_data == rank_candidates_data
        evaluation=True

    Candidate mode:
        train_candidates_data -> labelled data
        rank_candidates_data  -> candidates.jsonl
        evaluation=False
    """

    train_record_map = {r["candidate_id"]: r for r in train_records}

    rank_record_map = {r["candidate_id"]: r for r in rank_records}

    def _features(d, record_map):
        r = record_map.get(d["candidate_id"], {})
        sig = r.get("redrob_signals", {})
        p = r.get("profile", {})

        return {
            "Tech_Fit": d["Tech_Fit"],
            "Context_Fit": d["Context_Fit"],
            "Behavior_Fit": d["Behavior_Fit"],
            "Years_of_Experience": float(p.get("years_of_experience", 0) or 0),
            "Notice_Period": sig.get("notice_period_days", 90),
            "GitHub_Score": max(sig.get("github_activity_score", 0), 0),
            "Response_Rate": sig.get("recruiter_response_rate", 0) or 0,
            "Open_to_Work": int(sig.get("open_to_work_flag", False)),
        }

    # ------------------------------
    # Training features
    # ------------------------------

    df_train = pd.DataFrame(
        [_features(d, train_record_map) for d in train_candidates_data]
    )

    # ------------------------------
    # Ranking features
    # ------------------------------

    df_rank = pd.DataFrame(
        [_features(d, rank_record_map) for d in rank_candidates_data]
    )

    # Labels only exist for training data
    y = np.array([d["Final_Score_NDCG"] for d in train_candidates_data])

    ranker = xgb.XGBRanker(**LTR_PARAMS)

    if evaluation:

        X_tr = df_train.iloc[idx_tr]
        X_te = df_train.iloc[idx_te]

        y_tr = y[idx_tr]
        y_te = y[idx_te]

        qid_tr = np.ones(len(X_tr), dtype=np.int32)
        qid_te = np.ones(len(X_te), dtype=np.int32)

        print(f"\n[*] Training XGBRanker (LambdaMART) on {len(X_tr)} samples...")

        ranker.fit(
            X_tr,
            y_tr,
            qid=qid_tr,
            eval_set=[(X_te, y_te)],
            eval_qid=[qid_te],
            verbose=20,
        )

    else:

        print(f"\n[*] Training XGBRanker on all {len(df_train)} labelled samples...")

        qid_train = np.ones(len(df_train), dtype=np.int32)

        ranker.fit(
            df_train,
            y,
            qid=qid_train,
            verbose=20,
        )

    # ------------------------------
    # Predict on ranking dataset
    # ------------------------------

    y_pred_all = ranker.predict(df_rank)

    save_path = os.path.join(models_dir, RANKER_MODEL_FILE)
    os.makedirs(models_dir, exist_ok=True)

    ranker.save_model(save_path)
    print(f"[*] Model saved → {save_path}")

    print("\n[*] Feature Importances:")

    for name, imp in sorted(
        zip(df_train.columns, ranker.feature_importances_),
        key=lambda x: -x[1],
    ):
        if imp > 0:
            print(f"    {name:<28}: {imp:.4f}")

    df_out = pd.DataFrame(
        [{k: v for k, v in d.items() if k != "_record"} for d in rank_candidates_data]
    )

    df_out["ltr_score"] = y_pred_all

    return df_out.sort_values(
        "ltr_score",
        ascending=False,
    ).reset_index(drop=True)
