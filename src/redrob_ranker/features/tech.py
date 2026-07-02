"""Tech_Fit prediction — XGBoost classifier over all-MiniLM-L6-v2 embeddings."""

import numpy as np
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

from ..config import TECH_CLF_PARAMS


def _tech_extract_text(c: dict) -> str:
    profile = c.get("profile", {})
    parts = [profile.get("headline", ""), profile.get("summary", "")]
    for role in c.get("career_history", []):
        parts.append(role.get("title", ""))
        parts.append(role.get("description", ""))
    for skill in c.get("skills", []):
        parts.append(skill.get("name", ""))
    return " ".join(filter(None, parts))


def predict_tech_fit(
    train_records: list,
    records_to_rank: list,
    evaluation: bool = True,
) -> tuple:
    print("[*] Generating embeddings (all-MiniLM-L6-v2)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate embeddings
    train_texts = [_tech_extract_text(r) for r in train_records]
    rank_texts = [_tech_extract_text(r) for r in records_to_rank]

    X_train = embed_model.encode(
        train_texts,
        show_progress_bar=True,
        batch_size=64,
    )

    X_rank = embed_model.encode(
        rank_texts,
        show_progress_bar=True,
        batch_size=64,
    )

    # Labels only exist for the training records
    gt_tech = np.array([int(r["parsed"].get("Tech_Fit", 0)) for r in train_records])

    unique, counts = np.unique(gt_tech, return_counts=True)
    can_stratify = bool((counts >= 2).all() and len(train_records) >= 50)

    idx_tr, idx_te = train_test_split(
        np.arange(len(train_records)),
        test_size=0.2,
        random_state=42,
        stratify=gt_tech if can_stratify else None,
    )

    # Remap labels to contiguous [0..K-1] for XGBoost
    unique_classes = sorted(np.unique(gt_tech))
    label_to_idx = {c: i for i, c in enumerate(unique_classes)}
    idx_to_label = {i: c for i, c in enumerate(unique_classes)}
    gt_mapped = np.array([label_to_idx[c] for c in gt_tech])

    params = {**TECH_CLF_PARAMS, "num_class": len(unique_classes)}
    clf = xgb.XGBClassifier(**params)

    if evaluation:
        print(
            f"[*] Training XGBoost Tech_Fit on {len(idx_tr)} samples "
            f"(classes {[int(c) for c in unique_classes]})..."
        )

        # Train on 80% for evaluation
        clf.fit(X_train[idx_tr], gt_mapped[idx_tr])

    else:
        print(
            f"[*] Training XGBoost Tech_Fit on all {len(train_records)} labelled samples..."
        )

        # Train on ALL labelled data for production
        clf.fit(X_train, gt_mapped)

    # Predict on whichever dataset we are ranking
    pred_mapped = clf.predict(X_rank)
    pred_tech = np.array([idx_to_label[p] for p in pred_mapped])

    if evaluation:
        acc = (pred_tech[idx_te] == gt_tech[idx_te]).mean() * 100
        print(f"    Tech_Fit held-out accuracy (20%): {acc:.1f}%")
        return pred_tech, idx_tr, idx_te, embed_model

    return pred_tech, [], [], embed_model
