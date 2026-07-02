"""Streaming JSONL loaders and golden-label merging.

Loads the candidate pool and golden labels, normalises the two label record
shapes, and merges them by candidate_id.
"""

import json
import os

from .config import GOLDEN_COMBINED, GOLDEN_SPLIT_FILES, POOL_FILES


def _normalize_label_record(label_rec: dict):
    if "parsed" in label_rec:
        cid = label_rec.get("candidate_id") or label_rec.get("Candidate_ID")
        return cid, label_rec.get("parsed")
    if "Tech_Fit" in label_rec or "Candidate_ID" in label_rec:
        cid = label_rec.get("Candidate_ID") or label_rec.get("candidate_id")
        return cid, label_rec
    return None, None


def _ensure_combined_golden(golden_path: str, split_files: tuple) -> str:
    if os.path.exists(golden_path):
        return golden_path
    found = [p for p in split_files if os.path.exists(p)]
    if not found:
        print(f"[!] Golden labels not found at {golden_path} or split files.")
        return golden_path
    print(f"[*] Building {golden_path} from: {found}")
    with open(golden_path, "w", encoding="utf-8") as out:
        for p in found:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
    return golden_path


def load_golden_set(data_dir: str) -> list:
    golden_path = os.path.join(data_dir, GOLDEN_COMBINED)
    split_files = [os.path.join(data_dir, f) for f in GOLDEN_SPLIT_FILES]
    pool_files = [os.path.join(data_dir, f) for f in POOL_FILES]
    golden_path = _ensure_combined_golden(golden_path, split_files)

    pool: dict = {}
    for path in pool_files:
        if not os.path.exists(path):
            print(f"[!] Pool file not found, skipping: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    pool[rec["candidate_id"]] = rec

    merged, skipped = [], 0
    with open(golden_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            label_rec = json.loads(line)
            cid, parsed = _normalize_label_record(label_rec)
            if not cid or not parsed:
                skipped += 1
                continue
            full = pool.get(cid)
            if full is None:
                skipped += 1
                continue
            record = dict(full)
            record["candidate_id"] = cid
            record["parsed"] = parsed
            merged.append(record)

    if skipped:
        print(f"[!] Skipped {skipped} record(s).")
    return merged


def load_candidate_pool(data_dir: str) -> list:
    path = os.path.join(data_dir, "candidates.jsonl")

    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records
