"""Prepare the toxicity/safety detection dataset for ml/train_detector.py.

Downloads and merges two HuggingFace datasets:
  - google/jigsaw_toxicity_pred  (~160K Wikipedia comments, multi-label → binary)
  - toxigen/toxigen-data          (~274K implicit/adversarial hate speech, binary)

Produces data/toxicity.jsonl in the format required by ml/common.load_jsonl_records:
    {"text": "...", "label": 0|1, "group_id": "..."}

group_id is the toxicity category for label=1 rows (violence, harassment, self_harm, etc.)
and safe-<topic> for label=0 rows, keeping similar content together during group-aware
splitting so attack families don't leak across train/valid/test.

Usage:
    pip install -r ml/requirements-ml.txt
    python -m data.scripts.prepare_toxicity_data
    # or with options:
    python data/scripts/prepare_toxicity_data.py --output data/toxicity.jsonl --max-per-class 5000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


# Jigsaw multi-label columns -> binary
JIGSAW_LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
JIGSAW_TEXT_COL = "comment_text"

# ToxiGen columns
TOXIGEN_TEXT_COL = "text"
TOXIGEN_LABEL_COL = "toxicity_human"   # float 0-5; >=2.5 → toxic
TOXIGEN_GROUP_COL = "target_group"


def _normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


def _jigsaw_group(row: dict) -> str:
    """Map Jigsaw multi-label to a single most-specific group_id."""
    if int(row.get("threat", 0)):
        return "toxicity-threat"
    if int(row.get("severe_toxic", 0)):
        return "toxicity-severe"
    if int(row.get("identity_hate", 0)):
        return "toxicity-identity-hate"
    if int(row.get("insult", 0)):
        return "toxicity-insult"
    if int(row.get("obscene", 0)):
        return "toxicity-obscene"
    if int(row.get("toxic", 0)):
        return "toxicity-general"
    return "safe-general"


def load_jigsaw(max_rows: int = 10000) -> list[dict]:
    from datasets import load_dataset
    print("[INFO] Loading google/jigsaw_toxicity_pred (train split)...")
    try:
        ds = load_dataset("google/jigsaw_toxicity_pred", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"  [WARN] jigsaw_toxicity_pred unavailable: {e}")
        print("  Trying alternative: jigsaw-unintended-bias...")
        try:
            ds = load_dataset("jigsaw_unintended_bias", split="train", trust_remote_code=True)
        except Exception as e2:
            print(f"  [WARN] Both Jigsaw datasets unavailable: {e2} — skipping.")
            return []

    records = []
    for row in ds:
        text = str(row.get(JIGSAW_TEXT_COL, row.get("comment_text", ""))).strip()
        if not text or len(text) < 10:
            continue
        # Binary: toxic if any of the label cols is 1
        is_toxic = any(int(row.get(col, 0)) for col in JIGSAW_LABEL_COLS if col in row)
        label = 1 if is_toxic else 0
        group_id = _jigsaw_group(row)
        records.append({"text": text, "label": label, "group_id": group_id, "_source": "jigsaw"})
        if len(records) >= max_rows:
            break

    toxic_n = sum(1 for r in records if r["label"] == 1)
    safe_n = len(records) - toxic_n
    print(f"  -> {len(records)} rows from jigsaw ({toxic_n} toxic, {safe_n} safe)")
    return records


def load_toxigen(max_rows: int = 5000) -> list[dict]:
    from datasets import load_dataset
    print("[INFO] Loading toxigen/toxigen-data...")
    try:
        ds = load_dataset("toxigen/toxigen-data", name="train", split="train", trust_remote_code=True)
    except Exception:
        try:
            ds = load_dataset("toxigen/toxigen-data", split="train", trust_remote_code=True)
        except Exception as e:
            print(f"  [WARN] toxigen/toxigen-data unavailable: {e} — skipping.")
            return []

    records = []
    for row in ds:
        text = str(row.get(TOXIGEN_TEXT_COL, row.get("generation", ""))).strip()
        if not text or len(text) < 10:
            continue
        # toxicity_human is a float 0-5 (annotator average); >= 2.5 → toxic
        raw = row.get(TOXIGEN_LABEL_COL, row.get("toxicity_ai", 0))
        try:
            score = float(raw)
        except (TypeError, ValueError):
            score = 0.0
        label = 1 if score >= 2.5 else 0
        target = str(row.get(TOXIGEN_GROUP_COL, row.get("target_group", ""))).strip()
        group_id = f"implicit-hate-{target.lower().replace(' ', '-')}" if label == 1 else "safe-general"
        records.append({"text": text, "label": label, "group_id": group_id, "_source": "toxigen"})
        if len(records) >= max_rows:
            break

    toxic_n = sum(1 for r in records if r["label"] == 1)
    safe_n = len(records) - toxic_n
    print(f"  -> {len(records)} rows from toxigen ({toxic_n} toxic, {safe_n} safe)")
    return records


def dedupe_and_balance(records: list[dict], max_per_class: int, seed: int = 42) -> list[dict]:
    seen: set[str] = set()
    by_label: dict[int, list[dict]] = {0: [], 1: []}
    for rec in records:
        norm = _normalize(rec["text"])
        if not norm or norm in seen:
            continue
        seen.add(norm)
        lbl = rec["label"]
        if lbl in by_label:
            by_label[lbl].append(rec)

    rng = random.Random(seed)
    for lbl in by_label:
        rng.shuffle(by_label[lbl])

    target = min(max_per_class, min(len(by_label[0]), len(by_label[1])))
    balanced = by_label[0][:target] + by_label[1][:target]
    print(f"[INFO] After dedup+balance: {len(by_label[0])} safe, {len(by_label[1])} toxic -> {len(balanced)} total")
    return balanced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare toxicity/safety detection dataset.")
    parser.add_argument("--output", type=Path, default=Path("data/toxicity.jsonl"))
    parser.add_argument("--max-per-class", type=int, default=5000,
                        help="Max examples per class after balancing (default 5000)")
    parser.add_argument("--jigsaw-rows", type=int, default=15000,
                        help="Max rows to load from Jigsaw before filtering (default 15000)")
    parser.add_argument("--toxigen-rows", type=int, default=8000,
                        help="Max rows to load from ToxiGen before filtering (default 8000)")
    parser.add_argument("--no-toxigen", action="store_true",
                        help="Skip ToxiGen (use Jigsaw only)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import datasets  # noqa: F401
    except ImportError:
        print("[ERROR] datasets not installed. Run: pip install -r ml/requirements-ml.txt")
        sys.exit(1)

    records = load_jigsaw(max_rows=args.jigsaw_rows)
    if not args.no_toxigen:
        records += load_toxigen(max_rows=args.toxigen_rows)

    if not records:
        print("[ERROR] No records loaded. Check dataset availability and HuggingFace access.")
        sys.exit(1)

    balanced = dedupe_and_balance(records, max_per_class=args.max_per_class, seed=args.seed)

    output_records = [{k: v for k, v in r.items() if not k.startswith("_")} for r in balanced]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    label_counts = {0: sum(1 for r in output_records if r["label"] == 0),
                    1: sum(1 for r in output_records if r["label"] == 1)}
    group_counts = len({r["group_id"] for r in output_records})
    print(f"\n[OK] Wrote {len(output_records)} records to {args.output}")
    print(f"     Label 0 (safe):   {label_counts[0]}")
    print(f"     Label 1 (toxic):  {label_counts[1]}")
    print(f"     Unique group_ids: {group_counts}")
    print(f"\nNext step:")
    print(f"  python -m ml.train_detector --task toxicity --data {args.output} \\")
    print(f"    --output ml/artifacts/toxicity-v1 --lora")


if __name__ == "__main__":
    main()
