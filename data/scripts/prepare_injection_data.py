"""Prepare the injection detection dataset for ml/train_detector.py.

Downloads and merges two HuggingFace datasets:
  - deepset/prompt-injections  (662 rows, clean baseline, widely validated)
  - neuralchemy/Prompt-injection-dataset  (29 attack categories, 2025-era techniques)

Produces data/injection.jsonl in the format required by ml/common.load_jsonl_records:
    {"text": "...", "label": 0|1, "group_id": "..."}

group_id is the attack category (for injection=1 rows) or safe-<topic> (for label=0 rows).
This ensures the group-aware split keeps attack families together so variants of the
same attack don't leak across train/valid/test, preventing inflated metrics.

Usage:
    pip install -r ml/requirements-ml.txt
    python -m data.scripts.prepare_injection_data
    # or:
    python data/scripts/prepare_injection_data.py --output data/injection.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# Columns in deepset/prompt-injections
# text → text, label → label (already 0/1)
DEEPSET_COLUMNS = {"text": "text", "label": "label"}

# Columns in neuralchemy/Prompt-injection-dataset
NEURALCHEMY_TEXT_COL = "text"
NEURALCHEMY_LABEL_COL = "label"
NEURALCHEMY_CATEGORY_COL = "category"

# Safe topic keywords → group_id
_SAFE_KEYWORDS = [
    ("password", "safe-password"), ("sign in", "safe-password"), ("login", "safe-password"),
    ("refund", "safe-refund"), ("subscription", "safe-refund"), ("billing", "safe-refund"),
    ("travel", "safe-policy"), ("policy", "safe-policy"), ("leave", "safe-policy"),
    ("translate", "safe-translation"), ("summarize", "safe-general"),
    ("photosynthesis", "safe-education"), ("explain", "safe-education"),
    ("meeting", "safe-productivity"), ("agenda", "safe-productivity"),
    ("shipment", "safe-shipping"), ("order", "safe-shipping"),
]


def _normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


def _safe_group(text: str) -> str:
    lc = text.lower()
    for keyword, group in _SAFE_KEYWORDS:
        if keyword in lc:
            return group
    return "safe-general"


def _injection_group(text: str, category: str | None) -> str:
    if category:
        slug = re.sub(r"[^a-z0-9]+", "-", str(category).lower()).strip("-")
        return f"attack-{slug}" if slug else "attack-other"
    lc = text.lower()
    if any(p in lc for p in ["ignore previous", "disregard", "forget your", "override"]):
        return "attack-instruction-override"
    if any(p in lc for p in ["jailbreak", "developer mode", "no restrictions", "bypass"]):
        return "attack-jailbreak"
    if any(p in lc for p in ["system prompt", "hidden prompt", "developer message"]):
        return "attack-extraction"
    if any(p in lc for p in ["you are now", "act as", "pretend to be"]):
        return "attack-role-manipulation"
    return "attack-other"


def load_deepset() -> list[dict]:
    from datasets import load_dataset
    print("[INFO] Loading deepset/prompt-injections...")
    ds = load_dataset("deepset/prompt-injections", split="train")
    records = []
    for row in ds:
        text = str(row.get("text", row.get("prompt", ""))).strip()
        label = int(row.get("label", 0))
        if not text:
            continue
        group_id = _injection_group(text, None) if label == 1 else _safe_group(text)
        records.append({"text": text, "label": label, "group_id": group_id, "_source": "deepset"})
    print(f"  → {len(records)} rows from deepset/prompt-injections")
    return records


def load_neuralchemy() -> list[dict]:
    from datasets import load_dataset
    print("[INFO] Loading neuralchemy/Prompt-injection-dataset...")
    try:
        ds = load_dataset("neuralchemy/Prompt-injection-dataset", split="train")
    except Exception as e:
        print(f"  [WARN] Could not load neuralchemy dataset: {e} — skipping.")
        return []
    records = []
    for row in ds:
        text = str(row.get(NEURALCHEMY_TEXT_COL, "")).strip()
        raw_label = row.get(NEURALCHEMY_LABEL_COL, 0)
        label = int(bool(raw_label))  # normalize to 0/1
        category = str(row.get(NEURALCHEMY_CATEGORY_COL, "")).strip() or None
        if not text:
            continue
        group_id = _injection_group(text, category) if label == 1 else _safe_group(text)
        records.append({"text": text, "label": label, "group_id": group_id, "_source": "neuralchemy"})
    print(f"  → {len(records)} rows from neuralchemy/Prompt-injection-dataset")
    return records


def dedupe_and_balance(records: list[dict], max_per_class: int = 5000) -> list[dict]:
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

    # Balance and cap
    target = min(max_per_class, min(len(by_label[0]), len(by_label[1])))
    balanced = by_label[0][:target] + by_label[1][:target]
    print(f"[INFO] After dedup+balance: {len(by_label[0])} safe, {len(by_label[1])} injection → {len(balanced)} total")
    return balanced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare injection detection dataset.")
    parser.add_argument("--output", type=Path, default=Path("data/injection.jsonl"))
    parser.add_argument("--max-per-class", type=int, default=5000,
                        help="Max examples per class after balancing (default 5000)")
    parser.add_argument("--no-neuralchemy", action="store_true",
                        help="Skip the neuralchemy dataset (use deepset only)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import datasets  # noqa: F401
    except ImportError:
        print("[ERROR] datasets not installed. Run: pip install -r ml/requirements-ml.txt")
        sys.exit(1)

    records = load_deepset()
    if not args.no_neuralchemy:
        records += load_neuralchemy()

    if not records:
        print("[ERROR] No records loaded. Check dataset availability.")
        sys.exit(1)

    balanced = dedupe_and_balance(records, max_per_class=args.max_per_class)

    # Remove internal _source field before writing
    output_records = [{k: v for k, v in r.items() if not k.startswith("_")} for r in balanced]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    label_counts = {0: sum(1 for r in output_records if r["label"] == 0),
                    1: sum(1 for r in output_records if r["label"] == 1)}
    group_counts = len({r["group_id"] for r in output_records})
    print(f"\n[OK] Wrote {len(output_records)} records to {args.output}")
    print(f"     Label 0 (safe):      {label_counts[0]}")
    print(f"     Label 1 (injection): {label_counts[1]}")
    print(f"     Unique group_ids:    {group_counts}")
    print(f"\nNext step:")
    print(f"  python -m ml.train_detector --task injection --data {args.output} \\")
    print(f"    --output ml/artifacts/injection-v1 --lora")


if __name__ == "__main__":
    main()
