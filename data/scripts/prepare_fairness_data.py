"""Prepare the fairness/bias detection dataset for ml/train_detector.py.

Downloads HateXplain from HuggingFace:
  - Explainable Hate Speech Detection (Kennedy et al., 2020)
  - Contains hate/offensive/normal labels + target demographic + rationale spans

Produces data/fairness.jsonl in the format required by ml/common.load_jsonl_records:
    {"text": "...", "label": 0|1, "group_id": "...", "rationale": "..."}

label=1 for hate/offensive speech, label=0 for normal.
group_id = target demographic (race, religion, gender, sexuality, etc.) which keeps
demographic-specific content together during group-aware splitting.
The "rationale" field (extra, not required by ml.common) captures the annotator-highlighted
spans that explain WHY the content is hateful — useful for future evidence extraction
in DetectorResult.evidence without any schema changes.

Usage:
    pip install -r ml/requirements-ml.txt
    python -m data.scripts.prepare_fairness_data
    # or:
    python data/scripts/prepare_fairness_data.py --output data/fairness.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


HATEXPLAIN_DATASET = "hatexplain"

# HateXplain label mapping (majority vote over annotators)
# 0=hatespeech, 1=offensive, 2=normal → binary: 0+1 → label=1, 2 → label=0
HATEXPLAIN_HATE = 0
HATEXPLAIN_OFFENSIVE = 1
HATEXPLAIN_NORMAL = 2

# Target community → normalized group_id slug
_GROUP_MAP = {
    "African": "race-african", "Arab": "ethnicity-arab", "Asian": "race-asian",
    "Caucasian": "race-caucasian", "Hispanic": "ethnicity-hispanic",
    "Indian": "ethnicity-indian", "Jewish": "religion-jewish",
    "Islam": "religion-islam", "Christian": "religion-christian",
    "Buddhist": "religion-buddhist", "Hindu": "religion-hindu",
    "Women": "gender-women", "Men": "gender-men",
    "LGBTQ": "sexuality-lgbtq", "Refugee": "identity-refugee",
    "Disability": "disability", "Economic": "economic-class",
    "":  "hate-other",
}


def _normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


def _majority_label(annotator_labels: list[int]) -> int | None:
    """Return the majority annotator label (0=hate, 1=offensive, 2=normal)."""
    if not annotator_labels:
        return None
    counts = {}
    for lbl in annotator_labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    return max(counts, key=lambda k: (counts[k], -k))


def _group_id(targets: list[str], binary_label: int) -> str:
    if binary_label == 0:
        return "safe-normal"
    for t in targets:
        for key, slug in _GROUP_MAP.items():
            if key and key.lower() in str(t).lower():
                return slug
    return "hate-other"


def _rationale_text(post_tokens: list[str], rationale_spans: list[list[int]]) -> str:
    """Extract rationale words as a string from token list + binary span masks."""
    if not rationale_spans or not post_tokens:
        return ""
    # majority-vote over annotator rationale masks
    n = len(post_tokens)
    vote = [0] * n
    for mask in rationale_spans:
        for i, bit in enumerate(mask[:n]):
            vote[i] += int(bit)
    threshold = len(rationale_spans) / 2
    rationale_tokens = [post_tokens[i] for i in range(n) if vote[i] > threshold]
    return " ".join(rationale_tokens)


def load_hatexplain(max_rows: int = 10000) -> list[dict]:
    from datasets import load_dataset
    print(f"[INFO] Loading {HATEXPLAIN_DATASET}...")
    try:
        ds = load_dataset(HATEXPLAIN_DATASET, split="train", trust_remote_code=True)
    except Exception as e:
        print(f"  [WARN] Could not load hatexplain: {e} — skipping.")
        return []

    records = []
    for row in ds:
        # post_tokens is a list of word tokens
        tokens = row.get("post_tokens", [])
        text = " ".join(tokens).strip()
        if not text or len(text) < 10:
            continue

        # annotators field has label list and rationale masks
        annotators = row.get("annotators", {})
        raw_labels = annotators.get("label", [])
        rationale_spans = row.get("rationales", [])

        majority = _majority_label(raw_labels)
        if majority is None:
            continue

        # Binary label: hate(0) or offensive(1) → 1; normal(2) → 0
        binary_label = 0 if majority == HATEXPLAIN_NORMAL else 1

        targets = row.get("target", [])
        if isinstance(targets, str):
            targets = [targets]

        group_id = _group_id(targets, binary_label)
        rationale = _rationale_text(tokens, rationale_spans)

        records.append({
            "text": text,
            "label": binary_label,
            "group_id": group_id,
            "rationale": rationale,
            "_source": "hatexplain",
        })
        if len(records) >= max_rows:
            break

    hate_n = sum(1 for r in records if r["label"] == 1)
    safe_n = len(records) - hate_n
    print(f"  -> {len(records)} rows from hatexplain ({hate_n} hate/offensive, {safe_n} normal)")
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
    print(f"[INFO] After dedup+balance: {len(by_label[0])} normal, {len(by_label[1])} hate -> {len(balanced)} total")
    return balanced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fairness/bias detection dataset.")
    parser.add_argument("--output", type=Path, default=Path("data/fairness.jsonl"))
    parser.add_argument("--max-per-class", type=int, default=3000,
                        help="Max examples per class after balancing (default 3000)")
    parser.add_argument("--max-load", type=int, default=10000,
                        help="Max rows to load from HateXplain before filtering (default 10000)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import datasets  # noqa: F401
    except ImportError:
        print("[ERROR] datasets not installed. Run: pip install -r ml/requirements-ml.txt")
        sys.exit(1)

    records = load_hatexplain(max_rows=args.max_load)

    if not records:
        print("[ERROR] No records loaded from HateXplain. Check HuggingFace connectivity.")
        sys.exit(1)

    balanced = dedupe_and_balance(records, max_per_class=args.max_per_class, seed=args.seed)

    # Write to JSONL — include rationale as extra field (ml.common ignores unknown fields)
    output_records = [{k: v for k, v in r.items() if not k.startswith("_")} for r in balanced]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    label_counts = {0: sum(1 for r in output_records if r["label"] == 0),
                    1: sum(1 for r in output_records if r["label"] == 1)}
    group_counts = len({r["group_id"] for r in output_records})
    print(f"\n[OK] Wrote {len(output_records)} records to {args.output}")
    print(f"     Label 0 (normal):         {label_counts[0]}")
    print(f"     Label 1 (hate/offensive): {label_counts[1]}")
    print(f"     Unique group_ids:         {group_counts}")
    print(f"     (rationale spans preserved in 'rationale' field)")
    print(f"\nNext step:")
    print(f"  python -m ml.train_detector --task fairness --data {args.output} \\")
    print(f"    --output ml/artifacts/fairness-v1 --lora")


if __name__ == "__main__":
    main()
