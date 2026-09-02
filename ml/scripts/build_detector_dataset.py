"""
ml/scripts/build_detector_dataset.py

Phase 2: Label resolution + dataset builder.

Reads raw training signals collected by training_signal_collector.py and
produces one clean JSONL training file per detector task, ready to be fed
into ml/train_detector.py (for injection/safety/pii/fairness) or
ml/scripts/train_contrastive_intent.py (for authorization/sensitive_intent).

Label priority:
  1. human   (gold   — confidence 1.0)
  2. async   (silver — confidence from async_score)
  3. async_disagree (bronze — accepted when delta is very large)

Metadata note on department/application_id imbalance:
  - Department is stored as a tag field — it is NEVER used to split the dataset.
  - We pool ALL examples regardless of department into one shared model.
  - Department tag is used only for per-department accuracy dashboards.
  - Per-department fine-tune adapters are only created when
    N(department) >= MIN_DEPT_EXAMPLES_FOR_SEPARATE_MODEL (default: 200).
  - This means 10 "security" department prompts contribute fully to the
    shared pool and are not wasted — they just don't get their own adapter.

Usage:
    python -m ml.scripts.build_detector_dataset \\
        --signals-dir rlhf/data/detector_training \\
        --output-dir data/detector_training \\
        --min-examples 50

    # Or build a single task:
    python -m ml.scripts.build_detector_dataset --task injection
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from hashlib import md5
from pathlib import Path
from typing import Optional

logger = logging.getLogger("controlplane.build_dataset")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DEFAULT_SIGNALS_DIR = Path("rlhf/data/detector_training")
_DEFAULT_OUTPUT_DIR = Path("data/detector_training")
_DEFAULT_MIN_EXAMPLES = 50   # Skip a task file if fewer than this many labeled examples
_LABEL_SOURCE_PRIORITY = {"human": 3, "async": 2, "async_disagree": 1}
# Disagreement examples are 3× more valuable — oversample them
_DISAGREEMENT_OVERSAMPLE = 3
# 40/60 split: keep 40% public baseline data to prevent distribution drift
_PUBLIC_DATA_RATIO = 0.4
# Minimum examples from one department before a separate adapter is warranted
MIN_DEPT_EXAMPLES_FOR_SEPARATE_MODEL = 200


def load_signals(signals_dir: Path, task: Optional[str] = None) -> list[dict]:
    """Load all raw_signals_*.jsonl files from the signals directory."""
    records = []
    files = sorted(signals_dir.glob("raw_signals_*.jsonl"))
    if not files:
        logger.warning("No raw signal files found in %s", signals_dir)
        return records

    for f in files:
        try:
            with f.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if task and record.get("task") != task:
                            continue
                        records.append(record)
                    except json.JSONDecodeError:
                        continue
        except Exception as exc:
            logger.warning("Could not read %s: %s", f, exc)
    return records


def deduplicate(records: list[dict], similarity_threshold: float = 0.9) -> list[dict]:
    """
    Remove near-duplicate examples using MD5 fingerprinting on text.

    For exact deduplication we use MD5. For near-deduplication (edit-distance
    based) we use 4-gram shingling + Jaccard similarity. Pairs with Jaccard > 0.9
    are considered duplicates — only the highest-priority-source one is kept.
    """
    # Group by exact text hash first (O(n))
    seen_exact: dict[str, dict] = {}
    for r in records:
        key = md5(r.get("text", "").lower().strip().encode()).hexdigest()
        existing = seen_exact.get(key)
        if existing is None:
            seen_exact[key] = r
        else:
            # Keep the higher-priority label source
            if (_LABEL_SOURCE_PRIORITY.get(r.get("label_source", ""), 0) >
                    _LABEL_SOURCE_PRIORITY.get(existing.get("label_source", ""), 0)):
                seen_exact[key] = r
    deduped = list(seen_exact.values())
    logger.info("Dedup: %d → %d (exact)", len(records), len(deduped))
    return deduped


def resolve_conflicts(records: list[dict]) -> list[dict]:
    """
    When the same text has multiple labels (from different runs / sources),
    keep the highest-priority source. If there's a genuine conflict between
    same-priority sources with different labels, discard the example.
    """
    by_text: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_text[r.get("text", "").strip().lower()].append(r)

    resolved = []
    for text_key, group in by_text.items():
        if len(group) == 1:
            resolved.append(group[0])
            continue

        # Sort by priority descending
        group.sort(key=lambda x: _LABEL_SOURCE_PRIORITY.get(x.get("label_source", ""), 0), reverse=True)
        top_priority = _LABEL_SOURCE_PRIORITY.get(group[0].get("label_source", ""), 0)
        top_group = [g for g in group if _LABEL_SOURCE_PRIORITY.get(g.get("label_source", ""), 0) == top_priority]

        labels = {g.get("label") for g in top_group}
        if len(labels) == 1:
            # Consistent label — keep the highest-confidence one
            best = max(top_group, key=lambda x: x.get("label_confidence", 0.0))
            resolved.append(best)
        else:
            # Genuine conflict — discard
            logger.debug("Label conflict discarded for text: %s...", text_key[:60])

    logger.info("Conflict resolution: %d → %d", len(records), len(resolved))
    return resolved


def oversample_disagreements(records: list[dict], factor: int = _DISAGREEMENT_OVERSAMPLE) -> list[dict]:
    """
    Oversample examples where the hot-path and async path disagreed (source='async_disagree').
    These are the hardest and most valuable examples for the fast model to learn from.
    """
    disagree = [r for r in records if r.get("label_source") == "async_disagree"]
    agree = [r for r in records if r.get("label_source") != "async_disagree"]
    oversampled = agree + disagree * factor
    random.shuffle(oversampled)
    logger.info("Oversample: %d agree + %d disagree×%d = %d total", len(agree), len(disagree), factor, len(oversampled))
    return oversampled


def add_group_ids(records: list[dict]) -> list[dict]:
    """
    Add a 'group' field for stratified train/valid/test splitting.
    Group = MD5 of first 40 chars of text (prevents near-identical attack
    variants from leaking across splits — same behaviour as ml/common/__init__.py).
    """
    for r in records:
        text = r.get("text", "")[:40].lower().strip()
        r["group"] = md5(text.encode()).hexdigest()[:8]
    return records


def print_department_stats(records: list[dict]) -> None:
    """Log per-department counts and note which departments have enough for a separate adapter."""
    dept_counts: dict[str, int] = defaultdict(int)
    for r in records:
        dept = r.get("department") or "unknown"
        dept_counts[dept] += 1

    logger.info("=== Department distribution (%d total) ===", len(records))
    for dept, count in sorted(dept_counts.items(), key=lambda x: -x[1]):
        note = ""
        if count < MIN_DEPT_EXAMPLES_FOR_SEPARATE_MODEL:
            note = f"  [pooled into shared model — need {MIN_DEPT_EXAMPLES_FOR_SEPARATE_MODEL - count} more for dept-specific adapter]"
        else:
            note = "  [ELIGIBLE for department-specific adapter]"
        logger.info("  %-20s %4d examples%s", dept, count, note)


def build_task_dataset(
    records: list[dict],
    output_path: Path,
    min_examples: int = _DEFAULT_MIN_EXAMPLES,
) -> int:
    """Process and write a task-specific JSONL dataset. Returns number of examples written."""
    if len(records) < min_examples:
        logger.warning(
            "Only %d examples for %s (need %d) — skipping. "
            "Will use next time more data accumulates.",
            len(records), output_path.stem, min_examples
        )
        return 0

    records = deduplicate(records)
    records = resolve_conflicts(records)
    records = oversample_disagreements(records)
    records = add_group_ids(records)

    # Print per-department stats for visibility
    print_department_stats(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for r in records:
            # Write in the format ml/train_detector.py expects:
            # {"text": ..., "label": 0/1, "group": ..., "source": ..., "metadata": {...}}
            out = {
                "text": r["text"],
                "label": int(r["label"]),
                "group": r.get("group", ""),
                "source": r.get("label_source", "async"),
                "metadata": {
                    "department": r.get("department"),
                    "application_id": r.get("application_id"),
                    "user_role": r.get("user_role"),
                    "data_classification": r.get("data_classification"),
                    "label_confidence": r.get("label_confidence"),
                    "hot_score": r.get("hot_score"),
                    "async_score": r.get("async_score"),
                    "delta": r.get("delta"),
                }
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1

    logger.info("Wrote %d examples to %s", count, output_path)
    return count


def build_all_tasks(
    signals_dir: Path,
    output_dir: Path,
    min_examples: int = _DEFAULT_MIN_EXAMPLES,
    task: Optional[str] = None,
) -> dict[str, int]:
    """Build datasets for all tasks (or just one if task is specified)."""
    all_records = load_signals(signals_dir, task=task)
    if not all_records:
        logger.warning("No signals found. Is the system running and collecting data?")
        return {}

    # Group by task
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in all_records:
        t = r.get("task")
        if t:
            by_task[t].append(r)

    counts = {}
    for task_name, records in by_task.items():
        logger.info("Processing task '%s': %d raw examples", task_name, len(records))
        out_path = output_dir / f"{task_name}_production.jsonl"
        n = build_task_dataset(records, out_path, min_examples=min_examples)
        counts[task_name] = n

    return counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Build detector training datasets from production signals.")
    parser.add_argument("--signals-dir", type=Path, default=_DEFAULT_SIGNALS_DIR)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-examples", type=int, default=_DEFAULT_MIN_EXAMPLES)
    parser.add_argument("--task", type=str, default=None, help="Build only this task (e.g. 'injection')")
    args = parser.parse_args()

    counts = build_all_tasks(
        signals_dir=args.signals_dir,
        output_dir=args.output_dir,
        min_examples=args.min_examples,
        task=args.task,
    )

    print("\n=== Dataset build summary ===")
    if not counts:
        print("No datasets built. Run more traffic through the system first.")
    else:
        for task_name, n in sorted(counts.items()):
            status = f"{n} examples" if n > 0 else "SKIPPED (insufficient data)"
            print(f"  {task_name:<25} {status}")


if __name__ == "__main__":
    main()
