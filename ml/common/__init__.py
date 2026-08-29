"""ml/common -- shared ML utilities and evaluation routines."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

from ml.common.data_utils import default_group_key, group_aware_split, normalize_text
from ml.common.eval_utils import (
    calibration_curve_points,
    compare_models,
    compute_detection_metrics,
    find_threshold_for_target_recall,
)
from ml.common.lora_utils import (
    LoraTrainConfig,
    build_lora_model,
    run_training,
    tokenize_dataset,
)

REQUIRED_BASE_FIELDS = ("text", "label", "group_id")


def load_jsonl_records(
    path: str | Path,
    required_fields: Sequence[str] = REQUIRED_BASE_FIELDS,
    *,
    dedupe: bool = True,
    require_both_classes: bool = True,
) -> list[dict[str, Any]]:
    """Load and validate a JSONL dataset."""
    path = Path(path)
    required = set(required_fields)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        missing = required - set(record.keys())
        if missing:
            joined = ", ".join(sorted(missing))
            raise ValueError(
                f"Line {line_number} is missing required field(s): {joined}"
            )
        if record["label"] not in (0, 1):
            raise ValueError(f"Line {line_number} has a non-binary label")
        normalized = " ".join(str(record["text"]).lower().split())
        if not normalized:
            continue
        if dedupe and normalized in seen:
            continue
        seen.add(normalized)
        record = dict(record)
        record["label"] = int(record["label"])
        records.append(record)

    if not records:
        raise ValueError(f"No usable records loaded from {path}")
    if require_both_classes and len({r["label"] for r in records}) != 2:
        raise ValueError("Training data must contain both classes (0 and 1)")
    return records


def grouped_split(
    records: list[dict[str, Any]],
    *,
    test_size: float = 0.20,
    valid_size: float = 0.125,
    seed: int = 42,
):
    """Split into (train, valid, test) so no group_id spans two splits."""
    try:
        return _sklearn_grouped_split(records, test_size, valid_size, seed)
    except ImportError:
        return _pure_grouped_split(records, test_size, valid_size, seed)


def _sklearn_grouped_split(records, test_size, valid_size, seed):
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    labels = [r["label"] for r in records]
    groups = [r["group_id"] for r in records]

    if len(set(groups)) < 3:
        train_valid, test = train_test_split(
            records, test_size=test_size, random_state=seed, stratify=labels
        )
        tv_labels = [r["label"] for r in train_valid]
        train, valid = train_test_split(
            train_valid, test_size=valid_size, random_state=seed + 1, stratify=tv_labels
        )
        return train, valid, test

    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_valid_idx, test_idx = next(outer.split(records, labels, groups))
    train_valid = [records[i] for i in train_valid_idx]
    test = [records[i] for i in test_idx]

    inner = GroupShuffleSplit(n_splits=1, test_size=valid_size, random_state=seed + 1)
    tv_labels = [r["label"] for r in train_valid]
    tv_groups = [r["group_id"] for r in train_valid]
    train_idx, valid_idx = next(inner.split(train_valid, tv_labels, tv_groups))
    return (
        [train_valid[i] for i in train_idx],
        [train_valid[i] for i in valid_idx],
        test,
    )


def _pure_grouped_split(records, test_size, valid_size, seed):
    by_group: dict[Any, list[dict[str, Any]]] = {}
    for r in records:
        by_group.setdefault(r["group_id"], []).append(r)
    groups = list(by_group)
    random.Random(seed).shuffle(groups)
    total = len(records)

    def take(target_fraction: float, pool: list[Any]) -> list[Any]:
        target = target_fraction * total
        chosen, taken = [], 0
        for g in list(pool):
            if taken >= target and chosen:
                break
            chosen.append(g)
            taken += len(by_group[g])
            pool.remove(g)
        return chosen

    pool = list(groups)
    test_groups = take(test_size, pool)
    valid_groups = take(valid_size * (1 - test_size), pool)
    train_groups = pool

    def flatten(gs):
        return [r for g in gs for r in by_group[g]]

    train = flatten(train_groups)
    valid = flatten(valid_groups)
    test = flatten(test_groups)
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError("Not enough independent groups for train/valid/test")
    return train, valid, test


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def apply_temperature(margin: float, temperature: float) -> float:
    return _sigmoid(margin / (temperature or 1.0))


def fit_temperature(margins: Sequence[float], labels: Sequence[int]) -> float:
    margins = [float(m) for m in margins]
    labels = [int(y) for y in labels]
    eps = 1e-12

    def nll(T: float) -> float:
        total = 0.0
        for m, y in zip(margins, labels):
            p = min(max(_sigmoid(m / T), eps), 1 - eps)
            total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
        return total / max(len(margins), 1)

    best_T, best_nll = 1.0, nll(1.0)
    for i in range(1, 200):
        T = 0.05 * i
        v = nll(T)
        if v < best_nll:
            best_nll, best_T = v, T
    lo, hi = max(0.01, best_T - 0.05), best_T + 0.05
    for i in range(51):
        T = lo + (hi - lo) * i / 50
        v = nll(T)
        if v < best_nll:
            best_nll, best_T = v, T
    return round(best_T, 4)


def confusion_at_threshold(scores, labels, threshold) -> dict:
    tp = fp = tn = fn = 0
    for s, y in zip(scores, labels):
        pred = 1 if s >= threshold else 0
        if pred and y:
            tp += 1
        elif pred and not y:
            fp += 1
        elif not pred and not y:
            tn += 1
        else:
            fn += 1
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall)
        else 0.0
    )
    return {
        "threshold": round(float(threshold), 6),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "fpr": round(fpr, 6),
        "fnr": round(fnr, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def select_threshold_for_fnr(scores, labels, target_fnr: float = 0.05) -> dict:
    uniq = sorted({float(s) for s in scores})
    if not uniq:
        out = confusion_at_threshold(scores, labels, 0.5)
        out.update(target_fnr=target_fnr, target_met=out["fnr"] <= target_fnr)
        return out

    candidates = [0.0]
    for a, b in zip(uniq, uniq[1:]):
        candidates.append((a + b) / 2.0)
    candidates.append(uniq[-1] + 1e-9)

    satisfying = None
    fallback = None
    for thr in candidates:
        m = confusion_at_threshold(scores, labels, thr)
        if fallback is None or (m["fnr"], m["fpr"]) < (
            fallback["fnr"],
            fallback["fpr"],
        ):
            fallback = m
        if m["fnr"] <= target_fnr:
            satisfying = m

    chosen = dict(satisfying if satisfying is not None else fallback)
    chosen["target_fnr"] = target_fnr
    chosen["target_met"] = chosen["fnr"] <= target_fnr
    return chosen

