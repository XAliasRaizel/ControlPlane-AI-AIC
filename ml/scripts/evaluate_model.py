"""Evaluate a trained or pretrained ControlPlane detector artifact against a labeled JSONL file.

Loads a model artifact (same format as ml/train_detector.py output) via
backend/shared/model_backend.py's CalibratedClassifier, runs it against a
test set, and produces a full evaluation report including:

  - Confusion matrix (TP/FP/TN/FN)
  - FPR, FNR, precision, recall, F1 at the configured threshold
  - ROC-AUC, AUPRC
  - Per-group_id breakdown (detects category-specific or demographic biases)
  - Threshold sweep table (to help tune the operating point)

The threshold used is whatever is in calibration.json; pass --threshold to override.

Usage:
    python ml/scripts/evaluate_model.py \\
        --artifact ml/artifacts/injection-pretrained \\
        --data data/injection.jsonl

    # Evaluate on the held-out test split only (group-aware, same seed as trainer)
    python ml/scripts/evaluate_model.py \\
        --artifact ml/artifacts/injection-v1 \\
        --data data/injection.jsonl \\
        --split test \\
        --threshold 0.45

    # Save JSON report
    python ml/scripts/evaluate_model.py \\
        --artifact ml/artifacts/injection-v1 \\
        --data data/injection.jsonl \\
        --output ml/artifacts/injection-v1/eval_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a script from the repo root
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    from ml.common import (
        apply_temperature, confusion_at_threshold,
        grouped_split, load_jsonl_records, select_threshold_for_fnr,
    )
except ImportError:
    # Running as bare script with ml/ on path
    from common import (  # type: ignore
        apply_temperature, confusion_at_threshold,
        grouped_split, load_jsonl_records, select_threshold_for_fnr,
    )


def _sigmoid(x: float) -> float:
    import math
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def load_calibration(artifact_dir: Path) -> dict:
    for candidate in (artifact_dir.parent / "calibration.json",
                       artifact_dir / "calibration.json"):
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def score_all(model_dir: Path, calib: dict, records: list[dict]) -> list[tuple[float, int]]:
    """Return (calibrated_score, true_label) pairs using the CalibratedClassifier."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        print("[ERROR] torch/transformers not installed. Run: pip install -r ml/requirements-ml.txt")
        sys.exit(1)

    temperature = float(calib.get("temperature", 1.0)) or 1.0
    positive_index = int(calib.get("positive_index", 1))
    max_length = int(calib.get("max_length", 256))

    print(f"[INFO] Loading model from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    results = []
    for i, rec in enumerate(records, 1):
        if i % 100 == 0:
            print(f"  Scored {i}/{len(records)} ...", end="\r")
        enc = tokenizer(
            rec["text"], truncation=True, max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(**enc).logits[0]
        pos = float(logits[positive_index].item())
        neg = float(logits[1 - positive_index].item())
        score = _sigmoid((pos - neg) / temperature)
        results.append((score, rec["label"]))

    print(f"  Scored {len(records)}/{len(records)} — done.          ")
    return results


def threshold_sweep(scores: list[float], labels: list[int]) -> list[dict]:
    """Generate confusion metrics across a range of thresholds."""
    import numpy as np
    thresholds = sorted(set(
        [round(t, 3) for t in list(np.linspace(0.1, 0.9, 17))]
        + [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    ))
    rows = []
    for thr in thresholds:
        m = confusion_at_threshold(scores, labels, thr)
        rows.append(m)
    return rows


def per_group_metrics(scored_records: list[tuple[dict, float]], threshold: float) -> dict:
    """Compute confusion metrics per group_id."""
    groups: dict[str, list[tuple[float, int]]] = {}
    for rec, score in scored_records:
        g = rec.get("group_id", "unknown")
        groups.setdefault(g, []).append((score, rec["label"]))

    report = {}
    for g, pairs in sorted(groups.items()):
        s, l = zip(*pairs)
        m = confusion_at_threshold(list(s), list(l), threshold)
        m["n"] = len(pairs)
        report[g] = m
    return report


def safe_auc(fn_name: str, scores: list[float], labels: list[int]) -> float | None:
    if len(set(labels)) != 2:
        return None
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        fn = roc_auc_score if fn_name == "roc_auc" else average_precision_score
        return round(float(fn(labels, scores)), 6)
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a ControlPlane detector artifact on a labeled JSONL dataset."
    )
    parser.add_argument("--artifact", type=Path, required=True,
                        help="Artifact directory (contains model/ and calibration.json)")
    parser.add_argument("--data", type=Path, required=True,
                        help="JSONL dataset file ({text, label, group_id} per line)")
    parser.add_argument("--split", choices=["full", "train", "valid", "test"], default="full",
                        help="Which split to evaluate (default: full dataset)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for grouped_split (must match trainer)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override threshold from calibration.json")
    parser.add_argument("--output", type=Path, default=None,
                        help="Save JSON report to this path (optional)")
    parser.add_argument("--no-sweep", action="store_true",
                        help="Skip threshold sweep table (faster)")
    parser.add_argument("--no-per-group", action="store_true",
                        help="Skip per-group breakdown (faster)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Locate model directory
    model_dir = args.artifact / "model"
    if not model_dir.exists():
        model_dir = args.artifact  # allow pointing directly at model dir
    if not model_dir.exists():
        print(f"[ERROR] Model directory not found: {model_dir}")
        sys.exit(1)

    calib = load_calibration(args.artifact)
    if not calib:
        print(f"[WARN] No calibration.json found in {args.artifact} — using defaults.")

    threshold = args.threshold if args.threshold is not None else float(calib.get("threshold", 0.5))
    task = calib.get("task", "unknown")
    base_model = calib.get("base_model", str(model_dir))

    print(f"\n=== ControlPlane Detector Evaluation ===")
    print(f"Task:         {task}")
    print(f"Base model:   {base_model}")
    print(f"Artifact:     {args.artifact}")
    print(f"Dataset:      {args.data}")
    print(f"Split:        {args.split}")
    print(f"Threshold:    {threshold}")
    print()

    # Load and split records
    all_records = load_jsonl_records(args.data)
    if args.split == "full":
        eval_records = all_records
    else:
        train, valid, test = grouped_split(all_records, seed=args.seed)
        split_map = {"train": train, "valid": valid, "test": test}
        eval_records = split_map[args.split]

    print(f"[INFO] Evaluating on {len(eval_records)} records ({args.split} split)")
    label_dist = {0: sum(1 for r in eval_records if r["label"] == 0),
                  1: sum(1 for r in eval_records if r["label"] == 1)}
    print(f"[INFO] Label distribution: {label_dist[0]} negative, {label_dist[1]} positive")
    print()

    # Score
    scored_pairs = score_all(model_dir, calib, eval_records)
    scores = [s for s, _ in scored_pairs]
    labels = [l for _, l in scored_pairs]

    # Main metrics
    metrics = confusion_at_threshold(scores, labels, threshold)
    roc_auc = safe_auc("roc_auc", scores, labels)
    auprc = safe_auc("auprc", scores, labels)

    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS  (threshold={threshold:.4f})")
    print(f"{'='*50}")
    print(f"  TP: {metrics['tp']:5d}  FP: {metrics['fp']:5d}")
    print(f"  FN: {metrics['fn']:5d}  TN: {metrics['tn']:5d}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1:         {metrics['f1']:.4f}")
    print(f"  FPR:        {metrics['fpr']:.4f}  (false positives / all negatives)")
    print(f"  FNR:        {metrics['fnr']:.4f}  (false negatives / all positives) <- KEY")
    if roc_auc is not None:
        print(f"  ROC-AUC:    {roc_auc:.4f}")
    if auprc is not None:
        print(f"  AUPRC:      {auprc:.4f}")

    # Recommended threshold for 5% FNR target
    op = select_threshold_for_fnr(scores, labels, target_fnr=0.05)
    print(f"\n  [TIP] Threshold for ≤5% FNR: {op['threshold']:.4f}  "
          f"(FNR={op['fnr']:.4f}, FPR={op['fpr']:.4f}, met={op['target_met']})")

    # Threshold sweep
    sweep_rows = []
    if not args.no_sweep:
        try:
            import numpy  # noqa: F401
            sweep_rows = threshold_sweep(scores, labels)
            print(f"\n{'─'*70}")
            print(f"{'Threshold':>10} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} "
                  f"{'FNR':>7} {'FPR':>7} {'F1':>7}")
            print(f"{'─'*70}")
            for row in sweep_rows:
                print(f"  {row['threshold']:>8.3f} {row['tp']:>6} {row['fp']:>6} "
                      f"{row['fn']:>6} {row['tn']:>6} "
                      f"{row['fnr']:>7.4f} {row['fpr']:>7.4f} {row['f1']:>7.4f}")
        except ImportError:
            print("\n[WARN] numpy not installed — skipping threshold sweep table.")

    # Per-group breakdown
    group_report = {}
    if not args.no_per_group:
        scored_with_rec = list(zip(eval_records, scores))
        group_report = per_group_metrics(scored_with_rec, threshold)
        print(f"\n{'─'*60}")
        print("PER-GROUP BREAKDOWN")
        print(f"{'─'*60}")
        for g, gm in group_report.items():
            print(f"  {g:<35} n={gm['n']:>5}  FNR={gm['fnr']:.3f}  FPR={gm['fpr']:.3f}  F1={gm['f1']:.3f}")

    # Save report
    report = {
        "task": task,
        "base_model": base_model,
        "artifact": str(args.artifact),
        "dataset": str(args.data),
        "split": args.split,
        "n_eval": len(eval_records),
        "threshold": threshold,
        "metrics": metrics,
        "roc_auc": roc_auc,
        "auprc": auprc,
        "recommended_threshold_for_5pct_fnr": op,
        "per_group": group_report,
        "threshold_sweep": sweep_rows,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[OK] Full report saved to: {args.output}")

    print(f"\n{'='*50}\n")


if __name__ == "__main__":
    main()
