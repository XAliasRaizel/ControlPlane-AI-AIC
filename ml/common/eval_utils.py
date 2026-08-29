"""
ml/common/eval_utils.py

Shared evaluation for all four detectors — deliberately NOT just accuracy.

950 safe + 50 attacks => a model that says "everything is safe" scores 95%
accuracy and is useless for governance. These functions report
precision/recall/F1/FPR/FNR/ROC-AUC so a model can be judged on the failure
mode that actually matters (missed attacks).

Improvements over the original package version:
  - calibration_curve_points(): multi-threshold precision/recall curve for
    pitch-deck ROC slides, not just a single-threshold snapshot.
  - find_threshold_for_target_recall() now also tries threshold=0.0 as a
    last resort (catches the edge case where the positive class is extremely
    rare and standard threshold scanning misses it).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_detection_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """y_true: 0/1 labels. y_score: continuous risk score in [0, 1].

    Reports metrics AT the given threshold, plus threshold-independent
    ROC-AUC so you can compare models before picking an operating point.
    """
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    y_pred = (y_score_arr >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    n_pos = int((y_true_arr == 1).sum())
    n_neg = int((y_true_arr == 0).sum())

    return {
        "n": int(len(y_true_arr)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "threshold": float(threshold),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "roc_auc": (
            float(roc_auc_score(y_true_arr, y_score_arr))
            if n_pos > 0 and n_neg > 0
            else float("nan")
        ),
    }


def find_threshold_for_target_recall(
    y_true: Sequence[int],
    y_score: Sequence[float],
    target_recall: float = 0.95,
) -> Optional[float]:
    """Return the HIGHEST threshold that still achieves at least target_recall.

    For governance detectors, you pick the operating point by
    "what recall can I not go below" (missed attacks are worse than extra
    review queue items), not by maximizing F1.

    Returns None only if even threshold=0 (predict everything positive)
    cannot reach the target, which means the positive class has no signal
    whatsoever in y_score.

    Walk thresholds from strictest to loosest — recall only rises as
    threshold drops, so the first one that clears the bar is the best
    (highest precision) among all that do.
    """
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)

    candidates = sorted(set(y_score_arr.tolist()), reverse=True)
    # Add 0.0 explicitly as a last-resort fallback (predict everything positive)
    if 0.0 not in candidates:
        candidates.append(0.0)

    for t in candidates:
        y_pred = (y_score_arr >= t).astype(int)
        r = recall_score(y_true_arr, y_pred, zero_division=0)
        if r >= target_recall:
            return float(t)
    return None


def calibration_curve_points(
    y_true: Sequence[int],
    y_score: Sequence[float],
    n_thresholds: int = 20,
) -> List[Dict[str, float]]:
    """Compute precision/recall/FPR at n_thresholds evenly spaced thresholds.

    Returns a list of dicts (one per threshold) sorted from strictest to
    loosest — suitable for plotting a precision-recall or ROC curve for a
    pitch deck or evaluation report.

    Example usage:
        pts = calibration_curve_points(y_true, scores)
        # Each point: {"threshold": 0.9, "precision": 0.95, "recall": 0.62, ...}
    """
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)

    lo, hi = float(y_score_arr.min()), float(y_score_arr.max())
    if lo == hi:
        return [compute_detection_metrics(y_true_arr, y_score_arr, threshold=lo)]

    thresholds = np.linspace(hi, lo, n_thresholds)
    points = []
    for t in thresholds:
        m = compute_detection_metrics(y_true_arr, y_score_arr, threshold=float(t))
        points.append({k: m[k] for k in ("threshold", "precision", "recall",
                                          "f1", "false_positive_rate",
                                          "false_negative_rate")})
    return points


def compare_models(results_by_name: Dict[str, Dict[str, float]]) -> str:
    """Formatting helper for a before/after or model-vs-model comparison table.

    The kind of table you actually want to paste into a pitch deck.
    """
    header = (
        f"{'model':<28}{'precision':>10}{'recall':>10}"
        f"{'f1':>8}{'fpr':>8}{'fnr':>8}{'roc_auc':>9}"
    )
    lines = [header]
    for name, m in results_by_name.items():
        lines.append(
            f"{name:<28}{m['precision']:>10.3f}{m['recall']:>10.3f}{m['f1']:>8.3f}"
            f"{m['false_positive_rate']:>8.3f}{m['false_negative_rate']:>8.3f}"
            f"{m['roc_auc']:>9.3f}"
        )
    return "\n".join(lines)
