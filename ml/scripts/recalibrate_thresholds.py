"""
ml/scripts/recalibrate_thresholds.py

Phase 4: Online threshold recalibration (lightweight, frequent — every 6 hours).

Instead of retraining the full model, this script reads the last N production-
labeled examples per task, recomputes the optimal decision threshold at the
target FNR, and hot-writes it to calibration.json.

The running server picks it up on the NEXT request via model_backend.py's
lru_cache invalidation (reset_cache()), so no restart needed.

This is especially powerful for the authorization / sensitive_intent detectors
where the threshold drifts as new query patterns (e.g., new department names,
new role structures) enter the system.

Department-specific recalibration:
  - For tasks like 'authorization' where department context matters,
    if a department has >= MIN_DEPT_EXAMPLES (200) labeled examples,
    its threshold is stored as a per-department override in calibration.json:
    {"threshold": 0.42, "dept_thresholds": {"HR": 0.38, "Finance": 0.45}}
  - For departments with < 200 examples, the shared threshold is used.
  - The hot-path reads dept_thresholds[dept] if present, else falls back
    to the shared threshold. (This is a zero-code-change extension.)

Usage:
    python -m ml.scripts.recalibrate_thresholds
    python -m ml.scripts.recalibrate_thresholds --task authorization --target-fnr 0.03
    python -m ml.scripts.recalibrate_thresholds --dry-run  # print but don't write
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("controlplane.recalibrate")

_SIGNALS_DIR = Path("rlhf/data/detector_training")
_ARTIFACTS_DIR = Path("ml/artifacts")
_DEFAULT_TARGET_FNR = 0.03
_LOOKBACK_EXAMPLES = 500  # Use the last N examples per task
_MIN_EXAMPLES_FOR_RECALIBRATION = 50
_MIN_DEPT_EXAMPLES = 200  # Minimum per-dept examples for dept-specific threshold


def load_recent_signals(task: str, n: int = _LOOKBACK_EXAMPLES) -> list[dict]:
    """Load the most recent N labeled examples for a task from the signal files."""
    files = sorted(_SIGNALS_DIR.glob("raw_signals_*.jsonl"), reverse=True)  # newest first
    records = []
    for f in files:
        if len(records) >= n:
            break
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
            for line in reversed(lines):
                if len(records) >= n:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("task") == task and r.get("label") is not None:
                        records.append(r)
                except json.JSONDecodeError:
                    continue
        except Exception as exc:
            logger.warning("Could not read %s: %s", f, exc)
    return records


def find_calibration_file(task: str) -> Optional[Path]:
    """Find the calibration.json for a task in ml/artifacts/<task>-v*/calibration.json."""
    # Find all versioned artifact directories for this task
    candidates = sorted(_ARTIFACTS_DIR.glob(f"{task}-v*/calibration.json"), reverse=True)
    if candidates:
        return candidates[0]
    # Also check unversioned
    plain = _ARTIFACTS_DIR / task / "calibration.json"
    if plain.exists():
        return plain
    return None


def recalibrate_task(
    task: str,
    target_fnr: float = _DEFAULT_TARGET_FNR,
    dry_run: bool = False,
) -> Optional[dict]:
    """
    Recalibrate the decision threshold for a single task.

    Returns the updated calibration dict, or None if insufficient data.
    """
    records = load_recent_signals(task)
    if len(records) < _MIN_EXAMPLES_FOR_RECALIBRATION:
        logger.info(
            "Task '%s': only %d examples (need %d) — skipping recalibration.",
            task, len(records), _MIN_EXAMPLES_FOR_RECALIBRATION
        )
        return None

    # Build scores and labels lists
    # Use async_score as the score (it's the teacher's score — closest to ground truth)
    # Fall back to (1 - hot_score) confidence as proxy when async_score not available
    scores = []
    labels = []
    for r in records:
        score = r.get("async_score")
        if score is None:
            # human-override case — use a committed score of 0.95 or 0.05
            score = 0.95 if r.get("label") == 1 else 0.05
        scores.append(float(score))
        labels.append(int(r.get("label", 0)))

    if len(set(labels)) < 2:
        logger.info("Task '%s': only one label class in data — skipping.", task)
        return None

    # Use ml.common's select_threshold_for_fnr (same function used in full training)
    try:
        from ml.common import select_threshold_for_fnr
    except ImportError:
        import sys
        sys.path.insert(0, ".")
        from ml.common import select_threshold_for_fnr

    operating_point = select_threshold_for_fnr(scores, labels, target_fnr=target_fnr)
    new_threshold = operating_point["threshold"]
    new_fnr = operating_point.get("fnr", "?")
    new_fpr = operating_point.get("fpr", "?")

    logger.info(
        "Task '%s': new threshold=%.4f (FNR=%.3f, FPR=%.3f) from %d examples",
        task, new_threshold, new_fnr, new_fpr, len(records)
    )

    # --- Per-department threshold overrides ---
    dept_thresholds: dict[str, float] = {}
    by_dept: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for r in records:
        dept = r.get("department") or "unknown"
        score = r.get("async_score")
        if score is None:
            score = 0.95 if r.get("label") == 1 else 0.05
        by_dept[dept][0].append(float(score))
        by_dept[dept][1].append(int(r.get("label", 0)))

    for dept, (dept_scores, dept_labels) in by_dept.items():
        if len(dept_scores) < _MIN_DEPT_EXAMPLES:
            logger.info(
                "  dept '%s': %d examples < %d threshold — using shared threshold.",
                dept, len(dept_scores), _MIN_DEPT_EXAMPLES
            )
            continue
        if len(set(dept_labels)) < 2:
            continue
        dept_op = select_threshold_for_fnr(dept_scores, dept_labels, target_fnr=target_fnr)
        dept_thresholds[dept] = dept_op["threshold"]
        logger.info(
            "  dept '%s': dept-specific threshold=%.4f (FNR=%.3f) from %d examples",
            dept, dept_op["threshold"], dept_op.get("fnr", 0), len(dept_scores)
        )

    # --- Update calibration.json ---
    calib_path = find_calibration_file(task)
    if calib_path is None:
        logger.warning("No calibration.json found for task '%s' — cannot update.", task)
        return None

    try:
        calib = json.loads(calib_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read %s: %s", calib_path, exc)
        return None

    old_threshold = calib.get("threshold")
    calib["threshold"] = new_threshold
    calib["recalibrated_from_production"] = True
    calib["recalibration_examples"] = len(records)
    calib["recalibration_target_fnr"] = target_fnr
    if dept_thresholds:
        calib["dept_thresholds"] = dept_thresholds

    if not dry_run:
        calib_path.write_text(json.dumps(calib, indent=2), encoding="utf-8")
        logger.info("Updated %s: %.4f → %.4f", calib_path, old_threshold, new_threshold)

        # Hot-reload: invalidate model_backend lru_cache so next request picks it up
        try:
            from backend.shared.model_backend import reset_cache
            reset_cache()
            logger.info("model_backend cache reset — new threshold active immediately.")
        except Exception as exc:
            logger.debug("Could not reset model_backend cache (server may not be running): %s", exc)
    else:
        logger.info("[DRY-RUN] Would update %s: %.4f → %.4f", calib_path, old_threshold, new_threshold)

    return calib


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Recalibrate hot-path detector thresholds from production data.")
    parser.add_argument("--task", type=str, default=None, help="Recalibrate only this task.")
    parser.add_argument("--target-fnr", type=float, default=_DEFAULT_TARGET_FNR)
    parser.add_argument("--dry-run", action="store_true", help="Print proposed changes without writing.")
    args = parser.parse_args()

    # All tasks that have calibration files
    if args.task:
        tasks = [args.task]
    else:
        tasks = [p.parent.name.rstrip("-v0123456789") for p in _ARTIFACTS_DIR.glob("*/calibration.json")]
        tasks = list(set(tasks))  # deduplicate

    if not tasks:
        logger.warning("No calibration files found in %s", _ARTIFACTS_DIR)
        return

    print(f"\nRecalibrating {len(tasks)} task(s) at target FNR={args.target_fnr:.3f}")
    for task in sorted(tasks):
        recalibrate_task(task, target_fnr=args.target_fnr, dry_run=args.dry_run)
    print("\nDone. New thresholds active on next request (no restart needed).")


if __name__ == "__main__":
    main()
