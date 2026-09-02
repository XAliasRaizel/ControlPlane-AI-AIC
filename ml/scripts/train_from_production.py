"""
ml/scripts/train_from_production.py

Phase 3: Orchestrator — full training pipeline from production data.

Runs nightly (or on-demand) to:
  1. Build/refresh per-task datasets from production signals (Phase 2)
  2. Merge with curated public baselines (40% public / 60% production)
  3. Run SFT fine-tuning via ml/train_detector.py for classification tasks
  4. Recalibrate thresholds (Phase 4)
  5. Print a summary of what was trained and what was skipped

Usage:
    python -m ml.scripts.train_from_production
    python -m ml.scripts.train_from_production --tasks injection safety
    python -m ml.scripts.train_from_production --dry-run  # dataset build only, no training

What each task uses:
  injection   → SFT on (prompt, INJECTION/CLEAN)   — ml/train_detector.py
  safety      → SFT on (prompt, UNSAFE/CLEAN)      — ml/train_detector.py
  pii         → SFT on (prompt, PII/CLEAN)          — ml/train_detector.py
  fairness    → SFT on (prompt, BIASED/CLEAN)       — ml/train_detector.py
  authorization / sensitive_intent → threshold recalibration only (no SFT needed;
    these are embedding-distance classifiers, not transformer classifiers)

Note on small departments:
  Departments with < 200 examples are pooled into the shared model.
  The dataset builder logs their counts. Run this script to see the breakdown.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("controlplane.train_from_production")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_SIGNALS_DIR = Path("rlhf/data/detector_training")
_OUTPUT_DIR = Path("data/detector_training")
_PUBLIC_DATA_DIR = Path("data")   # existing curated dataset files
_ARTIFACTS_DIR = Path("ml/artifacts")
_MIN_PRODUCTION_EXAMPLES = 50
_PUBLIC_RATIO = 0.4  # Keep 40% public data to prevent drift

# Tasks that use transformer SFT (ml/train_detector.py)
_SFT_TASKS = ["injection", "safety", "pii", "fairness"]

# Tasks that only need threshold recalibration (embedding-based, no SFT)
_THRESHOLD_ONLY_TASKS = ["authorization", "sensitive_intent"]


def merge_with_public_baseline(
    production_path: Path,
    task: str,
    public_ratio: float = _PUBLIC_RATIO,
) -> Path:
    """
    Merge production data with existing public baseline dataset.

    Returns path to merged dataset (written alongside production file).
    Keeps public_ratio% from public data and (1-public_ratio)% from production.

    This prevents the model from forgetting hard public-dataset adversarial
    examples as it adapts to production patterns.
    """
    public_path = _PUBLIC_DATA_DIR / f"{task}.jsonl"
    merged_path = production_path.parent / f"{task}_merged.jsonl"

    prod_records = []
    if production_path.exists():
        with production_path.open("r", encoding="utf-8") as f:
            prod_records = [json.loads(l) for l in f if l.strip()]

    public_records = []
    if public_path.exists():
        with public_path.open("r", encoding="utf-8") as f:
            public_records = [json.loads(l) for l in f if l.strip()]

    if not prod_records and not public_records:
        logger.warning("No data for task '%s' — skipping.", task)
        return merged_path

    if not public_records:
        logger.info("No public baseline for '%s' — using production only.", task)
        merged = prod_records
    elif not prod_records:
        logger.info("No production data for '%s' — using public only.", task)
        merged = public_records
    else:
        # Blend: keep public_ratio from public, rest from production
        n_total = len(prod_records) + len(public_records)
        n_public = min(len(public_records), int(n_total * public_ratio))
        n_prod = min(len(prod_records), n_total - n_public)

        public_sample = random.sample(public_records, n_public) if len(public_records) > n_public else public_records
        prod_sample = random.sample(prod_records, n_prod) if len(prod_records) > n_prod else prod_records
        merged = public_sample + prod_sample
        random.shuffle(merged)
        logger.info(
            "Merged '%s': %d public + %d production = %d total",
            task, len(public_sample), len(prod_sample), len(merged)
        )

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with merged_path.open("w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return merged_path


def run_sft_training(task: str, data_path: Path) -> bool:
    """
    Run ml/train_detector.py for a classification task.
    Returns True if training succeeded.
    """
    output_dir = _ARTIFACTS_DIR / f"{task}-production"
    cmd = [
        sys.executable, "-m", "ml.train_detector",
        "--task", task,
        "--data", str(data_path),
        "--output", str(output_dir),
        "--epochs", "3",
        "--batch-size", "8",
        "--target-fnr", "0.03",
        "--lora",  # LoRA for efficiency — no full GPU required
    ]
    logger.info("Running SFT for task '%s': %s", task, " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            logger.error("SFT failed for '%s':\n%s", task, result.stderr[-2000:])
            return False
        logger.info("SFT complete for '%s'. Output:\n%s", task, result.stdout[-500:])
        return True
    except subprocess.TimeoutExpired:
        logger.error("SFT timed out for '%s' (1h limit).", task)
        return False
    except Exception as exc:
        logger.error("SFT error for '%s': %s", task, exc)
        return False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Full production training pipeline for hot-path detectors.")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Tasks to process. Default: all SFT tasks + threshold recalibration.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build datasets and print stats, but don't train models.")
    parser.add_argument("--min-examples", type=int, default=_MIN_PRODUCTION_EXAMPLES)
    parser.add_argument("--skip-recalibration", action="store_true")
    args = parser.parse_args()

    sft_tasks = args.tasks if args.tasks else _SFT_TASKS
    threshold_tasks = [] if args.tasks else _THRESHOLD_ONLY_TASKS

    print("=" * 60)
    print("ControlPlane.ai — Production Detector Training Pipeline")
    print("=" * 60)
    print(f"SFT tasks:        {sft_tasks}")
    print(f"Threshold-only:   {threshold_tasks}")
    print(f"Dry run:          {args.dry_run}")
    print(f"Min examples:     {args.min_examples}")
    print()

    # --- Step 1: Build production datasets ---
    print("Step 1: Building production datasets from signals...")
    try:
        from ml.scripts.build_detector_dataset import build_all_tasks
    except ImportError:
        sys.path.insert(0, ".")
        from ml.scripts.build_detector_dataset import build_all_tasks

    counts = build_all_tasks(
        signals_dir=_SIGNALS_DIR,
        output_dir=_OUTPUT_DIR,
        min_examples=args.min_examples,
        task=None,
    )
    print(f"\nDataset build results: {counts}\n")

    if args.dry_run:
        print("[DRY-RUN] Stopping before training. Check data quality above.")
        return

    # --- Step 2: Merge with public baselines and run SFT ---
    print("Step 2: Merging with public baselines and running SFT...")
    sft_results: dict[str, str] = {}
    for task in sft_tasks:
        prod_path = _OUTPUT_DIR / f"{task}_production.jsonl"
        n_prod = counts.get(task, 0)
        if n_prod < args.min_examples:
            sft_results[task] = f"SKIPPED (only {n_prod} production examples)"
            continue

        merged_path = merge_with_public_baseline(prod_path, task)
        success = run_sft_training(task, merged_path)
        sft_results[task] = "SUCCESS" if success else "FAILED"

    # --- Step 3: Threshold recalibration ---
    if not args.skip_recalibration:
        print("\nStep 3: Recalibrating thresholds for all tasks...")
        try:
            from ml.scripts.recalibrate_thresholds import recalibrate_task
        except ImportError:
            from ml.scripts.recalibrate_thresholds import recalibrate_task  # noqa

        all_recalib_tasks = list(sft_tasks) + list(threshold_tasks)
        for task in all_recalib_tasks:
            recalibrate_task(task)

    # --- Final summary ---
    print("\n" + "=" * 60)
    print("Training Pipeline Summary")
    print("=" * 60)
    print("\nSFT Results:")
    for task, result in sorted(sft_results.items()):
        print(f"  {task:<25} {result}")

    if not args.skip_recalibration:
        print("\nThreshold recalibration: complete (see logs above)")
        print("New thresholds are active on next request (no restart needed).")

    print("\nNew model artifacts in:", _ARTIFACTS_DIR)
    print("Restart the server to load new SFT model weights.")


if __name__ == "__main__":
    main()
