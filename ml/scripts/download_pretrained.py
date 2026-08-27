"""Download a pretrained HuggingFace sequence-classification model for use as a
ControlPlane detector artifact (Track A — no training required).

Creates the standard artifact layout consumed by backend/shared/model_backend.py:
    <output>/model/            HF model + tokenizer (save_pretrained)
    <output>/calibration.json  temperature, threshold, positive_label, etc.

Usage examples:

  # Injection detector
  python -m ml.scripts.download_pretrained \\
    --model-id protectai/deberta-v3-base-prompt-injection-v2 \\
    --output ml/artifacts/injection-pretrained \\
    --task injection \\
    --positive-index 1 \\
    --positive-label INJECTION_DETECTED \\
    --threshold 0.5

  # Safety / toxicity detector
  python -m ml.scripts.download_pretrained \\
    --model-id s-nlp/roberta_toxicity_classifier \\
    --output ml/artifacts/safety-pretrained \\
    --task toxicity \\
    --positive-index 1 \\
    --positive-label UNSAFE_CONTENT \\
    --threshold 0.5

  # Grounding NLI cross-encoder
  python -m ml.scripts.download_pretrained \\
    --model-id cross-encoder/nli-deberta-v3-base \\
    --output ml/artifacts/grounding-nli \\
    --task grounding \\
    --positive-index 1 \\
    --positive-label ENTAILMENT \\
    --threshold 0.5

After download, point the environment variable at the model/ subdirectory:
  CONTROLPLANE_MODEL_INJECTION=ml/artifacts/injection-pretrained/model
  CONTROLPLANE_MODEL_SAFETY=ml/artifacts/safety-pretrained/model
  CONTROLPLANE_MODEL_GROUNDING=ml/artifacts/grounding-nli/model
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


TASK_DEFAULTS = {
    "injection": {
        "recommended_model": "protectai/deberta-v3-base-prompt-injection-v2",
        "positive_label": "INJECTION_DETECTED",
        "positive_index": 1,
        "max_length": 512,
    },
    "toxicity": {
        "recommended_model": "s-nlp/roberta_toxicity_classifier",
        "positive_label": "UNSAFE_CONTENT",
        "positive_index": 1,
        "max_length": 512,
    },
    "fairness": {
        "recommended_model": "microsoft/deberta-v3-base",
        "positive_label": "BIASED",
        "positive_index": 1,
        "max_length": 256,
    },
    "grounding": {
        "recommended_model": "cross-encoder/nli-deberta-v3-base",
        "positive_label": "ENTAILMENT",
        "positive_index": 1,    # label_mapping: {0: contradiction, 1: entailment, 2: neutral}
        "max_length": 512,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a pretrained HF model as a ControlPlane detector artifact.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-id", required=True,
        help="HuggingFace model hub ID (e.g. protectai/deberta-v3-base-prompt-injection-v2)",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output artifact directory (model/ and calibration.json written here)",
    )
    parser.add_argument(
        "--task", choices=sorted(TASK_DEFAULTS), default=None,
        help="Detector task — used to set sensible defaults for calibration fields",
    )
    parser.add_argument(
        "--positive-index", type=int, default=None,
        help="Index in logits corresponding to the positive (risky) class (default 1)",
    )
    parser.add_argument(
        "--positive-label", default=None,
        help="Label string for positive detections (e.g. INJECTION_DETECTED)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Decision threshold (default 0.5 — tune after evaluating FPR/FNR)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Calibration temperature (default 1.0 — tune if model is overconfident)",
    )
    parser.add_argument(
        "--max-length", type=int, default=None,
        help="Tokenizer max_length (default per task or 512)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Merge task defaults with explicit overrides
    task_cfg = TASK_DEFAULTS.get(args.task, {}) if args.task else {}
    positive_index = args.positive_index if args.positive_index is not None else task_cfg.get("positive_index", 1)
    positive_label = args.positive_label or task_cfg.get("positive_label", "POSITIVE")
    max_length = args.max_length or task_cfg.get("max_length", 512)

    model_dir = args.output / "model"
    calib_path = args.output / "calibration.json"

    if model_dir.exists() and not args.force:
        print(f"[INFO] Model directory already exists: {model_dir}")
        print("       Use --force to re-download and overwrite.")
        sys.exit(0)

    # Heavy imports — only needed at download time
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        print("[ERROR] transformers not installed. Run: pip install -r ml/requirements-ml.txt")
        sys.exit(1)

    print(f"[INFO] Downloading model: {args.model_id}")
    print(f"[INFO] Output directory:  {args.output}")

    args.output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    print("[INFO] Loading model weights...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_id)

    # Inspect id2label to confirm positive_index
    id2label = {int(k): str(v) for k, v in getattr(model.config, "id2label", {}).items()}
    if id2label:
        print(f"[INFO] Model id2label: {id2label}")
        # Auto-detect entailment index for NLI models
        if args.task == "grounding":
            for idx, label in id2label.items():
                if "entail" in label.lower():
                    if args.positive_index is None:
                        positive_index = idx
                        print(f"[INFO] Auto-detected entailment index: {positive_index}")
                    break
        print(f"[INFO] Using positive_index={positive_index} -> '{id2label.get(positive_index, '?')}'")

    print("[INFO] Saving tokenizer and model...")
    tokenizer.save_pretrained(str(model_dir))
    model.save_pretrained(str(model_dir))

    calibration = {
        "task": args.task or "unknown",
        "base_model": args.model_id,
        "temperature": args.temperature,
        "threshold": args.threshold,
        "positive_label": positive_label,
        "positive_index": positive_index,
        "max_length": max_length,
        "lora": False,
        "pretrained": True,
    }
    calib_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    print(f"\n[OK] Artifact saved to: {args.output}")
    print(f"     Model:           {model_dir}")
    print(f"     Calibration:     {calib_path}")
    print(f"\nCalibration summary:")
    print(json.dumps(calibration, indent=2))
    print(f"\nActivate with:")
    if args.task:
        env_var = f"CONTROLPLANE_MODEL_{args.task.upper()}"
        print(f"  $env:{env_var} = \"{model_dir}\"")
    else:
        print(f"  $env:CONTROLPLANE_MODEL_<TASK> = \"{model_dir}\"")
    print()
    print("[TIP] Run ml/scripts/evaluate_model.py to tune the threshold before deploying.")


if __name__ == "__main__":
    main()
