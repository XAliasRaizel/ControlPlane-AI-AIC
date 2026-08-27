"""ControlPlane.ai — All-in-One Detector Training Script (Colab / Kaggle T4).

Run this on a free Kaggle or Google Colab GPU session (T4 is enough).
It trains all three classification detectors in sequence using LoRA:
  - injection  → microsoft/deberta-v3-base fine-tuned on injection.jsonl
  - toxicity   → s-nlp/roberta_toxicity_classifier fine-tuned on toxicity.jsonl
  - fairness   → microsoft/deberta-v3-base fine-tuned on fairness.jsonl

Prerequisites (run these cells first in Colab/Kaggle):
    !pip install -q torch transformers datasets scikit-learn accelerate peft sentencepiece

Usage — upload your prepared JSONL files and run:
    python ml/notebooks/train_detectors.py \\
        --injection-data /kaggle/input/controlplane-data/injection.jsonl \\
        --toxicity-data  /kaggle/input/controlplane-data/toxicity.jsonl \\
        --fairness-data  /kaggle/input/controlplane-data/fairness.jsonl \\
        --output-dir     /kaggle/working/artifacts

Or run only one task:
    python ml/notebooks/train_detectors.py --task injection \\
        --injection-data /kaggle/input/controlplane-data/injection.jsonl \\
        --output-dir /kaggle/working/artifacts

After training, download the artifacts/ directory and place into:
    ml/artifacts/injection-v1/
    ml/artifacts/toxicity-v1/
    ml/artifacts/fairness-v1/

Then activate with environment variables:
    $env:CONTROLPLANE_MODEL_INJECTION = "ml/artifacts/injection-v1/model"
    $env:CONTROLPLANE_MODEL_SAFETY    = "ml/artifacts/toxicity-v1/model"
    $env:CONTROLPLANE_MODEL_FAIRNESS  = "ml/artifacts/fairness-v1/model"

Expected T4 runtimes (5K samples per class, 3 epochs, LoRA r=8):
    injection : ~15-25 min
    toxicity  : ~20-30 min
    fairness  : ~15-20 min

The artifact layout is identical to ml/train_detector.py output so the
backend/shared/model_backend.py seam consumes it with no code changes.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


TASK_CONFIGS = {
    "injection": {
        "model": "microsoft/deberta-v3-base",
        "positive_label": "INJECTION_DETECTED",
        "epochs": 3,
        "batch_size": 8,
        "max_length": 256,
        "target_fnr": 0.05,
    },
    "toxicity": {
        # Start from the already-toxicity-aware checkpoint (saves training time)
        "model": "s-nlp/roberta_toxicity_classifier",
        "positive_label": "UNSAFE_CONTENT",
        "epochs": 3,
        "batch_size": 8,
        "max_length": 256,
        "target_fnr": 0.05,
    },
    "fairness": {
        "model": "microsoft/deberta-v3-base",
        "positive_label": "BIASED",
        "epochs": 3,
        "batch_size": 8,
        "max_length": 256,
        "target_fnr": 0.05,
    },
}


def check_gpu() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[GPU] {name} — {mem:.1f} GB VRAM")
        else:
            print("[WARN] No CUDA GPU detected. Training will be very slow on CPU.")
            print("       On Kaggle: Settings → Accelerator → GPU T4 x2")
            print("       On Colab:  Runtime → Change runtime type → T4 GPU")
    except ImportError:
        print("[ERROR] torch not installed. Run: pip install torch transformers datasets peft scikit-learn accelerate sentencepiece")
        sys.exit(1)


def train_task(task: str, data_path: Path, output_dir: Path, repo_root: Path) -> None:
    cfg = TASK_CONFIGS[task]
    artifact_out = output_dir / f"{task}-v1"
    artifact_out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"TRAINING: {task.upper()}")
    print(f"  data:   {data_path}")
    print(f"  model:  {cfg['model']}")
    print(f"  output: {artifact_out}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "-m", "ml.train_detector",
        "--task", task,
        "--data", str(data_path),
        "--output", str(artifact_out),
        "--model", cfg["model"],
        "--epochs", str(cfg["epochs"]),
        "--batch-size", str(cfg["batch_size"]),
        "--max-length", str(cfg["max_length"]),
        "--target-fnr", str(cfg["target_fnr"]),
        "--lora",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    result = subprocess.run(cmd, cwd=str(repo_root), env=env)

    if result.returncode != 0:
        print(f"\n[ERROR] Training failed for task={task} (exit code {result.returncode})")
        sys.exit(result.returncode)

    # Print summary from evaluation.json
    eval_path = artifact_out / "evaluation.json"
    if eval_path.exists():
        evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
        test = evaluation.get("test", {})
        print(f"\n{'─'*50}")
        print(f"  {task.upper()} TEST RESULTS")
        print(f"  FNR:       {test.get('fnr', '?'):.4f}  ← target ≤ {cfg['target_fnr']}")
        print(f"  FPR:       {test.get('fpr', '?'):.4f}")
        print(f"  F1:        {test.get('f1', '?'):.4f}")
        print(f"  ROC-AUC:   {evaluation.get('test', {}).get('roc_auc', '?')}")
        print(f"  Artifact:  {artifact_out}")
        print(f"{'─'*50}\n")

    print(f"[OK] {task} training complete → {artifact_out}")


def print_activation_instructions(tasks: list[str], output_dir: Path) -> None:
    print(f"\n{'='*60}")
    print("NEXT STEPS — Activate on your local machine:")
    print(f"{'='*60}")
    print("\n1. Download the artifacts/ directory from Kaggle/Colab working files.")
    print("   Place each task's folder in ml/artifacts/:")
    for task in tasks:
        src = output_dir / f"{task}-v1"
        print(f"   {src}  →  ml/artifacts/{task}-v1/")

    print("\n2. Set environment variables (PowerShell):")
    task_to_env = {
        "injection": "CONTROLPLANE_MODEL_INJECTION",
        "toxicity": "CONTROLPLANE_MODEL_SAFETY",
        "fairness": "CONTROLPLANE_MODEL_FAIRNESS",
    }
    for task in tasks:
        env_var = task_to_env.get(task, f"CONTROLPLANE_MODEL_{task.upper()}")
        print(f"   $env:{env_var} = \"ml/artifacts/{task}-v1/model\"")

    print("\n3. Verify with evaluate_model.py:")
    for task in tasks:
        data_file = f"data/{task}.jsonl"
        artifact = f"ml/artifacts/{task}-v1"
        print(f"   python ml/scripts/evaluate_model.py --artifact {artifact} --data {data_file} --split test")

    print("\n4. Compare regex vs model:")
    for task in tasks:
        data_file = f"data/{task}.jsonl"
        artifact = f"ml/artifacts/{task}-v1"
        if task != "grounding":
            print(f"   python ml/scripts/compare_detectors.py --task {task} --artifact {artifact} --data {data_file} --split test")

    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train all ControlPlane detector models (run on Colab/Kaggle T4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--task", choices=["injection", "toxicity", "fairness", "all"],
                        default="all",
                        help="Which task(s) to train (default: all)")
    parser.add_argument("--injection-data", type=Path, default=Path("data/injection.jsonl"),
                        help="Path to injection.jsonl (default: data/injection.jsonl)")
    parser.add_argument("--toxicity-data", type=Path, default=Path("data/toxicity.jsonl"),
                        help="Path to toxicity.jsonl (default: data/toxicity.jsonl)")
    parser.add_argument("--fairness-data", type=Path, default=Path("data/fairness.jsonl"),
                        help="Path to fairness.jsonl (default: data/fairness.jsonl)")
    parser.add_argument("--output-dir", type=Path, default=Path("ml/artifacts"),
                        help="Output directory for all artifacts (default: ml/artifacts)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Detect repo root — works both when running from repo root or from ml/notebooks/
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir
    for _ in range(5):
        if (repo_root / "ml" / "train_detector.py").exists():
            break
        repo_root = repo_root.parent
    else:
        print("[ERROR] Could not find repo root (ml/train_detector.py not found).")
        sys.exit(1)

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    check_gpu()

    task_data_map = {
        "injection": args.injection_data,
        "toxicity": args.toxicity_data,
        "fairness": args.fairness_data,
    }

    tasks_to_run = list(TASK_CONFIGS.keys()) if args.task == "all" else [args.task]

    for task in tasks_to_run:
        data_path = task_data_map[task]
        if not data_path.exists():
            print(f"\n[SKIP] {task}: data file not found: {data_path}")
            print(f"       Run: python data/scripts/prepare_{task}_data.py")
            continue
        train_task(task, data_path, args.output_dir, repo_root)

    trained = [t for t in tasks_to_run if (args.output_dir / f"{t}-v1" / "evaluation.json").exists()]
    if trained:
        print_activation_instructions(trained, args.output_dir)


if __name__ == "__main__":
    main()
