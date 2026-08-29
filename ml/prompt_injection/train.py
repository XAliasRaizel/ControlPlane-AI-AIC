"""
ml/prompt_injection/train.py

Fine-tunes a RoBERTa/DeBERTa sequence classifier for prompt injection
detection using LoRA. This is the unified training script — it replaces
the dataset-loading logic in ml/train_prompt_injection.py while keeping
the same group-aware split methodology that was already validated there.

Run on Colab/Kaggle (free T4 is enough for RoBERTa-base + LoRA at this
dataset size). NOT executed here — no torch/GPU in this environment.

Workflow:
  1. Load merged HF datasets via data.py
  2. Group-aware split (no near-duplicates across splits)
  3. LoRA fine-tune roberta-base
  4. Evaluate on held-out TEST set (not validation — the number that matters)
  5. Compare against deepset/deberta-v3-base-injection as baseline

If your fine-tuned model doesn't beat the deepset baseline on the same test
set, use deepset's model directly via model_backend.py — that's useful
information, not a failure.
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np

from ml.common.data_utils import group_aware_split
from ml.common.eval_utils import compare_models, compute_detection_metrics
from ml.common.lora_utils import LoraTrainConfig, build_lora_model, run_training, tokenize_dataset
from ml.prompt_injection.data import load_and_merge


def main() -> None:
    print("=== Prompt Injection Fine-Tuning ===")
    print("\n[1/5] Loading and merging datasets...")
    df = load_and_merge()

    print("\n[2/5] Group-aware split...")
    train_df, val_df, test_df = group_aware_split(
        df, text_col="text", test_size=0.15, val_size=0.15, seed=42
    )
    print(f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    from datasets import Dataset

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    test_ds = Dataset.from_pandas(test_df)

    print("\n[3/5] Building LoRA model...")
    cfg = LoraTrainConfig(
        base_model="roberta-base",
        num_labels=2,
        output_dir="ml/models/prompt_injection",
    )
    model, tokenizer = build_lora_model(cfg)

    train_ds = tokenize_dataset(train_ds, tokenizer, "text", cfg)
    val_ds = tokenize_dataset(val_ds, tokenizer, "text", cfg)
    test_ds = tokenize_dataset(test_ds, tokenizer, "text", cfg)

    print("\n[4/5] Training...")
    trainer, epoch_metrics = run_training(cfg, train_ds, val_ds, tokenizer, model)
    for epoch, m in sorted(epoch_metrics.items()):
        print(f"  epoch {epoch}: recall={m.get('eval_recall', 'n/a'):.3f}  "
              f"f1={m.get('eval_f1', 'n/a'):.3f}")

    print("\n[5/5] Held-out test evaluation...")
    predictions = trainer.predict(test_ds)
    logits = predictions.predictions
    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    fine_tuned_metrics = compute_detection_metrics(
        test_df["label"].tolist(), probs[:, 1].tolist()
    )

    # Baseline: deepset/deberta-v3-base-injection on the same test set.
    from transformers import pipeline as hf_pipeline

    baseline_clf = hf_pipeline(
        "text-classification",
        model="deepset/deberta-v3-base-injection",
        top_k=None,
        truncation=True,
    )
    baseline_scores = []
    for text in test_df["text"]:
        result = baseline_clf(text)[0]
        injection_score = max(
            (r["score"] for r in result if "injection" in r["label"].lower()),
            default=0.0,
        )
        baseline_scores.append(injection_score)
    baseline_metrics = compute_detection_metrics(
        test_df["label"].tolist(), baseline_scores
    )

    print("\n=== Model comparison on held-out test set ===")
    print(compare_models({
        "fine_tuned_roberta_lora": fine_tuned_metrics,
        "deepset/deberta-v3-base-injection": baseline_metrics,
    }))
    print(
        "\nIf the baseline beats your fine-tuned model, wire deepset's model directly "
        "via model_backend.py — that's the right engineering call, not a failure."
    )

    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"\nSaved LoRA adapter + tokenizer to {cfg.output_dir}")


if __name__ == "__main__":
    main()
