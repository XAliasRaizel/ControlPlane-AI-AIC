"""
ml/fairness/train.py

Fine-tunes a bias/fairness classifier on HateXplain using LoRA, and keeps
the dataset's human-annotated rationale spans as a training-analysis artifact.

RUNTIME CAVEAT: at inference time a new user query has no human rationale.
The rationale extraction below is for training-set analysis and demo narrative
("here is what the model was trained to key off"), NOT a runtime feature.
The live detector's `evidence` field reports the model's own predicted label
and confidence (via model_backend.py) — same as every other ML-backed detector.
If you want token-level evidence at runtime, that is an attention-based or
integrated-gradients explainability pass — a real, separate piece of work.

HateXplain label encoding: 0=hatespeech, 1=normal, 2=offensive.
is_flagged = int(majority_label != 1)  — "not normal" -> flagged.

NOT executed here — no torch/transformers/datasets in this sandbox.
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from ml.common.data_utils import group_aware_split
from ml.common.eval_utils import compute_detection_metrics
from ml.common.lora_utils import LoraTrainConfig, build_lora_model, run_training, tokenize_dataset


def load_hatexplain() -> pd.DataFrame:
    """Load HateXplain from HuggingFace and convert to a flat DataFrame.

    Majority label across 3 annotators decides is_flagged.
    Rationale spans are the words that >50% of annotators highlighted as the
    reason for the hate/offensive label — kept as a column for offline analysis,
    NOT used at inference time.

    Before trusting the column names verbatim, confirm with:
        ds["train"].features
    in your actual training environment — the schema can shift between
    dataset versions.
    """
    from datasets import load_dataset

    ds = load_dataset("hatexplain")
    rows = []
    for split_name in ("train", "validation", "test"):
        for row in ds[split_name]:
            tokens = row["post_tokens"]
            text = " ".join(tokens)

            labels = row["annotators"]["label"]
            majority_label = max(set(labels), key=labels.count)
            is_flagged = int(majority_label != 1)  # not "normal" -> flagged

            rationale_spans: list[str] = []
            if row["rationales"]:
                valid = [r for r in row["rationales"] if len(r) == len(tokens)]
                if valid:
                    avg_mask = np.array(valid, dtype=float).mean(axis=0)
                    rationale_spans = [
                        tok for tok, m in zip(tokens, avg_mask) if m > 0.5
                    ]

            rows.append({
                "text": text,
                "label": is_flagged,
                "rationale": " ".join(rationale_spans),
            })

    df = pd.DataFrame(rows)
    print(f"HateXplain loaded: {len(df):,} rows")
    print(df["label"].value_counts().to_string())
    return df


def main() -> None:
    print("=== Fairness Classifier Fine-Tuning (HateXplain) ===")
    print("\n[1/4] Loading HateXplain...")
    df = load_hatexplain()

    print("\n[2/4] Group-aware split...")
    train_df, val_df, test_df = group_aware_split(
        df, text_col="text", test_size=0.15, val_size=0.15, seed=42
    )
    print(f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    from datasets import Dataset

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    print("\n[3/4] Building LoRA model...")
    cfg = LoraTrainConfig(
        base_model="roberta-base",
        num_labels=2,
        output_dir="ml/models/fairness",
    )
    model, tokenizer = build_lora_model(cfg)
    train_ds = tokenize_dataset(train_ds, tokenizer, "text", cfg)
    val_ds = tokenize_dataset(val_ds, tokenizer, "text", cfg)

    print("\n[4/4] Training...")
    trainer, epoch_metrics = run_training(cfg, train_ds, val_ds, tokenizer, model)
    for epoch, m in sorted(epoch_metrics.items()):
        print(f"  epoch {epoch}: recall={m.get('eval_recall', 'n/a'):.3f}  "
              f"f1={m.get('eval_f1', 'n/a'):.3f}")

    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    # Sample (text, human_rationale) pairs for offline analysis/slide narrative.
    # NOT a runtime lookup table — see the module docstring.
    sample_path = f"{cfg.output_dir}/sample_rationales.csv"
    test_df[["text", "rationale"]].head(100).to_csv(sample_path, index=False)
    print(f"\nSaved model to {cfg.output_dir}")
    print(f"Sample rationales CSV (offline analysis only): {sample_path}")
    print("\nNOTE: At runtime, evidence = model label + confidence via model_backend.py.")
    print("      Do NOT surface human rationale as if it were available for new queries.")


if __name__ == "__main__":
    main()
