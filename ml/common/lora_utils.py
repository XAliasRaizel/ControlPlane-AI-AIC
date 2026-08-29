"""
ml/common/lora_utils.py

Shared LoRA fine-tuning scaffold for sequence classification.
Used by prompt_injection/train.py and fairness/train.py.
Safety and grounding are evaluate-first — do NOT wire LoRA into those
unless evaluation shows the pretrained model actually needs it.

NOT executed against a real model here — torch/transformers/peft are not
in the test environment. Written against stable HF/PEFT APIs (same pattern
used throughout the PEFT documentation). Two API-version caveats to verify
against your installed versions before running:

  1. Trainer's tokenizer argument was renamed from `tokenizer=` to
     `processing_class=` in newer transformers. Check your version — using
     the wrong one is an immediate crash, not a silent bug.
  2. TrainingArguments' `evaluation_strategy` was renamed to `eval_strategy`.
     Same rule: immediate crash on mismatch.

Both are handled with a runtime version check in run_training() below.

Improvement: run_training() now returns (trainer, metrics_summary) — a dict
of {epoch: eval_metrics} — so callers can log or compare training dynamics
without re-deriving them from trainer state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np


@dataclass
class LoraTrainConfig:
    base_model: str = "roberta-base"
    num_labels: int = 2
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    # Standard for RoBERTa / DeBERTa attention projections.
    target_modules: Tuple[str, ...] = ("query", "value")
    # LoRA typically wants a higher LR than full fine-tuning.
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    output_dir: str = "ml/models/output"
    max_length: int = 256
    seed: int = 42


def build_lora_model(cfg: LoraTrainConfig):
    """Load the base sequence-classification model and wrap with a LoRA adapter.

    Returns (model, tokenizer).
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model, num_labels=cfg.num_labels
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()  # sanity-check: should show a small % of total
    return model, tokenizer


def tokenize_dataset(dataset, tokenizer, text_col: str, cfg: LoraTrainConfig):
    def _tok(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            max_length=cfg.max_length,
            padding="max_length",
        )

    return dataset.map(_tok, batched=True)


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def compute_metrics_fn(eval_pred):
    """Passed to Trainer(compute_metrics=...).

    Uses eval_utils so the numbers reported during training match what you
    would compute yourself afterward — one source of truth for what "good" means.
    """
    from ml.common.eval_utils import compute_detection_metrics

    logits, labels = eval_pred
    probs = _softmax(np.asarray(logits))[:, 1]  # prob of positive/flagged class
    return compute_detection_metrics(labels, probs, threshold=0.5)


def run_training(
    cfg: LoraTrainConfig,
    train_dataset,
    eval_dataset,
    tokenizer,
    model,
) -> Tuple[Any, Dict[int, Dict[str, float]]]:
    """Run Trainer and return (trainer, metrics_summary).

    metrics_summary maps epoch (1-indexed int) to the eval metrics at that
    checkpoint — useful for logging training dynamics without re-deriving
    from trainer state.

    Handles the transformers API version differences for
    `eval_strategy` / `evaluation_strategy` and `processing_class` / `tokenizer`
    at runtime so the same script works across a range of transformers versions.
    """
    import transformers
    from transformers import Trainer, TrainingArguments
    from packaging.version import Version

    tf_version = Version(transformers.__version__)

    # TrainingArguments kwarg: eval_strategy (>=4.46) or evaluation_strategy (<4.46)
    eval_strat_kwarg = (
        "eval_strategy" if tf_version >= Version("4.46.0") else "evaluation_strategy"
    )

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=cfg.seed,
        report_to=[],
        **{eval_strat_kwarg: "epoch"},
    )

    # Trainer kwarg: processing_class (>=4.46) or tokenizer (<4.46)
    tok_kwarg = (
        "processing_class" if tf_version >= Version("4.46.0") else "tokenizer"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_fn,
        **{tok_kwarg: tokenizer},
    )
    trainer.train()

    # Collect per-epoch eval metrics from trainer log history
    metrics_summary: Dict[int, Dict[str, float]] = {}
    for entry in trainer.state.log_history:
        if "eval_loss" in entry and "epoch" in entry:
            epoch = int(round(entry["epoch"]))
            metrics_summary[epoch] = {
                k: v for k, v in entry.items() if k != "epoch"
            }

    return trainer, metrics_summary
