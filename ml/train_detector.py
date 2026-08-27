"""Generalized trainer for the optional learned ControlPlane detectors.

One entry point for the fine-tuning plan's classification detectors:
  injection  -> prompt-injection (RoBERTa / DeBERTa-v3)
  toxicity   -> safety / toxicity
  fairness   -> bias / fairness (rationale spans live in the dataset and are
                surfaced at inference time via DetectorResult.evidence)

It produces an artifact directory consumable, with zero code changes, by
backend/shared/model_backend.py:

    <output>/model/            HF model + tokenizer (save_pretrained)
    <output>/calibration.json  temperature, threshold (target-FNR), labels
    <output>/evaluation.json   validation + protected-test metrics

Heavy deps (torch/transformers/datasets/peft/sklearn) are imported lazily inside
main(), so importing this module never needs a GPU stack -- only ml.common
(dependency-light) is imported at module load.

Group-aware splitting stops near-identical attack variants leaking across
train/valid/test; temperature scaling calibrates the scores; the operating
threshold is chosen for a target false-negative rate, not a naive 0.5.

Run (Colab/Kaggle T4):
    python -m ml.train_detector --task injection \
        --data data/injection.jsonl --output ml/artifacts/injection-v0 --lora
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Dependency-light helpers. Support both `python -m ml.train_detector` (package
# import) and `python ml/train_detector.py` (script; ml/ is on sys.path[0]).
try:
    from ml.common import (apply_temperature, confusion_at_threshold,
                           fit_temperature, grouped_split, load_jsonl_records,
                           select_threshold_for_fnr)
except ImportError:  # pragma: no cover - exercised only as a bare script
    from common import (apply_temperature, confusion_at_threshold,  # type: ignore
                       fit_temperature, grouped_split, load_jsonl_records,
                       select_threshold_for_fnr)


TASK_DEFAULTS: dict[str, dict[str, str]] = {
    "injection": {"model": "roberta-base", "positive_label": "INJECTION_DETECTED"},
    "toxicity": {"model": "roberta-base", "positive_label": "UNSAFE_CONTENT"},
    "fairness": {"model": "roberta-base", "positive_label": "BIASED"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an optional learned detector.")
    parser.add_argument("--task", choices=sorted(TASK_DEFAULTS), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=None, help="HF model id (default per task)")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--target-fnr", type=float, default=0.05)
    parser.add_argument("--lora", action="store_true", help="Parameter-efficient LoRA fine-tune")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _margins_and_labels(trainer, dataset) -> tuple[list[float], list[int]]:
    """Per-example log-odds margin (logit_pos - logit_neg) and gold labels."""
    output = trainer.predict(dataset)
    logits = output.predictions
    labels = [int(y) for y in output.label_ids]
    margins = [float(row[1] - row[0]) for row in logits]
    return margins, labels


def _maybe_wrap_lora(model, enable: bool):
    """Wrap the model with a LoRA adapter when --lora is set and peft is present."""
    if not enable:
        return model, False
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError:
        print("peft not installed; falling back to a full fine-tune.")
        return model, False
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["query", "key", "value"],
    )
    return get_peft_model(model, config), True


def _safe_auc(fn, scores, labels):
    try:
        return round(float(fn(labels, scores)), 6) if len(set(labels)) == 2 else None
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    # Lazy heavy imports -- only needed when actually training.
    from datasets import Dataset
    from sklearn.metrics import average_precision_score, roc_auc_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    defaults = TASK_DEFAULTS[args.task]
    model_id = args.model or defaults["model"]
    positive_label = defaults["positive_label"]

    records = load_jsonl_records(args.data)
    train, valid, test = grouped_split(records, seed=args.seed)
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError("Not enough independent groups for train/valid/test")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def tokenize(rows: list[dict[str, Any]]):
        dataset = Dataset.from_list(rows)
        return dataset.map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=args.max_length),
            batched=True,
        )

    train_ds, valid_ds, test_ds = (tokenize(rows) for rows in (train, valid, test))
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    model, lora_used = _maybe_wrap_lora(model, args.lora)

    args.output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to=[],
        seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=valid_ds, tokenizer=tokenizer,
    )
    trainer.train()

    # --- Calibrate on validation, then choose the target-FNR threshold there ---
    valid_margins, valid_labels = _margins_and_labels(trainer, valid_ds)
    temperature = fit_temperature(valid_margins, valid_labels)
    valid_scores = [apply_temperature(m, temperature) for m in valid_margins]
    operating_point = select_threshold_for_fnr(valid_scores, valid_labels, args.target_fnr)
    threshold = operating_point["threshold"]

    # --- Protected test-set evaluation at the frozen threshold ---
    test_margins, test_labels = _margins_and_labels(trainer, test_ds)
    test_scores = [apply_temperature(m, temperature) for m in test_margins]
    test_metrics = confusion_at_threshold(test_scores, test_labels, threshold)
    test_metrics["roc_auc"] = _safe_auc(roc_auc_score, test_scores, test_labels)
    test_metrics["auprc"] = _safe_auc(average_precision_score, test_scores, test_labels)

    calibration = {
        "task": args.task,
        "temperature": temperature,
        "threshold": threshold,
        "positive_label": positive_label,
        "positive_index": 1,
        "max_length": args.max_length,
        "base_model": model_id,
        "lora": lora_used,
    }
    evaluation = {
        "task": args.task,
        "base_model": model_id,
        "lora": lora_used,
        "split_sizes": {"train": len(train), "valid": len(valid), "test": len(test)},
        "target_fnr": args.target_fnr,
        "validation_operating_point": operating_point,
        "test": test_metrics,
    }

    model_dir = args.output / "model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    (args.output / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    (args.output / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(json.dumps(evaluation, indent=2))
    print("Saved model to", model_dir)
    print("Calibration:", json.dumps(calibration))


if __name__ == "__main__":
    main()
