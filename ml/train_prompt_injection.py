"""Train the first optional learned ControlPlane detector.

Input JSONL records require text, label (0=safe, 1=injection), and group_id.
The group-level split prevents near-identical attack variations from leaking
between train, validation, and test data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="roberta-base")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    seen_texts = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not {"text", "label", "group_id"} <= record.keys():
            raise ValueError(f"Line {line_number} is missing text, label, or group_id")
        normalized = " ".join(record["text"].lower().split())
        if not normalized or normalized in seen_texts:
            continue
        if record["label"] not in (0, 1):
            raise ValueError(f"Line {line_number} has an invalid binary label")
        seen_texts.add(normalized)
        records.append({"text": record["text"], "label": int(record["label"]), "group_id": record["group_id"]})
    if len({record["label"] for record in records}) != 2:
        raise ValueError("Training data must contain both safe and injection samples")
    return records


def grouped_split(records: list[dict[str, Any]]):
    from sklearn.model_selection import GroupShuffleSplit

    labels = [record["label"] for record in records]
    groups = [record["group_id"] for record in records]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_valid_indices, test_indices = next(splitter.split(records, labels, groups))
    train_valid = [records[index] for index in train_valid_indices]
    test = [records[index] for index in test_indices]

    valid_size = 0.125  # 12.5% of 80% ~= 10% of full data
    splitter = GroupShuffleSplit(n_splits=1, test_size=valid_size, random_state=43)
    train_indices, valid_indices = next(
        splitter.split(train_valid, [record["label"] for record in train_valid], [record["group_id"] for record in train_valid])
    )
    return [train_valid[index] for index in train_indices], [train_valid[index] for index in valid_indices], test


def main() -> None:
    args = parse_args()
    # Imported here so normal API installs do not need ML packages.
    from datasets import Dataset
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    records = load_records(args.data)
    train, validation, test = grouped_split(records)
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Not enough independent groups for train/validation/test splitting")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenized(records: list[dict[str, Any]]):
        dataset = Dataset.from_list(records)
        return dataset.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=256), batched=True)

    train_dataset, validation_dataset, test_dataset = map(tokenized, (train, validation, test))
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    def metrics(prediction):
        import numpy as np

        labels = prediction.label_ids
        probabilities = np.exp(prediction.predictions[:, 1]) / np.exp(prediction.predictions).sum(axis=1)
        predicted = (probabilities >= 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average="binary", zero_division=0)
        return {
            "precision_injection": precision,
            "recall_injection": recall,
            "f1_injection": f1,
            "roc_auc": roc_auc_score(labels, probabilities) if len(set(labels)) == 2 else 0.0,
        }

    args.output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output),
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to=[],
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        compute_metrics=metrics,
    )
    trainer.train()
    report = {
        "validation": trainer.evaluate(validation_dataset),
        "test": trainer.evaluate(test_dataset, metric_key_prefix="test"),
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "model": args.model,
    }
    trainer.save_model(str(args.output / "model"))
    tokenizer.save_pretrained(str(args.output / "model"))
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
