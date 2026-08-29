"""
ml/safety/evaluate_pretrained.py

Measure-first discipline: evaluate off-the-shelf toxicity models BEFORE
deciding whether to fine-tune. If the best off-the-shelf model already meets
your recall/FPR bar, wire it into model_backend.py and stop — do not build
a training script you don't need.

This script IS the "should we fine-tune" decision, made empirically.

Usage:
    python -m ml.safety.evaluate_pretrained
    # or with a custom eval set:
    python -m ml.safety.evaluate_pretrained --eval-csv ml/safety/eval_set.csv

A pre-built eval_set.csv (30 rows) ships with this package so the script
runs immediately. Replace it with your own labeled set (50-200 rows, text/label
columns, label: 1=unsafe, 0=safe) for production results.

NOT executed here — no transformers/network in this sandbox.
"""
from __future__ import annotations

import argparse
import sys
sys.path.insert(0, ".")

import pandas as pd

from ml.common.eval_utils import compare_models, compute_detection_metrics

CANDIDATE_MODELS = [
    "unitary/toxic-bert",
    "s-nlp/roberta_toxicity_classifier",
]


def evaluate_model(model_name: str, eval_df: pd.DataFrame) -> dict:
    from transformers import pipeline

    clf = pipeline(
        "text-classification", model=model_name, top_k=None, truncation=True
    )
    scores = []
    for text in eval_df["text"]:
        result = clf(text)[0]
        # Normalize across different label conventions (TOXIC, LABEL_1, 1, unsafe...).
        toxic_score = max(
            (
                r["score"]
                for r in result
                if r["label"].lower().replace("label_", "") in ("toxic", "unsafe", "1")
            ),
            default=0.0,
        )
        scores.append(toxic_score)
    return compute_detection_metrics(eval_df["label"].tolist(), scores)


def main(eval_csv: str = "ml/safety/eval_set.csv") -> None:
    eval_df = pd.read_csv(eval_csv)
    print(f"Evaluating {len(CANDIDATE_MODELS)} models on {len(eval_df)} examples...\n")

    results = {}
    for model_name in CANDIDATE_MODELS:
        print(f"--- {model_name} ---")
        results[model_name] = evaluate_model(model_name, eval_df)
        for k, v in results[model_name].items():
            print(f"  {k}: {v}")
        print()

    print(compare_models(results))
    print(
        "\nDecision rule: if the best model's recall and FPR already meet your bar, "
        "wire it via model_backend.py (set CONTROLPLANE_MODEL_SAFETY to the artifact dir) "
        "and stop. Only build ml/safety/train.py if this shows a real, measured gap."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-csv", default="ml/safety/eval_set.csv")
    args = parser.parse_args()
    main(args.eval_csv)
