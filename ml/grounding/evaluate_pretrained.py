"""
ml/grounding/evaluate_pretrained.py

Measure-first discipline: compare the current token-overlap grounding formula
against an off-the-shelf NLI cross-encoder, on a labeled eval set.

If NLI clearly beats token-overlap (higher ROC-AUC on is_ungrounded detection),
swap the formula in backend/async_engines/grounding.py and pick a threshold
with find_threshold_for_target_recall() from ml/common/eval_utils.py.
This is a FORMULA SWAP + THRESHOLD PICK, not a training job.

Eval set columns: claim, context, is_ungrounded
  is_ungrounded: 1 = the claim is NOT supported by the context (hallucination)
                 0 = the claim is supported (grounded)

A pre-built eval_set.csv (20 rows) ships with this package for immediate use.

NOT executed here — no transformers/network in this sandbox.
"""
from __future__ import annotations

import argparse
import sys
sys.path.insert(0, ".")

import pandas as pd

from ml.common.eval_utils import (
    calibration_curve_points,
    compare_models,
    compute_detection_metrics,
    find_threshold_for_target_recall,
)


def token_overlap_score(claim: str, context: str) -> float:
    """Mirror of the existing formula: GroundingScore = 1 - TokenOverlap.

    High score = LESS grounded (matches existing convention so the two
    methods are directly comparable on one eval set).
    """
    claim_words = set(claim.lower().split())
    context_words = set(context.lower().split())
    if not claim_words:
        return 0.0
    overlap = len(claim_words & context_words) / len(claim_words)
    return 1.0 - overlap


def nli_entailment_score(claim: str, context: str, nli_pipeline) -> float:
    """Returns P(not entailed) — same direction as token_overlap_score
    (high = ungrounded), so both are directly comparable on the same eval set.
    """
    result = nli_pipeline({"text": context, "text_pair": claim})
    entail_prob = next(
        (r["score"] for r in result if r["label"].lower().startswith("entail")), 0.0
    )
    return 1.0 - entail_prob


def main(eval_csv: str = "ml/grounding/eval_set.csv") -> None:
    eval_df = pd.read_csv(eval_csv)
    y_true = eval_df["is_ungrounded"].tolist()

    print(f"Evaluating on {len(eval_df)} claim/context pairs...\n")

    # ---- Token overlap (current formula) ----
    token_scores = [
        token_overlap_score(c, x)
        for c, x in zip(eval_df["claim"], eval_df["context"])
    ]
    token_metrics = compute_detection_metrics(y_true, token_scores)

    # ---- NLI cross-encoder (candidate) ----
    from transformers import pipeline as hf_pipeline

    nli = hf_pipeline(
        "text-classification",
        model="cross-encoder/nli-deberta-v3-base",
        top_k=None,
        truncation=True,
    )
    nli_scores = [
        nli_entailment_score(c, x, nli)
        for c, x in zip(eval_df["claim"], eval_df["context"])
    ]
    nli_metrics = compute_detection_metrics(y_true, nli_scores)

    print(compare_models({
        "token_overlap (current)": token_metrics,
        "nli_cross_encoder (candidate)": nli_metrics,
    }))

    # If NLI wins, suggest the threshold to use.
    if nli_metrics["roc_auc"] > token_metrics["roc_auc"]:
        t = find_threshold_for_target_recall(y_true, nli_scores, target_recall=0.90)
        print(f"\nNLI wins on ROC-AUC. Suggested threshold for 90% recall: {t:.4f}")
        print("Swap async_engines/grounding.py to use nli_entailment_score() at this threshold.")
        pts = calibration_curve_points(y_true, nli_scores, n_thresholds=10)
        print("\nPrecision/Recall curve (NLI):")
        for p in pts:
            print(f"  t={p['threshold']:.3f}  prec={p['precision']:.3f}  recall={p['recall']:.3f}")
    else:
        print("\nToken overlap holds its own. Keep the existing formula.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-csv", default="ml/grounding/eval_set.csv")
    args = parser.parse_args()
    main(args.eval_csv)
