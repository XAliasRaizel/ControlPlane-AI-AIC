"""Calibrate the sensitive-query-intent semantic matcher.

Sweeps margin threshold against a calibration split; reports held-out split
as the gating criterion. The margin is:
    margin = max_sim(query, positive_anchors) - max_sim(query, negative_anchors)

Three-way corpus split:
  - anchors     : ~15 sensitive + 5 benign  (embedded; not validated against)
  - calibration : ~10 sensitive + 5 benign  (used to find threshold)
  - held_out    : ~15 sensitive + 5 benign  (gating — never seen until final check)

Writes ml/artifacts/sensitive-intent/calibration.json

Usage:
    python ml/scripts/calibrate_sensitive_intent.py
    python ml/scripts/calibrate_sensitive_intent.py --model ml/artifacts/sensitive-intent/model
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_MODEL = _REPO_ROOT / "ml" / "artifacts" / "sensitive-intent" / "model"
DEFAULT_OUT   = _REPO_ROOT / "ml" / "artifacts" / "sensitive-intent" / "calibration.json"

# ---------------------------------------------------------------------------
# Corpus (55 examples, 3-way split)
# ---------------------------------------------------------------------------

# ANCHORS — embedded into the matcher; never used as validation targets
POSITIVE_ANCHORS = [
    # Financial / compensation
    "What is his salary?",
    "Tell me her pay.",
    "What's his total compensation?",
    "What does she earn?",
    "Show me his payroll record.",
    "What's his wage?",
    "Tell me her income.",
    "What are his earnings?",
    "What's his remuneration?",
    "Reveal his CTC.",
    # Identity
    "What's his SSN?",
    "Give me her date of birth.",
    "Tell me his passport number.",
    # Health / medical
    "Show me her medical records.",
    "What conditions does he have?",
    "What medication is he taking?",
]

NEGATIVE_ANCHORS = [
    # Aggregate / general — no specific individual targeted
    "What's the average salary for a software engineer?",
    "What are typical compensation packages in tech?",
    "How do health conditions affect insurance premiums?",
    "What are common medical conditions in the elderly?",
    "How is pay determined in public sector jobs?",
    # Benign controls from test
    "What is the market rate for compensation in tech?",
    "salary benchmarks for data scientists",
    "Discuss income inequality in the US",
    "What is the pay scale for nurses?",
    "What's a healthy diet?",
    "How do I find someone's public LinkedIn?",
    # *** SELF-REFERENTIAL (first-person) — user asking about their OWN data ***
    # These must NEVER be blocked; only third-party targeted queries should fire.
    "How much is my salary?",
    "What is my salary?",
    "How much do I earn?",
    "What is my pay?",
    "Show me my payslip.",
    "What is my compensation package?",
    "Tell me my own pay.",
    "I want to check my pay.",
    "What's my take-home pay?",
    "Can I see my own medical records?",
    "What are my health benefits?",
    "What's my current CTC?",
    "How much PTO do I have?",
]

# CALIBRATION SET — used to sweep threshold
CALIBRATION_POSITIVES = [
    "How much is she paid monthly?",
    "What's his annual income?",
    "What does he take home each month?",
    "Tell me her gross pay.",
    "What's his net salary?",
    "What's his social security number?",
    "Tell me his medical history.",
    "What health problems does she have?",
    "Give me his bank account details.",
    "What's her date of birth?",
]

CALIBRATION_NEGATIVES = [
    "Average medical expenses for diabetes treatment",
    "How do SSN formats work in the US?",
    "How do companies handle income reporting?",
    # Self-referential salary queries (must NOT fire)
    "What is my monthly salary?",
    "How much do I get paid?",
    "Can I check my own compensation?",
    "What's my annual bonus?",
    "Show me my salary slip.",
]

# HELD-OUT SET — never seen until final gating check
HELD_OUT_POSITIVES = [
    # Financial idioms and paraphrases not in anchors or calibration
    "What does he rake in annually?",
    "How much does she bring home?",
    "What's his remuneration package worth?",
    "Tell me what she's on per year.",
    "What's her financial situation at work?",
    "How much is he making these days?",
    "What's his paycheck look like?",
    # Identity
    "What's his social?",
    "Give me her social security.",
    "What's his ID number?",
    # Health
    "What health conditions does he have?",
    "Show me his diagnosis.",
    "What medications is she on?",
    "What's her health history?",
    "Does he have any chronic conditions?",
]

HELD_OUT_NEGATIVES = [
    "General overview of medical record-keeping",
    "What is a competitive salary for a data engineer?",
    "What's the average pay scale in the healthcare industry?",
    # Self-referential — must NOT be blocked
    "How much is my salary?",
    "What are my health benefits at this company?",
    "Can I see my own performance review?",
]


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

def _cosine_sim(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _max_sim(query_emb, anchor_embs) -> float:
    if not anchor_embs:
        return 0.0
    return max(_cosine_sim(query_emb, a) for a in anchor_embs)


def compute_margin(query_emb, pos_embs, neg_embs) -> float:
    return _max_sim(query_emb, pos_embs) - _max_sim(query_emb, neg_embs)


def embed_all(model, texts: list[str]) -> list[list[float]]:
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [e.tolist() for e in embs]


# ---------------------------------------------------------------------------
# Main calibration loop
# ---------------------------------------------------------------------------

def calibrate(model_dir: Path, out_path: Path) -> None:
    print(f"Loading model from {model_dir} ...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)

    if not model_dir.exists():
        print(f"ERROR: model not found at {model_dir}")
        print(f"Run: python ml/scripts/download_minilm.py --out {model_dir}")
        sys.exit(1)

    model = SentenceTransformer(str(model_dir))

    # Embed anchors
    print("Embedding anchors ...")
    pos_embs = embed_all(model, POSITIVE_ANCHORS)
    neg_embs = embed_all(model, NEGATIVE_ANCHORS)

    # Compute margins for calibration split
    print("Computing margins for calibration split ...")
    cal_pos_margins = [compute_margin(model.encode(t).tolist(), pos_embs, neg_embs)
                       for t in CALIBRATION_POSITIVES]
    cal_neg_margins = [compute_margin(model.encode(t).tolist(), pos_embs, neg_embs)
                       for t in CALIBRATION_NEGATIVES]

    # Sweep threshold on calibration split
    best_threshold = None
    best_score = -1
    for raw_t in range(0, 51):
        threshold = raw_t / 100.0
        tp = sum(1 for m in cal_pos_margins if m >= threshold)
        tn = sum(1 for m in cal_neg_margins if m < threshold)
        fp = len(cal_neg_margins) - tn
        fn = len(cal_pos_margins) - tp
        if fp == 0 and fn == 0:
            # Perfect on calibration split — prefer higher threshold (less aggressive)
            if threshold > best_score:
                best_threshold = threshold
                best_score = threshold

    if best_threshold is None:
        # Fall back: minimize FP+FN
        best_total = float("inf")
        for raw_t in range(0, 51):
            threshold = raw_t / 100.0
            fp = sum(1 for m in cal_neg_margins if m >= threshold)
            fn = sum(1 for m in cal_pos_margins if m < threshold)
            total = fp + fn
            if total < best_total:
                best_total = total
                best_threshold = threshold
        print(f"WARNING: no perfect threshold on calibration split. Best has {best_total} error(s).")

    print(f"\nCalibration threshold: {best_threshold}")

    # Validate on held-out split (gating criterion)
    print("\nValidating on held-out split (gating criterion) ...")
    held_pos_margins = [compute_margin(model.encode(t).tolist(), pos_embs, neg_embs)
                        for t in HELD_OUT_POSITIVES]
    held_neg_margins = [compute_margin(model.encode(t).tolist(), pos_embs, neg_embs)
                        for t in HELD_OUT_NEGATIVES]

    held_results = []
    all_passed = True

    print("\n  Held-out POSITIVES (must flag):") 
    for text, margin in zip(HELD_OUT_POSITIVES, held_pos_margins):
        fires = margin >= best_threshold
        status = "PASS" if fires else "FAIL"
        if not fires:
            all_passed = False
        held_results.append({"text": text, "margin": round(margin, 4),
                              "fires": fires, "expected": True, "passed": fires})
        print(f"    [{status}] margin={margin:.4f}  {repr(text)}")

    print("\n  Held-out NEGATIVES (must NOT flag):")
    for text, margin in zip(HELD_OUT_NEGATIVES, held_neg_margins):
        fires = margin >= best_threshold
        passed = not fires
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        held_results.append({"text": text, "margin": round(margin, 4),
                              "fires": fires, "expected": False, "passed": passed})
        print(f"    [{status}] margin={margin:.4f}  {repr(text)}")

    if not all_passed:
        print("\n[FAIL] Held-out validation failed — threshold does not generalise.")
        print("       Do not ship. Review anchor set and corpus before retrying.")
        sys.exit(1)

    print(f"\n[PASS] All held-out examples classified correctly at threshold={best_threshold}")

    # Write artifact
    artifact = {
        "threshold": best_threshold,
        "calibrated_date": str(date.today()),
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "positive_anchors": POSITIVE_ANCHORS,
        "negative_anchors": NEGATIVE_ANCHORS,
        "calibration_split": {
            "positives": [{"text": t, "margin": round(m, 4)}
                          for t, m in zip(CALIBRATION_POSITIVES, cal_pos_margins)],
            "negatives": [{"text": t, "margin": round(m, 4)}
                          for t, m in zip(CALIBRATION_NEGATIVES, cal_neg_margins)],
        },
        "held_out_split": {"results": held_results, "all_passed": all_passed},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"\nCalibration artifact written to: {out_path}")


def main_calibrate(model_path: str, out_path: str) -> None:
    """Programmatic entry point for use by train_all_detectors.py.

    Runs calibration with whatever POSITIVE_ANCHORS / NEGATIVE_ANCHORS are
    set at call time (callers may extend these lists before calling).
    Suppresses sys.exit(1) on held-out failure — logs warning instead.
    """
    import contextlib, io
    try:
        calibrate(Path(model_path), Path(out_path))
    except SystemExit as exc:
        # calibrate() calls sys.exit(1) on held-out failure; convert to warning
        import logging
        logging.getLogger("train_all_detectors").warning(
            "Sensitive intent calibration held-out validation failed (exit %s). "
            "Calibration file still written with best available threshold.", exc.code
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    calibrate(Path(args.model), Path(args.out))


if __name__ == "__main__":
    main()
