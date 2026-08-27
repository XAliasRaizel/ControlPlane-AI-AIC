"""Side-by-side comparison: regex-only detector vs. pretrained/fine-tuned model.

Runs the same labeled JSONL dataset through:
  (a) The deterministic regex/rule detector (current production behaviour)
  (b) A trained/pretrained model artifact via CalibratedClassifier

Outputs a comparison table showing where each approach wins, ties, or loses,
helping you decide the operating threshold and whether Track B fine-tuning is worth it.

Usage:
    python ml/scripts/compare_detectors.py \\
        --task injection \\
        --artifact ml/artifacts/injection-pretrained \\
        --data data/injection.jsonl

    # Compare on the held-out test split only
    python ml/scripts/compare_detectors.py \\
        --task injection \\
        --artifact ml/artifacts/injection-v1 \\
        --data data/injection.jsonl \\
        --split test

    # Save JSON results
    python ml/scripts/compare_detectors.py \\
        --task injection \\
        --artifact ml/artifacts/injection-v1 \\
        --data data/injection.jsonl \\
        --output ml/artifacts/injection-v1/comparison.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    from ml.common import confusion_at_threshold, grouped_split, load_jsonl_records
except ImportError:
    from common import confusion_at_threshold, grouped_split, load_jsonl_records  # type: ignore


# ---------------------------------------------------------------------------
# Regex detectors — replicated from the live backend to avoid importing it
# (which would require the full FastAPI stack). Kept in sync by design.
# ---------------------------------------------------------------------------

import re


def _regex_injection_score(text: str) -> tuple[float, list[str]]:
    lc = text.lower()
    signatures = {
        "instruction_override": [
            "ignore previous instructions", "ignore all previous",
            "disregard previous instructions", "forget your instructions",
        ],
        "prompt_extraction": [
            "reveal your system prompt", "show me your system prompt",
            "reveal your hidden prompt", "developer message",
        ],
        "jailbreak": [
            "bypass safety", "jailbreak", "disable your restrictions",
            "no restrictions", "developer mode",
        ],
        "role_manipulation": ["you are now", "act as an unrestricted", "pretend to be"],
    }
    evidence = [
        cat for cat, patterns in signatures.items()
        if any(re.search(re.escape(p), lc) for p in patterns)
    ]
    score = min(1.0, 0.9 + 0.05 * (len(evidence) - 1)) if evidence else 0.0
    return score, evidence


def _regex_toxicity_score(text: str) -> tuple[float, list[str]]:
    lc = text.lower()
    _SIGS = {
        "violence": [r"\b(?:make\s+a\s+bomb|explosive\s+recipe|weapon\s+construction|harm\s+someone|hurt\s+someone|kill\s+someone)\b"],
        "illegal_activity": [r"\b(?:evade\s+law\s+enforcement|launder\s+money|forge\s+documents|counterfeit|drug\s+synthesis|smuggle)\b"],
        "hacking": [r"\b(?:hack|hacking|hacked|hacker|breach|exploit|sql\s+injection|crack\s+password|brute\s+force|steal\s+credentials|exfiltrate|break\s+into|penetrate\s+system|dump\s+database|rootkit|keylogger|phishing)\b"],
        "data_theft": [r"\b(?:steal\s+account|steal\s+information|steal\s+records|extract\s+private|extract\s+confidential|leak\s+data|expose\s+private|expose\s+confidential)\b"],
        "self_harm": [r"\b(?:self[- ]harm|suicide\s+method|how\s+to\s+end\s+my\s+life)\b"],
    }
    hits = []
    for cat, pats in _SIGS.items():
        for p in pats:
            if re.search(p, lc, re.I):
                hits.append(cat)
                break
    n = len(hits)
    score = min(1.0, 0.85 + 0.10 * (n - 1)) if n > 0 else 0.0
    return score, hits


def _regex_fairness_score(text: str) -> tuple[float, list[str]]:
    lc = text.lower()
    _TERMS = [
        "gender", "ethnicity", "religion", "race", "disability", "age",
        "because she is", "because he is", "too old", "too young",
    ]
    hits = [t for t in _TERMS if t in lc]
    score = round(min(1.0, 0.40 * len(hits)), 3)
    return score, hits


_REGEX_DETECTORS = {
    "injection": (_regex_injection_score, 0.5),
    "toxicity": (_regex_toxicity_score, 0.5),
    "fairness": (_regex_fairness_score, 0.5),
}


def regex_predict(task: str, text: str) -> dict:
    fn, threshold = _REGEX_DETECTORS.get(task, (_regex_injection_score, 0.5))
    score, evidence = fn(text)
    fires = score >= threshold
    return {"score": score, "fires": fires, "evidence": evidence, "threshold": threshold}


# ---------------------------------------------------------------------------
# Model scoring
# ---------------------------------------------------------------------------

def _load_calib(artifact_dir: Path) -> dict:
    for candidate in (artifact_dir.parent / "calibration.json",
                       artifact_dir / "calibration.json"):
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def _sigmoid(x: float) -> float:
    import math
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def load_model(model_dir: Path, calib: dict):
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        print("[ERROR] torch/transformers not installed. Run: pip install -r ml/requirements-ml.txt")
        sys.exit(1)

    print(f"[INFO] Loading model from {model_dir} ...")
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    mdl.eval()
    return tok, mdl


def model_score(text: str, tokenizer, model, calib: dict, torch) -> float:
    temperature = float(calib.get("temperature", 1.0)) or 1.0
    positive_index = int(calib.get("positive_index", 1))
    max_length = int(calib.get("max_length", 256))
    enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits[0]
    pos = float(logits[positive_index].item())
    neg = float(logits[1 - positive_index].item())
    return _sigmoid((pos - neg) / temperature)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(task: str, records: list[dict], artifact_dir: Path) -> dict:
    import torch

    model_dir = artifact_dir / "model"
    if not model_dir.exists():
        model_dir = artifact_dir
    calib = _load_calib(artifact_dir)
    model_threshold = float(calib.get("threshold", 0.5))

    tokenizer, mdl = load_model(model_dir, calib)

    regex_results, model_results, labels = [], [], []
    for i, rec in enumerate(records, 1):
        if i % 50 == 0:
            print(f"  Processing {i}/{len(records)} ...", end="\r")
        text = rec["text"]
        reg = regex_predict(task, text)
        mscore = model_score(text, tokenizer, mdl, calib, torch)
        regex_results.append(reg)
        model_results.append(mscore)
        labels.append(rec["label"])

    print(f"  Processing {len(records)}/{len(records)} — done.     ")

    regex_scores = [r["score"] for r in regex_results]
    regex_labels = [1 if r["fires"] else 0 for r in regex_results]

    regex_metrics = confusion_at_threshold(regex_scores, labels, 0.5)
    model_metrics = confusion_at_threshold(model_results, labels, model_threshold)

    # Agreement analysis
    agree = sum(1 for rl, ml in zip(regex_labels, [1 if s >= model_threshold else 0 for s in model_results]) if rl == ml)
    model_only_catch = sum(1 for rl, ms, y in zip(regex_labels, model_results, labels)
                           if rl == 0 and ms >= model_threshold and y == 1)
    regex_only_catch = sum(1 for rl, ms, y in zip(regex_labels, model_results, labels)
                           if rl == 1 and ms < model_threshold and y == 1)
    both_miss = sum(1 for rl, ms, y in zip(regex_labels, model_results, labels)
                    if rl == 0 and ms < model_threshold and y == 1)

    return {
        "task": task,
        "n_eval": len(records),
        "regex_metrics": regex_metrics,
        "model_metrics": model_metrics,
        "model_threshold": model_threshold,
        "agreement_rate": round(agree / max(len(records), 1), 4),
        "model_catches_regex_misses": model_only_catch,
        "regex_catches_model_misses": regex_only_catch,
        "both_miss": both_miss,
        "summary": {
            "regex_fnr": regex_metrics["fnr"],
            "model_fnr": model_metrics["fnr"],
            "regex_fpr": regex_metrics["fpr"],
            "model_fpr": model_metrics["fpr"],
            "regex_f1": regex_metrics["f1"],
            "model_f1": model_metrics["f1"],
        },
    }


def print_comparison(result: dict) -> None:
    rm = result["regex_metrics"]
    mm = result["model_metrics"]
    s = result["summary"]

    print(f"\n{'='*60}")
    print(f"COMPARISON: {result['task'].upper()}  (n={result['n_eval']})")
    print(f"{'='*60}")
    print(f"{'Metric':<18} {'Regex-only':>12} {'Model':>12}  {'Winner':>10}")
    print(f"{'─'*60}")

    def row(name: str, rv: float, mv: float, lower_better: bool = False) -> None:
        if lower_better:
            winner = "✓ model" if mv < rv else ("✓ regex" if rv < mv else "   tie")
        else:
            winner = "✓ model" if mv > rv else ("✓ regex" if rv > mv else "   tie")
        print(f"  {name:<16} {rv:>12.4f} {mv:>12.4f}  {winner:>10}")

    row("FNR ↓", s["regex_fnr"], s["model_fnr"], lower_better=True)
    row("FPR ↓", s["regex_fpr"], s["model_fpr"], lower_better=True)
    row("F1  ↑", s["regex_f1"], s["model_f1"], lower_better=False)

    print(f"\n  Agreement rate:            {result['agreement_rate']:.2%}")
    print(f"  Model catches regex miss:  {result['model_catches_regex_misses']}")
    print(f"  Regex catches model miss:  {result['regex_catches_model_misses']}")
    print(f"  Both miss (FN for both):   {result['both_miss']}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare regex vs. model detector performance.")
    parser.add_argument("--task", choices=["injection", "toxicity", "fairness"], required=True)
    parser.add_argument("--artifact", type=Path, required=True,
                        help="Artifact dir (contains model/ and calibration.json)")
    parser.add_argument("--data", type=Path, required=True,
                        help="Labeled JSONL dataset")
    parser.add_argument("--split", choices=["full", "train", "valid", "test"], default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None,
                        help="Save JSON comparison to this path (optional)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_records = load_jsonl_records(args.data)
    if args.split == "full":
        eval_records = all_records
    else:
        train, valid, test = grouped_split(all_records, seed=args.seed)
        split_map = {"train": train, "valid": valid, "test": test}
        eval_records = split_map[args.split]

    print(f"[INFO] Comparing on {len(eval_records)} records ({args.split} split)")

    result = compare(args.task, eval_records, args.artifact)
    print_comparison(result)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[OK] Comparison report saved to: {args.output}")


if __name__ == "__main__":
    main()
