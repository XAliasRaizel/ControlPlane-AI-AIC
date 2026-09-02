"""Train (continue fine-tuning) all ControlPlane hot-path detector models.

This script downloads the curated datasets for each detector, merges them,
fine-tunes the appropriate base model, evaluates it, and writes the artifact
in the exact layout expected by backend/shared/model_backend.py.

Detectors trained
-----------------
1. injection  — protectai/deberta-v3-base-prompt-injection-v2
                 on: deepset/prompt-injections + allenai/wildjailbreak
2. safety     — s-nlp/roberta_toxicity_classifier
                 on: lmsys/toxic-chat + PKU-Alignment/BeaverTails
3. fairness   — facebook/roberta-hate-speech-dynabench-r4-target
                 on: HateXplain + allenai/toxigen
4. sensitive  — MiniLM anchor-only recalibration from banking77 (no GPU training)

Usage
-----
  # Train all detectors (overnight)
  python ml/scripts/train_all_detectors.py

  # Train one detector only
  python ml/scripts/train_all_detectors.py --task injection

  # Quick smoke-test (100 examples, 1 epoch — verifies pipeline works)
  python ml/scripts/train_all_detectors.py --smoke-test

GPU notes
---------
  RTX 4050 Laptop (6 GB VRAM): runs all tasks with fp16 + batch_size=16.
  CPU fallback available but very slow.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_all_detectors")

# ---------------------------------------------------------------------------
# Detector configs
# ---------------------------------------------------------------------------

DETECTOR_CONFIGS = {
    "injection": {
        "base_model": "protectai/deberta-v3-base-prompt-injection-v2",
        "output_dir": _REPO_ROOT / "ml" / "artifacts" / "injection-v2",
        "positive_label": "INJECTION_DETECTED",
        "positive_index": 1,
        "num_labels": 2,
        "max_length": 512,
        "batch_size": 16,
        "epochs": 3,
        "lr": 2e-5,
        "max_samples": 60_000,
        "datasets": ["deepset/prompt-injections", "allenai/wildjailbreak"],
    },
    "safety": {
        "base_model": "s-nlp/roberta_toxicity_classifier",
        "output_dir": _REPO_ROOT / "ml" / "artifacts" / "toxicity-v2",
        "positive_label": "UNSAFE_CONTENT",
        "positive_index": 1,
        "num_labels": 2,
        "max_length": 256,
        "batch_size": 32,
        "epochs": 3,
        "lr": 2e-5,
        "max_samples": 60_000,
        "datasets": ["lmsys/toxic-chat", "PKU-Alignment/BeaverTails"],
    },
    "fairness": {
        "base_model": "facebook/roberta-hate-speech-dynabench-r4-target",
        "output_dir": _REPO_ROOT / "ml" / "artifacts" / "fairness-v2",
        "positive_label": "BIASED",
        "positive_index": 1,
        "num_labels": 2,
        "max_length": 256,
        "batch_size": 32,
        "epochs": 3,
        "lr": 2e-5,
        "max_samples": 50_000,
        "datasets": ["hatexplain", "allenai/toxigen"],
    },
}

# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _sample(rows: list, max_n: int, seed: int = 42) -> list:
    if len(rows) <= max_n:
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, max_n)


def load_injection_data(max_samples: int, smoke_test: bool) -> tuple[list, list]:
    from datasets import load_dataset  # type: ignore
    rows: list[dict] = []

    log.info("  Downloading deepset/prompt-injections ...")
    ds = load_dataset("deepset/prompt-injections", split="train")
    for ex in ds:
        rows.append({"text": ex["text"], "label": int(ex["label"])})
    log.info("    deepset: %d examples", len(rows))

    # ── jackhhao/jailbreak-classification (open, no auth required) ─────
    log.info("  Downloading jackhhao/jailbreak-classification ...")
    try:
        jb = load_dataset("jackhhao/jailbreak-classification", split="train")
        jb_rows = []
        for ex in jb:
            # label: "benign" = 0, "jailbreak" = 1
            label = 0 if str(ex.get("type", "")).lower() == "benign" else 1
            text = str(ex.get("prompt", ""))
            if text:
                jb_rows.append({"text": text, "label": label})
        cap = (max_samples // 2) if not smoke_test else 100
        jb_rows = _sample(jb_rows, cap)
        rows.extend(jb_rows)
        log.info("    jailbreak-classification: %d examples", len(jb_rows))
    except Exception as e:
        log.warning("  jailbreak-classification skipped: %s", e)

    # ── rubend18/ChatGPT-Jailbreak-Prompts (open collection) ───────────
    log.info("  Downloading rubend18/ChatGPT-Jailbreak-Prompts ...")
    try:
        cj = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts", split="train")
        cj_rows = []
        for ex in cj:
            text = str(ex.get("text") or ex.get("prompt") or "")
            if text:
                cj_rows.append({"text": text, "label": 1})  # all are jailbreaks
        cap = 2000 if not smoke_test else 50
        cj_rows = _sample(cj_rows, cap)
        rows.extend(cj_rows)
        log.info("    ChatGPT-Jailbreak-Prompts: %d examples", len(cj_rows))
    except Exception as e:
        log.warning("  ChatGPT-Jailbreak-Prompts skipped: %s", e)

    log.info("  Downloading allenai/wildjailbreak (may require HF_TOKEN) ...")
    try:
        wj = load_dataset("allenai/wildjailbreak", split="train", trust_remote_code=True)
        adv_rows = []
        for ex in wj:
            adv = ex.get("adversarial") or ""
            vanilla = ex.get("vanilla") or ""
            if adv:
                adv_rows.append({"text": adv, "label": 1})
            if vanilla:
                adv_rows.append({"text": vanilla, "label": 0})
        cap = (max_samples // 2) if not smoke_test else 100
        adv_rows = _sample(adv_rows, cap)
        rows.extend(adv_rows)
        log.info("    wildjailbreak: %d examples (capped)", len(adv_rows))
    except Exception as e:
        log.warning("  wildjailbreak skipped (gated — set HF_TOKEN to unlock): %s", e)

    rows = _sample(rows, max_samples if not smoke_test else 100)
    random.shuffle(rows)
    split = int(len(rows) * 0.9)
    return rows[:split], rows[split:]


def load_safety_data(max_samples: int, smoke_test: bool) -> tuple[list, list]:
    from datasets import load_dataset  # type: ignore
    rows: list[dict] = []

    log.info("  Downloading lmsys/toxic-chat ...")
    try:
        tc = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
        for ex in tc:
            rows.append({"text": str(ex.get("user_input", "")), "label": int(ex.get("toxicity", 0))})
        log.info("    toxic-chat: %d examples", len(rows))
    except Exception as e:
        log.warning("  toxic-chat skipped: %s", e)

    log.info("  Downloading PKU-Alignment/BeaverTails ...")
    try:
        bt = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
        bt_rows = []
        for ex in bt:
            bt_rows.append({"text": str(ex.get("prompt", "")), "label": 0 if ex.get("is_safe", True) else 1})
        cap = (max_samples // 2) if not smoke_test else 100
        bt_rows = _sample(bt_rows, cap)
        rows.extend(bt_rows)
        log.info("    BeaverTails: %d examples (capped)", len(bt_rows))
    except Exception as e:
        log.warning("  BeaverTails skipped: %s", e)

    rows = _sample(rows, max_samples if not smoke_test else 100)
    random.shuffle(rows)
    split = int(len(rows) * 0.9)
    return rows[:split], rows[split:]


def load_fairness_data(max_samples: int, smoke_test: bool) -> tuple[list, list]:
    from datasets import load_dataset  # type: ignore
    rows: list[dict] = []

    log.info("  Downloading hatexplain ...")
    try:
        hx = load_dataset("hatexplain", split="train")
        for ex in hx:
            annots = ex.get("annotators", {}).get("label", [])
            if not annots:
                continue
            majority = max(set(annots), key=annots.count)
            label = 0 if majority == 2 else 1
            tokens = ex.get("post_tokens", [])
            text = " ".join(tokens) if tokens else ""
            if text:
                rows.append({"text": text, "label": label})
        log.info("    hatexplain: %d examples", len(rows))
    except Exception as e:
        log.warning("  hatexplain skipped: %s", e)

    log.info("  Downloading allenai/toxigen ...")
    try:
        tg = load_dataset("allenai/toxigen", split="train", trust_remote_code=True)
        tg_rows = []
        for ex in tg:
            rating = ex.get("toxicity_human", 0) or 0
            label = 1 if float(rating) >= 0.5 else 0
            text = str(ex.get("text", ""))
            if text:
                tg_rows.append({"text": text, "label": label})
        cap = (max_samples // 2) if not smoke_test else 100
        tg_rows = _sample(tg_rows, cap)
        rows.extend(tg_rows)
        log.info("    toxigen: %d examples (capped)", len(tg_rows))
    except Exception as e:
        log.warning("  toxigen skipped: %s", e)

    rows = _sample(rows, max_samples if not smoke_test else 100)
    random.shuffle(rows)
    split = int(len(rows) * 0.9)
    return rows[:split], rows[split:]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_classifier(task: str, cfg: dict, smoke_test: bool) -> None:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )
    from datasets import Dataset  # type: ignore
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("=" * 60)
    log.info("Task: %s | device: %s", task, device)
    if device == "cuda":
        log.info("GPU: %s | VRAM: %.1f GB", torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)
    log.info("Base model: %s", cfg["base_model"])
    log.info("=" * 60)

    loader = {"injection": load_injection_data, "safety": load_safety_data, "fairness": load_fairness_data}[task]
    train_rows, eval_rows = loader(cfg["max_samples"], smoke_test)
    log.info("[%s] Train: %d  Eval: %d", task, len(train_rows), len(eval_rows))

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["base_model"], num_labels=cfg["num_labels"], ignore_mismatched_sizes=True
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=cfg["max_length"])

    train_ds = Dataset.from_list(train_rows).map(tokenize, batched=True, remove_columns=["text"])
    eval_ds  = Dataset.from_list(eval_rows).map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        accuracy  = (tp + tn) / (tp + fp + fn + tn + 1e-9)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    out_dir   = cfg["output_dir"]
    model_dir = out_dir / "model"
    out_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = (device == "cuda")
    actual_batch = cfg["batch_size"] if not smoke_test else 8
    grad_accum   = max(1, 32 // actual_batch)
    # warmup_steps ≈ 10% of training steps (warmup_ratio removed in Transformers 5.x)
    steps_per_epoch = max(1, len(train_ds) // (actual_batch * grad_accum))
    total_steps = steps_per_epoch * (1 if smoke_test else cfg["epochs"])
    warmup_steps = max(10, int(total_steps * 0.1))

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=1 if smoke_test else cfg["epochs"],
        per_device_train_batch_size=actual_batch,
        per_device_eval_batch_size=actual_batch * 2,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg["lr"],
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=use_fp16,
        gradient_checkpointing=(device == "cuda"),
        dataloader_pin_memory=(device == "cuda"),
        report_to="none",
        logging_steps=50,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,   # renamed from tokenizer= in Transformers 5.x
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    log.info("[%s] Training ...", task)
    trainer.train()
    metrics = trainer.evaluate()
    log.info("[%s] Eval: %s", task, {k: round(float(v), 4) for k, v in metrics.items() if isinstance(v, float)})

    log.info("[%s] Saving model to %s ...", task, model_dir)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    calib = {
        "task": task,
        "base_model": cfg["base_model"],
        "temperature": 1.0,
        "threshold": 0.5,
        "positive_label": cfg["positive_label"],
        "positive_index": cfg["positive_index"],
        "max_length": cfg["max_length"],
        "lora": False,
        "pretrained": False,
        "datasets": cfg["datasets"],
        "train_metrics": {k: round(float(v), 4) for k, v in metrics.items() if isinstance(v, float)},
    }
    (out_dir / "calibration.json").write_text(json.dumps(calib, indent=2), encoding="utf-8")
    log.info("[%s] Done. Activate: $env:CONTROLPLANE_MODEL_%s = \"%s\"", task, task.upper(), model_dir)


# ---------------------------------------------------------------------------
# Sensitive intent — anchor recalibration (no GPU needed)
# ---------------------------------------------------------------------------

def recalibrate_sensitive_intent(smoke_test: bool) -> None:
    log.info("=" * 60)
    log.info("Task: sensitive_intent (anchor recalibration, no GPU needed)")
    log.info("=" * 60)

    SENSITIVE_BANKING_INTENTS = {
        "balance_not_updated_after_payment", "beneficiary_not_allowed",
        "card_about_to_expire", "card_acceptance", "card_delivery_estimate",
        "card_linked_accounts", "card_not_working", "card_payment_fee_charged",
        "card_payment_not_recognised", "card_payment_wrong_exchange_rate",
        "card_swallowed", "compromised_card", "edit_personal_details",
        "get_disposable_virtual_card", "get_physical_card", "pending_card_payment",
        "pin_blocked", "reverted_card_payment?", "wrong_amount_of_cash_received",
        "wrong_exchange_rate_for_cash_withdrawal", "transaction_charged_twice",
        "transfer_not_received_by_recipient", "transfer_timing",
    }

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        log.error("datasets not installed.")
        return

    new_positives: list[str] = []
    new_negatives: list[str] = []

    log.info("  Downloading banking77 ...")
    try:
        b77 = load_dataset("banking77", split="train")
        label_names = b77.features["label"].names
        for ex in b77:
            intent = label_names[ex["label"]]
            text = ex["text"]
            if intent in SENSITIVE_BANKING_INTENTS:
                new_positives.append(text)
            else:
                new_negatives.append(text)
        log.info("    banking77: %d positive, %d negative intents sampled",
                 len(new_positives), len(new_negatives))
    except Exception as e:
        log.warning("  banking77 skipped: %s", e)
        return

    rng = random.Random(42)
    cap = 20 if smoke_test else 30
    extra_pos = rng.sample(new_positives, min(cap, len(new_positives)))
    extra_neg = rng.sample(new_negatives, min(cap, len(new_negatives)))

    # Dynamically extend the calibration script's anchor lists and re-run
    import importlib.util
    calib_path = _REPO_ROOT / "ml" / "scripts" / "calibrate_sensitive_intent.py"
    spec = importlib.util.spec_from_file_location("calibrate_si", str(calib_path))
    calibrate_si = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(calibrate_si)  # type: ignore

    calibrate_si.POSITIVE_ANCHORS.extend(extra_pos)
    calibrate_si.NEGATIVE_ANCHORS.extend(extra_neg)

    model_p = str(_REPO_ROOT / "ml" / "artifacts" / "sensitive-intent" / "model")
    out_p   = str(_REPO_ROOT / "ml" / "artifacts" / "sensitive-intent" / "calibration.json")
    calibrate_si.main_calibrate(model_p, out_p)
    log.info("sensitive_intent Done. Updated calibration.json at %s", out_p)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["injection", "safety", "fairness", "sensitive_intent", "all"],
                   default="all")
    p.add_argument("--smoke-test", action="store_true",
                   help="100 examples, 1 epoch — verify the pipeline works end-to-end")
    args = p.parse_args()

    tasks = (
        ["injection", "safety", "fairness", "sensitive_intent"]
        if args.task == "all" else [args.task]
    )

    log.info("ControlPlane detector training | tasks: %s", tasks)
    if args.smoke_test:
        log.info("SMOKE-TEST mode — 100 examples, 1 epoch")

    import torch
    if torch.cuda.is_available():
        log.info("GPU: %s | VRAM: %.1f GB",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)
    else:
        log.warning("No GPU detected — training will be slow on CPU.")

    for task in tasks:
        if task == "sensitive_intent":
            recalibrate_sensitive_intent(args.smoke_test)
        else:
            train_classifier(task, DETECTOR_CONFIGS[task], args.smoke_test)

    log.info("=" * 60)
    log.info("All done. Update start.ps1 env vars to point at new -v2 artifacts.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
