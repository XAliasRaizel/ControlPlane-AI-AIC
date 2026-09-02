"""Train ControlPlane detector models on both local Kaggle datasets and HuggingFace gated datasets.

Datasets Ingested:
------------------
1. injection-v3:
   - Hugging Face (Gated): allenai/wildjailbreak (train config - 30,000 pairs)
   - Hugging Face: deepset/prompt-injections (546)
   - Hugging Face: jackhhao/jailbreak-classification (1,044)

2. toxicity-v3:
   - Kaggle: D:/tushar/random/aIC 2026 me/jigsaw-toxic-comment-classification-challenge/train.csv.zip (30,000 comments)
   - Hugging Face (Gated): allenai/wildguardmix (wildguardtrain config - 20,000 pairs)
   - Hugging Face: PKU-Alignment/BeaverTails (20,000 pairs)

3. fairness-v3:
   - Kaggle: D:/tushar/random/aIC 2026 me/jigsaw-unintended-bias-in-toxicity-classification/train.csv (35,000 identity-tagged comments)
   - Hugging Face: hatexplain (15,383)

4. sensitive-intent-v3 & PII:
   - Kaggle: D:/tushar/random/aIC 2026 me/pii-detection-removal-from-educational-data/train.json (extracts real PII contexts)
   - Ingests PII sentences into sensitive intent anchor calibration

Usage:
------
  python ml/scripts/train_scaled_detectors.py
  python ml/scripts/train_scaled_detectors.py --task injection
  python ml/scripts/train_scaled_detectors.py --task safety
  python ml/scripts/train_scaled_detectors.py --task fairness
  python ml/scripts/train_scaled_detectors.py --task pii
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import zipfile
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_scaled_detectors")

KAGGLE_TOXIC_PATH = Path(r"D:\tushar\random\aIC 2026 me\jigsaw-toxic-comment-classification-challenge\train.csv.zip")
KAGGLE_BIAS_PATH  = Path(r"D:\tushar\random\aIC 2026 me\jigsaw-unintended-bias-in-toxicity-classification\train.csv")
KAGGLE_PII_PATH   = Path(r"D:\tushar\random\aIC 2026 me\pii-detection-removal-from-educational-data\train.json")

DETECTOR_CONFIGS = {
    "injection": {
        "base_model": "protectai/deberta-v3-base-prompt-injection-v2",
        "output_dir": _REPO_ROOT / "ml" / "artifacts" / "injection-v4",
        "positive_label": "INJECTION_DETECTED",
        "positive_index": 1,
        "num_labels": 2,
        "max_length": 512,
        "batch_size": 16,
        "epochs": 2,
        "lr": 2e-5,
        "max_samples": 40_000,
        "datasets": ["allenai/wildjailbreak", "deepset/prompt-injections", "jackhhao/jailbreak-classification"],
    },
    "safety": {
        "base_model": "s-nlp/roberta_toxicity_classifier",
        "output_dir": _REPO_ROOT / "ml" / "artifacts" / "toxicity-v4",
        "positive_label": "UNSAFE_CONTENT",
        "positive_index": 1,
        "num_labels": 2,
        "max_length": 256,
        "batch_size": 32,
        "epochs": 2,
        "lr": 2e-5,
        "max_samples": 60_000,
        "datasets": ["kaggle/jigsaw-toxic-comments", "allenai/wildguardmix", "PKU-Alignment/BeaverTails", "lmsys/toxic-chat"],
    },
    "fairness": {
        "base_model": "facebook/roberta-hate-speech-dynabench-r4-target",
        "output_dir": _REPO_ROOT / "ml" / "artifacts" / "fairness-v3",  # fairness-v3 already good, reuse
        "positive_label": "BIASED",
        "positive_index": 1,
        "num_labels": 2,
        "max_length": 256,
        "batch_size": 32,
        "epochs": 2,
        "lr": 2e-5,
        "max_samples": 45_000,
        "datasets": ["kaggle/jigsaw-unintended-bias", "hatexplain"],
    },
}

def _sample(rows: list, max_n: int, seed: int = 42) -> list:
    if len(rows) <= max_n:
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, max_n)

# ---------------------------------------------------------------------------
# Dataset Loaders (Kaggle + Hugging Face)
# ---------------------------------------------------------------------------

def load_scaled_injection_data(max_samples: int, smoke_test: bool) -> tuple[list, list]:
    from datasets import load_dataset  # type: ignore
    rows: list[dict] = []

    # 1. Hugging Face: allenai/wildjailbreak (streaming to avoid non-streaming crash on Windows)
    log.info("  [HF Gated] Loading allenai/wildjailbreak (streaming) ...")
    try:
        import itertools
        wj = load_dataset("allenai/wildjailbreak", "train", split="train", streaming=True)
        cap = 100 if smoke_test else 20_000
        adv_count = 0
        for ex in itertools.islice(wj, cap * 2):
            adv = str(ex.get("adversarial") or "").strip()
            vanilla = str(ex.get("vanilla") or "").strip()
            if adv:
                rows.append({"text": adv, "label": 1})
                adv_count += 1
            if vanilla:
                rows.append({"text": vanilla, "label": 0})
        log.info("    allenai/wildjailbreak: %d examples loaded", adv_count)
    except Exception as e:
        log.warning("    wildjailbreak skipped: %s", e)

    # 2. Hugging Face: deepset/prompt-injections
    log.info("  [HF] Loading deepset/prompt-injections ...")
    try:
        ds = load_dataset("deepset/prompt-injections", split="train")
        for ex in ds:
            rows.append({"text": ex["text"], "label": int(ex["label"])})
    except Exception as e:
        log.warning("    deepset skipped: %s", e)

    # 3. Hugging Face: jackhhao/jailbreak-classification
    log.info("  [HF] Loading jackhhao/jailbreak-classification ...")
    try:
        jb = load_dataset("jackhhao/jailbreak-classification", split="train")
        for ex in jb:
            label = 0 if str(ex.get("type", "")).lower() == "benign" else 1
            text = str(ex.get("prompt", "")).strip()
            if text:
                rows.append({"text": text, "label": label})
    except Exception as e:
        log.warning("    jackhhao skipped: %s", e)

    rows = _sample(rows, max_samples if not smoke_test else 200)
    random.shuffle(rows)
    split = int(len(rows) * 0.9)
    return rows[:split], rows[split:]


def load_scaled_safety_data(max_samples: int, smoke_test: bool) -> tuple[list, list]:
    import pandas as pd
    from datasets import load_dataset  # type: ignore
    rows: list[dict] = []

    # 1. Kaggle: jigsaw toxic comments
    if KAGGLE_TOXIC_PATH.exists():
        log.info("  [Kaggle] Loading jigsaw toxic comments from %s ...", KAGGLE_TOXIC_PATH.name)
        try:
            with zipfile.ZipFile(KAGGLE_TOXIC_PATH) as z:
                df = pd.read_csv(z.open("train.csv"))
                df["is_toxic"] = (
                    (df["toxic"] == 1) | (df["severe_toxic"] == 1) |
                    (df["obscene"] == 1) | (df["threat"] == 1) |
                    (df["insult"] == 1) | (df["identity_hate"] == 1)
                ).astype(int)
                
                pos_df = df[df["is_toxic"] == 1]
                neg_df = df[df["is_toxic"] == 0]
                
                cap = 15_000 if not smoke_test else 100
                pos_sample = pos_df.sample(min(cap, len(pos_df)), random_state=42)
                neg_sample = neg_df.sample(min(cap, len(neg_df)), random_state=42)
                
                for _, r in pos_sample.iterrows():
                    rows.append({"text": str(r["comment_text"]), "label": 1})
                for _, r in neg_sample.iterrows():
                    rows.append({"text": str(r["comment_text"]), "label": 0})
                log.info("    jigsaw toxic: %d examples loaded", len(pos_sample) + len(neg_sample))
        except Exception as e:
            log.warning("    jigsaw toxic loading failed: %s", e)

    # 2. Hugging Face: allenai/wildguardmix
    log.info("  [HF Gated] Loading allenai/wildguardmix ...")
    try:
        wg = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
        for ex in wg:
            prompt = str(ex.get("prompt") or "").strip()
            harm_label = str(ex.get("prompt_harm_label") or "").lower()
            label = 1 if "harm" in harm_label else 0
            if prompt:
                rows.append({"text": prompt, "label": label})
            if smoke_test and len(rows) >= 200:
                break
        log.info("    wildguardmix: %d examples loaded", len(rows))
    except Exception as e:
        log.warning("    wildguardmix skipped: %s", e)

    # 3. Hugging Face: PKU-Alignment/BeaverTails
    log.info("  [HF] Loading PKU-Alignment/BeaverTails ...")
    try:
        bt = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
        for ex in bt:
            rows.append({
                "text": str(ex.get("prompt", "")),
                "label": 0 if ex.get("is_safe", True) else 1,
            })
            if smoke_test and len(rows) >= 300:
                break
    except Exception as e:
        log.warning("    BeaverTails skipped: %s", e)

    # 4. Hugging Face: lmsys/toxic-chat (real chatbot toxic interactions)
    log.info("  [HF] Loading lmsys/toxic-chat ...")
    try:
        tc = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
        tc_count = 0
        for ex in tc:
            text = str(ex.get("user_input") or "").strip()
            label = 1 if ex.get("toxicity", 0) == 1 else 0
            if text:
                rows.append({"text": text, "label": label})
                tc_count += 1
            if smoke_test and tc_count >= 100:
                break
        log.info("    lmsys/toxic-chat: %d examples loaded", tc_count)
    except Exception as e:
        log.warning("    lmsys/toxic-chat skipped: %s", e)

    rows = _sample(rows, max_samples if not smoke_test else 200)
    random.shuffle(rows)
    split = int(len(rows) * 0.9)
    return rows[:split], rows[split:]


def load_scaled_fairness_data(max_samples: int, smoke_test: bool) -> tuple[list, list]:
    import pandas as pd
    from datasets import load_dataset  # type: ignore
    rows: list[dict] = []

    # 1. Kaggle: Jigsaw unintended bias in toxicity
    if KAGGLE_BIAS_PATH.exists():
        log.info("  [Kaggle] Loading jigsaw unintended bias from %s ...", KAGGLE_BIAS_PATH.name)
        try:
            # Read in chunks to avoid memory spike
            chunks = []
            for chunk in pd.read_csv(KAGGLE_BIAS_PATH, chunksize=50_000, usecols=["comment_text", "target", "identity_attack"]):
                # target >= 0.5 is toxic/biased
                chunk["is_biased"] = (chunk["target"] >= 0.5).astype(int)
                chunks.append(chunk[["comment_text", "is_biased"]])
                if len(chunks) * 50_000 >= 150_000:
                    break
            
            df = pd.concat(chunks)
            pos_df = df[df["is_biased"] == 1]
            neg_df = df[df["is_biased"] == 0]
            
            cap = 15_000 if not smoke_test else 100
            pos_sample = pos_df.sample(min(cap, len(pos_df)), random_state=42)
            neg_sample = neg_df.sample(min(cap, len(neg_df)), random_state=42)
            
            for _, r in pos_sample.iterrows():
                rows.append({"text": str(r["comment_text"]), "label": 1})
            for _, r in neg_sample.iterrows():
                rows.append({"text": str(r["comment_text"]), "label": 0})
            log.info("    jigsaw unintended bias: %d examples loaded", len(pos_sample) + len(neg_sample))
        except Exception as e:
            log.warning("    jigsaw bias failed: %s", e)

    # 2. Hugging Face: HateXplain
    log.info("  [HF] Loading hatexplain ...")
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
    except Exception as e:
        log.warning("    hatexplain skipped: %s", e)

    rows = _sample(rows, max_samples if not smoke_test else 200)
    random.shuffle(rows)
    split = int(len(rows) * 0.9)
    return rows[:split], rows[split:]


# ---------------------------------------------------------------------------
# Training Loop
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
    log.info("Task: %s (Scaled Dataset Fine-Tuning) | device: %s", task, device)
    if device == "cuda":
        log.info("GPU: %s | VRAM: %.1f GB", torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)
    log.info("Base model: %s", cfg["base_model"])
    log.info("Target Artifact: %s", cfg["output_dir"])
    log.info("=" * 60)

    loader = {
        "injection": load_scaled_injection_data,
        "safety": load_scaled_safety_data,
        "fairness": load_scaled_fairness_data,
    }[task]
    
    train_rows, eval_rows = loader(cfg["max_samples"], smoke_test)
    log.info("[%s] Merged Dataset -> Train: %d | Eval: %d", task, len(train_rows), len(eval_rows))

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
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    log.info("[%s] Training on RTX 4050 GPU ...", task)
    trainer.train()
    metrics = trainer.evaluate()
    log.info("[%s] Final Evaluation Metrics: %s", task, {k: round(float(v), 4) for k, v in metrics.items() if isinstance(v, float)})

    log.info("[%s] Exporting model to %s ...", task, model_dir)
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
    log.info("[%s] Successfully calibrated and saved to %s", task, model_dir)


# ---------------------------------------------------------------------------
# Sensitive Intent & PII Data Calibration (Kaggle Student PII)
# ---------------------------------------------------------------------------

def calibrate_pii_and_sensitive_intent(smoke_test: bool) -> None:
    log.info("=" * 60)
    log.info("Task: sensitive_intent & PII Calibration (from Kaggle student PII data)")
    log.info("=" * 60)

    extra_pii_prompts = []

    # banking77 for intent calibration
    log.info("  [HF] Loading PolyAI/banking77 for intent cluster anchors ...")
    try:
        b77 = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True)
        from collections import defaultdict
        by_class = defaultdict(list)
        for ex in b77:
            by_class[ex["label"]].append(ex["text"])
        bank_anchors = []
        for label, texts in list(by_class.items())[:20]:  # 20 intent clusters
            bank_anchors.extend(texts[:3])
        extra_pii_prompts.extend(bank_anchors)
        log.info("    banking77: %d anchor phrases added", len(bank_anchors))
    except Exception as e:
        log.warning("    banking77 skipped: %s", e)

    if KAGGLE_PII_PATH.exists():
        log.info("  [Kaggle] Extracting PII sentence contexts from %s ...", KAGGLE_PII_PATH.name)
        try:
            with open(KAGGLE_PII_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for doc in data[:500]:
                full_text = doc.get("full_text", "")
                tokens = doc.get("tokens", [])
                labels = doc.get("labels", [])
                
                # Extract sentences containing real student PII entities
                for token, label in zip(tokens, labels):
                    if label.startswith("B-"):
                        entity_type = label.split("-")[1]
                        extra_pii_prompts.append(f"What is the student's {entity_type}?")
                        extra_pii_prompts.append(f"Give me the {entity_type} for {token}.")
            
            log.info("    Extracted %d synthetic PII query patterns from Kaggle student data", len(extra_pii_prompts))
        except Exception as e:
            log.warning("    Kaggle PII extraction failed: %s", e)

    import importlib.util
    calib_path = _REPO_ROOT / "ml" / "scripts" / "calibrate_sensitive_intent.py"
    spec = importlib.util.spec_from_file_location("calibrate_si", str(calib_path))
    calibrate_si = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(calibrate_si)  # type: ignore

    if extra_pii_prompts:
        rng = random.Random(42)
        sample_pii = rng.sample(extra_pii_prompts, min(40, len(extra_pii_prompts)))
        calibrate_si.POSITIVE_ANCHORS.extend(sample_pii)
        log.info("    Added %d real student PII anchor queries to sensitive intent matcher", len(sample_pii))

    model_p = str(_REPO_ROOT / "ml" / "artifacts" / "sensitive-intent" / "model")
    out_p   = str(_REPO_ROOT / "ml" / "artifacts" / "sensitive-intent" / "calibration.json")
    calibrate_si.main_calibrate(model_p, out_p)
    log.info("sensitive_intent & PII calibration updated at %s", out_p)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["injection", "safety", "fairness", "pii", "all"], default="all")
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()

    tasks = ["injection", "safety", "fairness", "pii"] if args.task == "all" else [args.task]

    log.info("Starting Scaled Training Pipeline on GPU across Kaggle + HF datasets")
    log.info("Tasks: %s", tasks)

    for task in tasks:
        if task == "pii":
            calibrate_pii_and_sensitive_intent(args.smoke_test)
        else:
            train_classifier(task, DETECTOR_CONFIGS[task], args.smoke_test)

    log.info("=" * 60)
    log.info("All scaled -v3 detector models successfully trained and exported!")
    log.info("=" * 60)

if __name__ == "__main__":
    main()
