"""ControlPlane.ai RLHF — DPO fine-tuning entry point.

Loads a base model, applies LoRA via PEFT, loads the DPO dataset from an
export file, instantiates ``trl.DPOTrainer``, runs training, and saves
the adapter checkpoint.

This script is intended to be run manually / on-demand during the hackathon:

    python -m rlhf.training.train \\
        --category HR \\
        --export-path rlhf/data/exports/dpo_HR_20260830_120000.jsonl \\
        --base-model meta-llama/Llama-3.2-3B-Instruct

It can also be imported and called programmatically:

    from rlhf.training.train import run_dpo_training
    checkpoint = run_dpo_training(Category.HR, export_path, base_model)

Dependencies (not in requirements.txt — install separately):
    pip install trl peft transformers accelerate bitsandbytes datasets torch
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from rlhf.config import Category, CHECKPOINTS_DIR
from rlhf.training.dataset import load_dpo_dataset
from rlhf.training.dpo_config import get_dpo_config

logger = logging.getLogger(__name__)


def run_dpo_training(
    category: Category,
    export_path: str,
    base_model_name_or_path: str,
) -> str:
    """Run a full DPO fine-tuning job and save the adapter checkpoint.

    Steps
    -----
    1. Load the per-category ``DPORunConfig`` (beta, lr, LoRA config, etc.).
    2. Load the base model and tokenizer.
    3. Apply LoRA via ``peft.get_peft_model``.
    4. Load the DPO dataset via ``rlhf.training.dataset.load_dpo_dataset``.
    5. Instantiate ``trl.DPOTrainer`` and call ``.train()``.
    6. Save the adapter to ``data/checkpoints/<category>_<timestamp>/``.

    Args:
        category: The ``Category`` enum member for this training run.  Used
            to select hyperparameters and name the checkpoint directory.
        export_path: Path to the DPO JSONL file produced by
            ``rlhf.export.dpo_export.export_for_dpo``.
        base_model_name_or_path: HuggingFace model hub ID or local path to
            the base model checkpoint (e.g. ``"meta-llama/Llama-3.2-3B-Instruct"``).
            Overrides the value in ``dpo_config.py`` when provided.

    Returns:
        Absolute path (string) to the saved adapter checkpoint directory.

    Raises:
        ImportError: If any of ``trl``, ``peft``, ``transformers``, or
            ``torch`` are not installed.
        FileNotFoundError: If ``export_path`` does not exist.
    """
    # -----------------------------------------------------------------------
    # Import ML dependencies lazily so the rest of the rlhf package remains
    # importable without them (mirrors the pattern in ml/train_prompt_injection.py).
    # -----------------------------------------------------------------------
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOTrainer
        from trl import DPOConfig as TrlDPOConfig
    except ImportError as exc:
        raise ImportError(
            "DPO training requires trl, peft, transformers, and torch. "
            "Install with: pip install trl peft transformers accelerate bitsandbytes torch"
        ) from exc

    config = get_dpo_config(category)

    # Use the caller-provided base model if it differs from the config default.
    model_path = base_model_name_or_path or config.base_model_name_or_path

    logger.info("[RLHF/train] loading base model: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # DPOTrainer requires a pad token; use eos if not already set.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Apply LoRA.
    lora_cfg = LoraConfig(
        r=config.lora["r"],
        lora_alpha=config.lora["lora_alpha"],
        lora_dropout=config.lora["lora_dropout"],
        bias=config.lora["bias"],
        task_type=config.lora["task_type"],
        target_modules=config.lora["target_modules"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Load dataset.
    dataset = load_dpo_dataset(export_path, tokenizer=tokenizer)

    # Build checkpoint output path.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cat_str = category.value if isinstance(category, Category) else str(category)
    checkpoint_dir = CHECKPOINTS_DIR / f"{cat_str}_{ts}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Configure trl DPO training arguments.
    training_args = TrlDPOConfig(
        output_dir=str(checkpoint_dir),
        beta=config.beta,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        lr_scheduler_type=config.lr_scheduler_type,
        report_to=config.report_to,
        seed=config.seed,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    logger.info("[RLHF/train] starting DPO training (category=%s)", cat_str)
    trainer.train()

    trainer.save_model(str(checkpoint_dir / "adapter"))
    tokenizer.save_pretrained(str(checkpoint_dir / "adapter"))
    logger.info("[RLHF/train] checkpoint saved to %s", checkpoint_dir)

    return str(checkpoint_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DPO fine-tuning for a specific ControlPlane.ai category."
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        choices=[c.value for c in Category],
        help="Category to train (must match a Category enum value).",
    )
    parser.add_argument(
        "--export-path",
        type=str,
        required=True,
        help="Path to the DPO JSONL file produced by export_for_dpo.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="HuggingFace model ID or local path (overrides dpo_config default).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    category = Category(args.category)
    base_model = args.base_model or get_dpo_config(category).base_model_name_or_path
    out = run_dpo_training(
        category=category,
        export_path=args.export_path,
        base_model_name_or_path=base_model,
    )
    print(f"\n[RLHF] Training complete. Checkpoint at: {out}")  # noqa: T201
