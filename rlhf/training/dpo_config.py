"""ControlPlane.ai RLHF — Per-category DPO & LoRA hyperparameter config.

Design
------
* A ``_DEFAULTS`` dict defines the baseline hyperparameters.
* A ``_CATEGORY_OVERRIDES`` dict maps each ``Category`` to a partial dict of
  overrides.  Any key present in the override replaces the default; missing
  keys fall through to the default.
* ``get_dpo_config(category)`` merges the two and returns the final config.

This keeps category-specific tuning (e.g. a lower beta for HR to allow
looser alignment, a different base checkpoint for FINANCIAL) visible and
easy to extend without touching the training code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rlhf.config import Category

# ---------------------------------------------------------------------------
# Default hyperparameters — sensible small-model hackathon values
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    # ---- DPO core ----
    "beta": 0.1,                         # KL-divergence penalty weight
    "learning_rate": 5e-6,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,    # effective batch = 2 * 4 = 8
    "max_length": 1024,                  # max sequence length (prompt + response)
    "max_prompt_length": 512,
    # ---- Base model ----
    "base_model_name_or_path": "meta-llama/Llama-3.2-3B-Instruct",
    # ---- Training infrastructure ----
    "warmup_ratio": 0.1,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "lr_scheduler_type": "cosine",
    "report_to": [],                     # disable wandb/tensorboard by default
    "seed": 42,
    # ---- LoRA / PEFT ----
    "lora": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    },
}

# ---------------------------------------------------------------------------
# Per-category overrides
# ---------------------------------------------------------------------------

_CATEGORY_OVERRIDES: dict[Category, dict[str, Any]] = {
    Category.HR: {
        # HR responses are typically shorter; relax KL to allow more
        # expressive rewrites and reduce over-redaction.
        "beta": 0.07,
        "max_length": 768,
        "max_prompt_length": 384,
    },
    Category.FINANCIAL: {
        # Financial responses require stricter adherence to policy citations.
        # Higher beta keeps the fine-tuned model closer to the reference.
        "beta": 0.15,
        "learning_rate": 3e-6,
        # Optionally swap in a finance-specialised checkpoint.
        # Uncomment and set when one is available:
        # "base_model_name_or_path": "path/to/finance-llm-checkpoint",
    },
    Category.GENERAL: {
        # No overrides — use defaults.
    },
    Category.UNSPECIFIED: {
        # No overrides — use defaults.
    },
}


# ---------------------------------------------------------------------------
# Public config dataclass
# ---------------------------------------------------------------------------

@dataclass
class DPORunConfig:
    """Merged hyperparameter config for one DPO training run.

    All fields are readable as plain attributes.  The ``lora`` field is a
    dict ready to be unpacked into ``peft.LoraConfig(**config.lora)``.

    Attributes:
        category: The ``Category`` this config was built for.
        beta: DPO KL-divergence penalty.
        learning_rate: AdamW learning rate.
        num_train_epochs: Number of training epochs.
        per_device_train_batch_size: Batch size per GPU/CPU device.
        gradient_accumulation_steps: Steps before a gradient update.
        max_length: Maximum total sequence length (prompt + response).
        max_prompt_length: Maximum prompt-only length.
        base_model_name_or_path: HuggingFace model ID or local path.
        warmup_ratio: Fraction of steps used for LR warm-up.
        logging_steps: Log every N steps.
        save_strategy: Checkpoint save frequency (``"epoch"`` | ``"steps"``).
        lr_scheduler_type: LR decay schedule type.
        report_to: List of experiment trackers (empty list = disabled).
        seed: Random seed for reproducibility.
        lora: LoRA hyperparameter dict for ``peft.LoraConfig``.
    """

    category: Category
    beta: float
    learning_rate: float
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_length: int
    max_prompt_length: int
    base_model_name_or_path: str
    warmup_ratio: float
    logging_steps: int
    save_strategy: str
    lr_scheduler_type: str
    report_to: list = field(default_factory=list)
    seed: int = 42
    lora: dict = field(default_factory=dict)


def get_dpo_config(category: Category) -> DPORunConfig:
    """Return the merged DPO hyperparameter config for a given category.

    Starts from ``_DEFAULTS`` and applies any category-specific overrides
    from ``_CATEGORY_OVERRIDES``.  Falls back to defaults for any category
    not explicitly configured.

    Args:
        category: The ``Category`` enum member to build a config for.

    Returns:
        A ``DPORunConfig`` dataclass with all fields populated.
    """
    merged: dict[str, Any] = {**_DEFAULTS}
    overrides = _CATEGORY_OVERRIDES.get(category, {})
    # Apply overrides; LoRA is merged at the sub-dict level so partial
    # LoRA overrides are possible without respecifying every field.
    for key, value in overrides.items():
        if key == "lora" and isinstance(value, dict):
            merged["lora"] = {**merged.get("lora", {}), **value}
        else:
            merged[key] = value

    return DPORunConfig(
        category=category,
        beta=merged["beta"],
        learning_rate=merged["learning_rate"],
        num_train_epochs=merged["num_train_epochs"],
        per_device_train_batch_size=merged["per_device_train_batch_size"],
        gradient_accumulation_steps=merged["gradient_accumulation_steps"],
        max_length=merged["max_length"],
        max_prompt_length=merged["max_prompt_length"],
        base_model_name_or_path=merged["base_model_name_or_path"],
        warmup_ratio=merged["warmup_ratio"],
        logging_steps=merged["logging_steps"],
        save_strategy=merged["save_strategy"],
        lr_scheduler_type=merged["lr_scheduler_type"],
        report_to=merged.get("report_to", []),
        seed=merged["seed"],
        lora=merged["lora"],
    )
