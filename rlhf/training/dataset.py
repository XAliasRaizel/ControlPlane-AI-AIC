"""ControlPlane.ai RLHF — DPO dataset loader.

Loads an exported DPO JSONL file and wraps it as a Hugging Face
``datasets.Dataset`` with ``prompt`` / ``chosen`` / ``rejected`` columns,
which is the exact format expected by ``trl.DPOTrainer``.

Tokenization note
-----------------
This module does NOT tokenize the dataset.  ``trl.DPOTrainer`` handles
tokenization internally via the tokenizer passed to the trainer.  This
loader's job is:
  (a) parse the JSONL file,
  (b) validate that the required columns are present, and
  (c) return a ``Dataset`` object.

Adding manual tokenization here would likely conflict with DPOTrainer's
internal preprocessing and is therefore intentionally omitted.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_dpo_dataset(export_path: str, tokenizer=None):  # noqa: ANN001
    """Load a DPO-format JSONL file and return a Hugging Face ``Dataset``.

    The ``tokenizer`` argument is accepted but **not used** in this function
    (see the tokenization note in the module docstring).  It is kept as a
    parameter so the function signature is future-proof and consistent with
    the rest of the training pipeline.

    Args:
        export_path: Path to the DPO JSONL file produced by
            ``rlhf.export.dpo_export.export_for_dpo``.
        tokenizer: Optional tokenizer (accepted but unused — see note above).

    Returns:
        A ``datasets.Dataset`` with columns ``["prompt", "chosen", "rejected"]``.

    Raises:
        FileNotFoundError: If ``export_path`` does not exist.
        ValueError: If any line is missing a required column.
        ImportError: If the ``datasets`` package is not installed
            (``pip install datasets`` or ``pip install -r rlhf/training/requirements-training.txt``).
    """
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for DPO training. "
            "Install it with: pip install datasets"
        ) from exc

    path = Path(export_path)
    if not path.exists():
        raise FileNotFoundError(f"[RLHF/dataset] export file not found: {path}")

    required_columns = {"prompt", "chosen", "rejected"}
    records: list[dict] = []

    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("[RLHF/dataset] skipping malformed line %d: %s", lineno, exc)
                continue
            missing = required_columns - record.keys()
            if missing:
                raise ValueError(
                    f"[RLHF/dataset] line {lineno} is missing columns: {missing}"
                )
            records.append({col: record[col] for col in required_columns})

    if not records:
        raise ValueError(
            f"[RLHF/dataset] no valid records found in {path}. "
            "Run export_for_dpo first and ensure pairs are labelled."
        )

    dataset = Dataset.from_list(records)
    logger.info(
        "[RLHF/dataset] loaded %d records from %s", len(dataset), path
    )
    return dataset
