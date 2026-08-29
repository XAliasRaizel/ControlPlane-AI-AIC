"""ControlPlane.ai RLHF — DPO-format JSONL exporter.

Reads all preference pairs from the active storage backend, applies the
standard filter pipeline (``export.filters.filter_pairs``), reshapes
each surviving pair into the ``{"prompt", "chosen", "rejected"}`` format
that ``trl.DPOTrainer`` expects, and writes to a timestamped output file.

The output file is **never overwritten** — a new file is created on each
call.  This preserves the full history of exports even if the source data
changes.

Usage
-----
    from rlhf.export.dpo_export import export_for_dpo
    from rlhf.config import Category

    output_path = export_for_dpo(category=Category.HR)
    print(f"Exported to: {output_path}")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from rlhf.config import Category, EXPORTS_DIR, STORAGE_BACKEND
from rlhf.export.filters import filter_pairs
from rlhf.schema import PreferencePair

logger = logging.getLogger(__name__)


def _get_store():
    """Return the active storage module based on ``config.STORAGE_BACKEND``.

    Returns:
        Either ``rlhf.storage.json_store`` or ``rlhf.storage.sqlite_store``.

    Raises:
        ValueError: If ``STORAGE_BACKEND`` is set to an unrecognised value.
    """
    if STORAGE_BACKEND == "json":
        from rlhf.storage import json_store
        return json_store
    elif STORAGE_BACKEND == "sqlite":
        from rlhf.storage import sqlite_store
        return sqlite_store
    else:
        raise ValueError(
            f"[RLHF] Unknown STORAGE_BACKEND: {STORAGE_BACKEND!r}. "
            "Set RLHF_STORAGE_BACKEND to 'json' or 'sqlite'."
        )


def _reshape_for_dpo(pair: PreferencePair) -> dict:
    """Reshape a labelled pair into the trl.DPOTrainer input format.

    Args:
        pair: A labelled ``PreferencePair`` with ``chosen`` in ``{"a", "b"}``.

    Returns:
        A dict with keys ``"prompt"``, ``"chosen"``, ``"rejected"``.
    """
    if pair.chosen == "a":
        chosen_text   = pair.response_a.text
        rejected_text = pair.response_b.text
    else:
        chosen_text   = pair.response_b.text
        rejected_text = pair.response_a.text

    return {
        "prompt": pair.prompt,
        "chosen": chosen_text,
        "rejected": rejected_text,
    }


def export_for_dpo(
    category: Optional[Category] = None,
    output_dir: Optional[str] = None,
    store_path: Optional[Path] = None,
) -> str:
    """Export labelled preference pairs to a DPO-format JSONL file.

    Reads all pairs from the active backend (controlled by
    ``config.STORAGE_BACKEND``), applies ``filter_pairs``, reshapes each
    surviving pair into ``{"prompt", "chosen", "rejected"}``, and writes to
    a new timestamped file.  Prints a summary to stdout.

    Args:
        category: When provided, export only pairs with this category.  The
            category name is embedded in the output filename.
        output_dir: Override the default exports directory.  Useful in tests.
        store_path: Override the store's default data file path (useful in
            tests — pass the same temp path used with ``json_store.write_pair``).

    Returns:
        The absolute path of the file that was written (as a string).

    Raises:
        ValueError: If ``STORAGE_BACKEND`` is not recognised.
    """
    store = _get_store()
    if store_path is not None:
        all_pairs = store.read_all_pairs(path=store_path)
    else:
        all_pairs = store.read_all_pairs()
    total_read = len(all_pairs)

    filtered = filter_pairs(all_pairs, category=category)
    total_filtered = len(filtered)

    # Build output path — never overwrite.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cat_tag = category.value if category else "ALL"
    filename = f"dpo_{cat_tag}_{ts}.jsonl"

    out_dir = Path(output_dir) if output_dir else EXPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    with out_path.open("w", encoding="utf-8") as fh:
        for pair in filtered:
            fh.write(json.dumps(_reshape_for_dpo(pair)) + "\n")

    # --- Summary ---
    summary_lines = [
        "",
        "=" * 60,
        "  DPO Export Summary",
        "=" * 60,
        f"  Storage backend  : {STORAGE_BACKEND}",
        f"  Category filter  : {cat_tag}",
        f"  Pairs read       : {total_read}",
        f"  Pairs after filter: {total_filtered}",
        f"  Output file      : {out_path}",
        "=" * 60,
        "",
    ]
    summary = "\n".join(summary_lines)
    print(summary)           # noqa: T201
    logger.info("[RLHF/dpo_export] exported %d pairs to %s", total_filtered, out_path)

    return str(out_path)
