"""ControlPlane.ai RLHF — Active JSON (JSONL) storage backend.

This is the **currently active** storage backend.  All reads and writes
in the normal code path go through this file.

Design
------
The JSONL file is **append-only** — we never overwrite or re-write lines.

* ``write_pair`` appends a line tagged ``"record_type": "initial"``.
* ``update_label`` appends a *new* line tagged ``"record_type": "label_update"``
  with the same ``pair_id`` and only the updated label fields.  This mirrors
  the approach used by immutable audit ledgers (like the Merkle ledger in
  ``backend/audit_integrity/``) — the log grows forward, never backward.
* ``read_all_pairs`` reads every line, groups by ``pair_id``, and merges
  any ``"label_update"`` records onto their matching ``"initial"`` record
  (latest update wins).  The reconciled objects are returned as full
  ``PreferencePair`` instances.

Thread safety
-------------
File I/O is not locked; for a hackathon single-process setup this is fine.
Before making this multi-process or multi-host, add file locking or switch
to SQLite (which handles concurrent writers gracefully).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rlhf.config import Category, RAW_JSONL_PATH
from rlhf.schema import PreferencePair

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_file(path: Path) -> None:
    """Create the file and all parent directories if they do not exist.

    Args:
        path: The JSONL file path to ensure.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()


def _serialize(pair: PreferencePair) -> str:
    """Serialise a ``PreferencePair`` to a compact JSON string.

    Args:
        pair: The pair to serialise.

    Returns:
        A single JSON object string (no trailing newline — callers add that).
    """
    return pair.model_dump_json()


def _deserialize(line: str) -> dict:
    """Parse one JSONL line into a raw dict.

    Args:
        line: A single JSON line from the JSONL file.

    Returns:
        Parsed dict.
    """
    return json.loads(line)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def _ensure_trailing_newline(path: Path) -> None:
    """Ensure the file ends with a newline before appending a new record."""
    try:
        if path.exists() and path.stat().st_size > 0:
            with path.open("rb+") as fh:
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) != b"\n":
                    fh.write(b"\n")
    except Exception:
        pass


def write_pair(pair: PreferencePair, path: Path = RAW_JSONL_PATH) -> None:
    """Append a new, unlabelled pair to the JSONL file.

    This is the entry-point for generator output.  The pair is written with
    ``record_type = "initial"`` so that ``read_all_pairs`` can distinguish
    initial records from subsequent label updates.

    Args:
        pair: A ``PreferencePair`` with ``record_type == "initial"`` and
            ``chosen == None`` (i.e. not yet labelled).
        path: Override the default JSONL path (useful in tests).
    """
    _ensure_file(path)
    _ensure_trailing_newline(path)
    record = pair.model_copy(update={"record_type": "initial"})
    with path.open("a", encoding="utf-8") as fh:
        fh.write(_serialize(record) + "\n")
    logger.debug("[RLHF/json_store] wrote initial pair %s", pair.pair_id)


def update_label(
    pair_id: str,
    chosen: str,
    labeled_by: str,
    judge_metadata: Optional[dict] = None,
    path: Path = RAW_JSONL_PATH,
) -> None:
    """Append a label-update record for an existing pair.

    Because the JSONL is append-only we do NOT overwrite the original line.
    Instead we append a new ``"label_update"`` record with the same
    ``pair_id``; ``read_all_pairs`` will merge the two.

    A ``"label_update"`` record for a ``pair_id`` that does not yet exist in
    the file is still written — this is intentional (e.g. for importing
    external labels), but a warning is logged.

    Args:
        pair_id: UUID of the pair being labelled.
        chosen: ``"a"``, ``"b"``, or ``"tie"``.
        labeled_by: ``"human"`` or ``"llm_judge"``.
        judge_metadata: Optional dict with raw judge votes / position info.
        path: Override the default JSONL path (useful in tests).

    Raises:
        ValueError: If ``chosen`` or ``labeled_by`` are invalid values.
    """
    if chosen not in ("a", "b", "tie"):
        raise ValueError(f"[RLHF/json_store] invalid 'chosen' value: {chosen!r}")
    if labeled_by not in ("human", "llm_judge"):
        raise ValueError(f"[RLHF/json_store] invalid 'labeled_by' value: {labeled_by!r}")

    _ensure_file(path)
    _ensure_trailing_newline(path)
    update_record = {
        "pair_id": pair_id,
        "record_type": "label_update",
        "chosen": chosen,
        "labeled_by": labeled_by,
        "labeled_at": datetime.now(timezone.utc).isoformat(),
        "judge_metadata": judge_metadata or {},
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(update_record) + "\n")
    logger.debug("[RLHF/json_store] wrote label_update for pair %s", pair_id)


def read_all_pairs(path: Path = RAW_JSONL_PATH) -> list[PreferencePair]:
    """Read and reconcile all pairs from the JSONL file.

    Reconciliation strategy:
    1. Parse every line.
    2. Group by ``pair_id``.
    3. For each group, start from the ``"initial"`` record and apply any
       ``"label_update"`` records in the order they appear (last one wins).
    4. Return fully-resolved ``PreferencePair`` objects.

    Malformed lines are skipped with a warning rather than crashing the
    reader — append-only logs can accumulate partial writes on crash.

    Args:
        path: Override the default JSONL path (useful in tests).

    Returns:
        List of resolved ``PreferencePair`` objects, one per unique ``pair_id``.
    """
    if not path.exists():
        return []

    raw_lines: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_lines.append(_deserialize(line))
            except json.JSONDecodeError as exc:
                if "}{" in line:
                    for sub in line.replace("}{", "}\n{").split("\n"):
                        sub = sub.strip()
                        if sub:
                            try:
                                raw_lines.append(_deserialize(sub))
                            except Exception:
                                pass
                else:
                    logger.warning(
                        "[RLHF/json_store] skipping malformed line %d: %s", lineno, exc
                    )

    # Group: pair_id -> {"initial": dict, "updates": [dict, ...]}
    groups: dict[str, dict] = {}
    for record in raw_lines:
        pid = record.get("pair_id")
        if not pid:
            continue
        rtype = record.get("record_type", "initial")
        if pid not in groups:
            groups[pid] = {"initial": None, "updates": []}
        if rtype == "label_update":
            groups[pid]["updates"].append(record)
        else:
            groups[pid]["initial"] = record

    pairs: list[PreferencePair] = []
    for pid, group in groups.items():
        base = group["initial"]
        if base is None:
            logger.warning(
                "[RLHF/json_store] pair_id %s has label_update(s) but no initial record; skipping",
                pid,
            )
            continue
        # Apply updates in order — last update wins
        for upd in group["updates"]:
            for field in ("chosen", "labeled_by", "labeled_at", "judge_metadata"):
                if field in upd and upd[field] is not None:
                    base[field] = upd[field]
        # Ensure record_type is reset so the resolved object looks clean
        base["record_type"] = "initial"
        try:
            pairs.append(PreferencePair(**base))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[RLHF/json_store] failed to deserialise pair_id %s: %s", pid, exc
            )

    return pairs


def query(
    category: Optional[Category] = None,
    labeled_only: bool = False,
    path: Path = RAW_JSONL_PATH,
) -> list[PreferencePair]:
    """Convenience filter over ``read_all_pairs``.

    Args:
        category: When provided, return only pairs with this category.
        labeled_only: When True, return only pairs where ``chosen`` is not None.
        path: Override the default JSONL path (useful in tests).

    Returns:
        Filtered list of ``PreferencePair`` objects.
    """
    pairs = read_all_pairs(path=path)
    if labeled_only:
        pairs = [p for p in pairs if p.chosen is not None]
    if category is not None:
        pairs = [p for p in pairs if p.category == category.value or p.category == category]
    return pairs
