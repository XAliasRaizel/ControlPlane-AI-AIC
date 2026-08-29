# =============================================================================
# NOT ACTIVE YET
#
# This file mirrors json_store.py's complete public interface
# (write_pair / update_label / read_all_pairs / query) against a SQLite
# database.
#
# To activate:
#   1. Set RLHF_STORAGE_BACKEND="sqlite" in your environment (or change
#      STORAGE_BACKEND in rlhf/config.py).
#   2. Call init_db() once (e.g. from a migration script or startup hook)
#      to create the ``preference_pairs`` table.
#   3. Update rlhf/export/dpo_export.py and any other callers that read
#      STORAGE_BACKEND to import from here instead of json_store.
#
# Nothing in the active code path imports from this file.  It is ready
# to be swapped in without changing any other module's interface.
# =============================================================================
"""ControlPlane.ai RLHF — SQLite storage backend (NOT ACTIVE YET).

Drop-in replacement for ``json_store.py``.  Switching from JSON to SQLite
requires only flipping ``STORAGE_BACKEND`` in ``rlhf/config.py`` from
``"json"`` to ``"sqlite"``.  The table schema uses ``pair_id`` as the
primary key, enabling real ``UPDATE`` statements for label writing instead
of the append-and-reconcile trick used by the JSONL backend.

See the ``# NOT ACTIVE YET`` block at the top of this file.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rlhf.config import Category, SQLITE_DB_PATH
from rlhf.schema import PreferencePair

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS preference_pairs (
    pair_id                    TEXT PRIMARY KEY,
    session_id                 TEXT,
    prompt                     TEXT NOT NULL,
    -- response_a fields
    response_a_text            TEXT NOT NULL DEFAULT '',
    response_a_model_name      TEXT NOT NULL,
    response_a_version         TEXT NOT NULL DEFAULT 'unknown',
    response_a_hyperparameters TEXT NOT NULL DEFAULT '{}',
    response_a_is_error        INTEGER NOT NULL DEFAULT 0,
    response_a_error_message   TEXT,
    -- response_b fields
    response_b_text            TEXT NOT NULL DEFAULT '',
    response_b_model_name      TEXT NOT NULL,
    response_b_version         TEXT NOT NULL DEFAULT 'unknown',
    response_b_hyperparameters TEXT NOT NULL DEFAULT '{}',
    response_b_is_error        INTEGER NOT NULL DEFAULT 0,
    response_b_error_message   TEXT,
    -- label fields
    category                   TEXT NOT NULL DEFAULT 'UNSPECIFIED',
    chosen                     TEXT,           -- 'a' | 'b' | 'tie' | NULL
    labeled_by                 TEXT,           -- 'human' | 'llm_judge' | NULL
    judge_metadata             TEXT,           -- JSON blob
    -- timestamps
    created_at                 TEXT NOT NULL,
    labeled_at                 TEXT,
    source_pipeline            TEXT NOT NULL DEFAULT 'unknown'
);
"""


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def init_db(path: Path = SQLITE_DB_PATH) -> None:
    """Create the ``preference_pairs`` table if it does not already exist.

    This is intentionally *not* called automatically on import — invoke it
    manually from a migration script or startup hook when activating the
    SQLite backend.

    Args:
        path: Override the default SQLite database path (useful in tests).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as conn:
        conn.execute(_CREATE_TABLE_SQL)
        conn.commit()
    logger.info("[RLHF/sqlite_store] database initialised at %s", path)


def _get_conn(path: Path) -> sqlite3.Connection:
    """Open and return a SQLite connection with row_factory set.

    Args:
        path: Database file path.

    Returns:
        An open ``sqlite3.Connection``.
    """
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Public interface — mirrors json_store.py exactly
# ---------------------------------------------------------------------------

def write_pair(pair: PreferencePair, path: Path = SQLITE_DB_PATH) -> None:
    """Insert a new, unlabelled pair into the database.

    Uses ``INSERT OR IGNORE`` so calling ``write_pair`` twice with the same
    ``pair_id`` is idempotent (unlike the JSONL backend which would append
    a duplicate).

    Args:
        pair: A ``PreferencePair`` with ``chosen == None`` (not yet labelled).
        path: Override the default database path (useful in tests).
    """
    sql = """
    INSERT OR IGNORE INTO preference_pairs (
        pair_id, session_id, prompt,
        response_a_text, response_a_model_name, response_a_version,
        response_a_hyperparameters, response_a_is_error, response_a_error_message,
        response_b_text, response_b_model_name, response_b_version,
        response_b_hyperparameters, response_b_is_error, response_b_error_message,
        category, chosen, labeled_by, judge_metadata,
        created_at, labeled_at, source_pipeline
    ) VALUES (
        ?,?,?,  ?,?,?,?,?,?,  ?,?,?,?,?,?,  ?,?,?,?,  ?,?,?
    )
    """
    a = pair.response_a
    b = pair.response_b
    with _get_conn(path) as conn:
        conn.execute(sql, (
            pair.pair_id, pair.session_id, pair.prompt,
            a.text, a.model_name, a.model_version_or_checkpoint,
            json.dumps(a.hyperparameters), int(a.is_error), a.error_message,
            b.text, b.model_name, b.model_version_or_checkpoint,
            json.dumps(b.hyperparameters), int(b.is_error), b.error_message,
            pair.category if isinstance(pair.category, str) else pair.category.value,
            pair.chosen, pair.labeled_by,
            json.dumps(pair.judge_metadata) if pair.judge_metadata else None,
            pair.created_at.isoformat(),
            pair.labeled_at.isoformat() if pair.labeled_at else None,
            pair.source_pipeline,
        ))
        conn.commit()
    logger.debug("[RLHF/sqlite_store] inserted pair %s", pair.pair_id)


def update_label(
    pair_id: str,
    chosen: str,
    labeled_by: str,
    judge_metadata: Optional[dict] = None,
    path: Path = SQLITE_DB_PATH,
) -> None:
    """Update the label fields of an existing pair using a real SQL UPDATE.

    Unlike the JSONL backend, this is a true in-place update — no append-
    and-reconcile needed.

    Args:
        pair_id: UUID of the pair being labelled.
        chosen: ``"a"``, ``"b"``, or ``"tie"``.
        labeled_by: ``"human"`` or ``"llm_judge"``.
        judge_metadata: Optional dict with raw judge votes / position info.
        path: Override the default database path (useful in tests).

    Raises:
        ValueError: If ``chosen`` or ``labeled_by`` are invalid.
        RuntimeError: If no row with ``pair_id`` exists in the database.
    """
    if chosen not in ("a", "b", "tie"):
        raise ValueError(f"[RLHF/sqlite_store] invalid 'chosen': {chosen!r}")
    if labeled_by not in ("human", "llm_judge"):
        raise ValueError(f"[RLHF/sqlite_store] invalid 'labeled_by': {labeled_by!r}")

    sql = """
    UPDATE preference_pairs
    SET chosen=?, labeled_by=?, judge_metadata=?, labeled_at=?
    WHERE pair_id=?
    """
    with _get_conn(path) as conn:
        cursor = conn.execute(sql, (
            chosen, labeled_by,
            json.dumps(judge_metadata) if judge_metadata else None,
            datetime.now(timezone.utc).isoformat(),
            pair_id,
        ))
        conn.commit()
        if cursor.rowcount == 0:
            raise RuntimeError(
                f"[RLHF/sqlite_store] update_label: no row found for pair_id={pair_id!r}"
            )
    logger.debug("[RLHF/sqlite_store] labelled pair %s as %s", pair_id, chosen)


def read_all_pairs(path: Path = SQLITE_DB_PATH) -> list[PreferencePair]:
    """Read all rows from the database and return as ``PreferencePair`` objects.

    Args:
        path: Override the default database path (useful in tests).

    Returns:
        List of ``PreferencePair`` objects.
    """
    from rlhf.schema import ModelResponse  # local import avoids circular dep

    if not path.exists():
        logger.warning("[RLHF/sqlite_store] database not found at %s", path)
        return []

    with _get_conn(path) as conn:
        rows = conn.execute("SELECT * FROM preference_pairs").fetchall()

    pairs: list[PreferencePair] = []
    for row in rows:
        try:
            a = ModelResponse(
                text=row["response_a_text"],
                model_name=row["response_a_model_name"],
                model_version_or_checkpoint=row["response_a_version"],
                hyperparameters=json.loads(row["response_a_hyperparameters"] or "{}"),
                is_error=bool(row["response_a_is_error"]),
                error_message=row["response_a_error_message"],
            )
            b = ModelResponse(
                text=row["response_b_text"],
                model_name=row["response_b_model_name"],
                model_version_or_checkpoint=row["response_b_version"],
                hyperparameters=json.loads(row["response_b_hyperparameters"] or "{}"),
                is_error=bool(row["response_b_is_error"]),
                error_message=row["response_b_error_message"],
            )
            pairs.append(PreferencePair(
                pair_id=row["pair_id"],
                session_id=row["session_id"],
                prompt=row["prompt"],
                response_a=a,
                response_b=b,
                category=row["category"],
                chosen=row["chosen"],
                labeled_by=row["labeled_by"],
                judge_metadata=json.loads(row["judge_metadata"]) if row["judge_metadata"] else None,
                created_at=row["created_at"],
                labeled_at=row["labeled_at"],
                source_pipeline=row["source_pipeline"],
            ))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[RLHF/sqlite_store] failed to deserialise row pair_id=%s: %s",
                row["pair_id"], exc,
            )
    return pairs


def query(
    category: Optional[Category] = None,
    labeled_only: bool = False,
    path: Path = SQLITE_DB_PATH,
) -> list[PreferencePair]:
    """Convenience filter over ``read_all_pairs``.

    Args:
        category: When provided, return only pairs with this category.
        labeled_only: When True, return only pairs where ``chosen`` is not None.
        path: Override the default database path (useful in tests).

    Returns:
        Filtered list of ``PreferencePair`` objects.
    """
    pairs = read_all_pairs(path=path)
    if labeled_only:
        pairs = [p for p in pairs if p.chosen is not None]
    if category is not None:
        cat_val = category.value if isinstance(category, Category) else category
        pairs = [p for p in pairs if p.category == cat_val]
    return pairs
