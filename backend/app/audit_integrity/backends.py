"""
backend/app/audit_integrity/backends.py

Two separate storage backends, and that separation is the point:

  - AuditRecordBackend: the main audit log (SQLite here, matching the
    rest of ControlPlane's audit store). This is what a privileged
    insider -- a DBA, a compromised admin account, ControlPlane's own
    process if compromised -- could plausibly rewrite.
  - AnchorBackend: a SEPARATE, append-only store for sealed
    checkpoints. In this repo it's a different file in a different
    format (JSONL, opened with mode="a" only -- this code never opens
    it for writing in any mode that could truncate or rewrite it).

Putting checkpoints somewhere structurally different from the record
log is what makes tampering detectable even by someone who can rewrite
*every* row in the audit database, because they'd also have to reach
into a second, differently-shaped store to cover their tracks. In
production, make that separation real: a different account, a
write-once-read-many (WORM) bucket, a separate service -- see
docs/audit_integrity_spec.md for what Azure SQL Ledger and Certificate
Transparency do here.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from .models import AuditRecord, Checkpoint


class AuditRecordBackend:
    """SQLite-backed store for the hash-chained audit records themselves."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_records (
                seq INTEGER PRIMARY KEY,
                timestamp REAL NOT NULL,
                payload_json TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                record_hash TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def append(self, record: AuditRecord) -> None:
        self._conn.execute(
            "INSERT INTO audit_records (seq, timestamp, payload_json, prev_hash, record_hash) VALUES (?, ?, ?, ?, ?)",
            (record.seq, record.timestamp, json.dumps(record.payload, sort_keys=True), record.prev_hash, record.record_hash),
        )
        self._conn.commit()

    def count(self) -> int:
        (n,) = self._conn.execute("SELECT COUNT(*) FROM audit_records").fetchone()
        return n

    def get(self, seq: int) -> Optional[AuditRecord]:
        row = self._conn.execute(
            "SELECT seq, timestamp, payload_json, prev_hash, record_hash FROM audit_records WHERE seq = ?", (seq,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def get_range(self, from_seq: int, to_seq: int) -> List[AuditRecord]:
        rows = self._conn.execute(
            "SELECT seq, timestamp, payload_json, prev_hash, record_hash FROM audit_records "
            "WHERE seq BETWEEN ? AND ? ORDER BY seq ASC",
            (from_seq, to_seq),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_all(self) -> List[AuditRecord]:
        rows = self._conn.execute(
            "SELECT seq, timestamp, payload_json, prev_hash, record_hash FROM audit_records ORDER BY seq ASC"
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    @staticmethod
    def _row_to_record(row) -> AuditRecord:
        seq, timestamp, payload_json, prev_hash, record_hash = row
        return AuditRecord(seq=seq, timestamp=timestamp, payload=json.loads(payload_json),
                            prev_hash=prev_hash, record_hash=record_hash)

    # ------------------------------------------------------------------
    # NOT part of the normal API. This exists only so the demo script and
    # tests can simulate "an attacker with database write access edited
    # history" -- which is exactly the threat this whole feature exists
    # to catch. Nothing in ledger.py or verifier.py ever calls this.
    # ------------------------------------------------------------------
    def _simulate_attacker_overwrite(self, seq: int, new_payload: dict, recompute_hash: bool) -> None:
        record = self.get(seq)
        if record is None:
            raise ValueError(f"no such record: seq={seq}")
        if recompute_hash:
            from .hashing import compute_record_hash  # local import: this helper should only ever be reachable from here
            new_hash = compute_record_hash(record.prev_hash, seq, record.timestamp, new_payload)
        else:
            new_hash = record.record_hash  # attacker changes content but doesn't bother updating the hash -- the naive case
        self._conn.execute(
            "UPDATE audit_records SET payload_json = ?, record_hash = ? WHERE seq = ?",
            (json.dumps(new_payload, sort_keys=True), new_hash, seq),
        )
        self._conn.commit()

    def _simulate_attacker_rechain_from(self, start_seq: int) -> None:
        """The 'sophisticated' attack: after editing one record, recompute
        every record's hash from start_seq onward so the chain is
        internally self-consistent again. Only an externally anchored
        checkpoint sealed before start_seq can still catch this."""
        from .hashing import compute_record_hash
        records = self.get_all()
        by_seq = {r.seq: r for r in records}
        prev_hash = by_seq[start_seq - 1].record_hash if (start_seq - 1) in by_seq else None
        for r in records:
            if r.seq < start_seq:
                continue
            effective_prev = prev_hash if prev_hash is not None else r.prev_hash
            new_hash = compute_record_hash(effective_prev, r.seq, r.timestamp, r.payload)
            self._conn.execute(
                "UPDATE audit_records SET prev_hash = ?, record_hash = ? WHERE seq = ?",
                (effective_prev, new_hash, r.seq),
            )
            prev_hash = new_hash
        self._conn.commit()


class AnchorBackend:
    """Append-only JSONL store for sealed checkpoints. Deliberately a
    different file, a different format, and a different code path than
    AuditRecordBackend -- see the module docstring for why."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.touch(exist_ok=True)

    def append_checkpoint(self, checkpoint: Checkpoint) -> None:
        with open(self.path, "a", encoding="utf-8") as f:  # "a" only -- never "w", never "r+"
            f.write(json.dumps(checkpoint.__dict__, sort_keys=True) + "\n")

    def get_all_checkpoints(self) -> List[Checkpoint]:
        if not self.path.exists():
            return []
        checkpoints = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    checkpoints.append(Checkpoint(**json.loads(line)))
        return checkpoints
