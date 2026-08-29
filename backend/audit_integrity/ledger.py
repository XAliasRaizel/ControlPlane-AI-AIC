"""
backend/audit_integrity/ledger.py

TamperEvidentAuditLedger wraps your existing audit writes with two
things ControlPlane's current HMAC fingerprinting doesn't give you:

  1. A hash chain: every record's hash depends on the previous
     record's hash, so editing record #7 changes what #8, #9, ...
     would need to hash to in order to still look valid.
  2. Periodic checkpoints: every `checkpoint_interval` records, the
     Merkle root over that batch is computed, HMAC-signed, and written
     to a SEPARATE append-only store (see backends.AnchorBackend).
     This is the part that actually matters -- a hash chain alone only
     proves tampering if the attacker *doesn't* also rewrite everything
     downstream of their edit. An externally anchored checkpoint can't
     be fixed up that way, because fixing it up requires reaching into
     a second store the attacker may not control.

This mirrors, in miniature, what Certificate Transparency calls a
"Signed Tree Head" and what Azure SQL Database Ledger calls a
"database digest stored outside the database".
"""
from __future__ import annotations

import time
from typing import Optional

from .backends import AnchorBackend, AuditRecordBackend
from .hashing import GENESIS_HASH, compute_record_hash, hmac_sign_hex
from .merkle import leaf_hash, merkle_root
from .models import AuditRecord, Checkpoint


class TamperEvidentAuditLedger:
    def __init__(
        self,
        record_backend: AuditRecordBackend,
        anchor_backend: AnchorBackend,
        hmac_secret: bytes,
        checkpoint_interval: int = 5,
    ) -> None:
        self._records = record_backend
        self._anchors = anchor_backend
        self._hmac_secret = hmac_secret
        self.checkpoint_interval = checkpoint_interval
        self._last_checkpointed_seq = self._infer_last_checkpointed_seq()

    def _infer_last_checkpointed_seq(self) -> int:
        checkpoints = self._anchors.get_all_checkpoints()
        return checkpoints[-1].to_seq if checkpoints else 0

    def append(self, payload: dict) -> AuditRecord:
        """Append one audit event (whatever your existing audit.py already
        builds -- decision, risk, fingerprinted user/prompt, matched
        policy, timestamp) and chain it to the previous record."""
        last = self._records.get(self._records.count()) if self._records.count() else None
        prev_hash = last.record_hash if last else GENESIS_HASH
        seq = self._records.count() + 1
        timestamp = time.time()
        record_hash = compute_record_hash(prev_hash, seq, timestamp, payload)
        record = AuditRecord(seq=seq, timestamp=timestamp, payload=payload, prev_hash=prev_hash, record_hash=record_hash)
        self._records.append(record)

        if seq - self._last_checkpointed_seq >= self.checkpoint_interval:
            self.seal_checkpoint()
        return record

    def seal_checkpoint(self) -> Optional[Checkpoint]:
        """Manually seal whatever records have accumulated since the last
        checkpoint. Called automatically by append() at the configured
        interval, but you can also call this directly (e.g. on a timer,
        or before a shift handover, or before a compliance export)."""
        from_seq = self._last_checkpointed_seq + 1
        to_seq = self._records.count()
        if to_seq < from_seq:
            return None  # nothing new to seal

        batch = self._records.get_range(from_seq, to_seq)
        leaves = [leaf_hash(r.record_hash.encode("utf-8")) for r in batch]
        root = merkle_root(leaves)
        root_hex = root.hex()
        tree_size = len(batch)
        timestamp = time.time()

        signing_material = f"{from_seq}:{to_seq}:{root_hex}:{tree_size}:{timestamp}".encode("utf-8")
        signature_hex = hmac_sign_hex(self._hmac_secret, signing_material)

        checkpoint = Checkpoint(
            checkpoint_id=len(self._anchors.get_all_checkpoints()) + 1,
            from_seq=from_seq, to_seq=to_seq, merkle_root_hex=root_hex,
            tree_size=tree_size, timestamp=timestamp, signature_hex=signature_hex,
        )
        self._anchors.append_checkpoint(checkpoint)
        self._last_checkpointed_seq = to_seq
        return checkpoint
