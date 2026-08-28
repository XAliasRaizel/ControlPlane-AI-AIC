"""
backend/app/audit_integrity/models.py

Plain dataclasses -- no Pydantic dependency, same reasoning as the
agent-governance feature: this stays testable anywhere Python runs,
and wraps in Pydantic only at an HTTP boundary if you add one later.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AuditRecord:
    seq: int
    timestamp: float
    payload: Dict[str, Any]     # whatever your existing audit.py already logs (fingerprints, risk, decision, reason -- never raw PII)
    prev_hash: str
    record_hash: str = ""       # filled in by the ledger on append; empty here is just the pre-append shape


@dataclass
class Checkpoint:
    checkpoint_id: int
    from_seq: int                # inclusive
    to_seq: int                  # inclusive
    merkle_root_hex: str
    tree_size: int
    timestamp: float = field(default_factory=time.time)
    signature_hex: str = ""      # HMAC-SHA256 over the fields above, see hashing.hmac_sign_hex


@dataclass
class VerificationResult:
    ok: bool
    records_checked: int
    checkpoints_checked: int
    first_broken_seq: Optional[int] = None
    first_broken_checkpoint: Optional[int] = None
    details: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.ok:
            return (f"VALID -- {self.records_checked} record(s) and {self.checkpoints_checked} "
                     f"checkpoint(s) verified, chain and anchors consistent.")
        where = []
        if self.first_broken_seq is not None:
            where.append(f"record seq={self.first_broken_seq}")
        if self.first_broken_checkpoint is not None:
            where.append(f"checkpoint id={self.first_broken_checkpoint}")
        loc = " / ".join(where) if where else "unknown location"
        return f"TAMPERING DETECTED -- first break at {loc}. Details: {'; '.join(self.details)}"
