"""
Tamper-evident audit for ControlPlane.ai.

Wraps audit writes in a hash chain (each record's hash depends on the
one before it) plus periodic Merkle checkpoints anchored to a
separate append-only store -- the same combination Certificate
Transparency, Sigstore's Rekor, and Azure SQL Database Ledger all use
in one form or another.

Public entry points: TamperEvidentAuditLedger (writer) and
verify_ledger (independent, read-only checker) -- kept in separate
modules on purpose, see verifier.py's docstring.
"""
from .backends import AnchorBackend, AuditRecordBackend
from .ledger import TamperEvidentAuditLedger
from .models import AuditRecord, Checkpoint, VerificationResult
from .verifier import verify_ledger

__all__ = [
    "TamperEvidentAuditLedger",
    "AuditRecordBackend",
    "AnchorBackend",
    "AuditRecord",
    "Checkpoint",
    "VerificationResult",
    "verify_ledger",
]
