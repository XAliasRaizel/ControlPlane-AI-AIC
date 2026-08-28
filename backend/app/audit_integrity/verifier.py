"""
backend/app/audit_integrity/verifier.py

Deliberately kept separate from ledger.py: this module only ever
reads from AuditRecordBackend and AnchorBackend, never writes to
either. In production this should run as an independent job with
read-only credentials -- possibly on a different machine, run by a
different team (security/compliance, not the team that operates
ControlPlane) -- the same separation of duties Sigstore's
"rekor-monitor" and Certificate Transparency's "gossiping" auditors
rely on: the party checking the log is not the party who can write
to it.

Two independent checks, because they catch two different things:

  1. Chain integrity: for every record, recompute its hash from
     (prev_hash, seq, timestamp, payload) and confirm it matches what's
     stored, and confirm prev_hash matches the previous record's actual
     hash. This catches an edit where the attacker changed a record's
     content but didn't bother (or couldn't) recompute its hash and
     everything after it.
  2. Checkpoint consistency: for every sealed checkpoint, recompute
     the Merkle root over the *current* record hashes in that range and
     compare to the anchored, HMAC-signed root. This catches the more
     sophisticated version of the same attack, where the attacker
     edited a record *and* re-chained every record after it so the
     hashes are internally consistent again -- that trick still can't
     retroactively fix a checkpoint that was already written to the
     separate anchor store.

Neither check can see tampering that happens *and* is fully re-chained
entirely within records that haven't been checkpointed yet -- exactly
like Certificate Transparency's maximum merge delay, a smaller
checkpoint_interval shrinks this window at the cost of more
checkpoints. See docs/audit_integrity_spec.md.
"""
from __future__ import annotations

from .backends import AnchorBackend, AuditRecordBackend
from .hashing import GENESIS_HASH, compute_record_hash, hmac_verify_hex
from .merkle import leaf_hash, merkle_root
from .models import VerificationResult


def verify_chain_integrity(records) -> VerificationResult:
    expected_prev = GENESIS_HASH
    details = []
    for record in records:
        recomputed = compute_record_hash(record.prev_hash, record.seq, record.timestamp, record.payload)
        if record.prev_hash != expected_prev:
            details.append(f"seq={record.seq}: prev_hash does not match the actual previous record's hash")
            return VerificationResult(ok=False, records_checked=record.seq - 1, checkpoints_checked=0,
                                       first_broken_seq=record.seq, details=details)
        if recomputed != record.record_hash:
            details.append(f"seq={record.seq}: stored hash does not match the record's own content")
            return VerificationResult(ok=False, records_checked=record.seq - 1, checkpoints_checked=0,
                                       first_broken_seq=record.seq, details=details)
        expected_prev = record.record_hash
    return VerificationResult(ok=True, records_checked=len(records), checkpoints_checked=0)


def verify_checkpoints(record_backend: AuditRecordBackend, checkpoints, hmac_secret: bytes) -> VerificationResult:
    details = []
    for cp in checkpoints:
        signing_material = f"{cp.from_seq}:{cp.to_seq}:{cp.merkle_root_hex}:{cp.tree_size}:{cp.timestamp}".encode("utf-8")
        if not hmac_verify_hex(hmac_secret, signing_material, cp.signature_hex):
            details.append(f"checkpoint id={cp.checkpoint_id}: signature does not match its own contents "
                            f"(the anchor file itself may have been edited)")
            return VerificationResult(ok=False, records_checked=0, checkpoints_checked=cp.checkpoint_id - 1,
                                       first_broken_checkpoint=cp.checkpoint_id, details=details)

        current_records = record_backend.get_range(cp.from_seq, cp.to_seq)
        if len(current_records) != cp.tree_size:
            details.append(f"checkpoint id={cp.checkpoint_id}: expected {cp.tree_size} records in range "
                            f"{cp.from_seq}-{cp.to_seq}, found {len(current_records)}")
            return VerificationResult(ok=False, records_checked=0, checkpoints_checked=cp.checkpoint_id - 1,
                                       first_broken_checkpoint=cp.checkpoint_id, details=details)

        leaves = [leaf_hash(r.record_hash.encode("utf-8")) for r in current_records]
        recomputed_root_hex = merkle_root(leaves).hex()
        if recomputed_root_hex != cp.merkle_root_hex:
            details.append(f"checkpoint id={cp.checkpoint_id} (records {cp.from_seq}-{cp.to_seq}): "
                            f"anchored Merkle root does not match what those records currently hash to -- "
                            f"something in this range changed after it was sealed")
            return VerificationResult(ok=False, records_checked=0, checkpoints_checked=cp.checkpoint_id - 1,
                                       first_broken_checkpoint=cp.checkpoint_id, details=details)
    return VerificationResult(ok=True, records_checked=0, checkpoints_checked=len(checkpoints))


def verify_ledger(record_backend: AuditRecordBackend, anchor_backend: AnchorBackend, hmac_secret: bytes) -> VerificationResult:
    """The full check: chain integrity across every record, then every
    checkpoint's anchored root against current data. Returns the first
    problem found by either check, whichever comes first in the log."""
    records = record_backend.get_all()
    chain_result = verify_chain_integrity(records)
    if not chain_result.ok:
        return chain_result

    checkpoints = anchor_backend.get_all_checkpoints()
    checkpoint_result = verify_checkpoints(record_backend, checkpoints, hmac_secret)
    if not checkpoint_result.ok:
        checkpoint_result.records_checked = len(records)
        return checkpoint_result

    return VerificationResult(ok=True, records_checked=len(records), checkpoints_checked=len(checkpoints))
