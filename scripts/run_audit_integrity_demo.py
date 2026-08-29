#!/usr/bin/env python3
"""
scripts/run_audit_integrity_demo.py

Standalone, dependency-free (stdlib only) walkthrough of the
tamper-evident audit layer. Three acts:

  A. Append 12 audit records, sealing a checkpoint every 4. Verify --
     should be clean.
  B. A "naive" tamper: edit one record's content directly in the
     database, the way a DBA or a compromised admin account could.
     Verify -- chain integrity should catch it immediately.
  C. A "sophisticated" tamper: edit a record *and* re-chain every
     record after it, so the database is internally self-consistent
     again. Verify chain-only -- this now (wrongly) looks fine. Verify
     the full ledger, checkpoints included -- this is what actually
     catches it, because the checkpoint was already sealed to a
     separate store before the tamper happened.

    python scripts/run_audit_integrity_demo.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.audit_integrity import AnchorBackend, AuditRecordBackend, TamperEvidentAuditLedger, verify_ledger  # noqa: E402
from backend.audit_integrity.verifier import verify_chain_integrity  # noqa: E402

HMAC_SECRET = b"demo-secret-change-me-before-anything-real-touches-this"


def _sample_payload(i: int) -> dict:
    # Shaped like a real ControlPlane audit record -- fingerprints and
    # decisions, never raw PII -- so this drops in next to your existing
    # audit.py writes rather than inventing a new record shape.
    decisions = ["ALLOW", "ALLOW", "BLOCK", "HUMAN_REVIEW", "ALLOW"]
    return {
        "request_id": f"req_{1000 + i}",
        "user_fingerprint": f"fp_{i:04x}",
        "application": "hr-copilot",
        "decision": decisions[i % len(decisions)],
        "risk": round(0.1 + (i % 5) * 0.17, 2),
        "matched_rule": "hr-privacy-v2" if i % 3 == 0 else None,
    }


def build_ledger(workdir: Path, label: str) -> tuple[TamperEvidentAuditLedger, AuditRecordBackend, AnchorBackend]:
    records_db = workdir / f"{label}_records.sqlite"
    anchor_file = workdir / f"{label}_anchors.jsonl"
    record_backend = AuditRecordBackend(records_db)
    anchor_backend = AnchorBackend(anchor_file)
    ledger = TamperEvidentAuditLedger(record_backend, anchor_backend, HMAC_SECRET, checkpoint_interval=4)
    return ledger, record_backend, anchor_backend


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)

        print("=" * 72)
        print("ACT A -- append 12 records, checkpoint every 4, verify clean")
        print("=" * 72)
        ledger, records, anchors = build_ledger(workdir, "act_a")
        for i in range(12):
            ledger.append(_sample_payload(i))
        checkpoints = anchors.get_all_checkpoints()
        print(f"records written : {records.count()}")
        print(f"checkpoints sealed: {len(checkpoints)} "
              f"({', '.join(f'#{c.checkpoint_id} covers {c.from_seq}-{c.to_seq}' for c in checkpoints)})")
        result = verify_ledger(records, anchors, HMAC_SECRET)
        print(f"verify_ledger()  : {result.summary()}")

        print("\n" + "=" * 72)
        print("ACT B -- naive tamper: edit record #8's payload, leave its hash alone")
        print("=" * 72)
        original_8 = _sample_payload(7)
        print(f"record #8 originally: {original_8}  (decision was genuinely BLOCK)")
        tampered_payload = dict(original_8)
        tampered_payload["decision"] = "ALLOW"  # someone is covering something up
        records._simulate_attacker_overwrite(seq=8, new_payload=tampered_payload, recompute_hash=False)
        result = verify_ledger(records, anchors, HMAC_SECRET)
        print(f"verify_ledger()  : {result.summary()}")
        print("-> caught immediately: the stored hash no longer matches the record's own content.")

        print("\n" + "=" * 72)
        print("ACT C -- sophisticated tamper: edit #8 AND re-chain #8 onward to stay self-consistent")
        print("=" * 72)
        ledger2, records2, anchors2 = build_ledger(workdir, "act_c")
        for i in range(12):
            ledger2.append(_sample_payload(i))
        tampered_payload_2 = dict(_sample_payload(7))
        tampered_payload_2["decision"] = "ALLOW"  # same cover-up, this time done carefully
        records2._simulate_attacker_overwrite(seq=8, new_payload=tampered_payload_2, recompute_hash=False)
        records2._simulate_attacker_rechain_from(start_seq=8)  # recomputes #8's hash correctly, then re-chains 9-12

        chain_only = verify_chain_integrity(records2.get_all())
        print(f"verify_chain_integrity() alone : {chain_only.summary()}")
        print("   (this check alone is fooled -- the database is internally consistent again)")

        full_result = verify_ledger(records2, anchors2, HMAC_SECRET)
        print(f"verify_ledger() (chain + anchored checkpoints): {full_result.summary()}")
        print("   -> the checkpoint covering records 5-8 was sealed to a SEPARATE store before the")
        print("      tamper happened. Re-chaining the database doesn't -- and can't -- rewrite that.")

        # Close SQLite connections before temp directory cleanup (Windows file locking)
        records.close()
        records2.close()

        print("\n" + "=" * 72)
        print("Done. Act A stays clean throughout; Acts B and C both start from a fresh, honest ledger.")
        print("=" * 72)


if __name__ == "__main__":
    main()
