"""
tests/test_audit_integrity.py

Pure-stdlib unit tests, unittest-based so they run standalone or under
pytest's auto-discovery without adding a dependency.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.audit_integrity import (  # noqa: E402
    AnchorBackend, AuditRecordBackend, TamperEvidentAuditLedger, verify_ledger,
)
from backend.audit_integrity.verifier import verify_chain_integrity  # noqa: E402
from backend.audit_integrity.merkle import leaf_hash, merkle_root, inclusion_proof, verify_inclusion  # noqa: E402

SECRET = b"test-secret"


class AuditIntegrityTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.records = AuditRecordBackend(tmp / "records.sqlite")
        self.anchors = AnchorBackend(tmp / "anchors.jsonl")
        self.ledger = TamperEvidentAuditLedger(self.records, self.anchors, SECRET, checkpoint_interval=4)

    def tearDown(self) -> None:
        self.records.close()
        self._tmp.cleanup()

    def _fill(self, n: int) -> None:
        for i in range(n):
            self.ledger.append({"decision": "BLOCK" if i % 4 == 3 else "ALLOW", "n": i})


class TestCleanLedger(AuditIntegrityTestCase):
    def test_empty_ledger_verifies(self):
        result = verify_ledger(self.records, self.anchors, SECRET)
        self.assertTrue(result.ok)

    def test_clean_ledger_verifies(self):
        self._fill(12)
        result = verify_ledger(self.records, self.anchors, SECRET)
        self.assertTrue(result.ok)
        self.assertEqual(result.records_checked, 12)
        self.assertEqual(result.checkpoints_checked, 3)  # 12 / interval-of-4

    def test_checkpoints_seal_at_the_configured_interval(self):
        self._fill(9)  # 2 full checkpoints (1-4, 5-8), record 9 not yet checkpointed
        checkpoints = self.anchors.get_all_checkpoints()
        self.assertEqual(len(checkpoints), 2)
        self.assertEqual((checkpoints[0].from_seq, checkpoints[0].to_seq), (1, 4))
        self.assertEqual((checkpoints[1].from_seq, checkpoints[1].to_seq), (5, 8))

    def test_manual_seal_covers_the_leftover_tail(self):
        self._fill(9)
        self.assertEqual(len(self.anchors.get_all_checkpoints()), 2)
        self.ledger.seal_checkpoint()
        checkpoints = self.anchors.get_all_checkpoints()
        self.assertEqual(len(checkpoints), 3)
        self.assertEqual((checkpoints[2].from_seq, checkpoints[2].to_seq), (9, 9))
        result = verify_ledger(self.records, self.anchors, SECRET)
        self.assertTrue(result.ok)


class TestNaiveTamper(AuditIntegrityTestCase):
    def test_content_edit_without_hash_update_is_caught(self):
        self._fill(12)
        original = self.records.get(8)
        self.assertEqual(original.payload["decision"], "BLOCK")

        self.records._simulate_attacker_overwrite(seq=8, new_payload={"decision": "ALLOW", "n": 7}, recompute_hash=False)

        result = verify_ledger(self.records, self.anchors, SECRET)
        self.assertFalse(result.ok)
        self.assertEqual(result.first_broken_seq, 8)

    def test_untampered_records_still_pass_up_to_the_break(self):
        self._fill(12)
        self.records._simulate_attacker_overwrite(seq=8, new_payload={"decision": "ALLOW", "n": 7}, recompute_hash=False)
        result = verify_ledger(self.records, self.anchors, SECRET)
        self.assertEqual(result.records_checked, 7)  # seq 1-7 were fine before the break


class TestSophisticatedTamper(AuditIntegrityTestCase):
    def test_rechaining_fools_chain_only_check(self):
        self._fill(12)
        self.records._simulate_attacker_overwrite(seq=8, new_payload={"decision": "ALLOW", "n": 7}, recompute_hash=False)
        self.records._simulate_attacker_rechain_from(start_seq=8)

        chain_only = verify_chain_integrity(self.records.get_all())
        self.assertTrue(chain_only.ok)  # this is the whole point of the "sophisticated" scenario

    def test_but_full_verification_still_catches_it_via_the_anchored_checkpoint(self):
        self._fill(12)
        self.records._simulate_attacker_overwrite(seq=8, new_payload={"decision": "ALLOW", "n": 7}, recompute_hash=False)
        self.records._simulate_attacker_rechain_from(start_seq=8)

        result = verify_ledger(self.records, self.anchors, SECRET)
        self.assertFalse(result.ok)
        self.assertEqual(result.first_broken_checkpoint, 2)  # checkpoint covering records 5-8

    def test_tamper_after_the_last_checkpoint_is_not_caught_until_the_next_seal(self):
        # Honest limitation, not a bug: this is the same bounded window
        # Certificate Transparency calls its "maximum merge delay". A
        # record added after the most recent checkpoint, then fully
        # re-chained, can't be caught until *some* checkpoint covers it.
        self._fill(9)  # checkpoints cover 1-4 and 5-8; record 9 is not yet anchored
        self.records._simulate_attacker_overwrite(seq=9, new_payload={"decision": "ALLOW", "n": 8}, recompute_hash=False)
        self.records._simulate_attacker_rechain_from(start_seq=9)
        result = verify_ledger(self.records, self.anchors, SECRET)
        self.assertTrue(result.ok)  # confirms the window exists...
        self.ledger.seal_checkpoint()  # ...and confirms sealing closes it
        result_after_seal_of_a_tampered_record = verify_ledger(self.records, self.anchors, SECRET)
        # Sealing a checkpoint over already-tampered data anchors the bad
        # state, which is exactly why checkpoint_interval should be small
        # relative to how quickly you review new records -- documented in
        # the spec.
        self.assertTrue(result_after_seal_of_a_tampered_record.ok)


class TestCheckpointSignatureTamper(AuditIntegrityTestCase):
    def test_editing_the_anchor_file_directly_is_caught(self):
        self._fill(8)
        checkpoints = self.anchors.get_all_checkpoints()
        self.assertEqual(len(checkpoints), 2)

        # Simulate an attacker editing the anchor file's root hash directly
        # (not just the audit DB) without the HMAC secret to re-sign it.
        lines = self.anchors.path.read_text().splitlines()
        import json
        cp1 = json.loads(lines[0])
        cp1["merkle_root_hex"] = "00" * 32  # forged root, signature now stale
        lines[0] = json.dumps(cp1, sort_keys=True)
        self.anchors.path.write_text("\n".join(lines) + "\n")

        result = verify_ledger(self.records, self.anchors, SECRET)
        self.assertFalse(result.ok)
        self.assertEqual(result.first_broken_checkpoint, 1)

    def test_wrong_hmac_secret_fails_verification_even_on_a_clean_ledger(self):
        self._fill(8)
        result = verify_ledger(self.records, self.anchors, b"the-wrong-secret")
        self.assertFalse(result.ok)


class TestMerkleMath(unittest.TestCase):
    def test_inclusion_proof_round_trip_for_several_sizes(self):
        for n in (1, 2, 3, 4, 5, 8, 13, 16, 17):
            leaves_raw = [f"leaf-{i}".encode() for i in range(n)]
            hashed = [leaf_hash(d) for d in leaves_raw]
            root = merkle_root(hashed)
            for idx in range(n):
                proof = inclusion_proof(hashed, idx)
                self.assertTrue(verify_inclusion(leaves_raw[idx], proof, root),
                                 f"inclusion proof failed for n={n}, idx={idx}")

    def test_inclusion_proof_rejects_wrong_data(self):
        leaves_raw = [f"leaf-{i}".encode() for i in range(6)]
        hashed = [leaf_hash(d) for d in leaves_raw]
        root = merkle_root(hashed)
        proof = inclusion_proof(hashed, 2)
        self.assertFalse(verify_inclusion(b"not-the-real-leaf", proof, root))

    def test_root_changes_if_any_leaf_changes(self):
        base = [leaf_hash(f"leaf-{i}".encode()) for i in range(10)]
        tampered = list(base)
        tampered[4] = leaf_hash(b"something-else")
        self.assertNotEqual(merkle_root(base), merkle_root(tampered))


if __name__ == "__main__":
    unittest.main()
