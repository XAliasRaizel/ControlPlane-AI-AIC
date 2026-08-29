# Tamper-Evident Audit Log — Feature Spec

**Status:** implemented and unit-tested (14/14 passing, stdlib only — no new dependencies). Demo script runs three acts end to end and matches every expected outcome.

## 1. Why this exists

HMAC fingerprinting (already in ControlPlane's audit store) answers "who was this about, without storing raw PII." It does not answer "did anyone edit this record after it was written." Those are different problems. This feature closes the second one: it makes the audit log **tamper-evident** — any retroactive edit, by anyone, including someone with full database access, becomes detectable.

## 2. Research: this is not a novel idea, and that's the point

Three organizations run large parts of the internet's trust infrastructure on exactly this pattern (hash-chained records + periodic checkpoints anchored somewhere separate from the log itself):

- **Google's Certificate Transparency.** Every publicly-trusted TLS certificate is logged to an append-only Merkle tree; the root is periodically signed into a "Signed Tree Head" (STH), and independent parties "gossip" their observed STHs to each other so no single log operator can show different histories to different people without getting caught. The underlying implementation, **Trillian**, has since been generalized beyond certificates to binary and AI-model transparency.
- **Sigstore's Rekor** (Linux Foundation, backed by Google, Red Hat, Chainguard) applies the same transparency-log pattern to software supply chains — PyPI and npm both support publishing package-signing events to it. A companion tool, `rekor-monitor`, exists specifically to run as an **independent watcher** that checks consistency and flags anomalies — a different party than whoever writes to the log.
- **Microsoft's Azure SQL Database Ledger** hash-chains every transaction and, every 30 seconds, computes a Merkle-tree "database digest" that it explicitly recommends storing **outside the database** — in immutable blob storage or a separate ledger service. Microsoft's own documentation is candid about why: a sufficiently privileged database administrator or cloud operator can otherwise rewrite ledger tables directly.

There's also a cautionary tale worth knowing: **AWS's Quantum Ledger Database (QLDB)** — a managed service built around this exact idea, marketed to banks and healthcare for a compliance-grade audit trail — was fully discontinued on July 31, 2025. AWS's own migration guidance for QLDB customers admits the replacement it recommends (Aurora PostgreSQL) gives you audit logging but *not* the cryptographic verifiability QLDB had. The lesson: build this as a portable, self-contained layer around your own data, not a bet on a single vendor's managed ledger product.

This feature is a small, self-contained version of the same idea: hash chain + periodic Merkle checkpoint + a separate anchor store + an independent verifier, using only what's already in your project (SQLite, Python's stdlib `hmac`/`hashlib`).

## 3. What was built

```
backend/app/audit_integrity/
  __init__.py    # exports TamperEvidentAuditLedger (writer) and verify_ledger (checker)
  models.py      # AuditRecord, Checkpoint, VerificationResult
  hashing.py     # canonical JSON + SHA-256 + HMAC helpers
  merkle.py      # RFC 6962-style Merkle tree: root computation + inclusion proofs
  backends.py    # AuditRecordBackend (SQLite) + AnchorBackend (separate append-only JSONL)
  ledger.py      # TamperEvidentAuditLedger: append() + seal_checkpoint()
  verifier.py    # verify_chain_integrity() + verify_checkpoints() + verify_ledger()

scripts/run_audit_integrity_demo.py   # three-act demo, run it directly
tests/test_audit_integrity.py          # 14 unittest cases (pytest-discoverable)
```

## 4. How it works

**Writing (`ledger.py`):** every `append(payload)` computes `record_hash = SHA256(prev_hash ‖ seq ‖ timestamp ‖ payload)` and stores it alongside the record. Every `checkpoint_interval` records (default 5), the ledger computes a **Merkle root** over that batch's hashes, HMAC-signs it, and writes it to a **separate** append-only file — not a table in the same database.

**Verifying (`verifier.py`), and this part is deliberately a different module the writer never calls:**

1. **Chain integrity** — recompute every record's hash from its own stored content and confirm it matches, and confirm each record's `prev_hash` really is the previous record's hash. Catches a naive edit where the attacker changed content but didn't update the hash.
2. **Checkpoint consistency** — for every sealed checkpoint, recompute the Merkle root over the *current* record hashes in that range and compare to the anchored, HMAC-signed root. Catches the more careful version of the same attack, where the attacker also re-chained every record after their edit so the database looks internally consistent again. That trick can't reach into the separate anchor file to fix the checkpoint too.

## 5. The demo (`scripts/run_audit_integrity_demo.py`) — verified output

```
ACT A -- append 12 records, checkpoint every 4, verify clean
  -> VALID -- 12 records and 3 checkpoints verified

ACT B -- naive tamper: edit record #8's payload, leave its hash alone
  -> TAMPERING DETECTED -- first break at record seq=8

ACT C -- sophisticated tamper: edit #8 AND re-chain #8 onward
  verify_chain_integrity() alone -> VALID   (this check alone is fooled!)
  verify_ledger() (chain + anchored checkpoints) ->
      TAMPERING DETECTED -- first break at checkpoint id=2
      (records 5-8: anchored Merkle root no longer matches)
```

Act C is the point of the whole feature: a hash chain by itself can be defeated by anyone who can rewrite everything downstream of their edit. An externally anchored checkpoint can't be fixed the same way, because doing so means reaching into a second store.

## 6. Honest limitations (say these out loud in the pitch, don't wait for a judge to find them)

- **HMAC, not asymmetric signing.** Whoever holds the HMAC secret can forge a checkpoint as convincingly as they can verify one — the writer and the verifier share a key. Certificate Transparency and Azure Confidential Ledger both use asymmetric signing (and Azure additionally runs inside a hardware-backed secure enclave) specifically so *verifying* history never requires the same key that could *rewrite* it. Documented upgrade path: Ed25519 signing via Python's `cryptography` package (not installed in the sandbox this was tested in — add it when you wire this in), with the private key held only by the process that seals checkpoints.
- **A bounded detection window, not instant tamper-proofing.** Records added after the most recent checkpoint aren't anchored yet — if they're edited and re-chained before the next checkpoint seals, that specific edit won't be caught until a checkpoint is sealed over them (and if it seals *after* the edit, it anchors the tampered state as if it were correct). This is the same tradeoff Certificate Transparency calls its "maximum merge delay." Smaller `checkpoint_interval` shrinks the window; it doesn't remove it. (`tests/test_audit_integrity.py::test_tamper_after_the_last_checkpoint_is_not_caught_until_the_next_seal` demonstrates this directly rather than hiding it.)
- **The anchor store's separation is logical, not yet physical.** In this build, the anchor file lives on the same disk as everything else — it's a different *file and format*, opened only in append mode, but a root user on the same machine could still reach both. Real separation means a different account, a different service, or WORM-configured storage (Azure Blob with an immutability policy, AWS S3 Object Lock, or a genuinely external service like Rekor) — a deployment decision, not something one Python module can fully guarantee on its own.

## 7. Integration TODOs

- [ ] Wrap your existing `audit.py` write calls with `TamperEvidentAuditLedger.append(payload)` instead of (or alongside) the current insert — `payload` can be exactly what you already build (fingerprinted user/prompt, risk scores, decision, matched policy).
- [ ] Pick a real `checkpoint_interval` (record count) or add a time-based trigger (e.g. also seal every N minutes regardless of volume) depending on your actual request rate.
- [ ] Move `AnchorBackend`'s file to genuinely separate storage before this goes anywhere near production — see §6.
- [ ] Wire `verify_ledger()` into a scheduled job (cron / GitHub Actions on a timer) that runs independently of the main app process and alerts on `VerificationResult.ok is False` — this is your `rekor-monitor` equivalent.
- [ ] If you build the C2 session-risk tracker or the agent tool-governance layer, route *their* audit writes through this same ledger rather than starting a second, unrelated audit trail.

## 8. Acceptance criteria

- [x] Every record is hash-chained to the one before it
- [x] Checkpoints seal automatically at a configurable interval, and can also be sealed manually
- [x] Checkpoints are HMAC-signed and stored in a separate append-only store
- [x] A naive content edit (hash left stale) is caught by chain-integrity verification
- [x] A sophisticated edit-and-rechain is *not* caught by chain-integrity alone, but *is* caught by checkpoint verification
- [x] Editing the anchor file itself (forging a root) is caught via HMAC signature mismatch
- [x] Merkle inclusion proofs verified exhaustively (every leaf, tree sizes 1–100+, 820/820 correct, zero false positives on tampered data)
- [x] 14/14 unit tests passing; demo script runs end to end and matches every documented outcome
