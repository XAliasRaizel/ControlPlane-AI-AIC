# Quickstart: Tamper-Evident Audit Log

## Drop-in

Copy these into your repo at the **exact same relative paths**:

```
backend/app/audit_integrity/__init__.py
backend/app/audit_integrity/models.py
backend/app/audit_integrity/hashing.py
backend/app/audit_integrity/merkle.py
backend/app/audit_integrity/backends.py
backend/app/audit_integrity/ledger.py
backend/app/audit_integrity/verifier.py
scripts/run_audit_integrity_demo.py
tests/test_audit_integrity.py
```

No new dependencies — stdlib only (`hashlib`, `hmac`, `sqlite3`, `json`).

## Run it

```bash
python3 scripts/run_audit_integrity_demo.py
python3 -m unittest tests.test_audit_integrity -v
# or, since it's plain unittest, your existing pytest suite will pick it up too:
pytest tests/test_audit_integrity.py -v
```

Both were run and passed (14/14 tests; all three demo acts matching their documented outcome) before this was handed to you.

## Minimal usage

```python
from app.audit_integrity import AuditRecordBackend, AnchorBackend, TamperEvidentAuditLedger, verify_ledger

records = AuditRecordBackend("audit_records.sqlite")
anchors = AnchorBackend("audit_checkpoints.jsonl")   # keep this on genuinely separate storage in production -- see the spec, section 6
ledger = TamperEvidentAuditLedger(records, anchors, hmac_secret=b"...", checkpoint_interval=50)

# wherever you currently write an audit event:
ledger.append({
    "request_id": request_id,
    "user_fingerprint": fingerprint,
    "decision": decision,
    "risk": risk_score,
    "matched_rule": matched_rule,
})

# in a separate, independent process/job:
result = verify_ledger(records, anchors, hmac_secret=b"...")
if not result.ok:
    alert_someone(result.summary())
```

## If you want to hand this to an agentic coding tool (Antigravity, Claude Code, etc.)

Point it at `docs/audit_integrity_spec.md` — design rationale, the research it's grounded in, integration TODOs each marked `[ ]`, and acceptance criteria, written so an agent (or a teammate) can pick it up without this conversation as context.
