"""
backend/audit_integrity/hashing.py

Small helpers shared by ledger.py and verifier.py. Canonical JSON
matters more than it looks: if the same logical record can serialize
to two different byte strings (key order, float formatting, unicode
normalization), the same record could hash two different ways
depending on who computes it -- which would make verification
unreliable through no fault of the crypto. `canonical_json` pins all
of that down: sorted keys, fixed separators, str() fallback for
anything json can't natively serialize.
"""
from __future__ import annotations

import hashlib
import hmac as _hmac
import json
from typing import Any

GENESIS_HASH = "0" * 64  # the "previous hash" of the very first record in the chain


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def compute_record_hash(prev_hash: str, seq: int, timestamp: float, payload: dict) -> str:
    material = canonical_json({"prev_hash": prev_hash, "seq": seq, "timestamp": timestamp, "payload": payload})
    return sha256_hex(material)


def hmac_sign_hex(secret_key: bytes, message: bytes) -> str:
    return _hmac.new(secret_key, message, hashlib.sha256).hexdigest()


def hmac_verify_hex(secret_key: bytes, message: bytes, signature_hex: str) -> bool:
    expected = hmac_sign_hex(secret_key, message)
    return _hmac.compare_digest(expected, signature_hex)  # constant-time, avoids timing side-channels
