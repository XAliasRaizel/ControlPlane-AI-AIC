"""
backend/audit_integrity/merkle.py

Merkle tree construction, root computation, and inclusion proofs,
following RFC 6962's definitions -- the same math behind Certificate
Transparency, Trillian, and Sigstore's Rekor. Two deliberate choices
carried over from that lineage, both for the same reason (a subtly
malformed tree can be made to lie about its own contents):

  1. Domain-separated hashing: a leaf hash is SHA-256(0x00 || data),
     an internal node hash is SHA-256(0x01 || left || right). Without
     this prefix, a crafted leaf's bytes could be made to collide with
     a valid internal node, letting an attacker forge a tree that
     verifies but doesn't mean what it claims to.
  2. A left-heavy split for sizes that aren't a power of two, instead
     of duplicating the last leaf. Leaf-duplication is what several
     early Merkle tree implementations (including Bitcoin's) did, and
     it has a known weakness: an attacker can sometimes make two
     different sets of transactions hash to the same root. RFC 6962's
     split rule avoids that class of bug entirely.
"""
from __future__ import annotations

import hashlib
from typing import List, Tuple

LEAF_PREFIX = b"\x00"
NODE_PREFIX = b"\x01"
EMPTY_ROOT = hashlib.sha256(b"").digest()


def leaf_hash(data: bytes) -> bytes:
    return hashlib.sha256(LEAF_PREFIX + data).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(NODE_PREFIX + left + right).digest()


def _split_point(n: int) -> int:
    """Largest power of two strictly smaller than n (RFC 6962 calls this k)."""
    k = 1
    while k * 2 < n:
        k *= 2
    return k


def merkle_root(leaves: List[bytes]) -> bytes:
    """`leaves` must already be leaf-hashed (see leaf_hash)."""
    n = len(leaves)
    if n == 0:
        return EMPTY_ROOT
    if n == 1:
        return leaves[0]
    k = _split_point(n)
    return _node_hash(merkle_root(leaves[:k]), merkle_root(leaves[k:]))


def inclusion_proof(leaves: List[bytes], index: int) -> List[Tuple[str, bytes]]:
    """
    Audit path for leaves[index], as a list of (side, hash) pairs read
    bottom-to-top. `side` tells verify_inclusion which side of the
    running hash this sibling belongs on -- "R" means the sibling is
    to the right of everything hashed so far, "L" means it's to the left.
    """
    def _walk(subset: List[bytes], i: int) -> List[Tuple[str, bytes]]:
        n = len(subset)
        if n <= 1:
            return []
        k = _split_point(n)
        if i < k:
            return _walk(subset[:k], i) + [("R", merkle_root(subset[k:]))]
        return _walk(subset[k:], i - k) + [("L", merkle_root(subset[:k]))]
    return _walk(leaves, index)


def verify_inclusion(leaf_data: bytes, audit_path: List[Tuple[str, bytes]], expected_root: bytes) -> bool:
    """Recomputes a root from a single leaf's raw data + its audit path
    and checks it against a root you already trust (e.g. an anchored
    checkpoint). This is what lets you prove one record was in the log
    without handing over the whole log -- the same trick a browser uses
    to check one certificate against a Certificate Transparency log."""
    h = leaf_hash(leaf_data)
    for side, sibling in audit_path:
        h = _node_hash(h, sibling) if side == "R" else _node_hash(sibling, h)
    return h == expected_root
