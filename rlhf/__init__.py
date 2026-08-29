"""ControlPlane.ai — RLHF Preference-Data & DPO Fine-Tuning Module.

This package is fully self-contained.  Nothing in this package imports
from the rest of the ControlPlane codebase; external code may import
*from* this package once the stubs below are wired up.

See rlhf/README.md for the end-to-end flow.
"""

from rlhf.config import Category, STORAGE_BACKEND  # noqa: F401
from rlhf.schema import PreferencePair  # noqa: F401

__all__ = ["Category", "STORAGE_BACKEND", "PreferencePair"]
