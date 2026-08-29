"""ControlPlane.ai RLHF — Category-assignment enforcement.

This module is the *single* place where a pair's category gets validated
and attached before it reaches any storage backend.  Call
``assign_category`` before passing a pair to either ``json_store`` or
``sqlite_store`` — this is enforced by convention and by the generator
functions, which both call this function themselves.

Never allow a raw string to bypass this function.
"""

from __future__ import annotations

from rlhf.config import Category
from rlhf.schema import PreferencePair


def assign_category(pair: PreferencePair, category: Category) -> PreferencePair:
    """Validate and attach a ``Category`` to a ``PreferencePair``.

    This is the **only** path through which a category may be written to
    a pair.  Any attempt to pass an arbitrary string (rather than a
    ``Category`` enum member) will raise a ``ValueError``, preventing
    cross-domain contamination in the training data.

    Called automatically by both generator functions (``api_vs_api.py``
    and ``local_vs_local.py``) before they return — callers generally
    do not need to call this themselves, but may do so when re-categorising
    an existing pair.

    Args:
        pair: The ``PreferencePair`` whose category should be set.
        category: A valid ``Category`` enum member.  The function rejects
            plain strings, even if their value happens to match a valid
            member name.

    Returns:
        A new ``PreferencePair`` (Pydantic copy) with the validated
        category attached.

    Raises:
        ValueError: If ``category`` is not an instance of ``Category``.
    """
    if not isinstance(category, Category):
        # Attempt coercion for convenience (e.g. when the value comes from
        # an environment variable or config file), but be explicit about it.
        try:
            category = Category(category)
        except ValueError:
            raise ValueError(
                f"[RLHF] '{category!r}' is not a valid Category. "
                f"Valid choices: {[c.value for c in Category]}"
            )
    return pair.model_copy(update={"category": category})
