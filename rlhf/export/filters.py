"""ControlPlane.ai RLHF — Preference-pair filtering for DPO export.

All filtering decisions live here so they can be tested independently
of the export and storage layers.  Filters are applied in order and
do NOT mutate the input list or the storage backend.

Filter pipeline (applied in sequence)
--------------------------------------
1. Drop unlabelled pairs (``chosen is None``).
2. Drop ties (``chosen == "tie"``).
3. Drop pairs where either side ``is_error``.
4. Drop near-duplicate response pairs (normalised text equality).
5. Filter by category if provided.
"""

from __future__ import annotations

import re
from typing import Optional

from rlhf.config import Category
from rlhf.schema import PreferencePair


def _normalise(text: str) -> str:
    """Collapse whitespace and lowercase for near-duplicate detection.

    Args:
        text: Raw response text.

    Returns:
        A normalised string for comparison purposes only.
    """
    return re.sub(r"\s+", " ", text.strip().lower())


def filter_pairs(
    pairs: list[PreferencePair],
    category: Optional[Category] = None,
) -> list[PreferencePair]:
    """Apply the standard DPO-export filter pipeline to a list of pairs.

    Filters are applied in the following order:
      1. Drop unlabelled pairs (``chosen is None``).
      2. Drop ties (``chosen == "tie"``).
      3. Drop pairs where ``response_a.is_error`` or ``response_b.is_error``.
      4. Drop pairs where the normalised text of both responses is identical.
      5. Filter by ``category`` if provided.

    Args:
        pairs: Input list of ``PreferencePair`` objects.  Not mutated.
        category: When provided, keep only pairs with this category.

    Returns:
        A new list containing only the pairs that passed all filters.
    """
    result: list[PreferencePair] = []

    for pair in pairs:
        # 1. Must be labelled
        if pair.chosen is None:
            continue

        # 2. No ties — ties carry no training signal
        if pair.chosen == "tie":
            continue

        # 3. Neither side may be an error response
        if pair.response_a.is_error or pair.response_b.is_error:
            continue

        # 4. Responses must not be textually identical (no training signal)
        if _normalise(pair.response_a.text) == _normalise(pair.response_b.text):
            continue

        # 5. Category filter
        if category is not None:
            cat_val = category.value if isinstance(category, Category) else str(category)
            pair_cat = pair.category if isinstance(pair.category, str) else pair.category.value
            if pair_cat != cat_val:
                continue

        result.append(pair)

    return result
