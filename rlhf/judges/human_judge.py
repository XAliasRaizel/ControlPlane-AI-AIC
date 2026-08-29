"""ControlPlane.ai RLHF — Human-in-the-loop CLI judge.

Presents a ``PreferencePair`` in a readable terminal format and collects
a human preference label (``a``, ``b``, or ``tie``).

The ``input_fn`` parameter is injectable so the same function can be
driven by a Streamlit callback, a web form, or a test fixture without
changing any of the surrounding logic.

Usage
-----
    from rlhf.judges.human_judge import judge_pair_with_human

    labelled_pair = judge_pair_with_human(pair)           # CLI
    labelled_pair = judge_pair_with_human(pair, input_fn=my_streamlit_fn)  # UI
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable

from rlhf.schema import PreferencePair

logger = logging.getLogger(__name__)

_VALID_CHOICES: frozenset[str] = frozenset({"a", "b", "tie"})

_HEADER = "=" * 72
_SEP    = "-" * 72


def _render_pair(pair: PreferencePair) -> str:
    """Format a preference pair for human-readable terminal display.

    Args:
        pair: The pair to render.

    Returns:
        A multi-line string ready to print.
    """
    lines = [
        _HEADER,
        f"  Pair ID : {pair.pair_id}",
        f"  Category: {pair.category}",
        f"  Source  : {pair.source_pipeline}",
        _SEP,
        "  PROMPT",
        _SEP,
        pair.prompt,
        _SEP,
        "  RESPONSE A",
        f"  Model: {pair.response_a.model_name}  "
        f"({pair.response_a.model_version_or_checkpoint})",
        _SEP,
        pair.response_a.text if not pair.response_a.is_error else f"[ERROR] {pair.response_a.error_message}",
        _SEP,
        "  RESPONSE B",
        f"  Model: {pair.response_b.model_name}  "
        f"({pair.response_b.model_version_or_checkpoint})",
        _SEP,
        pair.response_b.text if not pair.response_b.is_error else f"[ERROR] {pair.response_b.error_message}",
        _HEADER,
    ]
    return "\n".join(lines)


def judge_pair_with_human(
    pair: PreferencePair,
    input_fn: Callable[..., str] = input,
) -> PreferencePair:
    """Collect a human preference label for a ``PreferencePair`` via a prompt.

    Displays the prompt and both responses, then repeatedly asks for input
    until the human types a valid choice (``a``, ``b``, or ``tie``).  The
    function never crashes on invalid input — it re-prompts instead.

    IMPORTANT: this function will NOT overwrite a pre-existing human label.
    If ``pair.labeled_by == "human"`` the pair is returned unchanged with a
    warning log.

    The ``input_fn`` dependency is injectable so this function can be driven
    by a UI callback (e.g. a Streamlit text_input that returns the user's
    choice string) without any changes to the surrounding logic.

    Args:
        pair: The ``PreferencePair`` to label.
        input_fn: Callable that accepts a prompt string and returns the
            human's raw input string.  Defaults to the built-in ``input()``.

    Returns:
        A copy of the pair with ``chosen``, ``labeled_by``, and
        ``labeled_at`` filled in.
    """
    # Do not overwrite existing human labels.
    if pair.labeled_by == "human":
        logger.warning(
            "[RLHF/human_judge] pair %s already has a human label; returning unchanged",
            pair.pair_id,
        )
        return pair

    print(_render_pair(pair))  # noqa: T201

    while True:
        raw = input_fn("  Your choice — type 'a', 'b', or 'tie', then press Enter: ")
        choice = raw.strip().lower()
        if choice in _VALID_CHOICES:
            break
        print(  # noqa: T201
            f"  [!] '{raw}' is not a valid choice.  "
            "Please type exactly 'a', 'b', or 'tie'."
        )

    print(f"\n  [OK] Recorded: {choice!r}  (pair {pair.pair_id})\n")  # noqa: T201
    logger.info("[RLHF/human_judge] pair %s labelled as %r by human", pair.pair_id, choice)

    return pair.model_copy(update={
        "chosen": choice,
        "labeled_by": "human",
        "labeled_at": datetime.now(timezone.utc),
        "judge_metadata": {"source": "human_cli"},
    })
