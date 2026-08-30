"""ControlPlane.ai RLHF — LLM-as-Judge labelling utility.

Position-bias mitigation
------------------------
Naive LLM judges systematically prefer whichever response they read
*first*, regardless of quality.  We mitigate this by running the judge
at least twice, swapping which response is presented first each time:

  Run 1: prompt says "Response A = response_a,  Response B = response_b"
  Run 2: prompt says "Response A = response_b,  Response B = response_a"

After accounting for the swap, a confident verdict requires *both* runs
to agree.  If they disagree, ``chosen`` is set to ``"tie"`` rather than
guessing.  This is consistent with the self-consistency approach used in
``backend/utils/llm_judge.py``.

Usage
-----
    from rlhf.judges.llm_judge import judge_pair_with_llm

    def my_llm(prompt: str) -> str:          # wire to real provider
        return openai_client.complete(prompt)

    labelled_pair = judge_pair_with_llm(pair, judge_model_call=my_llm)
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Callable

from rlhf.config import increment_judge_counter
from rlhf.schema import PreferencePair

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stub — wire to real LLM provider.
# ---------------------------------------------------------------------------

def call_judge_llm(prompt: str) -> str:
    """Call the judge LLM and return its raw text response.

    Wired to ``backend.utils.llm_judge.get_active_provider()``, which
    selects between ``OpenAIProvider``, ``AnthropicProvider``, and
    ``MockProvider`` based on the ``CP_JUDGE_PROVIDER`` environment
    variable.  Falls back to the ``MockProvider`` when no provider is
    configured — so this function never raises in normal operation.

    Args:
        prompt: The fully-assembled judge prompt (produced by
            ``_build_judge_prompt``).

    Returns:
        The LLM's raw text response.  Expected to contain one of ``"A"``,
        ``"B"``, or ``"tie"`` (case-insensitive) plus optional reasoning.
    """
    from backend.utils.llm_judge import get_active_provider
    provider = get_active_provider()
    # The judge prompt is the full combined system+user string; pass it as
    # the user message with an empty system prompt so we don't re-wrap it.
    text, _meta = provider.complete(system="", user=prompt)
    return text


# ---------------------------------------------------------------------------
# Judge prompt template
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """\
You are an expert, unbiased evaluator comparing two AI responses to the same prompt.
Your task is to decide which response is better or whether they are equally good.

Respond with a JSON object containing exactly these keys:
  "verdict"   : one of "A", "B", or "tie"
  "reasoning" : one sentence explaining your choice (max 80 words)

Do not add any other text outside the JSON object.
"""

_JUDGE_USER_TEMPLATE = """\
## Original Prompt
{prompt}

## Response A
{response_a}

## Response B
{response_b}

Which response is better? Reply with a JSON object: {{"verdict": "A" | "B" | "tie", "reasoning": "..."}}
"""


def _build_judge_prompt(prompt: str, first_text: str, second_text: str) -> str:
    """Assemble the full judge prompt with both responses.

    Args:
        prompt: The original user prompt.
        first_text: Text shown as Response A in this run.
        second_text: Text shown as Response B in this run.

    Returns:
        A formatted prompt string to pass to the judge LLM.
    """
    return _JUDGE_USER_TEMPLATE.format(
        prompt=prompt,
        response_a=first_text or "(empty / error)",
        response_b=second_text or "(empty / error)",
    )


def _parse_verdict(raw: str) -> str:
    """Extract the verdict string from a judge LLM response.

    Tries JSON parsing first, then falls back to simple regex for robustness.

    Args:
        raw: Raw text from the judge LLM.

    Returns:
        Normalised verdict: ``"a"``, ``"b"``, or ``"tie"``.
    """
    raw = raw.strip()
    try:
        data = json.loads(raw)
        verdict = str(data.get("verdict", "")).strip().lower()
    except (json.JSONDecodeError, AttributeError):
        m = re.search(r'\b(verdict\s*[":]\s*)?"?([ABab]|tie)\b', raw, re.IGNORECASE)
        verdict = m.group(2).lower() if m else "tie"

    if verdict not in ("a", "b", "tie"):
        verdict = "tie"
    return verdict


# ---------------------------------------------------------------------------
# Public labelling function
# ---------------------------------------------------------------------------

def judge_pair_with_llm(
    pair: PreferencePair,
    judge_model_call: Callable[[str], str] = call_judge_llm,
    n_calls: int = 2,
) -> PreferencePair:
    """Label a ``PreferencePair`` using an LLM judge with position-bias control.

    Runs the judge ``n_calls`` times (minimum 2), alternating which response
    is presented as "A" and which as "B".  A confident label is only recorded
    when both runs agree (after accounting for the swap); otherwise the pair
    is marked as ``"tie"``.

    The raw votes and orderings are stored in ``judge_metadata`` so the
    decision is fully auditable.

    IMPORTANT: this function will NOT overwrite a pre-existing human label.
    If ``pair.labeled_by == "human"`` the pair is returned unchanged with a
    warning log.

    Args:
        pair: The unlabelled ``PreferencePair`` to judge.
        judge_model_call: Callable that takes a prompt string and returns the
            LLM's raw text.  Defaults to the stub ``call_judge_llm``.
        n_calls: Number of judge calls to make (minimum 2, should be even).

    Returns:
        A copy of the pair with ``chosen``, ``labeled_by``, ``labeled_at``,
        and ``judge_metadata`` filled in.

    Raises:
        RuntimeError: If the daily judge-call cap has been reached.
    """
    # Do not overwrite a human label — safety rule.
    if pair.labeled_by == "human":
        logger.warning(
            "[RLHF/llm_judge] pair %s already has a human label; skipping LLM judge",
            pair.pair_id,
        )
        return pair

    n_calls = max(n_calls, 2)  # enforce minimum

    raw_votes: list[dict] = []
    # We'll accumulate normalised verdicts from the *perspective of response_a*.
    # i.e. if the order is swapped and the judge says "A", that maps to response_b
    # winning, so we record "b".
    normalised_verdicts: list[str] = []

    for call_idx in range(n_calls):
        increment_judge_counter()  # raises RuntimeError if cap reached

        # Alternate ordering each call to cancel position bias.
        swapped = (call_idx % 2 == 1)
        if swapped:
            first_text, second_text = pair.response_b.text, pair.response_a.text
            order_label = "b_first"
        else:
            first_text, second_text = pair.response_a.text, pair.response_b.text
            order_label = "a_first"

        prompt_text = _build_judge_prompt(pair.prompt, first_text, second_text)
        try:
            raw_response = judge_model_call(
                f"{_JUDGE_SYSTEM_PROMPT}\n\n{prompt_text}"
            )
            judge_verdict = _parse_verdict(raw_response)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[RLHF/llm_judge] call %d failed: %s", call_idx, exc)
            judge_verdict = "tie"
            raw_response = f"ERROR: {exc}"

        # Normalise verdict to response_a / response_b frame.
        if swapped:
            if judge_verdict == "a":
                normalised = "b"   # judge said A, but A was actually response_b
            elif judge_verdict == "b":
                normalised = "a"   # judge said B, but B was actually response_a
            else:
                normalised = "tie"
        else:
            normalised = judge_verdict

        raw_votes.append({
            "call_index": call_idx,
            "order": order_label,
            "raw_verdict": judge_verdict,
            "normalised_verdict": normalised,
            "raw_response": raw_response if isinstance(raw_response, str) else str(raw_response),
        })
        normalised_verdicts.append(normalised)

    # Determine final verdict — only accept non-tie if all runs agree.
    unique_verdicts = set(normalised_verdicts)
    if len(unique_verdicts) == 1:
        final_chosen = normalised_verdicts[0]
    else:
        final_chosen = "tie"  # disagreement → tie rather than guessing
        logger.info(
            "[RLHF/llm_judge] pair %s: judge runs disagreed %s → marking as tie",
            pair.pair_id, normalised_verdicts,
        )

    return pair.model_copy(update={
        "chosen": final_chosen,
        "labeled_by": "llm_judge",
        "labeled_at": datetime.now(timezone.utc),
        "judge_metadata": {
            "n_calls": n_calls,
            "normalised_verdicts": normalised_verdicts,
            "final_chosen": final_chosen,
            "raw_votes": raw_votes,
        },
    })
