"""ControlPlane.ai RLHF — DPO evaluation utilities.

Three complementary evaluation signals
---------------------------------------
1. **Reward margin** (``compute_reward_margin``): purely log-probability-based,
   no LLM calls.  Fast and objective.

2. **Human-prompt consistency check** (``human_prompt_consistency_check``):
   generates base vs fine-tuned responses for a list of human-curated prompts,
   runs the LLM judge multiple times with swapped ordering (position-bias
   control), and flags any prompt where the fine-tuned model is not
   *consistently* preferred above a threshold.

3. **Full evaluation report** (``run_full_evaluation``): combines both signals
   into one dict.  The ``pass`` field is True only when the average reward
   margin is positive AND the human-prompt consistency check leaves no items
   in the review queue.

Safety guarantee
----------------
No evaluation function in this module auto-approves a model swap.  The LLM
judge's role is strictly to *flag disagreement* for a human to review, not
to make a final decision.  The ``pass`` field in the full report is a
convenience indicator — any production swap should still require a human
to inspect ``flagged_for_human_review`` before proceeding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EvalResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Result of a single-prompt consistency evaluation.

    Attributes:
        prompt: The evaluation prompt text.
        reward_margin: DPO log-prob reward margin (positive = fine-tuned
            preferred).  May be None if not computed for this prompt.
        judge_votes: List of raw judge verdicts (each is ``"base"`` or
            ``"finetuned"`` or ``"tie"``) across all judge calls.
        consistent: True if the fine-tuned model was preferred in at least
            ``agreement_threshold`` fraction of judge calls.
        needs_review: True if ``consistent`` is False — this prompt should be
            inspected by a human before the fine-tuned model is deployed.
    """

    prompt: str
    reward_margin: Optional[float]
    judge_votes: list[str] = field(default_factory=list)
    consistent: bool = False
    needs_review: bool = True


# ---------------------------------------------------------------------------
# 1. Reward margin from log-probabilities
# ---------------------------------------------------------------------------

def compute_reward_margin(
    base_model: Any,
    finetuned_model: Any,
    tokenizer: Any,
    prompt: str,
    chosen: str,
    rejected: str,
) -> float:
    """Compute the DPO reward margin from model log-probabilities.

    The DPO reward margin is defined as:

        margin = log P_finetuned(chosen | prompt) / P_base(chosen | prompt)
                 - log P_finetuned(rejected | prompt) / P_base(rejected | prompt)

    A positive margin means the fine-tuned model assigns relatively more
    probability to the chosen (preferred) response than the base model does —
    which is the goal of DPO training.

    No LLM API calls are made — this uses the model's ``forward`` pass directly.

    Args:
        base_model: The reference (un-fine-tuned) causal LM (``torch`` model).
        finetuned_model: The DPO fine-tuned causal LM.
        tokenizer: The shared tokenizer.
        prompt: The original user prompt.
        chosen: Text of the preferred response.
        rejected: Text of the rejected response.

    Returns:
        The reward margin as a float.  Positive = fine-tuned model is better.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("torch is required for compute_reward_margin.") from exc

    def _log_prob(model: Any, full_text: str) -> float:
        """Compute sum of log-probs for ``full_text`` under ``model``."""
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        # outputs.loss is mean NLL per token; convert to total log-prob.
        n_tokens = inputs["input_ids"].shape[1]
        return float(-outputs.loss.item() * n_tokens)

    chosen_full  = prompt + "\n" + chosen
    rejected_full = prompt + "\n" + rejected

    logp_base_chosen    = _log_prob(base_model,      chosen_full)
    logp_ft_chosen      = _log_prob(finetuned_model, chosen_full)
    logp_base_rejected  = _log_prob(base_model,      rejected_full)
    logp_ft_rejected    = _log_prob(finetuned_model, rejected_full)

    margin = (logp_ft_chosen - logp_base_chosen) - (logp_ft_rejected - logp_base_rejected)
    logger.debug("[RLHF/evaluate] reward_margin=%.4f for prompt %.60s…", margin, prompt)
    return margin


# ---------------------------------------------------------------------------
# 2. Human-prompt consistency check
# ---------------------------------------------------------------------------

def human_prompt_consistency_check(
    human_prompts: list[str],
    base_model_call: Callable[[str], str],
    finetuned_model_call: Callable[[str], str],
    llm_judge_call: Callable[[str, str, str], str],
    n_judge_calls: int = 3,
    agreement_threshold: float = 0.66,
) -> list[EvalResult]:
    """Check fine-tuned model consistency across human-curated prompts.

    For each prompt:
      1. Generate a response from the base model and the fine-tuned model.
      2. Run the LLM judge ``n_judge_calls`` times, alternating response order
         each time to control for position bias.
      3. Tally votes: a vote is ``"finetuned"`` if the judge preferred the
         fine-tuned response, ``"base"`` if it preferred the base response,
         or ``"tie"`` if neither was preferred.
      4. The fine-tuned model is ``consistent`` for this prompt only if the
         fraction of ``"finetuned"`` votes >= ``agreement_threshold``.
      5. ``needs_review=True`` is set whenever the model is *not* consistent.

    The LLM judge's role is *only* to flag disagreement.  It does NOT
    auto-approve — a human must inspect the flagged prompts before any
    deployment decision is made.

    Args:
        human_prompts: List of prompts chosen by human evaluators.  These are
            NOT labelled pairs — just raw prompts used to spot-check quality.
        base_model_call: Callable ``(prompt: str) -> str`` generating a base
            model response.
        finetuned_model_call: Callable ``(prompt: str) -> str`` generating a
            fine-tuned model response.
        llm_judge_call: Callable ``(prompt, response_a, response_b) -> str``
            returning a raw judge verdict string containing ``"A"``, ``"B"``,
            or ``"tie"`` (or ``"a"``/``"b"``).  The caller is responsible for
            wiring this to a real LLM — use the stub in
            ``rlhf.judges.llm_judge.call_judge_llm`` as a starting point.
        n_judge_calls: Number of judge calls per prompt.  Must be >= 2.
        agreement_threshold: Fraction of ``"finetuned"`` votes required to
            consider the result ``consistent``.  Default 0.66 ≈ 2 out of 3.

    Returns:
        A list of ``EvalResult`` objects, one per prompt.
    """
    import re

    n_judge_calls = max(n_judge_calls, 2)
    results: list[EvalResult] = []

    for prompt in human_prompts:
        try:
            base_resp = base_model_call(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[RLHF/evaluate] base_model_call failed: %s", exc)
            base_resp = ""

        try:
            ft_resp = finetuned_model_call(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[RLHF/evaluate] finetuned_model_call failed: %s", exc)
            ft_resp = ""

        votes: list[str] = []
        for call_idx in range(n_judge_calls):
            swapped = (call_idx % 2 == 1)
            if swapped:
                resp_a, resp_b = ft_resp, base_resp
            else:
                resp_a, resp_b = base_resp, ft_resp

            try:
                raw = llm_judge_call(prompt, resp_a, resp_b)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[RLHF/evaluate] llm_judge_call failed: %s", exc)
                votes.append("tie")
                continue

            # Parse verdict — normalise to "base" / "finetuned" / "tie"
            match = re.search(r'\b([ABab]|tie)\b', raw or "")
            if not match:
                votes.append("tie")
                continue
            raw_verdict = match.group(1).lower()
            if raw_verdict == "tie":
                votes.append("tie")
            elif (raw_verdict == "a" and not swapped) or (raw_verdict == "b" and swapped):
                votes.append("base")
            else:
                votes.append("finetuned")

        total_non_tie = sum(1 for v in votes if v != "tie") or 1  # avoid div-zero
        finetuned_frac = sum(1 for v in votes if v == "finetuned") / n_judge_calls
        consistent = finetuned_frac >= agreement_threshold

        results.append(EvalResult(
            prompt=prompt,
            reward_margin=None,  # computed separately in run_full_evaluation
            judge_votes=votes,
            consistent=consistent,
            needs_review=not consistent,
        ))

    return results


# ---------------------------------------------------------------------------
# 3. Full evaluation report
# ---------------------------------------------------------------------------

def run_full_evaluation(
    human_prompts: list[str],
    labeled_pairs: list,           # list[PreferencePair] — avoid circular import
    base_model: Any,
    finetuned_model: Any,
    tokenizer: Any,
    base_model_call: Callable[[str], str],
    finetuned_model_call: Callable[[str], str],
    llm_judge_call: Callable[[str, str, str], str],
    n_judge_calls: int = 3,
    agreement_threshold: float = 0.66,
) -> dict:
    """Run the full two-signal evaluation and return a combined report.

    Signal 1 — Reward margin:
        Computed via ``compute_reward_margin`` for every ``labeled_pairs``
        entry.  Average across all pairs gives one scalar.

    Signal 2 — Human-prompt consistency:
        Computed via ``human_prompt_consistency_check`` for every prompt in
        ``human_prompts``.

    The top-level ``pass`` key is only True when:
      - The average reward margin is positive (fine-tuned model is
        directionally better on the training distribution), AND
      - No prompt in ``human_prompts`` is flagged for review
        (fine-tuned model is consistently preferred).

    Safety note:
        ``pass: True`` is a necessary but NOT sufficient condition for
        deployment.  A human MUST inspect ``flagged_for_human_review`` before
        any model swap.  This function should never be wired to auto-deploy.

    Args:
        human_prompts: List of raw prompts for consistency checking.
        labeled_pairs: List of ``PreferencePair`` objects with ``chosen``
            set (used for reward-margin computation).
        base_model: The reference causal LM (torch model).
        finetuned_model: The DPO fine-tuned causal LM.
        tokenizer: Shared tokenizer for both models.
        base_model_call: Callable generating base model responses.
        finetuned_model_call: Callable generating fine-tuned responses.
        llm_judge_call: Callable for LLM judging (see
            ``human_prompt_consistency_check`` for signature).
        n_judge_calls: Number of judge calls per prompt.
        agreement_threshold: Fraction of votes required for consistency.

    Returns:
        A dict with keys:
          - ``average_reward_margin`` (float | None)
          - ``consistency_results`` (list of ``EvalResult``)
          - ``flagged_for_human_review`` (list of prompt strings)
          - ``pass`` (bool)
    """
    # --- Reward margin ---
    margins: list[float] = []
    for pair in labeled_pairs:
        if pair.chosen not in ("a", "b"):
            continue
        chosen_text   = pair.response_a.text if pair.chosen == "a" else pair.response_b.text
        rejected_text = pair.response_b.text if pair.chosen == "a" else pair.response_a.text
        try:
            m = compute_reward_margin(
                base_model, finetuned_model, tokenizer,
                pair.prompt, chosen_text, rejected_text,
            )
            margins.append(m)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[RLHF/evaluate] reward_margin failed for pair %s: %s",
                           pair.pair_id, exc)

    avg_margin: Optional[float] = (sum(margins) / len(margins)) if margins else None

    # --- Consistency check ---
    consistency_results = human_prompt_consistency_check(
        human_prompts=human_prompts,
        base_model_call=base_model_call,
        finetuned_model_call=finetuned_model_call,
        llm_judge_call=llm_judge_call,
        n_judge_calls=n_judge_calls,
        agreement_threshold=agreement_threshold,
    )

    flagged = [r.prompt for r in consistency_results if r.needs_review]

    # Overall pass: positive margin AND clean review queue.
    passed = bool(avg_margin is not None and avg_margin > 0 and len(flagged) == 0)

    logger.info(
        "[RLHF/evaluate] avg_reward_margin=%.4f  flagged=%d  pass=%s",
        avg_margin or 0.0, len(flagged), passed,
    )

    return {
        "average_reward_margin": avg_margin,
        "consistency_results": consistency_results,
        "flagged_for_human_review": flagged,
        "pass": passed,
    }
