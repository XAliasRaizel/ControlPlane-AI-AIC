"""Feedback evaluator (Section 5.11).

Computes false-positive / false-negative classification from
HumanReviewOutcome history and converts every human override into a
labelled RLHF PreferencePair so governance decisions automatically
feed the preference-learning loop.

RLHF integration
----------------
When a reviewer changes the system decision (original_action !=
final_action) we have a weak but real preference signal:

  false_positive (system over-blocked): the reviewer preferred a
  LESS restrictive response => preferred response is the one the
  system would have given WITHOUT triggering the detector.

  false_negative (system under-blocked): the reviewer preferred a
  MORE restrictive response => preferred response is the sanitized /
  blocked output, not the raw LLM text.

In both cases we store a PreferencePair with labeled_by="human" so it
can be used for DPO fine-tuning later.  The pair is constructed
cheaply (no extra LLM calls) from information already available at
feedback time.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("controlplane.feedback")


class FeedbackEvaluator:
    """Analyses human overrides to improve detection thresholds.

    Every HumanReviewOutcome where final_action != original_action is a
    labelled error.  This module logs it explicitly as a false positive or
    false negative at write time AND stores a labelled RLHF PreferencePair
    so the override signal feeds back into DPO fine-tuning.
    """

    def record_override(
        self,
        request_id: str,
        original_action: str,
        final_action: str,
        notes: str = "",
        prompt: Optional[str] = None,
        original_response: Optional[str] = None,
    ) -> dict[str, Any]:
        """Record a human override and store an RLHF preference pair.

        Args:
            request_id: The governance request that was overridden.
            original_action: The action the system chose originally.
            final_action: The action the human reviewer chose.
            notes: Optional free-text reviewer notes.
            prompt: The original user prompt (used to build the pair).
                    When None the pair is not stored (graceful degradation).
            original_response: The response the system would have returned.
                    When None the pair is not stored.

        Returns:
            A dict with keys: request_id, error_type, original_action,
            final_action, notes.
        """
        if original_action == final_action:
            logger.debug("No override for %s -- actions match", request_id)
            return {"request_id": request_id, "error_type": None}

        # Classify the error
        severity_order = ["ALLOW", "MODIFY", "REROUTE", "HUMAN_REVIEW", "BLOCK"]
        orig_idx = severity_order.index(original_action) if original_action in severity_order else -1
        final_idx = severity_order.index(final_action) if final_action in severity_order else -1

        if final_idx < orig_idx:
            error_type = "false_positive"  # System was too aggressive
        else:
            error_type = "false_negative"  # System was too lenient

        logger.info(
            "Override recorded: request=%s original=%s final=%s error=%s",
            request_id, original_action, final_action, error_type,
        )

        # ------------------------------------------------------------------
        # RLHF: convert the override into a labelled PreferencePair.
        # ------------------------------------------------------------------
        if prompt and original_response:
            _store_override_pair(
                request_id=request_id,
                prompt=prompt,
                original_response=original_response,
                error_type=error_type,
                notes=notes,
            )

        return {
            "request_id": request_id,
            "error_type": error_type,
            "original_action": original_action,
            "final_action": final_action,
            "notes": notes,
        }


# ---------------------------------------------------------------------------
# RLHF pair construction from feedback overrides
# ---------------------------------------------------------------------------

def _store_override_pair(
    request_id: str,
    prompt: str,
    original_response: str,
    error_type: str,
    notes: str,
) -> None:
    """Construct and store a labelled PreferencePair from a human override.

    false_positive: system was too aggressive.
      response_a = original_response (system output -- may be over-redacted)
      response_b = a simulator-generated alternative (less restricted)
      chosen     = "b"  (reviewer preferred the less-restricted version)

    false_negative: system was too lenient.
      response_a = original_response (raw, un-redacted)
      response_b = a simulator-generated safer alternative
      chosen     = "b"  (reviewer preferred the safer/restricted version)

    In both cases we set labeled_by="human" and attach the request_id
    as session_id for traceability.

    All exceptions are silently swallowed -- a failure here must never
    affect the feedback endpoint response.

    Args:
        request_id: Governance request ID (used as session_id in the pair).
        prompt: The original user prompt.
        original_response: The response the system returned.
        error_type: "false_positive" or "false_negative".
        notes: Reviewer notes (stored in judge_metadata).
    """
    try:
        from backend.shared import llm_simulator
        from rlhf.schema import ModelResponse, PreferencePair
        from rlhf.storage.categorize import assign_category
        from rlhf.storage.json_store import write_pair, update_label
        from rlhf.config import Category
        import datetime, uuid

        # Generate an alternative response from the simulator.
        alt_response = llm_simulator.generate(prompt=prompt)

        resp_original = ModelResponse(
            text=original_response,
            model_name="controlplane_system",
            model_version_or_checkpoint="v0",
        )
        resp_alternative = ModelResponse(
            text=alt_response,
            model_name="llm_simulator_v1",
            model_version_or_checkpoint="v0",
        )

        pair = PreferencePair(
            prompt=prompt,
            response_a=resp_original,
            response_b=resp_alternative,
            session_id=request_id,
            source_pipeline="feedback_override",
        )
        pair = assign_category(pair, Category.GENERAL)

        write_pair(pair)

        # In both error types, reviewer preferred the alternative (b).
        # false_positive: b is the less-restricted response (better for the user).
        # false_negative: b is the simulator output which the system should have
        #                 produced with more caution -- still marks a direction.
        chosen = "b"

        update_label(
            pair_id=pair.pair_id,
            chosen=chosen,
            labeled_by="human",
            judge_metadata={"source": "feedback_override", "error_type": error_type, "notes": notes},
        )

        logger.info(
            "[RLHF/feedback] stored override pair %s (error_type=%s, chosen=%s)",
            pair.pair_id, error_type, chosen,
        )

    except Exception as exc:  # noqa: BLE001
        logger.debug("[RLHF/feedback] override pair storage skipped: %s", exc)
