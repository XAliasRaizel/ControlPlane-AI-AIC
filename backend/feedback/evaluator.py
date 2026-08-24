"""Feedback evaluator (Section 5.11 — stretch Phase 7).

Computes FPR/FNR from HumanReviewOutcome history and suggests threshold
changes.  Stub implementation for now.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("controlplane.feedback")


class FeedbackEvaluator:
    """Analyzes human overrides to improve detection thresholds.

    Every HumanReviewOutcome where final_action != original_action is a
    labeled error.  This module logs it explicitly as a false positive or
    false negative at write time.

    TODO (Phase 7): compute FPR/FNR per detector, surface suggested
    threshold changes.
    """

    def record_override(
        self,
        request_id: str,
        original_action: str,
        final_action: str,
        notes: str = "",
    ) -> dict[str, Any]:
        if original_action == final_action:
            logger.debug("No override for %s — actions match", request_id)
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
        return {
            "request_id": request_id,
            "error_type": error_type,
            "original_action": original_action,
            "final_action": final_action,
            "notes": notes,
        }
