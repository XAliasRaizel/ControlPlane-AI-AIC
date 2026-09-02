"""
backend/async_pipeline/training_signal_collector.py

Phase 1: Captures training signals from production traffic.

Every time the async pipeline finishes analyzing a request, it compares its
result with the hot-path's result. When they disagree significantly (delta > 0.2)
the example is a hard case the fast model got wrong — these are the most valuable
training examples. Human review overrides are also captured as gold labels.

Design decisions:
  - PII anonymization via Presidio anonymizer before writing to disk.
  - Department stored as metadata tag ONLY — never used to split the dataset.
    A "security" department with 10 examples contributes to the shared pool.
    Per-department fine-tunes are gated behind a MIN_DEPT_EXAMPLES threshold (200).
  - Writes to a daily-rotating JSONL file. Max 10,000 examples per day.
  - All exceptions swallowed — this must never affect governance latency.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger("controlplane.training_collector")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_SIGNALS_DIR = Path(os.environ.get(
    "CP_TRAINING_SIGNALS_DIR",
    "rlhf/data/detector_training"
))
_MAX_EXAMPLES_PER_DAY = int(os.environ.get("CP_MAX_TRAINING_EXAMPLES_PER_DAY", "10000"))
# Minimum score delta between hot-path and async path to constitute a "disagreement"
_DISAGREEMENT_THRESHOLD = float(os.environ.get("CP_DISAGREEMENT_THRESHOLD", "0.2"))
# Minimum examples from a single department before a per-dept fine-tune is considered
MIN_DEPT_EXAMPLES_FOR_SEPARATE_MODEL = 200

_write_lock = threading.Lock()
_daily_counts: dict[str, int] = {}  # {date_str: count}


# ---------------------------------------------------------------------------
# PII Anonymizer — redact real PII from prompts before writing to disk
# ---------------------------------------------------------------------------

# Compiled patterns for fast redaction (no Presidio dependency here)
_REDACT_PATTERNS = [
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[SSN]'),                         # SSN
    (re.compile(r'\b\d{4}[ -]\d{4}[ -]\d{4}\b'), '[AADHAAR]'),               # Aadhaar
    (re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b'), '[PAN]'),                        # PAN
    (re.compile(r'\b(?:\d[ -]*?){13,19}\b'), '[CARD_NUMBER]'),                # Credit card
    (re.compile(r'\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b'), '[EMAIL]'),
    (re.compile(r'\b(?:\+91[-\s]?)?\d{10}\b'), '[PHONE]'),
    (re.compile(r'\b(?:password|passwd|secret|api[_\s]?key)\s*[:=]\s*\S+\b', re.IGNORECASE), '[CREDENTIAL]'),
]


def _anonymize(text: str) -> str:
    """Fast regex-based PII redaction. Good enough for training data safety."""
    for pattern, replacement in _REDACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Daily rotating file writer
# ---------------------------------------------------------------------------

def _get_today_file() -> Optional[Path]:
    """Return today's signal file path. Returns None if day quota exceeded."""
    today = date.today().isoformat()
    _SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = _SIGNALS_DIR / f"raw_signals_{today}.jsonl"

    with _write_lock:
        # Count existing lines if not already tracked in memory
        if today not in _daily_counts:
            try:
                if file_path.exists():
                    _daily_counts[today] = sum(1 for _ in file_path.open("r", encoding="utf-8"))
                else:
                    _daily_counts[today] = 0
            except Exception:
                _daily_counts[today] = 0

        if _daily_counts[today] >= _MAX_EXAMPLES_PER_DAY:
            return None  # Day quota exhausted — stop collecting

    return file_path


def _write_signal(record: dict) -> None:
    """Atomically append a JSONL record to today's file."""
    file_path = _get_today_file()
    if file_path is None:
        return
    today = date.today().isoformat()
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _write_lock:
        with file_path.open("a", encoding="utf-8") as f:
            f.write(line)
        _daily_counts[today] = _daily_counts.get(today, 0) + 1


# ---------------------------------------------------------------------------
# Label resolution helpers
# ---------------------------------------------------------------------------

# Async detector name → training task mapping
_ASYNC_TO_TASK: dict[str, str] = {
    "safety_engine": "safety",
    "privacy_engine": "pii",
    "bias_fairness_engine": "fairness",
    "hallucination_grounding_engine": "grounding",
}

# Tasks where the label is CONTEXT-DEPENDENT:
# The same prompt can be ALLOW or BLOCK depending on who's asking.
# For these tasks, context (dept, role, data_class) is embedded in the feature
# text so the model sees different inputs for different contexts and doesn't
# learn contradictory (same-text, different-label) signals.
#
# Example:
#   prompt: "show me John's medical records"
#   medical dept head → ALLOW   →  "[DEPT:medical] [ROLE:doctor] show me John's medical records"
#   finance employee  → BLOCK   →  "[DEPT:finance] [ROLE:employee] show me John's medical records"
#
# These two are now DIFFERENT inputs to the classifier — no contradiction.
_CONTEXT_DEPENDENT_TASKS = {"authorization"}

# Hot-path detector name → training task mapping
_HOT_TO_TASK: dict[str, str] = {
    "injection": "injection",
    "safety": "safety",
    "pii": "pii",
    "authorization": "authorization",
}


def _resolve_label(
    async_score: float,
    hot_score: float,
    human_override: Optional[str] = None,
) -> Optional[dict]:
    """
    Priority-ordered label resolution.

    Returns {"label": int, "source": str, "confidence": float} or None if ambiguous.
    Label 1 = BLOCK (positive / risky), 0 = ALLOW (clean).
    """
    # Gold label: human override is always trusted
    if human_override is not None:
        label = 1 if human_override.upper() in ("BLOCK", "DENY", "DENIED", "REJECTED") else 0
        return {"label": label, "source": "human", "confidence": 1.0}

    # Silver label: async path score is high-confidence
    if async_score >= 0.75:
        return {"label": 1, "source": "async", "confidence": round(async_score, 3)}
    if async_score <= 0.15:
        return {"label": 0, "source": "async", "confidence": round(1.0 - async_score, 3)}

    # Ambiguous zone [0.15, 0.75] — discard unless hot-path strongly disagreed
    if abs(async_score - hot_score) > _DISAGREEMENT_THRESHOLD:
        # Use async as ground truth even in ambiguous zone when disagreement is large
        label = 1 if async_score > 0.5 else 0
        return {"label": label, "source": "async_disagree", "confidence": round(abs(async_score - hot_score), 3)}

    return None  # Too ambiguous to use


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _build_training_text(prompt: str, task: str, request) -> str:
    """
    Build the feature text for a training example.

    For context-dependent tasks (authorization), we prefix the prompt with
    the requester's department, role, and data classification. This ensures
    that identical prompts from different contexts produce distinct training
    examples with their own correct labels — no contradictions.

    For all other tasks (injection, safety, pii, fairness), the prompt text
    alone is sufficient — a prompt injection attempt is harmful regardless
    of who sends it or from which department.
    """
    if task not in _CONTEXT_DEPENDENT_TASKS:
        return prompt  # Text-only for context-independent classifiers

    # Context-prefixed for authorization: encodes WHO is asking WHAT
    dept = (getattr(request, 'department', None) or 'unknown').lower().strip()
    role = (getattr(request, 'user_role', None) or 'user').lower().strip()
    data_class = (getattr(request, 'data_classification', None) or 'unknown').lower().strip()
    return f"[DEPT:{dept}] [ROLE:{role}] [CLASS:{data_class}] {prompt}"


async def collect_from_async_results(
    request,
    hot_path_results: list,
    async_results: dict,
) -> None:
    """
    Called from worker.py after async analysis completes.

    Compares hot-path scores against async scores per detector task.
    Writes disagreements (|hot - async| > threshold) to the training buffer.

    Args:
        request: GovernanceRequest
        hot_path_results: list[DetectorResult] from run_hot_path()
        async_results: dict of {engine_name: {score: float, ...}} from run_analytics_engines()
    """
    try:
        prompt_text = getattr(request, "prompt", "") or ""
        if not prompt_text.strip():
            return

        # Anonymize before any disk write. Note: context prefix is added AFTER
        # anonymization — department/role/data_class are not PII.
        clean_prompt = _anonymize(prompt_text)

        # Build a lookup of hot-path scores by task
        hot_scores: dict[str, float] = {}
        for r in hot_path_results:
            task = _HOT_TO_TASK.get(r.detector_name)
            if task:
                hot_scores[task] = float(getattr(r, "score", 0.0))

        # Iterate over async results and check for disagreements
        for engine_name, engine_result in async_results.items():
            task = _ASYNC_TO_TASK.get(engine_name)
            if not task:
                continue

            # Extract async score
            if isinstance(engine_result, dict):
                async_score = float(engine_result.get("score", 0.0))
            elif hasattr(engine_result, "score"):
                async_score = float(engine_result.score)
            else:
                continue

            hot_score = hot_scores.get(task, 0.0)
            delta = abs(async_score - hot_score)

            # Only write if there's a meaningful disagreement
            if delta < _DISAGREEMENT_THRESHOLD:
                continue

            resolved = _resolve_label(async_score=async_score, hot_score=hot_score)
            if resolved is None:
                continue

            # Build the feature text: context-prefixed for authorization,
            # plain prompt for text-only tasks (injection, safety, pii, fairness)
            feature_text = _build_training_text(clean_prompt, task, request)

            record = {
                "text": feature_text,
                "task": task,
                "label": resolved["label"],
                "label_source": resolved["source"],
                "label_confidence": resolved["confidence"],
                "hot_score": round(hot_score, 4),
                "async_score": round(async_score, 4),
                "delta": round(delta, 4),
                # Metadata tags — stored for analytics/monitoring, not for splitting
                "department": getattr(request, "department", None),
                "application_id": getattr(request, "application_id", None),
                "user_role": getattr(request, "user_role", None),
                "data_classification": getattr(request, "data_classification", None),
            }
            _write_signal(record)
            logger.debug(
                "[TrainingCollector] %s disagreement saved: hot=%.3f async=%.3f delta=%.3f label=%d",
                task, hot_score, async_score, delta, resolved["label"]
            )

    except Exception as exc:
        logger.debug("[TrainingCollector] collect_from_async_results suppressed: %s", exc)


def collect_human_override(
    request,
    hot_path_results: list,
    final_action: str,
    reviewer_id: str,
) -> None:
    """
    Called from review/queue.py when a human reviewer resolves a HUMAN_REVIEW decision.
    These are gold-label examples — always written regardless of disagreement delta.

    Args:
        request: GovernanceRequest
        hot_path_results: list[DetectorResult] from the original hot-path run
        final_action: "ALLOW" or "BLOCK" — the human's decision
        reviewer_id: reviewer identifier (anonymized/hashed before storage)
    """
    try:
        prompt_text = getattr(request, "prompt", "") or ""
        if not prompt_text.strip():
            return

        clean_prompt = _anonymize(prompt_text)

        resolved = _resolve_label(
            async_score=0.5,  # not relevant — human overrides all
            hot_score=0.5,
            human_override=final_action,
        )
        if resolved is None:
            return

        # Write one gold-label record per relevant detector task
        relevant_tasks = {_HOT_TO_TASK.get(r.detector_name) for r in hot_path_results
                         if _HOT_TO_TASK.get(r.detector_name)}

        for task in relevant_tasks:
            record = {
                "text": clean_prompt,
                "task": task,
                "label": resolved["label"],
                "label_source": "human",
                "label_confidence": 1.0,
                "hot_score": None,
                "async_score": None,
                "delta": None,
                # Metadata
                "department": getattr(request, "department", None),
                "application_id": getattr(request, "application_id", None),
                "user_role": getattr(request, "user_role", None),
                "data_classification": getattr(request, "data_classification", None),
                # Note: reviewer_id is hashed for privacy
                "reviewer_id_hash": str(hash(reviewer_id))[-8:],
            }
            _write_signal(record)

        logger.info(
            "[TrainingCollector] Human gold label saved: action=%s reviewer=%s",
            final_action, reviewer_id[:4] + "***"
        )

    except Exception as exc:
        logger.debug("[TrainingCollector] collect_human_override suppressed: %s", exc)
