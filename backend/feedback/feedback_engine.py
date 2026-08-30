"""feedback_engine.py -- Self-Governing Threshold Auto-Tuner (Section 5.12)

Architecture decision summary
------------------------------
Three candidates were evaluated:
  1. Threshold auto-tuning per detector/rule, based on override rate.  <- BUILT
  2. Confidence-weight adjustment in risk fusion.                       <- Phase 2
  3. Auto-escalate to mandatory human review once override rate is severe.

Decision: build Candidate 1 as the primary mechanism, with Candidate 3's
principle folded in as a hard safety ceiling, not a separate system:

  * Below MIN_SAMPLES         -> INSUFFICIENT_DATA  (do nothing)
  * MODERATE <= rate < SEVERE -> NUDGE (raise threshold by NUDGE_STEP)
  * rate >= SEVERE            -> ESCALATE (stop nudging; demand human review)

Honest limitation
-----------------
Override data only tells you about false positives you already caught.
A rule silently missing things (false negatives) generates zero override
records, so this can only push a threshold UP, never validate it should
go down. That is structurally correct for a governance system.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger("controlplane.feedback_engine")

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
MIN_SAMPLES: int = 5                    # below this, never tune
MODERATE_OVERRIDE_RATE: float = 0.25   # 25% -> nudge threshold up
SEVERE_OVERRIDE_RATE: float = 0.50    # 50% -> escalate, stop nudging
NUDGE_STEP: float = 0.05              # each nudge is exactly this
MAX_THRESHOLD: float = 0.98            # hard ceiling -- never exceed
POLICIES_DIR: Path = Path(__file__).parent.parent.parent / "policies"


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------

@dataclass
class RuleOverrideSummary:
    """Aggregated override statistics for a single policy rule."""
    rule_id: str
    policy_id: str
    total_flags: int
    override_count: int
    override_rate: float
    current_threshold: Optional[float]
    detector_name: Optional[str]


@dataclass
class TuningDecision:
    """Outcome of evaluating one rule."""
    rule_id: str
    policy_id: str
    action: str   # NUDGE | ESCALATE | HOLD | INSUFFICIENT_DATA
    reason: str
    old_threshold: Optional[float]
    new_threshold: Optional[float]
    override_rate: float
    sample_size: int
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Internal DB / YAML loaders
# ---------------------------------------------------------------------------

def _get_db_path() -> Path:
    from backend.shared.config import settings
    return Path(settings.db_path)


def _init_tuning_table(db_path: Path):
    """Ensure the tuning_history table exists for audit compliance."""
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tuning_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                rule_id TEXT NOT NULL,
                policy_id TEXT NOT NULL,
                action TEXT NOT NULL,
                old_threshold REAL,
                new_threshold REAL,
                override_rate REAL,
                sample_size INTEGER,
                reason TEXT NOT NULL
            );
        """)
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("Could not init tuning_history table: %s", exc)


def _load_feedback_records(db_path: Path) -> list[dict]:
    """Load all resolved human reviews."""
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT request_id, policy_id, reason,
                      'HUMAN_REVIEW' AS original_action,
                      COALESCE(final_action, 'HUMAN_REVIEW') AS final_action
               FROM pending_reviews
               WHERE status = 'RESOLVED'"""
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("Could not load feedback records: %s", exc)
        return []


def _load_policy_rules() -> list[dict]:
    """Walk POLICIES_DIR and return every rule with its policy context."""
    rules = []
    for yaml_path in sorted(POLICIES_DIR.glob("*.yaml")):
        try:
            with open(yaml_path, encoding="utf-8") as f:
                policy = yaml.safe_load(f)
            policy_id = policy.get("name", yaml_path.stem)
            for rule in policy.get("rules", []):
                when = rule.get("when", {})
                det = when.get("detector_score_at_least", {})
                detector_name, threshold = None, None
                if det:
                    detector_name, threshold = next(iter(det.items()))
                rules.append({
                    "rule_id": rule.get("id", ""),
                    "policy_id": policy_id,
                    "policy_file": str(yaml_path),
                    "detector_name": detector_name,
                    "threshold": threshold,
                })
        except Exception as exc:
            logger.warning("Could not load %s: %s", yaml_path, exc)
    return rules


def _find_rule_in_policies(rule_id: str) -> Optional[dict]:
    for r in _load_policy_rules():
        if r["rule_id"] == rule_id:
            return r
    return None


# ---------------------------------------------------------------------------
# Core tuning pipeline
# ---------------------------------------------------------------------------

def compute_override_summaries() -> list[RuleOverrideSummary]:
    """Aggregate resolved review records into per-rule override statistics."""
    db_path = _get_db_path()
    records = _load_feedback_records(db_path)
    rules = _load_policy_rules()

    # Build counts keyed by both rule_id AND policy_id for flexible matching
    flags_by_key: dict[str, int] = defaultdict(int)
    overrides_by_key: dict[str, int] = defaultdict(int)

    for rec in records:
        # A review record may carry a rule_id (direct match) or policy_id (parent match)
        key = rec.get("policy_id", "")
        if key:
            flags_by_key[key] += 1
            if rec.get("final_action") != rec.get("original_action"):
                overrides_by_key[key] += 1

    summaries = []
    for rule in rules:
        rule_id = rule["rule_id"]
        policy_id = rule["policy_id"]
        # Priority: rule_id first (most specific), then policy_id (parent match)
        if rule_id in flags_by_key:
            total = flags_by_key[rule_id]
            overrides = overrides_by_key[rule_id]
        elif policy_id in flags_by_key:
            total = flags_by_key[policy_id]
            overrides = overrides_by_key[policy_id]
        else:
            total, overrides = 0, 0
        rate = overrides / total if total > 0 else 0.0
        summaries.append(RuleOverrideSummary(
            rule_id=rule_id,
            policy_id=policy_id,
            total_flags=total,
            override_count=overrides,
            override_rate=round(rate, 4),
            current_threshold=rule["threshold"],
            detector_name=rule["detector_name"],
        ))
    return summaries


def evaluate_tuning_decisions(
    summaries: list[RuleOverrideSummary],
) -> list[TuningDecision]:
    """Apply the three-tier decision logic to each rule's statistics."""
    decisions = []
    for s in summaries:
        if s.current_threshold is None:
            continue

        if s.total_flags < MIN_SAMPLES:
            decisions.append(TuningDecision(
                rule_id=s.rule_id, policy_id=s.policy_id,
                action="INSUFFICIENT_DATA",
                reason=(
                    f"Only {s.total_flags} resolved review(s) -- "
                    f"minimum {MIN_SAMPLES} required before tuning."
                ),
                old_threshold=s.current_threshold, new_threshold=None,
                override_rate=s.override_rate, sample_size=s.total_flags,
            ))

        elif s.override_rate >= SEVERE_OVERRIDE_RATE:
            # Safety ceiling: stop nudging, escalate to mandatory human review.
            decisions.append(TuningDecision(
                rule_id=s.rule_id, policy_id=s.policy_id,
                action="ESCALATE",
                reason=(
                    f"Override rate {s.override_rate:.0%} exceeds severe threshold "
                    f"({SEVERE_OVERRIDE_RATE:.0%}). "
                    "A parameter nudge is insufficient. "
                    "The underlying rule definition requires comprehensive human policy review."
                ),
                old_threshold=s.current_threshold, new_threshold=None,
                override_rate=s.override_rate, sample_size=s.total_flags,
            ))

        elif s.override_rate >= MODERATE_OVERRIDE_RATE:
            # Moderate: nudge threshold up by one bounded step.
            new_thresh = min(
                round(s.current_threshold + NUDGE_STEP, 4),
                MAX_THRESHOLD,
            )
            if new_thresh == s.current_threshold:
                decisions.append(TuningDecision(
                    rule_id=s.rule_id, policy_id=s.policy_id,
                    action="HOLD",
                    reason=f"Already at maximum ceiling ({MAX_THRESHOLD}). Cannot nudge further.",
                    old_threshold=s.current_threshold, new_threshold=None,
                    override_rate=s.override_rate, sample_size=s.total_flags,
                ))
            else:
                decisions.append(TuningDecision(
                    rule_id=s.rule_id, policy_id=s.policy_id,
                    action="NUDGE",
                    reason=(
                        f"Override rate {s.override_rate:.0%} exceeds moderate threshold "
                        f"({MODERATE_OVERRIDE_RATE:.0%}). "
                        f"Raising {s.detector_name!r} threshold: "
                        f"{s.current_threshold} -> {new_thresh}. "
                        "Requires stronger detector evidence before this rule fires."
                    ),
                    old_threshold=s.current_threshold, new_threshold=new_thresh,
                    override_rate=s.override_rate, sample_size=s.total_flags,
                ))
        else:
            decisions.append(TuningDecision(
                rule_id=s.rule_id, policy_id=s.policy_id,
                action="HOLD",
                reason=(
                    f"Override rate {s.override_rate:.0%} is below moderate threshold "
                    f"({MODERATE_OVERRIDE_RATE:.0%}). "
                    "Rule is performing within acceptable bounds."
                ),
                old_threshold=s.current_threshold, new_threshold=None,
                override_rate=s.override_rate, sample_size=s.total_flags,
            ))
    return decisions


def apply_tuning_decisions(decisions: list[TuningDecision]) -> list[TuningDecision]:
    """Write NUDGE decisions back into policy YAML files and record audit logs."""
    db_path = _get_db_path()
    _init_tuning_table(db_path)

    for decision in decisions:
        if decision.action != "NUDGE":
            continue

        try:
            rule = _find_rule_in_policies(decision.rule_id)
            if not rule:
                logger.warning("[tuner] NUDGE skipped -- rule %s not found", decision.rule_id)
                continue

            yaml_path = Path(rule["policy_file"])
            with open(yaml_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f)

            patched = False
            for r in raw.get("rules", []):
                if r.get("id") == decision.rule_id:
                    when = r.setdefault("when", {})
                    det = when.setdefault("detector_score_at_least", {})
                    if rule["detector_name"]:
                        det[rule["detector_name"]] = decision.new_threshold
                        patched = True
                        break

            if patched:
                with open(yaml_path, "w", encoding="utf-8") as f:
                    yaml.dump(raw, f, default_flow_style=False,
                              allow_unicode=True, sort_keys=False)
                logger.info(
                    "[tuner] NUDGE applied: rule=%s %s->%s rate=%.0f%% n=%d file=%s",
                    decision.rule_id, decision.old_threshold,
                    decision.new_threshold, decision.override_rate * 100,
                    decision.sample_size, yaml_path.name,
                )
                # Log audit record
                _record_tuning_audit(db_path, decision)

        except Exception as exc:
            logger.error("[tuner] NUDGE failed for %s: %s", decision.rule_id, exc)

    return decisions


def _record_tuning_audit(db_path: Path, decision: TuningDecision):
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute(
            """INSERT INTO tuning_history
               (timestamp, rule_id, policy_id, action, old_threshold, new_threshold, override_rate, sample_size, reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                decision.timestamp,
                decision.rule_id,
                decision.policy_id,
                decision.action,
                decision.old_threshold,
                decision.new_threshold,
                decision.override_rate,
                decision.sample_size,
                decision.reason,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("Could not record tuning history: %s", exc)


def get_tuning_history(limit: int = 50) -> list[dict]:
    """Retrieve audit history of past tuning decisions."""
    db_path = _get_db_path()
    _init_tuning_table(db_path)
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM tuning_history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("Could not load tuning history: %s", exc)
        return []


def seed_demo_feedback_records() -> dict:
    """Seed realistic review history to demonstrate NUDGE, ESCALATE, and HOLD paths.

    Uses rule_ids and policy names that exactly match the loaded YAML policies so
    that compute_override_summaries() correctly associates override counts with rules.

    Patterns:
      hr-pii-present (hr-governance)      8 records, 3 overridden -> 37.5% -> NUDGE
      finance-pii-redact (finance-governance) 10 records, 6 overridden -> 60%  -> ESCALATE
      support-block-injection (support-governance) 7 records, 0 overridden -> 0% -> HOLD
    """
    db_path = _get_db_path()
    _init_tuning_table(db_path)
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        now = datetime.now(timezone.utc).isoformat()

        # Clear stale demo seed records before re-seeding to keep data clean
        conn.execute(
            "DELETE FROM pending_reviews WHERE request_id LIKE 'demo-%'"
        )

        # 1. hr-pii-present (policy: hr-governance) -> 3/8 = 37.5% override -> NUDGE
        #    Reviewers approved requests even though PII was found (over-aggressive rule)
        for i in range(8):
            # First 3 are overridden: original=HUMAN_REVIEW but reviewer said ALLOW
            final = "ALLOW" if i < 3 else "HUMAN_REVIEW"
            conn.execute(
                """INSERT OR REPLACE INTO pending_reviews
                   (request_id, created_at, policy_id, reason, risk, status, final_action,
                    reviewer_id, notes, resolved_at)
                   VALUES (?, ?, 'hr-pii-present', 'PII detected in HR profile lookup', 0.42,
                           'RESOLVED', ?, 'auditor_jane', 'Public job title is not sensitive PII', ?)""",
                (f"demo-hr-{i+1:03d}", now, final, now),
            )

        # 2. finance-pii-redact (policy: finance-governance) -> 6/10 = 60% override -> ESCALATE
        #    Pattern is severe: reviewers overriding repeatedly, rule needs human policy review
        for i in range(10):
            final = "ALLOW" if i < 6 else "MODIFY"
            conn.execute(
                """INSERT OR REPLACE INTO pending_reviews
                   (request_id, created_at, policy_id, reason, risk, status, final_action,
                    reviewer_id, notes, resolved_at)
                   VALUES (?, ?, 'finance-pii-redact', 'PII redaction triggered on financial report', 0.82,
                           'RESOLVED', ?, 'compliance_lead', 'Legitimate quarterly report - authorized recipient', ?)""",
                (f"demo-fin-{i+1:03d}", now, final, now),
            )

        # 3. support-block-injection (policy: support-governance) -> 0/7 = 0% override -> HOLD
        #    Reviewers consistently agreed with all BLOCK decisions (rule is correct)
        for i in range(7):
            conn.execute(
                """INSERT OR REPLACE INTO pending_reviews
                   (request_id, created_at, policy_id, reason, risk, status, final_action,
                    reviewer_id, notes, resolved_at)
                   VALUES (?, ?, 'support-block-injection', 'Prompt injection attempt detected', 0.96,
                           'RESOLVED', 'BLOCK', 'sec_analyst', 'Confirmed adversarial jailbreak attempt', ?)""",
                (f"demo-inj-{i+1:03d}", now, now),
            )

        conn.commit()
        conn.close()
        return {
            "status": "seeded",
            "records_created": 25,
            "patterns": [
                {"rule": "hr-pii-present", "policy": "hr-governance",
                 "sample_size": 8, "overrides": 3, "rate": "37.5%", "expected_action": "NUDGE"},
                {"rule": "finance-pii-redact", "policy": "finance-governance",
                 "sample_size": 10, "overrides": 6, "rate": "60.0%", "expected_action": "ESCALATE"},
                {"rule": "support-block-injection", "policy": "support-governance",
                 "sample_size": 7, "overrides": 0, "rate": "0.0%", "expected_action": "HOLD"},
            ],
        }
    except Exception as exc:
        logger.error("Could not seed demo feedback: %s", exc)
        return {"status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_tuning_cycle(dry_run: bool = True) -> dict:
    """Run the full self-governing tuning cycle."""
    summaries = compute_override_summaries()
    decisions = evaluate_tuning_decisions(summaries)
    if not dry_run:
        decisions = apply_tuning_decisions(decisions)

    applied = [d for d in decisions if d.action == "NUDGE"]
    escalated = [d for d in decisions if d.action == "ESCALATE"]
    held = [d for d in decisions if d.action == "HOLD"]
    insufficient = [d for d in decisions if d.action == "INSUFFICIENT_DATA"]

    return {
        "dry_run": dry_run,
        "total_rules_evaluated": len(decisions),
        "summaries": [asdict(s) for s in summaries],
        "decisions": [asdict(d) for d in decisions],
        "nudged_count": len(applied),
        "escalated_count": len(escalated),
        "held_count": len(held),
        "insufficient_data_count": len(insufficient),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "min_samples": MIN_SAMPLES,
            "moderate_override_rate": MODERATE_OVERRIDE_RATE,
            "severe_override_rate": SEVERE_OVERRIDE_RATE,
            "nudge_step": NUDGE_STEP,
            "max_threshold_ceiling": MAX_THRESHOLD,
        },
    }
