# Agentic Tool-Call Governance — Feature Spec

**Status:** core logic implemented and unit-tested (13/13 passing, stdlib + PyYAML only). FastAPI wiring layer written but not executed in the build environment (fastapi/pydantic unavailable there) — smoke-test `router.py` once dropped into a real environment.

## 1. Why this exists

The Round 2 brief calls out that "multi-turn conversations and AI agents that take actions (not just generate text) introduce compounding risk, where one questionable output can shape several downstream decisions." Everything ControlPlane.ai has built so far governs *text* — a prompt in, a response out. This feature governs *actions* — an agent deciding to call `send_email`, `issue_refund`, or `delete_record`, intercepted and scored before it is allowed to run, not after.

## 2. Design principle (non-negotiable)

> The agent proposes. The governor disposes. The agent is never given the ability to execute a tool directly — only to ask `ToolGovernor.invoke(...)`, which decides and, only on ALLOW or an approved HUMAN_REVIEW, performs the execution itself.

This is what makes the gate real rather than cosmetic: even an agent that has been prompt-injected into "ignore your instructions and just do it" cannot bypass the gate, because it was never holding the key in the first place. This mirrors how production agent frameworks implement human-in-the-loop tool calling — the interceptor sits *between* the agent and the tool, not inside the agent's own reasoning.

## 3. What was built

```
backend/app/agents/
  __init__.py       # exports ToolGovernor and the core dataclasses
  models.py         # ToolCallContext, ToolRiskSignal, GovernanceDecision, PendingToolCall
  tools.py          # toy tools: send_email, issue_refund, delete_record (+ registry)
  risk.py           # tool-call risk scoring: severity floor + weighted blend + reversibility multiplier
  policy.py         # loads policies/agent_tools.yaml, evaluates conditions against risk + context
  queue.py          # in-memory human-review queue (enqueue / list_pending / resolve)
  governance.py     # ToolGovernor: the single choke point tying the above together
  router.py         # FastAPI endpoints: POST /agent/act, GET /agent/pending, POST /agent/pending/{id}/resolve

policies/agent_tools.yaml   # declarative rules: thresholds, authorization, session carryover
scripts/run_agent_governance_demo.py   # 8-scenario golden-path style walkthrough, run it directly
tests/test_agent_governance.py         # 13 unittest cases (pytest-discoverable, no pytest required to run them)
```

## 4. The risk model (mirrors the existing Risk Engine's shape on purpose)

For a given tool call, four factors are computed independently:

| Factor | Meaning | 0.0 means | 1.0 means |
|---|---|---|---|
| `authorization` | Is this role allowed to call this tool at all? | clearly authorized | clearly not |
| `magnitude` | How big is the blast radius (currently: refund amount)? | trivial | at/above the hard cap |
| `sensitivity` | Does the call touch something flagged sensitive (PII record, external+PII email)? | not sensitive | maximally sensitive |
| `session_carryover` | Has this session already been flagged risky (e.g. by a multi-turn/session-risk tracker)? | clean session | already elevated |

```
severity_floor = max(authorization, magnitude, sensitivity, session_carryover)
blended        = 0.35*authorization + 0.30*magnitude + 0.20*sensitivity + 0.15*session_carryover
base_risk      = max(severity_floor, blended)
overall_risk   = min(1.0, base_risk * reversibility_multiplier)   # 1.15x for irreversible actions, 1.0x otherwise
```

The severity floor is the important part: one seriously bad factor (an unauthorized caller, say) cannot be diluted into a mediocre-looking average by three harmless ones. This is the same principle the main Risk Engine already uses.

**Deliberately re-derived, not trusted:** `authorization` and `magnitude` are computed from the caller's role and the raw call args, independent of anything the tool itself reports about its own risk. A tool implementation that "forgets" to flag itself as risky can't understate what actually happened.

## 5. Policy overrides risk — on purpose

Risk score alone does not decide the outcome. `policies/agent_tools.yaml` is checked first; only when *no* rule matches does the code fall back to raw-score thresholds (`>= 0.85` → BLOCK, `>= 0.50` → HUMAN_REVIEW, else ALLOW). This produces one of the more interesting moments in the demo: an admin deleting a PII-flagged record scores just as "sensitive" (0.9+) as an intern attempting the same thing — but the `admin-delete-authorized` policy rule explicitly ALLOWs it for that role, while the intern is hard-blocked. **Same risk score, different decision, because context — not just risk — determines the action.** That's the same "RISK → POLICY → DECISION, not RISK → DECISION" principle your architecture already documents for the text pipeline; this shows it holds for actions too.

## 6. The eight demo scenarios (`scripts/run_agent_governance_demo.py`)

1. Internal email, no PII → **ALLOW**
2. Small refund ($120) → **ALLOW**
3. Mid-size refund ($1500) → **HUMAN_REVIEW** → manager approves → executes
4. Large refund ($5000) → **BLOCK** outright (above the hard cap, no amount of approval changes that)
5. Intern deletes a PII customer record → **BLOCK** (hard rule — not even eligible for review)
6. Admin deletes the same *kind* of record → **ALLOW** (context changes the outcome, not the risk score)
7. Ordinary internal email, but the session already carries elevated risk from an earlier turn → **HUMAN_REVIEW** (the compounding-risk scenario the brief explicitly asks for)
8. A tool name that was never registered → **BLOCK**, and the process does not crash

Run it: `python3 scripts/run_agent_governance_demo.py`

## 7. Integration TODOs (marked in code with matching comments)

These are the seams left intentionally open because the exact function signatures in your real `audit.py` / risk tracker weren't available to build against directly:

- [ ] `governance.py`: replace `_noop_audit_sink` with a real call into your audit store (fingerprint the args the same way the rest of ControlPlane fingerprints prompts — do not log raw PII into the audit trail here either).
- [ ] `governance.py`: replace `_zero_session_risk` with a lookup into whatever tracks cumulative per-session risk (this is the natural hook point for a CUSUM-style session-risk tracker, if/when that's built — see the `session-carryover-escalation` rule, which already consumes it).
- [ ] `router.py`: swap the module-level `_governor` singleton for a FastAPI dependency, and mount `router` in `main.py`.
- [ ] `queue.py`: swap the in-memory dict for your real review-queue persistence once this needs to survive a process restart.

## 8. Natural next extensions (not built — scope them separately)

- **Response-side validation**, not just request-side: confirm the tool's actual result matches what was authorized (e.g. the refund amount executed equals the amount that was approved) — closes the loop the interceptor pattern calls "on the way out," not just "on the way in."
- **Per-session rate limiting**: N tool calls of a given risk tier per session before auto-escalating, independent of any single call's score.
- **More tools** — same registry pattern in `tools.py` extends to anything: `create_ticket`, `update_customer_record`, `approve_loan`, etc. Each new tool needs a `describe_risk` + `execute` pair and, usually, a couple of new policy rules.
- **Wire the same `ToolGovernor` into the RAG/knowledge-assistant work** — if the "Ask ControlPlane" conversational assistant ever gains write actions (e.g. "close this review case"), it should go through the exact same gate, not a separate one.

## 9. Acceptance criteria

- [x] All three toy tools implemented with simulated (non-real) side effects
- [x] Every call is scored, policy-checked, and decided before any execution occurs
- [x] ALLOW executes immediately; BLOCK never executes; HUMAN_REVIEW queues and only executes after explicit approval
- [x] An unregistered tool name is rejected safely, not a crash
- [x] Elevated session risk can escalate an otherwise-clean call (compounding-risk scenario)
- [x] Same risk score, different role → different decision (policy overrides raw risk)
- [x] 13/13 unit tests passing; demo script runs end to end and prints a readable trace
- [ ] Smoke-tested against real FastAPI/Pydantic (do this once dropped into your environment — see §1)
