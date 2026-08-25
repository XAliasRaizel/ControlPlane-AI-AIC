# Architecture and execution model

## Request lifecycle

1. The client submits a `GovernanceRequest` to `POST /v1/govern`.
2. The gateway assigns a request ID and concurrently runs all hot-path
   detectors with `asyncio.gather`.
3. The risk engine combines normalized signals using fixed, documented weights.
4. The policy engine evaluates every file in `policies/` (currently
   `global.yaml`, `hr.yaml`, `finance.yaml`, `support.yaml`), most-specific
   scope first (application > department > global), each in priority order.
5. The decision engine returns the required action and, when appropriate,
   sanitizes the candidate response.
6. The gateway writes privacy-conscious audit metadata and enqueues a separate
   deep-analysis job. It does not wait for that job before replying.
7. A reviewer can submit feedback linked to the request ID.

## Hot path contract

Each detector returns the same contract:

```json
{
  "detector": "prompt_injection",
  "score": 0.9,
  "severity": "CRITICAL",
  "triggered": true,
  "evidence": ["instruction_override"],
  "confidence": 0.95
}
```

`evidence` contains categories or reasons, never a copied raw prompt or a PII
value. The current detectors are deterministic, CPU-friendly baseline
implementations. `GPUAdapter` is an intentionally stable seam for a trained
detector; production model inference should be moved behind an isolated,
versioned service with timeouts and circuit breaking.

## Risk fusion

The aggregate score is a severity floor blended with a weighted signal, then
escalated by context:

```
severity_floor  = max(privacy, injection, safety, authorization, fairness)
blended         = 0.25 privacy + 0.25 injection + 0.15 safety
                + 0.25 authorization + 0.10 fairness
base_risk       = max(severity_floor, blended)
overall_risk    = min(1.0, base_risk * (1.15 if critical_context else 1.0))
```

A single maximal-confidence finding (e.g. a confirmed authorization denial)
now sets the risk floor directly, instead of being diluted by unrelated
clean detectors — a straight weighted average previously let a
100%-confidence unauthorized-access case land around 0.46, which meant the
risk-threshold HUMAN_REVIEW rules could almost never fire for the single
most common violation shape in this system. Policy rules that check raw
detector scores directly (rather than overall_risk) are unaffected either
way — an explicit authorization failure or injection match still blocks
regardless of the aggregate. Async fairness, grounding, cost and
performance assessments are recorded after the response and are not
treated as real-time enforcement signals.

## Operational transition

For a production deployment replace in-process background tasks with a durable
queue and independently scaled workers. Keep model-provider proxying separate
from governance evaluation, use authenticated identity rather than caller
claims, protect audit stores with encryption and retention controls, and stage
policy/model changes in observe-only mode before enforcing them.
