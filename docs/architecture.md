# Architecture and execution model

## Request lifecycle

1. The client submits a `GovernanceRequest` to `POST /v1/govern`.
2. The gateway assigns a request ID and concurrently runs all hot-path
   detectors with `asyncio.gather`.
3. The risk engine combines normalized signals using fixed, documented weights.
4. The policy engine evaluates `policies/default.yaml` in priority order.
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

The current aggregate is deliberately explainable:

```
overall = 0.20 privacy + 0.30 injection + 0.20 safety
        + 0.20 authorization + 0.10 context
```

It is capped at `1.0`. Policy rules can override it: for example an explicit
authorization failure always blocks, even if the aggregated score is low.
Async fairness, grounding, cost and performance assessments are recorded after
the response and are not treated as real-time enforcement signals.

## Operational transition

For a production deployment replace in-process background tasks with a durable
queue and independently scaled workers. Keep model-provider proxying separate
from governance evaluation, use authenticated identity rather than caller
claims, protect audit stores with encryption and retention controls, and stage
policy/model changes in observe-only mode before enforcing them.
