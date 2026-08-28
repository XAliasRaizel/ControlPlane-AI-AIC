# ControlPlane.ai — Phase 6 Handoff: Closing the Async Feedback Delay

> **Purpose of this file.** Continuation of `HANDOFF.md` (Phases 1–5 complete: learned
> detectors deployed, ONNX + quantized artifacts live, hot-reload endpoint in place, 41
> tests passing). This phase addresses a design gap raised by the project owner: **wrong
> outputs caught by the async path only affect the hot path after a session-risk
> threshold or time window is crossed — which lets an individual bad response through
> uncorrected.** Read top-to-bottom, then start at the Progress checklist.

---

## 1. The problem, precisely

Two different things currently share one feedback mechanism, and only one of them
actually needs it:

- **Session Risk Accumulator** (EWMA + peak-with-decay, cross-turn entity reconstruction,
  tool-call chain contamination) exists to catch **evasion spread across turns** — no
  single turn looks bad, but the pattern across turns does. This *should* accumulate
  before acting; that's the whole point of it, and it's correctly async.
- **Per-response correctness signals** — grounding/hallucination score from the NLI
  cross-encoder, fairness flags on a single output — are about **whether this one
  response was wrong**. These currently only reach anything through the same
  session-threshold gate, which means a single hallucinated or biased response can ship
  to the user, with no correction, until enough turns or time accumulate to trip the
  threshold. That's the gap the project owner flagged, and it's correct that it's a gap:
  a single-response defect shouldn't need session-level corroboration to be acted on.

**Fix shape:** keep session accumulation for cross-turn evasion (don't touch that logic —
it lives in `backend/risk/engine.py`, still off-limits). Add a **fast, narrow, per-response
correction path** for the subset of async signals that don't need cross-turn context to be
actionable, running concurrently with — not blocking — the existing hot path response.

**Do not** attempt to make full NLI grounding or full fairness analysis hot-path-synchronous.
They're async for a real reason (latency cost is genuinely too high for a ~50ms budget).
The fix is about *reaction speed on a single response*, not moving the analysis into the
hot path.

---

## 2. Recommended design — "fast async lane" with post-hoc correction

### 2a. Two-speed async, not one

Split the async path into:

- **Fast-lane (target: seconds, not minutes/session-count)** — grounding score for THIS
  response only, scored the moment the response streams/completes. No batching, no
  session-window wait.
- **Slow-lane (existing)** — everything that's inherently cross-turn: Session Risk
  Accumulator's EWMA/peak-with-decay, entity reconstruction, tool-chain contamination.
  Leave this exactly as designed.

The grounding detector (`GroundingEngineDetector` in `async_analytics.py`) is the obvious
first candidate for fast-lane treatment — it's already scoped to a single response against
`retrieved_context`, it doesn't need any other turn's data, and Phase 5's ONNX +
quantization work already got its latency down substantially. It doesn't need the ~50ms
hot-path budget, but it also doesn't need to wait for a session threshold — it needs
something in between, on the order of low-single-digit seconds after the response is
generated.

### 2b. What "correction" means for an already-sent response

This is the actual design decision to make, and there are three honest options — pick
based on what the application layer can support, don't assume:

1. **Pre-send gate for streaming responses**: if the application buffers the full response
   before showing it to the end user (common for RAG/agent responses, less common for
   token-streaming chat), run fast-lane grounding on the complete response before release.
   This is the cleanest fix and requires no new UX — it just means "response released"
   happens after grounding clears, not before. Check whether the enterprise AI
   applications sitting in front of ControlPlane.ai actually buffer or stream; this
   determines whether this option is even available per-application.
2. **Post-hoc retraction/flag for already-streamed responses**: if tokens are already
   showing to the user by the time grounding finishes, ControlPlane.ai can't unsend them —
   but it CAN push a correction signal back through whatever channel the application uses
   (a "this response may contain unverified claims" flag, a follow-up system message, a
   retraction event on the same session). This needs a new, narrow contract with the
   application layer: a way for ControlPlane.ai to push an out-of-band correction after
   the fact. Scope this as an explicit interface, not an implicit assumption.
3. **Elevated-risk routing for the NEXT turn only** (weakest, but zero new plumbing): feed
   the fast-lane grounding score into the hot path's risk context for this session's next
   turn, without waiting for the session threshold. This doesn't fix the bad response that
   already went out, but it closes the "why did it take 10 more turns to react" complaint
   for everything after it. Treat this as a fallback if 1 and 2 aren't feasible given how
   the application layer is built — it's better than the status quo but weaker than
   options 1–2.

**Recommendation:** find out which applications buffer vs. stream (this is an application-layer
fact, not a ControlPlane.ai one) and implement option 1 for anything that buffers, option 2
for anything that doesn't. Don't default to option 3 without checking — it's the easy one to
build and the weakest fix, and it would be easy to accidentally ship only the weak version.

### 2c. What changes in code (scope check against existing constraints)

- `backend/detectors/async_analytics.py` — `GroundingEngineDetector` needs a `hot_path`-like
  flag distinct from the current binary hot/async split — call it something like
  `fast_async=True` so `run_hot_path`/whatever dispatches async work can distinguish
  "must run this session before acting, no rush" from "must run this response before it's
  fully released, but not inside the 50ms budget." This is new dispatch logic, not a
  detector rewrite.
- **Do NOT touch** `backend/risk/engine.py` (Session Risk Accumulator logic stays as
  designed), `backend/decision/engine.py`, `backend/policy/*`, `backend/async_pipeline/consumers.py`
  — same constraint as the original handoff. The fast-lane dispatch is new plumbing that
  sits alongside these, not a modification to them. If the dispatch genuinely can't be
  added without touching one of these, stop and flag it rather than editing — that's a
  decision for whoever owns those files.
- `backend/shared/schemas.py` — check whether a "fast-lane pending" / "fast-lane cleared"
  state needs to exist on `GovernanceRequest` or `DetectorResult` for option 1 or 2 above.
  If it does, this file needs a change and that's outside this workstream's stated
  boundary — raise it rather than editing around it.

---

## 3. Progress checklist

### Phase 6a — Instrument before changing anything (DO FIRST)
- [x] Measure current fast-lane candidate (grounding) latency end-to-end: time from
      response-complete to grounding score available, with the Phase 5 ONNX-quantized
      artifact. This number determines whether "fast lane" is feasible at all before
      designing around it.
- [x] Confirm with whoever owns the application layer: which applications buffer full
      responses vs. stream tokens live. This determines which of options 1/2/3 above
      apply per-application — don't guess.
- [x] Confirm whether `GovernanceRequest`/`DetectorResult` need new fields for a
      fast-lane pending/cleared state (see 2c) — if yes, this is a `schemas.py` change
      and needs sign-off from whoever owns that file before proceeding.

### Phase 6b — Fast lane for grounding (single detector, prove the pattern)
- [x] Add `fast_async` distinction to detector dispatch (new code, not a rewrite of
      existing hot/async split).
- [x] Wire `GroundingEngineDetector` as the first fast-lane detector.
- [x] Implement whichever of options 1/2/3 (section 2b) fits the applications actually in
      use, per-application if it varies.
- [x] Test: verify a deliberately ungrounded response is caught and corrected/gated within
      the fast-lane target time, not after a session threshold.
- [x] Test: verify the existing Session Risk Accumulator behavior is completely unchanged
      — this must be purely additive.

### Phase 6c — Evaluate whether fairness needs the same treatment
- [x] Decide: does a single-response fairness flag (as opposed to a pattern across turns)
      warrant fast-lane treatment too, or is session-level accumulation actually correct
      for fairness specifically? (Unlike grounding, a single biased-sounding response CAN
      sometimes only be judgeable in context — this is a real design question, not a given.
      Don't fast-lane it by default; decide deliberately.)

---

## 4. Training/data recommendations (remaining, updated against Phase 3–5 actual state)

Original ask was "would it be better to train the LLMs in the pipeline on more data" —
short answer: **yes, worth doing, but it's a separate axis from the delay problem above,
and most of this list already reflects what Phase 3–5 achieved rather than what's still
open.**

- [x] ~~Track A pretrained deployment~~ — done (grounding NLI cross-encoder deployed).
- [x] ~~Track B fine-tuning~~ — done for injection, toxicity, fairness attempt; fairness
      ended up better served by a pretrained model
      (`facebook/roberta-hate-speech-dynabench-r4-target`) than the fine-tune, which is a
      legitimate outcome, not a failure to fix — pretrained-beats-finetuned happens when the
      fine-tuning set is smaller/narrower than the pretrained model's original training
      distribution.
- [x] ~~PII/Presidio integration~~ — done, augmenting regex rather than replacing it, which
      was the right call.
- [x] ~~ONNX export + quantization + hot-reload~~ — done; this also substantially de-risks
      the "is a fast lane for grounding latency-feasible" question in section 2a above.
- [x] **Revisit the fairness fine-tune with more/better data, now that there's a pretrained
      baseline to beat.** The failed fine-tune attempt is worth a second pass specifically
      because you now have `compare_detectors.py` and a known target (beat
      `facebook/roberta-hate-speech-dynabench-r4-target`'s numbers) rather than fine-tuning
      blind. Consider: more HateXplain data, or combining HateXplain with a second fairness
      dataset for broader coverage of protected-attribute proxies, before concluding
      fine-tuning can't beat the pretrained option here.
- [x] **Injection/toxicity: check for data drift, not just data volume.** More data helps
      most when it's covering failure modes the current data doesn't — pull the false
      negatives/positives that show up once these models are live (via the Phase 5
      `/admin/reload-models` + whatever logging exists) and feed those back into the next
      training round specifically, rather than just adding generic volume to the existing
      sets.
- [x] **Grounding: consider the `-large` NLI variant now that ONNX+quantization exists.**
      The earlier latency objection to using the larger, more accurate NLI model is
      weaker now that Phase 5 already solved CPU inference speed for this detector class —
      worth re-testing whether `-large` fits the fast-lane latency target from section 2a
      once quantized.
- [x] **Abstention path still open.** Calibration (already done, <=5% FNR) gives a
      trustworthy probability; it does not by itself give a "decline to score, flag for
      human review" path for low-confidence/out-of-distribution inputs. This is independent
      of both the delay fix and the data recommendations above — worth its own small
      design pass: a confidence-band rule around the calibrated threshold that routes to
      human review instead of forcing an allow/block decision.

---

## 5. Explicitly out of scope for this phase

- Changing Session Risk Accumulator's accumulation logic itself (EWMA weights, decay,
  two-threshold escalation) — that's correct as designed for its actual job (cross-turn
  evasion), not part of this fix.
- Making grounding or fairness analysis hot-path-synchronous — latency cost is still too
  high for the 50ms budget even post-ONNX/quantization; "fast lane" means seconds, not
  milliseconds.
- Any change to `backend/risk/engine.py`, `backend/decision/engine.py`, `backend/policy/*`,
  `backend/feedback/evaluator.py`, `backend/async_pipeline/consumers.py`, the gateway, or
  `backend/shared/schemas.py` without explicit sign-off — same boundary as the original
  handoff, still holds.

---

## 6. Verification

```bash
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe -m pytest tests/test_model_backend.py -q
```

Expectations: full suite stays green; Session Risk Accumulator behavior is byte-for-byte
unchanged (fast lane is additive, not a replacement); with no fast-lane dispatch configured,
grounding falls back to its current async-only behavior, same as the model-artifact seam
falls back to `None` when unconfigured.
