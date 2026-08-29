# ControlPlane.ai — Phase 9 (Final): Session Risk Accumulator — Full Implementation Spec

> **Purpose of this file.** This supersedes the earlier Phase 9 draft. That version
> named what needed to be built; this version specifies it completely enough to
> implement in one pass — exact files, exact data structures, exact algorithms, exact
> config, exact tests, exact rollout steps — so this can be the **last handoff file**
> before the accumulator ships. Read the File Manifest first for the shape of the
> change, then work section by section; each section is self-contained enough to
> implement without needing another round-trip.

---

## Table of contents

0. Pre-flight checks (automatable, do first)
1. File manifest
2. Design recap (condensed — full rationale is in the previous handoff, not repeated here)
3. Config & environment variables
4. Session store (`backend/risk/session_store.py`)
5. Accumulator core (`backend/risk/accumulator.py`)
6. Entity reconstruction tracking
7. Tool-chain contamination tracking
8. Adaptive permissioning bands — computed, not enforced
9. The one deliberate touch to `engine.py`
10. Wiring in `main.py`
11. Schema additions (`schemas.py`) — needs sign-off
12. Calibration script (`ml/scripts/calibrate_session_accumulator.py`)
13. Test suite — every test, named and specified
14. Telemetry additions
15. Rollout runbook
16. Explicitly out of scope
17. Final verification checklist

---

## 0. Pre-flight checks (automatable, do first)

This is the one place a stop condition can override "do it all in one go" — but the
check itself is automatable, not a request for a human conversation:

```bash
git log --all --oneline -- backend/risk/engine.py | head -50
git log --all --oneline --grep="session" --grep="EWMA" --grep="accumulator" -i
git branch -a
grep -rn "session_id\|EWMA\|ewma\|accumulator" backend/risk/ backend/shared/schemas.py
```

- **If any of this turns up in-progress work on session accumulation that isn't on
  the current branch:** stop, don't implement — surface what was found instead. This
  document becomes a spec to reconcile against, not something to build over.
- **If nothing turns up (the expected case, per Phase 8's finding that this is
  currently unbuilt):** proceed through the rest of this document autonomously.

This check should take under a minute and doesn't require waiting on a person unless
it actually finds something.

---

## 1. File manifest

**New files:**

| File | Purpose |
|---|---|
| `backend/risk/session_store.py` | Session state storage: in-memory default, optional Redis-backed |
| `backend/risk/accumulator.py` | EWMA + peak-with-decay core, entity reconstruction, tool-chain contamination, band classification |
| `ml/scripts/calibrate_session_accumulator.py` | Synthetic scenario sweep, writes calibration artifact |
| `ml/artifacts/session-accumulator/calibration.json` | Output of the calibration script — checked in or generated at setup time (decide per repo convention; see §12) |
| `tests/test_session_accumulator.py` | Full test suite for this phase |

**Modified files:**

| File | Change |
|---|---|
| `backend/shared/schemas.py` | Add `session_id`, `session_risk`, `session_band` fields — additive only, **needs sign-off** (§11) |
| `backend/risk/engine.py` | One additive branch, gated by env var and `session_id` presence — existing noisy-OR path untouched (§9) |
| `backend/main.py` | Extract/pass `session_id`, call accumulator after risk fusion, extend structured logging (§10, §14) |

**Untouched, and staying that way:** `backend/decision/engine.py`, `backend/policy/*`,
`backend/feedback/evaluator.py`, `backend/async_pipeline/consumers.py`, the gateway
(beyond the `main.py` wiring above). See §16.

---

## 2. Design recap (condensed)

Dual-signal scoring (EWMA + peak-with-decay) because a single moving average lets one
bad turn dilute below threshold if enough benign turns follow — the exact evasion
pattern this exists to catch. Peak-with-decay preserves a bad turn's influence longer
than a plain average would. Both signals combine via `max()`. Full rationale is in the
prior Phase 9 draft; this document assumes it and moves straight to implementation.

---

## 3. Config & environment variables

Extend the existing "Env vars (the seam)" table with:

| Var | Consumed by | Default when unset |
|---|---|---|
| `CONTROLPLANE_SESSION_ACCUMULATOR_ENABLED` | `engine.py` gate | `false` — accumulator branch never executes; `session_risk`/`session_band` are `None` |
| `CONTROLPLANE_SESSION_STORE` | `session_store.py` | unset → `InMemorySessionStore` |
| `CONTROLPLANE_SESSION_TTL_SECONDS` | `session_store.py` | `1800` (30 min) |
| `CONTROLPLANE_SESSION_ACCUMULATOR_CONFIG` | `accumulator.py` | unset → built-in provisional defaults (§5) |

`CONTROLPLANE_SESSION_ACCUMULATOR_CONFIG` points at a `calibration.json`-style file
(same idiom as `CONTROLPLANE_MODEL_<TASK>` pointing at an artifact dir) — reuses an
existing pattern rather than inventing a new one.

**Default-off is load-bearing, not cosmetic.** With `CONTROLPLANE_SESSION_ACCUMULATOR_ENABLED`
unset, `engine.py`'s new branch must not execute even if `session_id` is present on the
request — this is what keeps the "byte-for-byte unchanged by default" guarantee intact
without relying on `session_id` being absent as the only safety net.

---

## 4. Session store — `backend/risk/session_store.py`

```python
"""
Session state storage for the Session Risk Accumulator.
Every method must fail closed to "no session state" rather than raising into the
hot/fast path — mirrors model_backend.py's consult() fail-safe philosophy.
"""
from __future__ import annotations
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Protocol


@dataclass
class SessionState:
    session_id: str
    created_at: float
    last_updated_at: float
    ewma_score: float = 0.0
    peak_score: float = 0.0
    turn_count: int = 0
    fragment_window: list[str] = field(default_factory=list)
    contaminated_tools: list[str] = field(default_factory=list)
    contamination_active: bool = False
    fast_lane_correction_count: int = 0
    last_band: int = 1

    @property
    def session_risk(self) -> float:
        return max(self.ewma_score, self.peak_score)

    def to_json(self) -> str: ...   # json.dumps(asdict(self))
    @classmethod
    def from_json(cls, raw: str) -> "SessionState": ...


class SessionStore(Protocol):
    def get(self, session_id: str) -> SessionState | None: ...
    def set(self, session_id: str, state: SessionState, ttl_seconds: int) -> None: ...
    def delete(self, session_id: str) -> None: ...


class InMemorySessionStore:
    """
    Zero-dependency default. Not shared across processes — see the concurrency
    caveat below, which must be surfaced in logs/docs, not left implicit.
    """
    def __init__(self):
        self._data: dict[str, tuple[SessionState, float]] = {}  # id -> (state, expires_at)

    def get(self, session_id: str) -> SessionState | None:
        entry = self._data.get(session_id)
        if entry is None:
            return None
        state, expires_at = entry
        if time.time() > expires_at:
            del self._data[session_id]
            return None
        return state

    def set(self, session_id: str, state: SessionState, ttl_seconds: int) -> None:
        self._data[session_id] = (state, time.time() + ttl_seconds)

    def delete(self, session_id: str) -> None:
        self._data.pop(session_id, None)


class RedisSessionStore:
    """
    Only imports redis lazily inside __init__ — never at module load — same lazy-import
    discipline used for torch/transformers elsewhere in this repo. If redis import
    fails or connection fails, __init__ should raise so get_session_store() can log and
    fall back to InMemorySessionStore rather than crashing the request path.
    """
    def __init__(self, url: str):
        import redis  # lazy import
        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._client.ping()  # fail fast at construction, not at first request

    def get(self, session_id: str) -> SessionState | None:
        raw = self._client.get(f"cp:session:{session_id}")
        return SessionState.from_json(raw) if raw else None

    def set(self, session_id: str, state: SessionState, ttl_seconds: int) -> None:
        self._client.set(f"cp:session:{session_id}", state.to_json(), ex=ttl_seconds)

    def delete(self, session_id: str) -> None:
        self._client.delete(f"cp:session:{session_id}")


_store_singleton: SessionStore | None = None

def get_session_store() -> SessionStore:
    """
    Seam entry point, mirrors get_detector_model()/get_grounding_scorer(). Caches a
    singleton. On any Redis construction failure, logs a warning and falls back to
    InMemorySessionStore — never raises out of this function.
    """
    global _store_singleton
    if _store_singleton is not None:
        return _store_singleton
    url = os.environ.get("CONTROLPLANE_SESSION_STORE")
    if url:
        try:
            _store_singleton = RedisSessionStore(url)
            return _store_singleton
        except Exception:
            # log.warning("Redis session store unavailable, falling back to in-memory")
            pass
    _store_singleton = InMemorySessionStore()
    return _store_singleton

def reset_store_cache() -> None:
    """For tests — mirrors model_backend.py's reset_cache()."""
    global _store_singleton
    _store_singleton = None
```

**Concurrency caveat — put this in the module docstring and in `HANDOFF.md`, not just
here:** `InMemorySessionStore` does not share state across worker processes. If the
deployment runs more than one worker, sessions whose turns land on different workers
will under-count. This is a known, documented limitation of the default store, not a
silent gap — call it out at startup (a log line at boot if `CONTROLPLANE_SESSION_ACCUMULATOR_ENABLED=true`
and `CONTROLPLANE_SESSION_STORE` is unset and the process is configured for >1 worker,
if that's detectable; otherwise document it prominently).

---

## 5. Accumulator core — `backend/risk/accumulator.py`

```python
"""
Dual-signal session risk accumulator: EWMA + peak-with-decay.
Every function here is pure (no I/O) except update_session, which is the only one
that touches the store — keeps the math independently testable.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from backend.risk.session_store import SessionState, SessionStore, get_session_store


@dataclass
class AccumulatorConfig:
    alpha: float = 0.3               # EWMA decay rate — provisional, see §12
    peak_decay: float = 0.9          # per-turn peak decay — provisional, see §12
    threshold_medium: float = 0.4    # provisional, see §12
    threshold_high: float = 0.7      # provisional, see §12
    ttl_seconds: int = 1800
    fragment_window_turns: int = 5


_DEFAULT_CONFIG = AccumulatorConfig()

def load_accumulator_config() -> AccumulatorConfig:
    """
    Reads CONTROLPLANE_SESSION_ACCUMULATOR_CONFIG (path to calibration.json, produced
    by ml/scripts/calibrate_session_accumulator.py). Falls back to _DEFAULT_CONFIG on
    any failure (unset, missing file, malformed JSON) — never raises. Same fail-safe
    posture as model_backend.py.
    """
    path = os.environ.get("CONTROLPLANE_SESSION_ACCUMULATOR_CONFIG")
    if not path:
        return _DEFAULT_CONFIG
    try:
        with open(path) as f:
            data = json.load(f)
        return AccumulatorConfig(
            alpha=data.get("alpha", _DEFAULT_CONFIG.alpha),
            peak_decay=data.get("peak_decay", _DEFAULT_CONFIG.peak_decay),
            threshold_medium=data.get("threshold_medium", _DEFAULT_CONFIG.threshold_medium),
            threshold_high=data.get("threshold_high", _DEFAULT_CONFIG.threshold_high),
            ttl_seconds=data.get("ttl_seconds", _DEFAULT_CONFIG.ttl_seconds),
            fragment_window_turns=data.get("fragment_window_turns", _DEFAULT_CONFIG.fragment_window_turns),
        )
    except Exception:
        return _DEFAULT_CONFIG


def update_ewma(prev_score: float, signal: float, alpha: float) -> float:
    return alpha * signal + (1 - alpha) * prev_score


def update_peak(prev_peak: float, signal: float, decay: float) -> float:
    return max(signal, prev_peak * decay)


def classify_band(session_risk: float, cfg: AccumulatorConfig) -> int:
    """
    Returns 1 (baseline), 2 (elevated), or 3 (high) based on the accumulator only.
    Band 4 (critical) is NOT decided here — it's the existing, unchanged, per-turn
    CRITICAL escalation already in engine.py. Don't conflate the two: if a single turn
    already trips CRITICAL, that path fires regardless of session_band, and
    session_band only describes the new middle ground the accumulator adds.
    """
    if session_risk >= cfg.threshold_high:
        return 3
    if session_risk >= cfg.threshold_medium:
        return 2
    return 1


def update_session(
    store: SessionStore,
    session_id: str,
    turn_signal: float,
    fast_lane_correction_fired: bool = False,
    pii_fragment: str | None = None,
    tool_name: str | None = None,
    data_classification: str | None = None,
    pii_detector_fn=None,
    cfg: AccumulatorConfig | None = None,
) -> SessionState:
    """
    The single entry point engine.py calls. Loads existing state or creates new,
    updates both signals, folds in fast-lane corrections, updates the entity-
    reconstruction window and tool-chain contamination flag, persists, returns state.
    """
    import time
    cfg = cfg or load_accumulator_config()
    state = store.get(session_id)
    now = time.time()
    if state is None:
        state = SessionState(session_id=session_id, created_at=now, last_updated_at=now)

    state.ewma_score = update_ewma(state.ewma_score, turn_signal, cfg.alpha)
    state.peak_score = update_peak(state.peak_score, turn_signal, cfg.peak_decay)
    state.turn_count += 1
    state.last_updated_at = now

    if fast_lane_correction_fired:
        state.fast_lane_correction_count += 1
        # Fold into EWMA as an additional signal spike — treat a fired correction as
        # turn_signal=1.0 blended in on top of the detector-driven update above, not a
        # replacement for it:
        state.ewma_score = update_ewma(state.ewma_score, 1.0, cfg.alpha)
        state.peak_score = update_peak(state.peak_score, 1.0, cfg.peak_decay)

    if pii_fragment:
        state.fragment_window.append(pii_fragment)
        state.fragment_window = state.fragment_window[-cfg.fragment_window_turns:]

    if data_classification in ("sensitive", "restricted") and tool_name:
        if tool_name not in state.contaminated_tools:
            state.contaminated_tools.append(tool_name)
        state.contamination_active = True

    state.last_band = classify_band(state.session_risk, cfg)
    store.set(session_id, state, ttl_seconds=cfg.ttl_seconds)
    return state
```

---

## 6. Entity reconstruction tracking

Reuses the existing PII detector — no new ML surface, per the scope decision in the
prior draft.

```python
def check_entity_reconstruction(state: SessionState, pii_detector_fn) -> bool:
    """
    pii_detector_fn: the existing PII detector's callable (whatever `pii.py` already
    exposes for a single text — reuse it as-is, don't reimplement).
    Concatenates the rolling fragment window and re-runs the existing check. Returns
    True only if the concatenation trips a violation that no individual fragment did
    (the whole thing is stronger evidence than the sum of checking each turn alone).
    """
    if len(state.fragment_window) < 2:
        return False
    concatenated = " ".join(state.fragment_window)
    result = pii_detector_fn(concatenated)
    return bool(getattr(result, "triggered", False))
```

**Call this from `main.py`, not from inside `update_session`** — keep the accumulator
core free of a hard dependency on the PII detector's exact interface; pass the boolean
result back in as part of the next turn's signal instead, e.g. by boosting
`turn_signal` before calling `update_session` when reconstruction is detected. This
keeps `accumulator.py` decoupled and independently testable.

---

## 7. Tool-chain contamination tracking

Already covered by `update_session`'s `data_classification`/`tool_name` parameters
above — `contaminated_tools` and `contamination_active` persist on `SessionState` for
the life of the session (bounded by TTL), not reset per-call. Confirm before wiring:
whether `data_classification` already exists on `GovernanceRequest` (the architecture
diagram lists it among fields the Gateway captures/derives) or needs to be added
alongside `session_id` in §11 — check the current schema rather than assuming either
way.

---

## 8. Adaptive permissioning bands — computed, not enforced

This is a deliberate scope boundary, mirroring the abstention band's precedent from
Phase 7d (computed and documented, not wired into `decision/engine.py`, because that
file is off-limits without separate sign-off):

- `state.last_band` (1/2/3) and `state.session_risk` are computed by this phase and
  exposed on the response (via the schema additions in §11).
- **This phase does not implement enforcement** — no tool-list filtering, no
  RAG-only forcing, no auto human-review routing based on band. That requires wiring
  into `backend/decision/engine.py`'s existing MODIFY vocabulary (alongside
  `Redact PII / Filter / Rewrite / Retry / Reroute`), which is explicitly out of scope
  here per the standing boundary — same reasoning as the abstention band.
- **This is still useful on its own**, and worth building even without enforcement:
  `session_band` changing visibly across turns in the API response and audit trail is
  itself a demonstrable artifact of the accumulator working — which matters given this
  was selected as the prototype's core demo mechanism. A visible, correct signal that
  isn't yet wired to automatic enforcement is real, useful progress, not a placeholder.
- Recommended band semantics for whoever later wires enforcement (documented here so
  the intent isn't lost, not implemented here):
  - Band 1: no change.
  - Band 2: restrict to a defined lower-risk tool subset; still auto-allow.
  - Band 3: force RAG-grounded-only responses; require fast-lane clearance before
    release regardless of the application's normal buffer/stream mode.
  - Band 4 (existing, unchanged): today's static per-turn CRITICAL path.

---

## 9. The one deliberate touch to `engine.py`

This is the only edit to a previously off-limits file in this phase (see §11 for the
schema addition, which is the other one). It must be additive and gated so the
existing path is provably untouched:

```python
# backend/risk/engine.py

import os

def _session_accumulator_enabled() -> bool:
    return os.environ.get("CONTROLPLANE_SESSION_ACCUMULATOR_ENABLED", "").lower() == "true"


def fuse_risk(detector_results: list[DetectorResult], request: GovernanceRequest) -> RiskScore:
    # --- EXISTING NOISY-OR FUSION LOGIC: DO NOT MODIFY ANYTHING ABOVE OR WITHIN THIS BLOCK ---
    turn_risk = existing_noisy_or_fusion(detector_results)
    # --- END EXISTING LOGIC ---

    session_risk = None
    session_band = None

    if request.session_id and _session_accumulator_enabled():
        from backend.risk.accumulator import update_session, load_accumulator_config
        from backend.risk.session_store import get_session_store

        cfg = load_accumulator_config()
        state = update_session(
            store=get_session_store(),
            session_id=request.session_id,
            turn_signal=turn_risk.score,
            fast_lane_correction_fired=bool(request.context.get("fast_lane_corrections", 0)) if request.context else False,
            cfg=cfg,
        )
        session_risk = state.session_risk
        session_band = state.last_band

    return RiskScore(
        # ...all existing fields, unchanged...
        session_risk=session_risk,
        session_band=session_band,
    )
```

**Verification that this stayed additive:** `git diff backend/risk/engine.py` should
show only new lines (the import, the helper function, the `if` block, and the two new
fields on the `RiskScore(...)` construction) — zero lines removed or altered inside
`existing_noisy_or_fusion` or anywhere else in the file. If the diff shows anything
else changing, that's a signal to stop and reconsider, not to proceed.

---

## 10. Wiring in `main.py`

- **Extract `session_id`** from the incoming request. Decide the source explicitly
  rather than guessing: a header (e.g. `X-ControlPlane-Session-Id`), a field in the
  request body, or derived from an existing field already present (e.g. an
  application-supplied conversation/thread ID, if one already flows through). Check
  what's already available before inventing a new required field on callers — if
  nothing suitable exists today, a header is the least invasive to add, since it
  doesn't require every calling application to change its request body shape.
- **Pass it through** to `GovernanceRequest.session_id`.
- **After `run_hot_path`/`run_fast_lane` complete**, if `fuse_risk` returned a non-null
  `session_risk`, this is available for inclusion in the response and in structured
  logs (§14). No new call is needed here beyond what `fuse_risk` already does — the
  accumulator update happens inside risk fusion, not as a separate step in `main.py`.
- **Entity reconstruction hook (§6):** call `check_entity_reconstruction` after
  fetching the session's current state (if `session_id` present and feature enabled),
  using the existing PII detector's callable, and fold a positive result into the
  *next* turn's `turn_signal` before it reaches `fuse_risk` — e.g. boost it toward 1.0
  — rather than trying to retroactively re-score a turn that's already been fused.

---

## 11. Schema additions — needs sign-off

```python
# backend/shared/schemas.py

class GovernanceRequest(BaseModel):
    # ...existing fields, unchanged...
    session_id: str | None = None   # NEW — additive, optional, default None

class RiskScore(BaseModel):
    # ...existing fields, unchanged...
    session_risk: float | None = None   # NEW
    session_band: int | None = None     # NEW, 1-3 (4 remains the existing static CRITICAL path)
```

This is the same category and shape of change already approved once in this project
(the `fast_lane_pending`/`fast_lane_webhook` additions, signed off by Tushar on
2026-08-28) — additive, optional, `None`-defaulted fields with zero effect when unset.
Given that precedent exists, this should be a fast approval, not a fresh debate — but
it still needs to happen and be recorded the same way, not silently merged. Confirm
also whether `data_classification` (referenced in §7) already exists on
`GovernanceRequest`; add it here alongside `session_id` if not, under the same
sign-off.

---

## 12. Calibration script — `ml/scripts/calibrate_session_accumulator.py`

Mirrors the existing `evaluate_model.py` / `compare_detectors.py` pattern: produces a
report and writes the artifact `accumulator.py` reads via
`CONTROLPLANE_SESSION_ACCUMULATOR_CONFIG`.

**Synthetic scenarios to generate and sweep against:**

1. **`multi_turn_evasion`** — N turns each at `turn_signal ≈ 0.35` (below a
   single-turn hot-path threshold of ~0.5, i.e. individually unremarkable). Sweep
   `alpha` to find the smallest value where `session_risk` crosses
   `threshold_medium` within a small number of turns (target: within 5) without
   the `pure_benign_control` scenario below also crossing it.
2. **`peak_dilution`** — one turn at `turn_signal = 0.9`, followed by M turns at
   `turn_signal = 0.05`. Sweep `peak_decay` to find the value that keeps
   `session_risk` (via the peak component) at or above `threshold_high` for at least
   M = 10 follow-up turns.
3. **`pure_benign_control`** — 50 turns, all at `turn_signal ≈ 0.05–0.10`.
   False-positive check: `session_risk` must stay below `threshold_medium`
   throughout, for every candidate `(alpha, peak_decay)` pair under consideration.
4. **`steady_medium_control`** — turns sustained at `turn_signal ≈ 0.35–0.45`.
   Verify this lands in Band 2 and does not spuriously escalate to Band 3.

**Output:** `ml/artifacts/session-accumulator/calibration.json`:

```json
{
  "alpha": 0.0,
  "peak_decay": 0.0,
  "threshold_medium": 0.0,
  "threshold_high": 0.0,
  "ttl_seconds": 1800,
  "fragment_window_turns": 5,
  "calibrated_on": "YYYY-MM-DD",
  "scenarios": {
    "multi_turn_evasion": {"turns_to_trigger": 0, "passed": true},
    "peak_dilution": {"turns_survived_at_threshold": 0, "passed": true},
    "pure_benign_control": {"max_session_risk_observed": 0.0, "passed": true},
    "steady_medium_control": {"band_reached": 2, "passed": true}
  }
}
```

Run this once locally (no GPU/heavy deps required — this is pure arithmetic over
synthetic sequences, no model inference involved) and commit the resulting
`calibration.json`, the same way other calibration artifacts in this repo are treated.
Don't hand-pick the four config values without running the sweep — the whole point of
this script is to replace "provisional, guessed" with "measured against defined
scenarios."

---

## 13. Test suite — `tests/test_session_accumulator.py`

Every test below should exist, named exactly as given, with the described assertion:

```python
def test_no_session_id_unchanged_behavior():
    """A GovernanceRequest with session_id=None produces identical RiskScore output
    (session_risk=None, session_band=None, all other fields unchanged) to the
    pre-Phase-9 system — compare against a fixture of pre-Phase-9 expected output."""

def test_accumulator_disabled_by_default():
    """CONTROLPLANE_SESSION_ACCUMULATOR_ENABLED unset, session_id IS present -> the
    engine.py branch still does not execute; session_risk/session_band are None."""

def test_ewma_multi_turn_evasion_triggers():
    """Using the calibrated config from calibration.json, replay the
    multi_turn_evasion scenario turn-by-turn through update_session and assert
    session_risk crosses threshold_medium at the exact turn recorded in
    calibration.json's scenarios.multi_turn_evasion.turns_to_trigger."""

def test_peak_with_decay_survives_dilution():
    """Replay peak_dilution. Assert session_risk (peak component) stays >=
    threshold_high through turn 10. Separately compute what an EWMA-only score
    (ignoring peak entirely) would have been at turn 10 using the same alpha, and
    assert it has already dropped below threshold_high -- proving the dual-signal
    design does something the EWMA alone would not."""

def test_pure_benign_no_false_trigger():
    """Replay pure_benign_control (50 turns). Assert session_risk never reaches
    threshold_medium at any point in the sequence."""

def test_entity_reconstruction_catches_split_pii():
    """Two turns, each containing one half of a synthetic SSN-shaped fragment
    (e.g. '123-45-' then '6789'), neither of which trips the existing PII detector
    individually. Assert check_entity_reconstruction returns True once both are in
    the fragment window."""

def test_entity_reconstruction_no_false_positive_on_unrelated_fragments():
    """Two turns with unrelated, individually-benign fragments that do NOT
    concatenate into anything sensitive. Assert check_entity_reconstruction
    returns False."""

def test_tool_chain_contamination_flag_persists():
    """Turn 1: update_session called with data_classification='sensitive',
    tool_name='read_customer_record'. Turn 2: unrelated tool_name='send_email',
    data_classification=None. Assert state.contamination_active is True after turn 2
    and 'read_customer_record' remains in state.contaminated_tools."""

def test_fast_lane_correction_feeds_accumulator():
    """Two otherwise-identical sessions, N turns each with equal turn_signal. Session
    A has fast_lane_correction_fired=True on turn 2; Session B never does. Assert
    Session A's session_risk is strictly higher than Session B's from turn 2 onward."""

def test_ttl_expiry():
    """Set state via store.set with a short ttl_seconds (e.g. 1), mock time forward
    past expiry, assert store.get returns None afterward."""

def test_band_classification_boundaries():
    """Directly test classify_band() at session_risk values just below and at
    threshold_medium and threshold_high, using calibrated config values -- assert
    band 1/2/3 boundaries are exactly where the config says they are."""

def test_band_4_untouched_by_accumulator():
    """A single turn whose per-turn detector score already trips the EXISTING static
    CRITICAL path: assert that path's behavior (whatever it does today) fires
    unchanged regardless of session_band's value -- proves band 4 wasn't
    accidentally routed through the new code."""

def test_in_memory_store_basic_roundtrip():
    """set() then get() returns an equal SessionState; delete() then get() returns
    None."""

def test_redis_store_parity_if_available():
    """Skip if redis isn't reachable in the test environment. Otherwise: replay the
    same turn sequence through both InMemorySessionStore and RedisSessionStore-backed
    accumulator runs, assert identical session_risk trajectories at every turn --
    mirrors the existing parity-test pattern used for the fairness detector's
    consult() in test_model_backend.py."""

def test_redis_construction_failure_falls_back_to_memory():
    """CONTROLPLANE_SESSION_STORE set to an unreachable URL -- assert
    get_session_store() returns an InMemorySessionStore instance rather than raising,
    and logs a warning (assert via caplog or equivalent)."""

def test_config_load_falls_back_on_missing_file():
    """CONTROLPLANE_SESSION_ACCUMULATOR_CONFIG pointing at a nonexistent path --
    assert load_accumulator_config() returns _DEFAULT_CONFIG rather than raising."""

def test_config_load_falls_back_on_malformed_json():
    """Same as above but the file exists and contains invalid JSON."""
```

After writing these, update the total test count in `HANDOFF.md` (currently 43) to
whatever the new total is, and record the exact `pytest -k` invocation used to isolate
this suite, matching the precedent set by `test_fast_lane_timeout_fail_open` in
Phase 8.

---

## 14. Telemetry additions

Extend the structured logging already emitted from `run_fast_lane` (or wherever the
per-request log line is emitted in `main.py`) with, when a session is active:

`session_id`, `session_risk`, `session_band`, `ewma_component`, `peak_component`,
`turn_count`, `contamination_active`, `entity_reconstruction_triggered`,
`fast_lane_correction_count`.

Same posture as Phase 8's fast-lane logging: log-only for this phase is acceptable,
aggregation into dashboards/metrics is a separate follow-on, but state that
explicitly in whatever tracker exists rather than leaving it implicit.

---

## 15. Rollout runbook

Once implemented and tested, enabling this in a demo/staging environment:

1. Run `ml/scripts/calibrate_session_accumulator.py`; confirm all four scenarios pass;
   commit `ml/artifacts/session-accumulator/calibration.json`.
2. Set `CONTROLPLANE_SESSION_ACCUMULATOR_CONFIG=ml/artifacts/session-accumulator/calibration.json`.
3. Decide and set `CONTROLPLANE_SESSION_STORE` — leave unset for a single-process
   demo; set to a Redis URL for anything with more than one worker.
4. Set `CONTROLPLANE_SESSION_ACCUMULATOR_ENABLED=true`.
5. Confirm the application layer is sending `session_id` (per whatever mechanism was
   decided in §10) — without it, the accumulator has nothing to key on even when
   enabled.
6. Run the full test suite plus `-k session` in isolation; confirm the fast-lane and
   pre-Phase-9 suites are still green (no regression from the additive branch).
7. Watch the new telemetry fields (§14) on a live multi-turn session to visually
   confirm `session_risk`/`session_band` move as expected before calling this done.

---

## 16. Explicitly out of scope

- Enforcement of adaptive permissioning bands inside `backend/decision/engine.py` —
  computed and exposed only (§8), same boundary as the abstention band.
- Cross-session tracking (evasion spread across multiple distinct sessions, not turns
  within one).
- Wiring the still-open abstention band (Phase 7d) into live routing — unrelated
  dependency, still blocked on `decision/engine.py`'s owner.
- Any change to `backend/policy/*`, `backend/feedback/evaluator.py`,
  `backend/async_pipeline/consumers.py`, or the gateway beyond the `main.py` wiring
  in §10.
- Redis deployment/ops (provisioning, HA) — this spec assumes a reachable URL if one
  is configured; standing it up is infrastructure work outside this phase.

---

## 17. Final verification checklist

```bash
./.venv/Scripts/python.exe -m py_compile backend/risk/session_store.py
./.venv/Scripts/python.exe -m py_compile backend/risk/accumulator.py
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe -m pytest -k "session" -q
git diff --stat -- backend/risk/engine.py backend/shared/schemas.py
git diff --stat -- backend/decision/engine.py backend/policy \
  backend/feedback/evaluator.py backend/async_pipeline/consumers.py
```

Expectations:
- Full suite green, including every test named in §13.
- `engine.py` diff shows only additive lines, per §9's verification note.
- `schemas.py` diff shows only the fields listed in §11, with sign-off recorded.
- The second `git diff --stat` (the still-untouchable files) comes back completely
  empty — if it doesn't, stop and flag it rather than explaining it away.
- `calibration.json` exists, is committed, and its four scenario results are recorded
  in it (§12) — not hand-picked constants.
- A live multi-turn session visibly moves `session_band` from 1 → 2 → 3 in the
  telemetry/response output when driven through a scenario matching
  `multi_turn_evasion` — this is the concrete, demoable proof that the core mechanism
  the prototype was built around now actually runs.
