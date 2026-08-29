"""Dual-signal session risk accumulator: EWMA + peak-with-decay.

Every function here is pure (no I/O) except update_session, which is the only
one that touches the store -- keeps the math independently testable.

Design rationale (condensed from HANDOFF_phase9_full_implementation_spec.md):
A single EWMA lets one bad turn dilute below threshold if enough benign turns
follow -- the exact evasion pattern this exists to catch. Peak-with-decay
preserves a bad turn's influence longer than a plain average would. Both
signals combine via max(). The dual-signal dual-threshold design is what
separates "session went bad 10 turns ago but is now pretending to be benign"
(peak remains elevated) from "session has consistently moderate risk" (EWMA
remains elevated).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from backend.risk.session_store import SessionState, SessionStore, get_session_store


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AccumulatorConfig:
    """Tunable parameters for the dual-signal accumulator.

    Provisional defaults below are replaced by calibrated values when
    CONTROLPLANE_SESSION_ACCUMULATOR_CONFIG is set (see load_accumulator_config).
    """
    alpha: float = 0.3              # EWMA decay rate (0=no memory, 1=no smoothing)
    peak_decay: float = 0.9         # per-turn peak decay (1=no decay, 0=instant reset)
    threshold_medium: float = 0.4   # session_risk >= this -> band 2
    threshold_high: float = 0.7     # session_risk >= this -> band 3
    ttl_seconds: int = 1800         # session TTL (30 min default)
    fragment_window_turns: int = 5  # PII fragment rolling window size


_DEFAULT_CONFIG = AccumulatorConfig()


def load_accumulator_config() -> AccumulatorConfig:
    """Read calibration.json pointed at by CONTROLPLANE_SESSION_ACCUMULATOR_CONFIG.

    Falls back to _DEFAULT_CONFIG on any failure (unset, missing file, malformed
    JSON) -- never raises. Same fail-safe posture as model_backend.py.
    """
    path = os.environ.get("CONTROLPLANE_SESSION_ACCUMULATOR_CONFIG", "").strip()
    if not path:
        return _DEFAULT_CONFIG
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return AccumulatorConfig(
            alpha=float(data.get("alpha", _DEFAULT_CONFIG.alpha)),
            peak_decay=float(data.get("peak_decay", _DEFAULT_CONFIG.peak_decay)),
            threshold_medium=float(data.get("threshold_medium", _DEFAULT_CONFIG.threshold_medium)),
            threshold_high=float(data.get("threshold_high", _DEFAULT_CONFIG.threshold_high)),
            ttl_seconds=int(data.get("ttl_seconds", _DEFAULT_CONFIG.ttl_seconds)),
            fragment_window_turns=int(data.get("fragment_window_turns", _DEFAULT_CONFIG.fragment_window_turns)),
        )
    except Exception:
        return _DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Pure math functions (no I/O — independently testable)
# ---------------------------------------------------------------------------

def update_ewma(prev_score: float, signal: float, alpha: float) -> float:
    """Standard exponentially-weighted moving average.

    alpha=0.3 means a new signal contributes 30% of the new value; the
    previous EWMA carries 70%. Higher alpha = faster response to new signals.
    """
    return alpha * signal + (1.0 - alpha) * prev_score


def update_peak(prev_peak: float, signal: float, decay: float) -> float:
    """Peak-with-decay: remember the worst signal seen, but let it slowly fade.

    If the current turn is worse than the decayed peak, the new turn wins.
    If the current turn is benign, the old peak decays by `decay` per turn.
    peak_decay=0.9 means a peak of 0.9 takes ~22 turns to decay to 0.1.
    """
    return max(signal, prev_peak * decay)


def classify_band(session_risk: float, cfg: AccumulatorConfig) -> int:
    """Classify a session_risk score into band 1, 2, or 3.

    Band 1: baseline -- no change to decision routing.
    Band 2: elevated -- intended for tool-list restriction (not enforced here).
    Band 3: high -- intended for RAG-grounded-only + fast-lane clearance (not enforced here).
    Band 4: NOT decided here -- that is the existing static per-turn CRITICAL
            path in engine.py. Do not conflate the two.
    """
    if session_risk >= cfg.threshold_high:
        return 3
    if session_risk >= cfg.threshold_medium:
        return 2
    return 1


# ---------------------------------------------------------------------------
# Entity reconstruction check
# ---------------------------------------------------------------------------

def check_entity_reconstruction(state: SessionState, pii_detector_fn) -> bool:
    """Detect PII spread across turns that no single turn would have triggered.

    pii_detector_fn: a callable that accepts a single text string and returns
    an object with a `triggered` bool attribute. Use the existing PII detector's
    regex patterns (see backend/detectors/pii.py) -- reuse, don't reimplement.

    Design: concatenates the rolling fragment_window and re-runs the check.
    Returns True only if the concatenation triggers a violation that no
    individual fragment would have (i.e., the window needed at least 2 turns).

    Called from main.py, NOT from inside update_session -- keeps accumulator.py
    decoupled from the PII detector's exact interface and independently testable.
    """
    if len(state.fragment_window) < 2:
        return False
    # Empty-string join: fragments are stored as raw text slices, not
    # sentence-level chunks, so direct concatenation is correct. A space join
    # would break split tokens like "123-45-" + "6789" -> "123-45- 6789"
    # which the SSN regex would not match.
    concatenated = "".join(state.fragment_window)
    result = pii_detector_fn(concatenated)
    return bool(getattr(result, "triggered", False))


# ---------------------------------------------------------------------------
# Main entry point — the one function engine.py calls
# ---------------------------------------------------------------------------

def update_session(
    store: SessionStore,
    session_id: str,
    turn_signal: float,
    fast_lane_correction_fired: bool = False,
    pii_fragment: str | None = None,
    tool_name: str | None = None,
    data_classification: str | None = None,
    cfg: AccumulatorConfig | None = None,
) -> SessionState:
    """Load-or-create session state, apply dual-signal update, persist, return.

    This is the single entry point engine.py calls. All side effects (store
    read/write) are isolated here; all pure math is above.

    Parameters
    ----------
    store:                     The session store (InMemory or Redis).
    session_id:                Unique session identifier from the request.
    turn_signal:               The per-turn risk score from noisy-OR fusion (0-1).
    fast_lane_correction_fired: Whether fast-lane analysis fired a correction
                               on this turn. If True, treated as an additional
                               turn_signal=1.0 spike on top of the detector signal.
    pii_fragment:              Optional PII-relevant text fragment to add to the
                               rolling entity-reconstruction window.
    tool_name:                 Tool used this turn (for contamination tracking).
    data_classification:       Data classification of this turn's context.
    cfg:                       AccumulatorConfig. If None, loads from env/defaults.
    """
    import time

    cfg = cfg or load_accumulator_config()
    now = time.time()

    # Load existing state or initialise a fresh one
    state = store.get(session_id)
    if state is None:
        state = SessionState(
            session_id=session_id,
            created_at=now,
            last_updated_at=now,
        )

    # --- Dual-signal update ---
    state.ewma_score = update_ewma(state.ewma_score, turn_signal, cfg.alpha)
    state.peak_score = update_peak(state.peak_score, turn_signal, cfg.peak_decay)
    state.turn_count += 1
    state.last_updated_at = now

    # Fast-lane correction: treat a fired correction as an ADDITIONAL signal
    # spike of 1.0, blended on top of the detector-driven update above.
    if fast_lane_correction_fired:
        state.fast_lane_correction_count += 1
        state.ewma_score = update_ewma(state.ewma_score, 1.0, cfg.alpha)
        state.peak_score = update_peak(state.peak_score, 1.0, cfg.peak_decay)

    # --- Entity reconstruction window ---
    if pii_fragment:
        state.fragment_window.append(pii_fragment)
        # Keep only the last N turns to bound memory
        state.fragment_window = state.fragment_window[-cfg.fragment_window_turns:]

    # --- Tool-chain contamination tracking ---
    # Once sensitive data touches a tool, that tool is marked contaminated for
    # the life of the session (bounded by TTL). This is intentionally sticky.
    if data_classification and data_classification.lower() in ("sensitive", "restricted", "high", "restricted"):
        if tool_name and tool_name not in state.contaminated_tools:
            state.contaminated_tools.append(tool_name)
        if tool_name:
            state.contamination_active = True

    # --- Band classification ---
    state.last_band = classify_band(state.session_risk, cfg)

    # Persist
    store.set(session_id, state, ttl_seconds=cfg.ttl_seconds)
    return state
