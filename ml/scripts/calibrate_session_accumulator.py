"""Calibration script for the Session Risk Accumulator.

Sweeps (alpha, peak_decay) pairs against four synthetic scenarios to find
calibrated config values. Pure arithmetic — no GPU, no model inference, no
heavy dependencies required. Mirrors the evaluate_model.py / compare_detectors.py
pattern: produces a report and writes the artifact that accumulator.py reads via
CONTROLPLANE_SESSION_ACCUMULATOR_CONFIG.

Usage:
    python ml/scripts/calibrate_session_accumulator.py
    python ml/scripts/calibrate_session_accumulator.py --out ml/artifacts/session-accumulator/calibration.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import date
from itertools import product
from pathlib import Path

# Make sure repo root is on sys.path when run directly
_REPO_ROOT = Path(__file__).resolve().parents[2]  # ml/scripts/.. -> ml/.. -> repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.risk.accumulator import (
    AccumulatorConfig,
    update_ewma,
    update_peak,
    classify_band,
)

DEFAULT_OUT = _REPO_ROOT / "ml" / "artifacts" / "session-accumulator" / "calibration.json"


# ---------------------------------------------------------------------------
# Scenario runners (pure arithmetic, no I/O)
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    name: str
    passed: bool
    detail: dict


def run_multi_turn_evasion(
    alpha: float,
    peak_decay: float,
    threshold_medium: float,
    signal: float = 0.45,
    max_turns: int = 20,
    target_turns: int = 5,
) -> ScenarioResult:
    """N turns each at turn_signal ~0.45: above the session threshold_medium (0.4)
    but below the per-turn enforcement threshold (~0.5). Each turn is
    individually unremarkable; the accumulator catches the pattern.

    Find (alpha, peak_decay) where session_risk crosses threshold_medium within
    target_turns. With signal=0.45 and threshold_medium=0.4, the EWMA
    asymptote is 0.45, so it will always eventually cross 0.4 -- the question
    is how fast (controlled by alpha).
    """
    ewma, peak = 0.0, 0.0
    trigger_turn = None
    for t in range(1, max_turns + 1):
        ewma = update_ewma(ewma, signal, alpha)
        peak = update_peak(peak, signal, peak_decay)
        session_risk = max(ewma, peak)
        if session_risk >= threshold_medium and trigger_turn is None:
            trigger_turn = t

    passed = trigger_turn is not None and trigger_turn <= target_turns
    return ScenarioResult(
        name="multi_turn_evasion",
        passed=passed,
        detail={"turns_to_trigger": trigger_turn, "target_turns": target_turns, "passed": passed},
    )


def run_peak_dilution(
    alpha: float,
    peak_decay: float,
    threshold_medium: float,
    threshold_high: float,
    spike_signal: float = 0.9,
    benign_signal: float = 0.05,
    required_survival_turns: int = 10,
) -> ScenarioResult:
    """One turn at 0.9, then M turns at 0.05.

    The dual-signal design should keep session_risk >= threshold_medium for at
    least required_survival_turns follow-up turns (via the peak component),
    while EWMA-only would have dropped below threshold_medium by turn 10.
    This proves the dual-signal design does something EWMA alone would not.
    """
    ewma, peak = 0.0, 0.0
    # Spike turn
    ewma = update_ewma(ewma, spike_signal, alpha)
    peak = update_peak(peak, spike_signal, peak_decay)

    survived = 0
    ewma_only_scores = []
    for _ in range(required_survival_turns):
        ewma = update_ewma(ewma, benign_signal, alpha)
        peak = update_peak(peak, benign_signal, peak_decay)
        ewma_only_scores.append(round(ewma, 6))
        if max(ewma, peak) >= threshold_high:
            survived += 1

    ewma_only_at_10 = ewma_only_scores[-1] if ewma_only_scores else 0.0
    # Two conditions: dual-signal survives, AND EWMA-alone would have dropped
    ewma_would_have_dropped = ewma_only_at_10 < threshold_high
    passed = survived >= required_survival_turns and ewma_would_have_dropped
    return ScenarioResult(
        name="peak_dilution",
        passed=passed,
        detail={
            "turns_survived_at_threshold_high": survived,
            "ewma_only_at_turn_10": ewma_only_at_10,
            "ewma_only_below_threshold_high_at_turn_10": ewma_would_have_dropped,
            "passed": passed,
        },
    )


def run_pure_benign_control(
    alpha: float,
    peak_decay: float,
    threshold_medium: float,
    turns: int = 50,
    signal_low: float = 0.05,
    signal_high: float = 0.10,
) -> ScenarioResult:
    """50 turns all at 0.05-0.10. session_risk must never reach threshold_medium."""
    import random
    rng = random.Random(42)  # deterministic
    ewma, peak = 0.0, 0.0
    max_risk = 0.0
    for _ in range(turns):
        signal = rng.uniform(signal_low, signal_high)
        ewma = update_ewma(ewma, signal, alpha)
        peak = update_peak(peak, signal, peak_decay)
        max_risk = max(max_risk, ewma, peak)

    passed = max_risk < threshold_medium
    return ScenarioResult(
        name="pure_benign_control",
        passed=passed,
        detail={"max_session_risk_observed": round(max_risk, 6), "passed": passed},
    )


def run_steady_medium_control(
    alpha: float,
    peak_decay: float,
    threshold_medium: float,
    threshold_high: float,
    turns: int = 20,
    signal_low: float = 0.35,
    signal_high: float = 0.45,
) -> ScenarioResult:
    """Sustained turns at 0.35-0.45. Must land in band 2, NOT spuriously reach band 3."""
    import random
    rng = random.Random(99)  # deterministic
    cfg = AccumulatorConfig(
        alpha=alpha,
        peak_decay=peak_decay,
        threshold_medium=threshold_medium,
        threshold_high=threshold_high,
    )
    ewma, peak = 0.0, 0.0
    reached_band3 = False
    final_band = 1
    for _ in range(turns):
        signal = rng.uniform(signal_low, signal_high)
        ewma = update_ewma(ewma, signal, alpha)
        peak = update_peak(peak, signal, peak_decay)
        session_risk = max(ewma, peak)
        band = classify_band(session_risk, cfg)
        if band == 3:
            reached_band3 = True
        final_band = band

    passed = not reached_band3 and final_band == 2
    return ScenarioResult(
        name="steady_medium_control",
        passed=passed,
        detail={"band_reached": final_band, "spurious_band3": reached_band3, "passed": passed},
    )


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_all_scenarios(
    alpha: float,
    peak_decay: float,
    threshold_medium: float,
    threshold_high: float,
) -> tuple[list[ScenarioResult], bool]:
    results = [
        run_multi_turn_evasion(alpha, peak_decay, threshold_medium),
        run_peak_dilution(alpha, peak_decay, threshold_medium, threshold_high),
        run_pure_benign_control(alpha, peak_decay, threshold_medium),
        run_steady_medium_control(alpha, peak_decay, threshold_medium, threshold_high),
    ]
    all_passed = all(r.passed for r in results)
    return results, all_passed


def sweep(
    alphas: list[float],
    peak_decays: list[float],
    threshold_medium: float,
    threshold_high: float,
) -> tuple[float, float, list[ScenarioResult]] | None:
    """Return the first (alpha, peak_decay) combination where all scenarios pass."""
    best = None
    for alpha, peak_decay in product(alphas, peak_decays):
        results, passed = run_all_scenarios(alpha, peak_decay, threshold_medium, threshold_high)
        if passed:
            best = (alpha, peak_decay, results)
            break  # take first passing combination
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the Session Risk Accumulator.")
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Output path for calibration.json",
    )
    parser.add_argument(
        "--threshold-medium", type=float, default=0.4,
        help="Band 1->2 boundary to test against (default: 0.4)",
    )
    parser.add_argument(
        "--threshold-high", type=float, default=0.7,
        help="Band 2->3 boundary to test against (default: 0.7)",
    )
    args = parser.parse_args()

    threshold_medium: float = args.threshold_medium
    threshold_high: float = args.threshold_high

    # Sweep grid — finer granularity to find peak_decay >= 0.98
    alphas = [round(v * 0.01, 2) for v in range(1, 100)]        # 0.01 .. 0.99
    peak_decays = [round(v * 0.01, 2) for v in range(1, 100)]  # 0.01 .. 0.99

    print(f"Sweeping {len(alphas)} x {len(peak_decays)} = {len(alphas)*len(peak_decays)} combinations...")

    result = sweep(alphas, peak_decays, threshold_medium, threshold_high)

    if result is None:
        print("\n[FAIL] No (alpha, peak_decay) combination passed all scenarios.")
        print("       Consider adjusting threshold_medium / threshold_high or expanding the grid.")
        sys.exit(1)

    best_alpha, best_peak_decay, scenario_results = result

    print(f"\n[PASS] Calibrated values found:")
    print(f"       alpha        = {best_alpha}")
    print(f"       peak_decay   = {best_peak_decay}")
    print(f"       threshold_medium = {threshold_medium}")
    print(f"       threshold_high   = {threshold_high}")
    print()

    for sr in scenario_results:
        status = "PASS" if sr.passed else "FAIL"
        print(f"  [{status}] {sr.name}: {sr.detail}")

    # Build output artifact
    calibration = {
        "alpha": best_alpha,
        "peak_decay": best_peak_decay,
        "threshold_medium": threshold_medium,
        "threshold_high": threshold_high,
        "ttl_seconds": 1800,
        "fragment_window_turns": 5,
        "calibrated_on": str(date.today()),
        "scenarios": {sr.name: sr.detail for sr in scenario_results},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print(f"\nCalibration artifact written to: {out_path}")


if __name__ == "__main__":
    main()
