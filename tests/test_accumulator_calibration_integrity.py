import json
from pathlib import Path

import pytest

from backend.risk.accumulator import update_ewma, update_peak

_CALIB_PATH = (
    Path(__file__).parent.parent
    / "ml" / "artifacts" / "session-accumulator" / "calibration.json"
)


@pytest.mark.skipif(
    not _CALIB_PATH.exists(),
    reason="ml/artifacts/ is gitignored; calibration.json not present in CI — "
           "run locally after generating the artifact.",
)
def test_peak_dilution_guarantee():
    """
    Verifies that the dual-signal correctly holds session_risk > threshold_high
    for 10 turns after a 0.9 spike, proving the peak-dilution fix is in effect.

    Asserts against the actual prod calibration.json values.  Skipped in CI
    because ml/artifacts/ is gitignored.
    """
    data = json.loads(_CALIB_PATH.read_text(encoding="utf-8"))
    alpha = data["alpha"]
    peak_decay = data["peak_decay"]
    threshold_high = data.get("threshold_high", 0.7)

    ewma, peak = 0.0, 0.0

    # Turn 1: spike of 0.9
    spike = 0.9
    ewma = update_ewma(ewma, spike, alpha)
    peak = update_peak(peak, spike, peak_decay)
    assert max(ewma, peak) >= spike

    # 10 follow-up benign turns
    benign = 0.05
    for _ in range(10):
        ewma = update_ewma(ewma, benign, alpha)
        peak = update_peak(peak, benign, peak_decay)

    session_risk = max(ewma, peak)

    assert session_risk >= threshold_high, (
        f"Peak dilution failed: risk after 10 benign turns was {session_risk:.4f}, "
        f"expected >= {threshold_high}. "
        f"(alpha={alpha}, peak_decay={peak_decay})"
    )


def test_peak_dilution_math_baseline():
    """
    Pure-math baseline test that runs in CI without any artifact.
    Verifies that with Option-B-calibrated values (alpha=0.01, peak_decay=0.99,
    threshold_high=0.7), the guarantee holds for 10+ turns.
    These are the same values recorded in the calibration.json artifact.
    """
    alpha = 0.01
    peak_decay = 0.99
    threshold_high = 0.7

    ewma, peak = 0.0, 0.0
    spike = 0.9
    ewma = update_ewma(ewma, spike, alpha)
    peak = update_peak(peak, spike, peak_decay)
    assert max(ewma, peak) >= spike

    benign = 0.05
    for _ in range(10):
        ewma = update_ewma(ewma, benign, alpha)
        peak = update_peak(peak, benign, peak_decay)

    session_risk = max(ewma, peak)

    assert session_risk >= threshold_high, (
        f"Baseline math check failed: {session_risk:.4f} < {threshold_high} "
        f"(alpha={alpha}, peak_decay={peak_decay})"
    )
