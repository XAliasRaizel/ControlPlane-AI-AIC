import json
from pathlib import Path

from backend.risk.accumulator import AccumulatorConfig, update_ewma, update_peak

def test_peak_dilution_guarantee():
    """
    Verifies that the dual-signal correctly holds risk > 0.7 for 10 turns
    after a 0.9 spike, proving peak_dilution is fixed.
    Loads calibration.json to assert against actual prod config values.
    """
    calib_path = Path("ml/artifacts/session-accumulator/calibration.json")
    assert calib_path.exists(), "Calibration artifact missing"
    
    data = json.loads(calib_path.read_text(encoding="utf-8"))
    
    alpha = data["alpha"]
    peak_decay = data["peak_decay"]
    threshold_high = data.get("threshold_high", 0.7)
    
    ewma, peak = 0.0, 0.0
    
    # 1. Turn 1: Spike of 0.9
    spike = 0.9
    ewma = update_ewma(ewma, spike, alpha)
    peak = update_peak(peak, spike, peak_decay)
    
    # Session risk must be exactly the spike (or very close)
    assert max(ewma, peak) >= spike
    
    # 2. 10 follow-up turns at benign signal (0.05)
    benign = 0.05
    for i in range(10):
        ewma = update_ewma(ewma, benign, alpha)
        peak = update_peak(peak, benign, peak_decay)
        
    session_risk = max(ewma, peak)
    
    # 3. Assert the risk is STILL > threshold_high
    assert session_risk >= threshold_high, (
        f"Peak dilution failed: risk after 10 turns was {session_risk}, "
        f"expected >= {threshold_high}. "
        f"(alpha={alpha}, peak_decay={peak_decay})"
    )
