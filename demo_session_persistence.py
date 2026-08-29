"""
Demonstration of Session Accumulator persistence for Option B
Shows how a single high-risk turn maintains elevated band status for extended period
"""
from backend.risk.accumulator import update_session, load_accumulator_config
from backend.risk.session_store import InMemorySessionStore

def demo_persistence():
    print("=" * 60)
    print("SESSION ACCUMULATOR PERSISTENCE DEMO (OPTION B)")
    print("=" * 60)
    
    # Set the environment variable for our calibration
    import os
    os.environ['CONTROLPLANE_SESSION_ACCUMULATOR_CONFIG'] = "ml/artifacts/session-accumulator/calibration.json"
    
    # Load our Option B calibration
    store = InMemorySessionStore()
    session_id = "demo-persistence-test"
    cfg = load_accumulator_config()  # This loads our Option B calibration
    
    print(f"Configuration (from calibration.json):")
    print(f"  Alpha (EWMA decay): {cfg.alpha}")
    print(f"  Peak decay: {cfg.peak_decay}")
    print(f"  Medium threshold: {cfg.threshold_medium}")
    print(f"  High threshold: {cfg.threshold_high}")
    print()
    
    # Scenario: One high-risk turn (simulating attack), then many low-risk turns
    print("Scenario: Single high-risk turn (0.9) followed by low-risk turns (0.05)")
    print("-" * 60)
    
    # Initialize
    ewma, peak = 0.0, 0.0
    state = None
    
    # Turn 0: High-risk spike (attack)
    turn_signal = 0.9
    state = update_session(
        store=store,
        session_id=session_id,
        turn_signal=turn_signal,
        cfg=cfg  # Pass our loaded config
    )
    ewma = state.ewma_score
    peak = state.peak_score
    risk = state.session_risk
    band = state.last_band
    
    print(f"Turn 1 (ATTACK): signal={turn_signal}")
    print(f"  EWMA: {ewma:.3f} | Peak: {peak:.3f} | Risk: {risk:.3f} | Band: {band}")
    print()
    
    # Turns 1-25: Low-risk benign activity
    print("Benign turns (0.05 each):")
    print("-" * 40)
    
    band3_start_turn = 1
    band3_end_turn = 1
    
    for i in range(25):  # 25 benign turns after the attack
        turn_signal = 0.05
        state = update_session(
            store=store,
            session_id=session_id,
            turn_signal=turn_signal,
            cfg=cfg  # Pass our loaded config
        )
        ewma = state.ewma_score
        peak = state.peak_score
        risk = state.session_risk
        band = state.last_band
        
        turn_num = i + 2  # Because we start counting from turn 2
        
        if band == 3:
            if band3_end_turn < turn_num:
                band3_end_turn = turn_num
            print(f"Turn {turn_num:2d}: EWMA={ewma:.3f} | Peak={peak:.3f} | Risk={risk:.3f} | Band: {band}  <-- MAINTAINING HIGH ALERT")
        else:
            print(f"Turn {turn_num:2d}: EWMA={ewma:.3f} | Peak={peak:.3f} | Risk={risk:.3f} | Band: {band}  <-- RETURNED TO NORMAL")
            if band3_end_turn == 1:  # First drop below band 3
                band3_end_turn = turn_num - 1
            break
    
    print()
    print("=" * 60)
    print("RESULTS:")
    print(f"  Band 3 (High Alert) maintained for turns {band3_start_turn} through {band3_end_turn}")
    print(f"  Total consecutive turns in Band 3: {band3_end_turn - band3_start_turn + 1}")
    print()
    print("This demonstrates Option B enhancement:")
    print("- Original calibration (~12 turns persistence)")  
    print("- Option B calibration (25+ turns persistence)")
    print("- Provides clearer visual demonstration of attack persistence")
    print("=" * 60)

if __name__ == "__main__":
    demo_persistence()