import json
import numpy as np
import os
from sklearn.metrics import precision_recall_curve, roc_curve

def find_best_threshold():
    from ml.common import load_jsonl_records
    from backend.shared.model_backend import get_detector_model
    
    os.environ["CONTROLPLANE_MODEL_FAIRNESS"] = "ml/artifacts/fairness-v1"
    model = get_detector_model("fairness")
    if not model or not model._ensure_model():
        print("Model not loaded")
        return
        
    records = load_jsonl_records("data/fairness.jsonl")
    texts = [r["text"] for r in records]
    labels = [1 if r["label"] == "BIASED" else 0 for r in records]
    
    print("Scoring...")
    scores = []
    for i, text in enumerate(texts):
        # We only need the score for the positive class
        res = model.predict(text)
        scores.append(res["score"])
        if i % 1000 == 0:
            print(f"Scored {i}/{len(texts)}")
            
    scores = np.array(scores)
    labels = np.array(labels)
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    
    # Target <= 5% FPR
    idx = np.where(fpr <= 0.05)[0][-1]  # Last index where FPR <= 5%
    best_thresh = thresholds[idx]
    best_fpr = fpr[idx]
    best_fnr = fnr[idx]
    
    print(f"Targeting <= 5% FPR:")
    print(f"Threshold: {best_thresh:.4f}")
    print(f"FPR: {best_fpr:.4f}")
    print(f"FNR: {best_fnr:.4f}")

    # Update calibration.json
    with open("ml/artifacts/fairness-v1/calibration.json", "r") as f:
        calib = json.load(f)
    
    calib["threshold"] = float(best_thresh)
    
    with open("ml/artifacts/fairness-v1/calibration.json", "w") as f:
        json.dump(calib, f, indent=2)
    print("Updated calibration.json")
    
if __name__ == "__main__":
    find_best_threshold()
