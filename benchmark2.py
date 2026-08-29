import os
import time
import asyncio
from backend.shared.model_backend import get_grounding_scorer

os.environ["CONTROLPLANE_MODEL_GROUNDING"] = "ml/artifacts/grounding-nli-large"

scorer = get_grounding_scorer("grounding")
print("Loading model...")
if scorer and scorer._ensure_model():
    print("Model loaded successfully")
    print("Warmup...")
    scorer.entailment("Paris is the capital and most populous city of France.", "The capital of France is Paris.")
    
    print("Benchmarking 10 iterations...")
    latencies = []
    for _ in range(10):
        t0 = time.time()
        res = scorer.entailment("Paris is the capital and most populous city of France.", "The capital of France is Paris.")
        t1 = time.time()
        latencies.append((t1 - t0) * 1000)
    
    avg_latency = sum(latencies) / len(latencies)
    print(f"Average latency: {avg_latency:.2f} ms")
    print(f"Result: {res}")
else:
    print("Failed to load model")
