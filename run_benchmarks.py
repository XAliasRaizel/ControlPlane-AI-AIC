import os
import time
import numpy as np
from backend.shared.model_backend import get_grounding_scorer

def run_benchmark(model_path, num_iterations=100):
    os.environ["CONTROLPLANE_MODEL_GROUNDING"] = model_path
    scorer = get_grounding_scorer("grounding")
    print(f"Loading model from {model_path}...")
    if scorer and scorer._ensure_model():
        print("Warmup...")
        for _ in range(5):
            scorer.entailment("Paris is the capital and most populous city of France.", "The capital of France is Paris.")
        
        print(f"Benchmarking {num_iterations} iterations...")
        latencies = []
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            scorer.entailment("Paris is the capital and most populous city of France.", "The capital of France is Paris.")
            latencies.append((time.perf_counter() - t0) * 1000)
        
        latencies = np.array(latencies)
        print(f"Model: {model_path}")
        print(f"Average: {np.mean(latencies):.2f} ms")
        print(f"p50: {np.percentile(latencies, 50):.2f} ms")
        print(f"p95: {np.percentile(latencies, 95):.2f} ms")
        print(f"p99: {np.percentile(latencies, 99):.2f} ms")
        print("-" * 40)
    else:
        print(f"Failed to load {model_path}")

if __name__ == "__main__":
    run_benchmark("ml/artifacts/grounding-nli")
    run_benchmark("ml/artifacts/grounding-nli-large")
