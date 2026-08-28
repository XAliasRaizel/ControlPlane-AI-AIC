import os
import time
import asyncio
from backend.shared.model_backend import get_grounding_scorer
from backend.shared.schemas import GovernanceRequest
from backend.detectors.async_analytics import GroundingEngineDetector

os.environ["CONTROLPLANE_MODEL_GROUNDING"] = "ml/artifacts/grounding-nli"

async def benchmark():
    scorer = get_grounding_scorer("grounding")
    print("Scorer loaded:", scorer is not None)
    
    req = GovernanceRequest(
        request_id="req-123",
        user_id="u1",
        application_id="app1",
        prompt="What is the capital of France?",
        response="The capital of France is Paris.",
        retrieved_context=["Paris is the capital and most populous city of France."]
    )
    
    detector = GroundingEngineDetector()
    
    # Warmup
    print("Running warmup...")
    await detector.analyze(req, {})
    
    print("Running benchmark (10 iterations)...")
    latencies = []
    for _ in range(10):
        t0 = time.time()
        result = await detector.analyze(req, {})
        t1 = time.time()
        latencies.append((t1 - t0) * 1000)
        
    avg_latency = sum(latencies) / len(latencies)
    print(f"Average latency: {avg_latency:.2f} ms")
    print(f"Result score: {result.score}, label: {result.label}, evidence: {result.evidence}")

if __name__ == "__main__":
    asyncio.run(benchmark())
