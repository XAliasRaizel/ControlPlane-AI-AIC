"""Optional GPU adapter.

The default prototype does not require PyTorch. If PyTorch is installed,
this module reports whether CUDA is available. Replace the `score_with_model`
method with a real Transformer classifier in a production/research prototype.
"""


class GPUAdapter:
    def __init__(self):
        self.available = False
        self.device = "cpu"
        try:
            import torch
            self.available = bool(torch.cuda.is_available())
            self.device = "cuda" if self.available else "cpu"
        except ImportError:
            pass

    def status(self):
        return {
            "gpu_available": self.available,
            "device": self.device,
        }

    def score_with_model(self, text: str) -> float:
        # Placeholder for Transformer inference.
        # Keep this interface stable when replacing with a real GPU model.
        return 0.0
