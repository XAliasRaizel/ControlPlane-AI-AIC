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
            "injection_model_loaded": self._injection_model_loaded(),
        }

    @staticmethod
    def _injection_model_loaded() -> bool:
        """True only if a CONTROLPLANE_MODEL_INJECTION artifact loaded successfully."""
        try:
            from backend.shared.model_backend import get_detector_model
            return get_detector_model("injection") is not None
        except Exception:
            return False

    def score_with_model(self, text: str) -> float:
        """Calibrated injection risk when a model artifact is configured.

        Interface unchanged; still returns 0.0 by default (no artifact / no ML
        stack), so existing callers behave exactly as before. Delegates to the
        model_backend seam rather than hosting inference itself.
        """
        try:
            from backend.shared.model_backend import consult
            prediction = consult("injection", text)
            if prediction is not None:
                return float(prediction["score"])
        except Exception:
            pass
        return 0.0
