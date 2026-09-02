"""Optional learned-detector inference seam (import is dependency-free).

Importing this module never imports torch/transformers. A model loads lazily
only when (a) a detector asks for its task model AND (b) an artifact directory
is configured via the environment variable CONTROLPLANE_MODEL_<TASK>, e.g.
CONTROLPLANE_MODEL_INJECTION=ml/artifacts/injection-v0/model

Every failure path -- variable unset, ML stack absent, artifact missing or
corrupt, inference error -- resolves to None / a no-op. So the deterministic
regex/rule pipeline keeps working unchanged when nothing is configured, and a
fine-tuned model can be dropped in later with no code change (a staged,
observe-first rollout). This is the runtime half of ml/train_detector.py.

Artifact layout produced by ml/train_detector.py:
    <artifact-dir>/model/            HF model + tokenizer (save_pretrained)
    <artifact-dir>/calibration.json  temperature, threshold, positive_label ...
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import json
import math
import os
import re
import threading
from pathlib import Path
from typing import Any, Optional

# Shared thread-pool for blocking ML inference calls (Presidio, SentenceTransformer).
# max_workers=4 is enough: we have at most 2 heavy calls per request in parallel.
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="cp-ml")

ENV_PREFIX = "CONTROLPLANE_MODEL_"

_cache: dict[str, Any] = {}
_lock = threading.Lock()


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _calibration_path(artifact_dir: Path) -> Optional[Path]:
    for candidate in (artifact_dir.parent / "calibration.json",
                      artifact_dir / "calibration.json"):
        if candidate.exists():
            return candidate
    return None


class CalibratedClassifier:
    """Lazy wrapper around a fine-tuned HF sequence classifier + calibration.

    score/predict return None if the ML stack or artifact cannot be loaded, so
    callers stay on their deterministic fallback.
    """

    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._device = "cpu"  # updated in _ensure_loaded
        self.temperature = 1.0
        self.threshold = 0.5
        self.positive_label = "POSITIVE"
        self.positive_index = 1
        self.max_length = 256
        self._read_calibration()

    def _read_calibration(self) -> None:
        path = _calibration_path(self.artifact_dir)
        if path is None:
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self.temperature = float(data.get("temperature", 1.0)) or 1.0
        self.threshold = float(data.get("threshold", 0.5))
        self.positive_label = str(data.get("positive_label", self.positive_label))
        self.positive_index = int(data.get("positive_index", 1))
        self.max_length = int(data.get("max_length", 256))

    @classmethod
    def try_load(cls, artifact_dir: str | Path) -> "Optional[CalibratedClassifier]":
        try:
            if not Path(artifact_dir).exists():
                return None
            return cls(artifact_dir)
        except Exception:
            return None

    def _ensure_loaded(self):
        if self._model is not None:
            return

        calib_path = _calibration_path(self.artifact_dir)
        if calib_path:
            with open(calib_path, "r", encoding="utf-8") as f:
                self.calibration = json.load(f)

        quantized_onnx_dir = self.artifact_dir / "model_quantized_onnx"
        onnx_dir = self.artifact_dir / "model_onnx"
        model_dir = self.artifact_dir / "model"
        
        import torch
        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Try INT8 Quantized ONNX first
        if self.calibration.get("quantized_onnx", False) and quantized_onnx_dir.exists():
            try:
                from optimum.onnxruntime import ORTModelForSequenceClassification
                from transformers import AutoTokenizer
                self._model = ORTModelForSequenceClassification.from_pretrained(str(quantized_onnx_dir), file_name="model_quantized.onnx")
                self._tokenizer = AutoTokenizer.from_pretrained(str(quantized_onnx_dir))
                return
            except ImportError:
                pass # fallback to fp32 onnx or pytorch

        # Try FP32 ONNX second
        if self.calibration.get("onnx", False) and onnx_dir.exists():
            try:
                from optimum.onnxruntime import ORTModelForSequenceClassification
                from transformers import AutoTokenizer
                self._model = ORTModelForSequenceClassification.from_pretrained(str(onnx_dir))
                self._tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))
                return
            except ImportError:
                pass # fallback to pytorch if optimum not installed

        # PyTorch fallback — move to GPU if available
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self._model.eval()
        self._model = self._model.to(self._device)

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            self._ensure_loaded()
            return True
        except Exception:
            self._model = None
            return False

    def score(self, text: str) -> Optional[float]:
        """Calibrated probability of the positive (risky) class, or None."""
        if not self._ensure_model():
            return None
        try:
            torch = self._torch
            enc = self._tokenizer(text, truncation=True,
                                  max_length=self.max_length, return_tensors="pt")
            # Move input tensors to the same device as the model
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self._model(**enc).logits[0]
            pos = float(logits[self.positive_index].item())
            neg = float(logits[1 - self.positive_index].item())
            return _sigmoid((pos - neg) / (self.temperature or 1.0))
        except Exception:
            return None

    def predict(self, text: str) -> Optional[dict]:
        s = self.score(text)
        if s is None:
            return None
        fires = s >= self.threshold
        denom = max(self.threshold, 1 - self.threshold, 1e-6)
        margin = min(1.0, abs(s - self.threshold) / denom)
        return {
            "score": round(float(s), 6),
            "fires": bool(fires),
            "confidence": round(0.5 + 0.5 * margin, 6),
            "label": self.positive_label if fires else "CLEAN",
            "threshold": round(float(self.threshold), 6),
        }


class GroundingScorer:
    """Optional NLI groundedness scorer (fine-tuning plan: grounding = NLI
    cross-encoder / HHEM-style classifier scored per-claim against retrieved
    chunks). groundedness() returns risk in [0,1] where higher = LESS grounded
    (more likely hallucinated), so it composes with the other risk signals.
    """

    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        self._model = None
        self._tokenizer = None
        self._torch = None
        self.max_length = 512
        self.entail_index: Optional[int] = None

    @classmethod
    def try_load(cls, artifact_dir: str | Path) -> "Optional[GroundingScorer]":
        try:
            if not Path(artifact_dir).exists():
                return None
            return cls(artifact_dir)
        except Exception:
            return None

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torch
            self._torch = torch
            
            quantized_onnx_dir = self.artifact_dir / "model_quantized_onnx"
            onnx_dir = self.artifact_dir / "model_onnx"
            model_dir = self.artifact_dir / "model"
            
            loaded = False
            # Try INT8 Quantized ONNX first
            if quantized_onnx_dir.exists():
                try:
                    from optimum.onnxruntime import ORTModelForSequenceClassification
                    from transformers import AutoTokenizer
                    self._model = ORTModelForSequenceClassification.from_pretrained(str(quantized_onnx_dir), file_name="model_quantized.onnx")
                    self._tokenizer = AutoTokenizer.from_pretrained(str(quantized_onnx_dir))
                    loaded = True
                except ImportError:
                    pass

            # Try FP32 ONNX second
            if not loaded and onnx_dir.exists():
                try:
                    from optimum.onnxruntime import ORTModelForSequenceClassification
                    from transformers import AutoTokenizer
                    self._model = ORTModelForSequenceClassification.from_pretrained(str(onnx_dir))
                    self._tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))
                    loaded = True
                except ImportError:
                    pass

            # PyTorch fallback
            if not loaded:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
                self._model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
                self._model.eval()

            id2label = {int(k): str(v).lower()
                        for k, v in getattr(self._model.config, "id2label", {}).items()}
            for i, label in id2label.items():
                if "entail" in label:
                    self.entail_index = i
            return True
        except Exception:
            self._model = None
            return False

    def entailment(self, premise: str, hypothesis: str) -> Optional[float]:
        if not self._ensure_model():
            return None
        try:
            torch = self._torch
            enc = self._tokenizer(premise, hypothesis, truncation=True,
                                  max_length=self.max_length, return_tensors="pt")
            with torch.no_grad():
                probs = torch.softmax(self._model(**enc).logits[0], dim=-1)
            idx = self.entail_index if self.entail_index is not None else probs.shape[-1] - 1
            return float(probs[idx].item())
        except Exception:
            return None

    @staticmethod
    def _claims(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
        return [p for p in parts if len(p.split()) >= 3]

    def groundedness(self, response: str, contexts: list[str]) -> Optional[dict]:
        if not contexts:
            return None
        claims = self._claims(response) or [response or ""]
        weakest, per_claim = 1.0, []
        for claim in claims:
            best = 0.0
            for ctx in contexts:
                e = self.entailment(ctx, claim)
                if e is None:
                    return None
                best = max(best, e)
            per_claim.append({"claim": claim, "entailment": round(best, 4)})
            weakest = min(weakest, best)
        return {
            "risk": round(1.0 - weakest, 6),
            "weakest_entailment": round(weakest, 6),
            "claims": per_claim,
        }


class SensitiveIntentScorer:
    """Optional semantic matcher for sensitive query intent (PII/Auth)."""

    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        self._model = None
        self.threshold = 0.5
        self.positive_anchors = []
        self.negative_anchors = []
        self.pos_embs = []
        self.neg_embs = []
        self._read_calibration()

    @classmethod
    def try_load(cls, artifact_dir: str | Path) -> "Optional[SensitiveIntentScorer]":
        try:
            if not Path(artifact_dir).exists():
                return None
            return cls(artifact_dir)
        except Exception:
            return None

    def _read_calibration(self) -> None:
        path = _calibration_path(self.artifact_dir)
        if path is None:
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self.threshold = float(data.get("threshold", 0.5))
        self.positive_anchors = data.get("positive_anchors", [])
        self.negative_anchors = data.get("negative_anchors", [])

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            # Try to load from "model" subdirectory, otherwise from the artifact_dir directly
            model_path = self.artifact_dir / "model"
            if not model_path.exists():
                model_path = self.artifact_dir
            # Prefer GPU for fast embedding inference (~5ms vs ~25ms on CPU)
            try:
                import torch
                _device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                _device = "cpu"
            self._model = SentenceTransformer(str(model_path), device=_device)

            # Pre-compute anchor embeddings as numpy 2D arrays for vectorized cosine sim
            if self.positive_anchors:
                pos = self._model.encode(self.positive_anchors, convert_to_numpy=True, show_progress_bar=False)
                # Normalize rows so cosine sim = dot product
                norms = np.linalg.norm(pos, axis=1, keepdims=True)
                self.pos_embs = pos / np.where(norms == 0, 1.0, norms)  # shape [n_pos, dim]
            else:
                self.pos_embs = None
            if self.negative_anchors:
                neg = self._model.encode(self.negative_anchors, convert_to_numpy=True, show_progress_bar=False)
                norms = np.linalg.norm(neg, axis=1, keepdims=True)
                self.neg_embs = neg / np.where(norms == 0, 1.0, norms)  # shape [n_neg, dim]
            else:
                self.neg_embs = None
            return True
        except Exception:
            self._model = None
            return False

    def _max_sim_np(self, query_emb_normalized, anchor_embs_normalized) -> float:
        """Vectorized max cosine similarity using pre-normalized numpy arrays."""
        if anchor_embs_normalized is None or len(anchor_embs_normalized) == 0:
            return 0.0
        # query: [dim], anchors: [n, dim] -> sims: [n]
        sims = anchor_embs_normalized @ query_emb_normalized
        return float(sims.max())

    def is_targeted_request(self, text: str) -> Optional[tuple[float, bool]]:
        """Returns (margin, fires) or None if model unavailable."""
        if not self._ensure_model():
            return None
        try:
            import numpy as np
            # Encode query and normalize
            query_emb = self._model.encode(text, convert_to_numpy=True, show_progress_bar=False)
            norm = np.linalg.norm(query_emb)
            if norm > 0:
                query_emb = query_emb / norm
            # Vectorized cosine sim against all anchors in one matmul
            margin = self._max_sim_np(query_emb, self.pos_embs) - self._max_sim_np(query_emb, self.neg_embs)
            fires = margin >= self.threshold
            return float(margin), bool(fires)
        except Exception:
            return None


def reset_cache() -> None:
    """Clear all lazy-loaded models and caches (e.g. for testing or hot-reloading)."""
    with _lock:
        _cache.clear()
    consult.cache_clear()
    consult_presidio.cache_clear()


# ---------------------------------------------------------------------------
# Module-level API used by detectors. Everything here is cached and thread-safe
# and NEVER raises: an unset env var, a missing artifact, or an unavailable ML
# stack all resolve to None, so the deterministic pipeline is unaffected.
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1000)
def consult(task: str, text: str) -> Optional[dict]:
    """Guarded convenience wrapper for detectors.

    Returns the calibrated predict() dict (score/fires/confidence/label/
    threshold) when a model is configured and usable, otherwise None. Swallows
    every error so a detector can call this inline without its own try/except
    and always fall back to its deterministic result.
    """
    try:
        clf = get_detector_model(task)
        if clf is None:
            return None
        return clf.predict(text)
    except Exception:
        return None


@functools.lru_cache(maxsize=1000)
def _consult_presidio_sync(text: str) -> list[str]:
    """Synchronous Presidio call (cached). Called via executor from async wrapper."""
    key = "presidio::analyzer"
    with _lock:
        analyzer = _cache.get(key)

    if analyzer is None:
        try:
            from presidio_analyzer import AnalyzerEngine
            analyzer = AnalyzerEngine()
            with _lock:
                _cache[key] = analyzer
        except ImportError:
            return []
        except Exception:
            return []

    try:
        results = analyzer.analyze(text=text, language="en")
        seen = set()
        unique = []
        for r in results:
            if r.entity_type not in seen:
                seen.add(r.entity_type)
                unique.append(r.entity_type)
        return unique
    except Exception:
        return []


def consult_presidio(text: str) -> list[str]:
    """Sync shim — kept for backward-compat with tests and non-async callers."""
    return _consult_presidio_sync(text)


async def aconsult_presidio(text: str) -> list[str]:
    """Async wrapper: runs Presidio in a thread so the event loop stays free."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_EXECUTOR, _consult_presidio_sync, text)


@functools.lru_cache(maxsize=1000)
def _consult_sensitive_intent_sync(text: str) -> Optional[tuple[float, bool]]:
    """Synchronous intent scorer call (cached). Called via executor from async wrapper."""
    try:
        scorer = get_sensitive_intent_scorer()
        if scorer is None:
            return None
        return scorer.is_targeted_request(text)
    except Exception:
        return None


def consult_sensitive_intent(text: str) -> Optional[tuple[float, bool]]:
    """Sync shim — kept for backward-compat with tests and non-async callers."""
    return _consult_sensitive_intent_sync(text)


async def aconsult_sensitive_intent(text: str) -> Optional[tuple[float, bool]]:
    """Async wrapper: runs SentenceTransformer.encode() in a thread so the event loop stays free."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_EXECUTOR, _consult_sensitive_intent_sync, text)


def artifact_dir_for(task: str) -> Optional[str]:
    """Return the configured artifact directory for a task, or None.

    Reads CONTROLPLANE_MODEL_<TASK> (task upper-cased). Unset resolves to None,
    which keeps the deterministic pipeline fully in charge.
    """
    if not task:
        return None
    value = os.environ.get(ENV_PREFIX + task.upper())
    if not value:
        return None
    value = value.strip()
    return value or None


def get_detector_model(task: str) -> Optional[CalibratedClassifier]:
    """Cached, never-raising accessor for a task classifier.

    Returns None when the task is not configured, the artifact is missing, or
    the ML stack is absent, so a caller can write:

        model = get_detector_model("injection")
        if model is not None:
            hit = model.predict(text)

    and otherwise stay entirely on the deterministic regex path.
    """
    key = "clf::" + (task or "")
    with _lock:
        if key in _cache:
            return _cache[key]
    model = None
    try:
        artifact = artifact_dir_for(task)
        if artifact is not None:
            model = CalibratedClassifier.try_load(artifact)
    except Exception:
        model = None
    with _lock:
        _cache[key] = model
    return model


def get_grounding_scorer(task: str = "grounding") -> Optional[GroundingScorer]:
    """Cached, never-raising accessor for the NLI grounding scorer."""
    key = "nli::" + (task or "grounding")
    with _lock:
        if key in _cache:
            return _cache[key]
    scorer = None
    try:
        artifact = artifact_dir_for(task)
        if artifact is not None:
            scorer = GroundingScorer.try_load(artifact)
    except Exception:
        scorer = None
    with _lock:
        _cache[key] = scorer
    return scorer


def get_sensitive_intent_scorer(task: str = "sensitive_intent") -> Optional[SensitiveIntentScorer]:
    """Cached, never-raising accessor for the sensitive intent scorer."""
    key = "intent::" + (task or "sensitive_intent")
    with _lock:
        if key in _cache:
            return _cache[key]
    scorer = None
    try:
        artifact = artifact_dir_for(task)
        if artifact is not None:
            scorer = SensitiveIntentScorer.try_load(artifact)
    except Exception:
        scorer = None
    with _lock:
        _cache[key] = scorer
    return scorer


def reset_cache() -> None:
    """Drop cached models. Used by tests and after environment changes."""
    with _lock:
        _cache.clear()
    try:
        _consult_sensitive_intent_sync.cache_clear()
    except Exception:
        pass
