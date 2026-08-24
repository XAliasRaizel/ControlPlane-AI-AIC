"""Auto-import all detector modules to trigger @register decorators."""
from backend.detectors import pii  # noqa: F401
from backend.detectors import injection  # noqa: F401
from backend.detectors import authorization  # noqa: F401
from backend.detectors import safety  # noqa: F401

from backend.detectors.base import DETECTOR_REGISTRY, run_hot_path  # noqa: F401
