"""Compatibility package for app.* imports."""
import sys
from backend import agents, audit, decision, detectors, feedback, gateway, policy, review, risk, shared

# Alias backend submodules as app submodules
sys.modules.setdefault("app", sys.modules[__name__])
sys.modules.setdefault("app.agents", agents)
