import traceback
from backend.shared.model_backend import get_sensitive_intent_scorer

try:
    scorer = get_sensitive_intent_scorer()
    print("Scorer loaded:", scorer)
except Exception as e:
    traceback.print_exc()
