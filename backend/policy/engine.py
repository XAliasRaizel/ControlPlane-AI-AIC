from typing import Any
from backend.shared.schemas import GovernanceRequest, RiskAssessment, DetectorResult, PolicyMatch, PolicySummary
from backend.policy.loader import PolicyLoader, PolicyConfigurationError

class PolicyEngine:
    def __init__(self):
        self.loader = PolicyLoader()

    def evaluate(self, request: GovernanceRequest, risk: RiskAssessment, context: dict) -> PolicyMatch:
        detector_map = {d.detector_name: d for d in risk.detector_results}
        
        # Section 8 precedence: application-scoped > department-scoped > global.
        def scope_priority(pset):
            scope = pset.get("scope", {})
            if isinstance(scope, dict):
                apps = scope.get("applications", [])
                depts = scope.get("departments", [])
                if apps and request.application_id in apps:
                    return 3  # application-scoped match
                if depts and request.department in depts:
                    return 2  # department-scoped match
                if not apps and not depts:
                    return 1  # global (no scope restrictions)
                return 0  # scoped but doesn't match this request
            return 1  # treat as global if scope is not a dict
            
        # Filter to only matching policy sets, then sort by specificity
        applicable = [(scope_priority(ps), ps) for ps in self.loader.policy_sets]
        applicable = [(p, ps) for p, ps in applicable if p > 0]
        applicable.sort(key=lambda x: x[0], reverse=True)
        
        for _, pset in applicable:
            # Sort rules top-to-bottom by priority desc
            rules = sorted(pset["rules"], key=lambda r: int(r.get("priority", 0)), reverse=True)
            for rule in rules:
                if self._matches(rule["when"], request, risk, detector_map):
                    return PolicyMatch(
                        policy_id=rule["id"],
                        policy_name=pset["policy_set"],
                        matched_condition=str(rule["when"]),
                        recommended_action=rule["action"]
                    )
        
        return PolicyMatch(
            policy_id='default-allow',
            policy_name='global',
            matched_condition='no rule matched',
            recommended_action='ALLOW'
        )

    @staticmethod
    def _matches(
        when: dict[str, Any],
        request: GovernanceRequest,
        risk: RiskAssessment,
        detector_map: dict[str, DetectorResult],
    ) -> bool:
        if not isinstance(when, dict):
            return False

        detector_name = when.get("detector_triggered")
        if detector_name:
            detector = detector_map.get(detector_name)
            # Triggered if score > 0 and label != CLEAN
            if not detector or detector.score <= 0 or detector.label == "CLEAN":
                return False

        for name in when.get("all_detectors_triggered", []):
            detector = detector_map.get(name)
            if not detector or detector.score <= 0 or detector.label == "CLEAN":
                return False

        any_detectors = when.get("any_detector_triggered")
        if any_detectors:
            triggered = False
            for name in any_detectors:
                d = detector_map.get(name)
                if d and d.score > 0 and d.label != "CLEAN":
                    triggered = True
                    break
            if not triggered:
                return False

        for name, minimum in when.get("detector_score_at_least", {}).items():
            if not detector_map.get(name) or detector_map[name].score < float(minimum):
                return False

        if "risk_at_least" in when and risk.overall_risk < float(when["risk_at_least"]):
            return False
        if "application_in" in when and request.application_id not in set(when["application_in"]):
            return False
        if "data_classification_in" in when and (request.data_classification or "").upper() not in {
            value.upper() for value in when["data_classification_in"]
        }:
            return False
        return True

    def summary(self) -> PolicySummary:
        # Just returning first set summary for simplicity or aggregate
        if not self.loader.policy_sets:
            return PolicySummary(policy_name="empty", policy_version="unknown", rules=[])
        pset = self.loader.policy_sets[0]
        return PolicySummary(
            policy_name=pset["policy_set"],
            policy_version=pset["version"],
            rules=pset["rules"]
        )

policy_engine = PolicyEngine()

def evaluate_policy(request: GovernanceRequest, risk: RiskAssessment, context: dict) -> PolicyMatch:
    return policy_engine.evaluate(request, risk, context)
