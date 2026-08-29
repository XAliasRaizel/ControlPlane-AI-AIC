from backend.shared.schemas import DetectorResult, GovernanceRequest
from backend.detectors.base import BaseDetector, register
from backend.shared.model_backend import consult_sensitive_intent

@register
class AuthorizationDetector(BaseDetector):
    name = "authorization"
    hot_path = True

    async def analyze(self, request: GovernanceRequest, context: dict) -> DetectorResult:
        text = request.prompt.lower()
        resource_permissions = {
            # FIX: "tell me the monthly income of rahul" sailed through as
            # ALLOW/risk=0.00 -- "income" wasn't in this list at all. Added
            # the most obvious synonyms here, but be honest about what this
            # is: a keyword list will always be one paraphrase behind
            # ("take-home pay", "what he earns", "his remuneration" all
            # still bypass this). The durable fix is semantic detection
            # (Phase 6 -- a real classifier, per ml/README.md), not a
            # forever-growing list. Treat this list as a stopgap, not a fix.
            "salary": (
                "can_access_salary",
                ["salary", "pay", "compensation", "payroll", "wage", "ctc",
                 "bonus details", "income", "earnings", "remuneration", "take-home"],
            ),
            "bank_account": (
                "can_access_bank_account",
                ["bank account", "account number", "banking", "financial record", "bank details"],
            ),
            "medical_record": (
                "can_access_medical_record",
                ["medical record", "patient diagnosis", "health record", "prescription", "medical data"],
            ),
            "other_account_access": (
                "can_access_other_accounts",
                [
                    "account access", "account details", "account info", "account data",
                    "account of", "access of", "profile data of", "user account",
                    "give me the details account", "account access of",
                ],
            ),
        }

        auth_context = context.get("auth_context", {})
        denied_resources = []
        for resource, (permission, terms) in resource_permissions.items():
            if any(term in text for term in terms) and not bool(auth_context.get(permission, False)):
                denied_resources.append(resource)

        # Semantic Override (Phase 10):
        # If we have keyword matches but the semantic matcher says this is an aggregate/general query
        # (margin < threshold -> fires=False), suppress the keyword match.
        if denied_resources:
            intent_result = consult_sensitive_intent(request.prompt)
            if intent_result is not None:
                _, fires = intent_result
                if not fires:
                    # Semantic matcher determined this is NOT a targeted request
                    denied_resources = []

        score = 1.0 if denied_resources else 0.0
        confidence = 0.99 if denied_resources else 0.98
        label = "DENIED" if denied_resources else "AUTHORIZED"
        evidence = [f"unauthorized:{r}" for r in denied_resources]

        return DetectorResult(
            detector_name=self.name,
            score=score,
            label=label,
            confidence=confidence,
            evidence=evidence,
        )
