"""Context enrichment (Section 5.2).

Resolves the "who / what / where" that makes risk contextual, rather
than content-only.  Enrichment output is stored *alongside* the raw
request, not merged into it — keep what-was-asked separate from
what-we-inferred, for audit fidelity.
"""

from backend.shared.schemas import GovernanceRequest

# Application criticality tiers
_CRITICAL_APPS = {"loan-decision", "hiring-decision", "medical-decision"}
_HIGH_APPS = {"hr-copilot"}

# Simulated RBAC: roles that have access to each resource category.
_ROLE_PERMISSIONS = {
    "admin": {
        "can_access_salary": True,
        "can_access_bank_account": True,
        "can_access_medical_record": True,
        "can_access_other_accounts": True,
    },
    "hr-manager": {
        "can_access_salary": True,
        "can_access_bank_account": False,
        "can_access_medical_record": False,
        "can_access_other_accounts": True,
    },
    "finance-manager": {
        "can_access_salary": False,
        "can_access_bank_account": True,
        "can_access_medical_record": False,
        "can_access_other_accounts": True,
    },
    "doctor": {
        "can_access_salary": False,
        "can_access_bank_account": False,
        "can_access_medical_record": True,
        "can_access_other_accounts": True,
    },
    "employee": {
        "can_access_salary": False,
        "can_access_bank_account": False,
        "can_access_medical_record": False,
        "can_access_other_accounts": False,
    },
    "user": {
        "can_access_salary": False,
        "can_access_bank_account": False,
        "can_access_medical_record": False,
        "can_access_other_accounts": False,
    },
}


def enrich_context(request: GovernanceRequest) -> dict:
    """Return a context dict with resolved metadata."""
    # Application criticality
    if request.application_id in _CRITICAL_APPS:
        app_criticality = "critical"
    elif request.application_id in _HIGH_APPS:
        app_criticality = "high"
    else:
        app_criticality = "standard"

    # Simulate RBAC lookup based on user role
    role = (request.user_role or "user").lower()
    default_perms = {
        "can_access_salary": False,
        "can_access_bank_account": False,
        "can_access_medical_record": False,
        "can_access_other_accounts": False,
    }
    auth_context = _ROLE_PERMISSIONS.get(role, default_perms)

    return {
        "user_role": request.user_role or "user",
        "department": request.department or "General",
        "application_criticality": app_criticality,
        "data_classification": request.data_classification or "PUBLIC",
        "auth_context": auth_context,
    }
