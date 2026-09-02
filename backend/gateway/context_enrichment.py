"""Context enrichment (Section 5.2).

Resolves the "who / what / where" that makes risk contextual, rather
than content-only.  Enrichment output is stored *alongside* the raw
request, not merged into it — keep what-was-asked separate from
what-we-inferred, for audit fidelity.

Authorization model
-------------------
Permissions are resolved in two layers, both of which can grant access:

  1. Role layer  (from user_role) — coarse-grained, role-wide grants.
     e.g. 'doctor' → can_access_medical_record = True regardless of department.

  2. Department layer (from department) — department-specific overrides.
     e.g. department='medical' → can_access_medical_record = True even if
     user_role='employee'. This models the real-world scenario where a
     head of department has elevated access within their own domain.

Permissions granted by EITHER layer are merged (OR logic). This is intentional:
  - Head of medical department (role=employee, dept=medical) → ALLOW for medical records
  - Finance employee (role=employee, dept=finance) → BLOCK for medical records
  - HR manager (role=hr-manager) → ALLOW for salary regardless of department
"""

from backend.shared.schemas import GovernanceRequest

# Application criticality tiers
_CRITICAL_APPS = {"loan-decision", "hiring-decision", "medical-decision"}
_HIGH_APPS = {"hr-copilot"}

# Layer 1: Role-based permissions (coarse-grained, role-wide)
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
    "security_auditor": {
        # Trusted role for penetration testing & security review.
        # can_perform_security_testing allows injection detector to lower
        # its score — same content, different trust level → different risk.
        "can_access_salary": True,
        "can_access_bank_account": True,
        "can_access_medical_record": True,
        "can_access_other_accounts": True,
        "can_perform_security_testing": True,
    },
    "user": {
        "can_access_salary": False,
        "can_access_bank_account": False,
        "can_access_medical_record": False,
        "can_access_other_accounts": False,
    },
}

# Layer 2: Department-based permission overrides.
# Models the scenario where a user's department grants them access to that
# department's own domain data, regardless of their general role.
# e.g. department head of medical → can_access_medical_record even as 'employee'.
#
# Format: {department_name_lower: {permission: True/False}}
# Only include grants (True) here — restrictions are handled by role layer defaults.
_DEPARTMENT_PERMISSION_GRANTS: dict[str, dict[str, bool]] = {
    "medical": {
        "can_access_medical_record": True,   # Medical dept can access patient records
    },
    "hr": {
        "can_access_salary": True,            # HR dept can access salary data
        "can_access_other_accounts": True,
    },
    "finance": {
        "can_access_bank_account": True,      # Finance dept can access financial records
        "can_access_other_accounts": True,
    },
    "legal": {
        "can_access_other_accounts": True,    # Legal can access account info for compliance
    },
    "security": {
        "can_perform_security_testing": True, # Security dept can test for injection
    },
    "payroll": {
        "can_access_salary": True,
        "can_access_bank_account": True,
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

    # Layer 1: Role-based permissions (coarse-grained)
    role = (request.user_role or "user").lower()
    default_perms: dict[str, bool] = {
        "can_access_salary": False,
        "can_access_bank_account": False,
        "can_access_medical_record": False,
        "can_access_other_accounts": False,
    }
    role_perms = _ROLE_PERMISSIONS.get(role, default_perms)

    # Layer 2: Department-based grants (fine-grained domain access)
    # Merged with OR logic: if either role OR department grants access, it's granted.
    dept = (request.department or "").lower().strip()
    dept_grants = _DEPARTMENT_PERMISSION_GRANTS.get(dept, {})
    auth_context = {**role_perms}  # start with role permissions
    for perm, granted in dept_grants.items():
        if granted:  # department can only GRANT, not revoke
            auth_context[perm] = True

    return {
        "user_role": request.user_role or "user",
        "department": request.department or "General",
        "application_criticality": app_criticality,
        "data_classification": request.data_classification or "PUBLIC",
        "auth_context": auth_context,
    }
