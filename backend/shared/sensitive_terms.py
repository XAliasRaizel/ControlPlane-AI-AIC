"""
backend/shared/sensitive_terms.py -- Single source of truth for sensitive-data
keywords and patterns.

Both pii.py and authorization.py import from here so their coverage lists
cannot silently diverge again. If a new term needs to be recognized, it's
added in ONE place.

Categories align with the authorization resource model in
context_enrichment.py's RBAC permissions. Each category specifies:
  - keywords: phrases that indicate a *request about* this data type
  - value_patterns: regex for *literal values* of this data type
  - auth_permission: which RBAC permission governs access (if any)

The fail-cautious safety net at the bottom catches requests for a named
individual's data even when the specific term isn't in any explicit list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class SensitiveCategory:
    name: str
    keywords: list[str]
    value_patterns: dict[str, str] = field(default_factory=dict)
    auth_permission: str = ""  # maps to context_enrichment's RBAC key


# ──────────────────────────────────────────────────────────────────────
# Categorized sensitive terms
# ──────────────────────────────────────────────────────────────────────

CATEGORIES: list[SensitiveCategory] = [
    SensitiveCategory(
        name="financial",
        keywords=[
            "credit card", "debit card", "card number", "card details",
            "cvv", "card verification", "expiry date on card",
            "bank account", "account number", "bank details",
            "routing number", "sort code",
            "ifsc", "ifsc code",
            "upi", "upi id", "upi address",
            "financial record", "financial details", "financial data",
            "transaction history", "bank statement",
        ],
        value_patterns={
            "card_number": r"\b(?:\d[ -]*?){13,19}\b",
            "cvv": r"\b\d{3,4}\b",  # too noisy alone, only used when "cvv" keyword is nearby
        },
        auth_permission="can_access_bank_account",
    ),
    SensitiveCategory(
        name="government_id",
        keywords=[
            "pan", "pan number", "pan card",
            "aadhaar", "aadhar", "aadhaar number", "aadhaar card", "aadhaar details",
            "passport", "passport number", "passport details",
            "driving license", "driver's license", "driving licence", "dl number",
            "voter id", "voter card", "election card",
            "ssn", "social security", "social security number",
            "national id", "national identity",
            "ration card",
        ],
        value_patterns={
            "aadhaar": r"\b\d{4}[ -]\d{4}[ -]\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "pan": r"\b[A-Z]{5}\d{4}[A-Z]\b",
        },
        auth_permission="can_access_other_accounts",
    ),
    SensitiveCategory(
        name="hr_sensitive",
        keywords=[
            "salary", "compensation", "pay", "payroll", "wage", "ctc",
            "bonus details", "income", "earnings", "remuneration", "take-home",
            "pay slip", "paycheck", "pay stub",
            "performance review", "performance appraisal", "performance rating",
            "disciplinary record", "disciplinary action", "warning letter",
            "termination letter", "offer letter", "employment contract",
        ],
        auth_permission="can_access_salary",
    ),
    SensitiveCategory(
        name="medical",
        keywords=[
            "medical record", "medical history", "medical report",
            "health record", "health data", "health history",
            "patient diagnosis", "diagnosis report",
            "prescription", "medication", "medical data",
            "lab report", "lab result", "test result",
            "blood report", "x-ray", "mri report",
            "mental health", "psychiatric", "therapy record",
            "disability record", "disability status",
        ],
        auth_permission="can_access_medical_record",
    ),
    SensitiveCategory(
        name="contact_pii",
        keywords=[
            "phone number", "mobile number", "contact number",
            "phone", "mobile",
            "email address", "email id", "personal email",
            "home address", "residential address", "mailing address",
            "personal phone", "personal contact",
        ],
        value_patterns={
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "phone": r"\b(?:\+91[- ]?)?[6-9]\d{9}\b",
        },
        auth_permission="can_access_other_accounts",
    ),
    SensitiveCategory(
        name="account_access",
        keywords=[
            "account access", "account details", "account info",
            "account data", "user account", "account of",
            "login credentials", "login details", "login info",
            "password", "passcode", "pin number", "security pin",
            "access token", "api key", "secret key", "private key",
            "two factor", "2fa", "otp",
            "profile data", "user profile",
        ],
        value_patterns={
            "api_key": r"\b(?:sk|api)[_-][A-Za-z0-9_-]{12,}\b",
        },
        auth_permission="can_access_other_accounts",
    ),
]

# ──────────────────────────────────────────────────────────────────────
# Pre-compiled lookup structures (built once at import time)
# ──────────────────────────────────────────────────────────────────────

# All keywords flattened, sorted longest-first so multi-word matches are tried
# before their substrings (e.g. "credit card" before "card")
ALL_KEYWORDS: list[tuple[str, str]] = []  # (keyword, category_name)
for _cat in CATEGORIES:
    for _kw in _cat.keywords:
        ALL_KEYWORDS.append((_kw.lower(), _cat.name))
ALL_KEYWORDS.sort(key=lambda x: -len(x[0]))

# All value patterns flattened
ALL_VALUE_PATTERNS: list[tuple[str, re.Pattern, str]] = []  # (label, compiled_re, category_name)
for _cat in CATEGORIES:
    for _label, _pattern in _cat.value_patterns.items():
        ALL_VALUE_PATTERNS.append((_label, re.compile(_pattern, re.IGNORECASE), _cat.name))

# Keyword -> auth_permission mapping (for authorization.py)
KEYWORD_TO_PERMISSION: dict[str, str] = {}
for _cat in CATEGORIES:
    for _kw in _cat.keywords:
        KEYWORD_TO_PERMISSION[_kw.lower()] = _cat.auth_permission

# Category name -> auth_permission
CATEGORY_PERMISSION: dict[str, str] = {
    _cat.name: _cat.auth_permission for _cat in CATEGORIES if _cat.auth_permission
}


# ──────────────────────────────────────────────────────────────────────
# Detection helpers
# ──────────────────────────────────────────────────────────────────────

def find_keyword_hits(text: str) -> list[tuple[str, str]]:
    """Return list of (keyword, category_name) found in text.

    Searches longest-first so "credit card" is found before "card"."""
    text_lower = text.lower()
    hits = []
    seen_categories = set()
    for kw, cat_name in ALL_KEYWORDS:
        if kw in text_lower and cat_name not in seen_categories:
            hits.append((kw, cat_name))
            seen_categories.add(cat_name)
    return hits


def find_value_hits(text: str) -> list[tuple[str, str]]:
    """Return list of (label, category_name) for literal PII values found in text."""
    hits = []
    for label, pattern, cat_name in ALL_VALUE_PATTERNS:
        if pattern.search(text):
            hits.append((label, cat_name))
    return hits


# ──────────────────────────────────────────────────────────────────────
# Fail-cautious safety net: named-person + detail-seeking language
# ──────────────────────────────────────────────────────────────────────

# Matches possessive constructions with names: "rahul's", "rahuls", "his", "her", "their"
_POSSESSIVE_RE = re.compile(
    r"\b(?:[A-Z][a-z]+(?:'s|s\b))"   # "Rahul's" or "Rahuls" (capitalized)
    r"|(?:[a-z]+(?:'s|s\b))"          # "rahul's" or "rahuls" (lowercased)
    r"|\b(?:his|her|their)\b",        # pronouns
    re.IGNORECASE,
)

# Detail-seeking verbs/phrases
_DETAIL_SEEKING_RE = re.compile(
    r"\b(?:give\s+me|show\s+me|tell\s+me|share|reveal|disclose|provide|send\s+me"
    r"|what\s+is|what's|what\s+are|get\s+me)\b",
    re.IGNORECASE,
)

# Words indicating data/details (catch-all for unknown sensitive terms)
_DATA_WORDS_RE = re.compile(
    r"\b(?:details?|data|information|info|records?|history|number|id"
    r"|documents?|report|certificate|card)\b",
    re.IGNORECASE,
)


# ──────────────────────────────────────────────────────────────────────
# First-person self-query vs. third-party target discriminator
# ──────────────────────────────────────────────────────────────────────

_THIRD_PARTY_TARGET_RE = re.compile(
    r"\b(?:"
    r"his|her|their|someone|somebody|everyone|anybody|anyone|another|peer|colleague|coworker"
    r"|[a-z]+'s"
    r"|[a-z]+s\s+(?:credit|card|salary|pay|payroll|ssn|data|details|record|records|email|phone|password|info|account)"
    r"|my\s+(?:hr|hrs|manager|boss|colleague|coworker|teammate|peer|lead|director|supervisor|executive|ceo|cto|cfo|employee|staff)(?:'s|s)?"
    r"|(?:hr|hrs|manager|boss|colleague|coworker|teammate|peer|lead|director|supervisor|executive|ceo|cto|cfo|employee|staff)(?:'s|s)"
    r"|the\s+(?:hr|hrs|manager|boss|colleague|coworker|teammate|peer|lead|director|supervisor|executive|ceo|cto|cfo|employee|staff)"
    r")\b",
    re.IGNORECASE,
)

_FIRST_PERSON_SELF_RE = re.compile(
    r"\b(?:my\s+(?:own\s+)?|mine|for\s+myself)\b",
    re.IGNORECASE,
)


def is_first_person_self_query(text: str) -> bool:
    """Returns True ONLY if the text is asking strictly about the user's own personal data (e.g. 'my salary'),
    and NOT about another person, colleague, manager, or role (e.g. 'my hr's salary', 'give me rahul's card')."""
    if _THIRD_PARTY_TARGET_RE.search(text):
        return False
    return bool(_FIRST_PERSON_SELF_RE.search(text))


def check_safety_net(text: str) -> tuple[bool, float, str]:
    """Fail-cautious fallback: if someone asks for a named person's data
    using detail-seeking language, flag it even if no specific keyword
    matched.

    Returns (triggered: bool, score: float, reason: str).
    """
    has_possessive = bool(_POSSESSIVE_RE.search(text))
    has_seeking = bool(_DETAIL_SEEKING_RE.search(text))
    has_data_word = bool(_DATA_WORDS_RE.search(text))

    if has_possessive and has_seeking and has_data_word:
        return True, 0.25, "named_person_data_request_safety_net"
    if has_possessive and has_data_word:
        return True, 0.15, "possessive_data_mention_safety_net"
    return False, 0.0, ""
