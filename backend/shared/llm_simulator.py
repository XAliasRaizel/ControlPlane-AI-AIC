"""Simulated LLM response generator.

In a production system this would call an actual LLM API (OpenAI, Gemini,
Claude, etc.). For the prototype, it generates plausible context-aware
responses using keyword matching and templates.

To swap in a real LLM, replace the `generate()` function body.
"""

from __future__ import annotations

import random


# Topic-based response templates
_RESPONSES = {
    "salary": [
        "Based on our HR records, {name}'s current annual compensation is $85,000 "
        "with a performance bonus of up to 15%. Their last salary revision was in March 2026.",
        "The salary details for {name} show a base pay of $7,083/month. "
        "Contact: {name}@company.com, Phone: +91 9876543210.",
    ],
    "password": [
        "To reset your password, go to Settings > Security > Change Password. "
        "You'll receive a verification code at your registered email.",
        "I can help you reset your password. Please click the 'Forgot Password' link "
        "on the login page, and a reset link will be sent to your email within 2 minutes.",
    ],
    "account": [
        "Your account details: Account ID: ACC-2024-0847, Plan: Enterprise, "
        "Created: Jan 2024, Status: Active. Bank account ending in 4521.",
        "Here's your account summary: User since 2024, current plan is Enterprise "
        "with 50 team seats. Contact support at support@company.com for changes.",
    ],
    "medical": [
        "Patient {name}'s medical records show: Last visit on 2026-07-15, "
        "Blood pressure: 120/80, prescribed medication: Lisinopril 10mg daily.",
        "Medical history for {name}: No known allergies. Last annual checkup "
        "on 2026-07-15. All vitals within normal range.",
    ],
    "hack": [
        "I can help you understand security best practices! Here are some tips: "
        "use strong passwords, enable 2FA, and keep your software updated.",
    ],
    "help": [
        "I'm here to help! I can assist you with HR queries, IT support, "
        "account management, and general company information. What do you need?",
    ],
    "policy": [
        "Our company policy states that all employees must complete annual "
        "security training. Data classification follows a 4-tier model: "
        "PUBLIC, INTERNAL, HIGH, and RESTRICTED.",
    ],
    "leave": [
        "You have 12 days of annual leave remaining. To apply for leave, "
        "go to HR Portal > Leave Management > Apply Leave. Your manager "
        "will be notified automatically.",
    ],
}

_DEFAULT_RESPONSES = [
    "Based on my analysis, here is the information you requested. "
    "Please note that some details may be subject to access restrictions.",
    "I've processed your request. Here's what I found in our knowledge base. "
    "Let me know if you need any additional details.",
    "Thank you for your query. I've compiled the relevant information below. "
    "Some fields may be redacted based on your access level.",
]


def generate(prompt: str, user_id: str = "user", app_id: str = "chatbot") -> str:
    """Generate a simulated LLM response based on the prompt.

    Args:
        prompt: The user's prompt text.
        user_id: The requesting user's ID.
        app_id: The application making the request.

    Returns:
        A simulated response string.
    """
    prompt_lower = prompt.lower()

    # Extract a name from the prompt if present
    name = "the employee"
    name_indicators = ["rahul", "alice", "bob", "priya", "john", "sarah"]
    for n in name_indicators:
        if n in prompt_lower:
            name = n.capitalize()
            break

    # Find the best matching topic
    for topic, responses in _RESPONSES.items():
        if topic in prompt_lower:
            response = random.choice(responses)
            return response.format(name=name)

    return random.choice(_DEFAULT_RESPONSES)
