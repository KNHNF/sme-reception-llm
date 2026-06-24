"""
Profanity detection and de-escalation.

Lightweight keyword-based filter - no external library needed.
The pipeline checks this BEFORE the LLM so no GPU call is wasted on an abusive turn.

Three-strike rule:
  Strike 1: gentle redirect
  Strike 2: firm reminder
  Strike 3: end the call
"""

_PROFANITY = {
    "fuck", "fucking", "fucked", "fucker", "shit", "shitting", "bullshit",
    "ass", "asshole", "bitch", "bastard", "crap", "damn", "bloody hell",
    "cunt", "dick", "piss", "pissed", "wanker", "twat", "arse",
}

DE_ESCALATION = [
    (
        "I understand you may be frustrated, and I'm here to help. "
        "Could I assist you with booking or checking an appointment?"
    ),
    (
        "I want to help you, but I do need us to keep our conversation respectful. "
        "How can I assist you today?"
    ),
    (
        "I'm afraid I'm unable to continue this call. "
        "Please call back when you're ready, and we'll be happy to help. Goodbye."
    ),
]


def contains_profanity(text: str) -> bool:
    """Return True if text contains a profanity word."""
    words = set(text.lower().split())
    # also check multi-word phrases
    lower = text.lower()
    return bool(words & _PROFANITY) or "bloody hell" in lower


def de_escalate(strike: int) -> str:
    """
    Return the appropriate de-escalation message for this strike number (1-indexed).
    Strike 3 = end the call.
    """
    idx = min(strike, len(DE_ESCALATION)) - 1
    return DE_ESCALATION[idx]


def is_terminal_strike(strike: int) -> bool:
    """Return True if this strike should end the call."""
    return strike >= len(DE_ESCALATION)
