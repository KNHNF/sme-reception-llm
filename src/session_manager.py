"""
Session Manager
In-memory multi-turn conversation state.

Keyed by session_id. Stores partial entities across clarification turns.
Terminal actions (book, cancel, out_of_scope) clear the session.
Sessions expire after TIMEOUT_SECONDS of inactivity.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

TIMEOUT_SECONDS = 300  # 5 minutes -- clear stale sessions


@dataclass
class Session:
    session_id: str
    partial_action: Optional[str] = None       # e.g. "book_appointment"
    partial_entities: dict = field(default_factory=dict)
    missing_fields: list = field(default_factory=list)
    turn_count: int = 0
    last_updated: float = field(default_factory=time.time)

    # Caller identity
    caller_name: Optional[str] = None          # captured on the first turn
    awaiting_name: bool = False                # True if we just asked for the name
    awaiting_name_confirm: bool = False        # True after capturing name, waiting for yes/no
    name_reask_count: int = 0                  # times we have re-asked for name (booking intent)

    # Calendar suggestion state
    pending_suggestion: Optional[dict] = None  # slot dict we just offered the caller
    suggestion_index: int = 0                  # how many slots we have already skipped

    # Profanity tracking
    profanity_strikes: int = 0

    # Confusion / out-of-scope retry tracking
    confusion_count: int = 0           # increments each time the caller hits out_of_scope
    last_out_of_scope_hint: str = ""   # last field hint given (avoids identical retries)

    def is_expired(self) -> bool:
        return (time.time() - self.last_updated) > TIMEOUT_SECONDS

    def touch(self):
        self.last_updated = time.time()
        self.turn_count += 1

    def clear(self):
        """Clear booking state. Name and profanity count persist for the whole call."""
        self.partial_action = None
        self.partial_entities = {}
        self.missing_fields = []
        self.pending_suggestion = None
        self.suggestion_index = 0
        self.turn_count = 0


TERMINAL_ACTIONS = {"book_appointment", "cancel_appointment", "out_of_scope"}

_sessions: dict[str, Session] = {}


def get_or_create(session_id: str) -> Session:
    if session_id not in _sessions or _sessions[session_id].is_expired():
        _sessions[session_id] = Session(session_id=session_id)
    return _sessions[session_id]


def update(session_id: str, llm_output: dict, entities: dict) -> dict:
    """
    Merge LLM output and new entities into the session.

    Call this after every LLM inference step.

    Returns the merged action dict that should be sent to the backend,
    or the clarify dict that should trigger a follow-up question.
    """
    session = get_or_create(session_id)
    session.touch()

    action = llm_output.get("action")

    if action == "clarify":
        # Store what we know so far, ready to merge on the next turn.
        session.partial_action = session.partial_action or "book_appointment"
        session.partial_entities.update({
            k: v for k, v in entities.items() if v is not None
        })
        session.missing_fields = llm_output.get("missing_fields", [])
        return llm_output

    if action in TERMINAL_ACTIONS:
        # Merge any previously stored partial context with the new output.
        merged = {**session.partial_entities, **llm_output}
        session.clear()
        return merged

    # check_availability does not clear the session.
    return llm_output


def get_context(session_id: str) -> dict:
    """
    Return partial entities already collected for this session.
    Used to augment the LLM prompt on clarification turns.
    """
    session = get_or_create(session_id)
    return {
        "partial_action":   session.partial_action,
        "partial_entities": session.partial_entities,
        "missing_fields":   session.missing_fields,
        "turn":             session.turn_count,
    }


def close(session_id: str):
    """Explicitly close and remove a session (e.g. caller hung up)."""
    _sessions.pop(session_id, None)


def _purge_expired():
    """Remove all expired sessions. Call periodically if running long."""
    expired = [sid for sid, s in _sessions.items() if s.is_expired()]
    for sid in expired:
        del _sessions[sid]


if __name__ == "__main__":
    sid = "test-session-001"

    print("Turn 1: caller says 'Book me in for a consultation on Monday'")
    entities_1 = {"date_resolved": "2026-06-22", "service": "consultation", "time_resolved": None}
    llm_out_1  = {"action": "clarify", "missing_fields": ["time"]}
    result_1   = update(sid, llm_out_1, entities_1)
    print(f"Result: {result_1}")
    print(f"Context: {get_context(sid)}\n")

    print("Turn 2: caller says '3pm please'")
    entities_2 = {"time_resolved": "15:00"}
    llm_out_2  = {"action": "book_appointment", "date": "2026-06-22", "time": "15:00", "service": "consultation"}
    result_2   = update(sid, llm_out_2, entities_2)
    print(f"Result: {result_2}")
    print(f"Context after terminal: {get_context(sid)}")
