"""
Action schema and spoken confirmation templates.
LLM outputs action JSON only. Backend fills spoken strings from templates.
"""

from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, field_validator
import re

# Enums

class ActionType(str, Enum):
    check_availability = "check_availability"
    book_appointment   = "book_appointment"
    cancel_appointment = "cancel_appointment"
    clarify            = "clarify"         # fallback: LLM could not extract a required field
    out_of_scope       = "out_of_scope"    # caller intent is not appointment-related
    end_call           = "end_call"        # caller said goodbye / wants to hang up

class ServiceType(str, Enum):
    general      = "general"       # 30 min
    consultation = "consultation"  # 60 min
    follow_up    = "follow_up"     # 15 min

# Action models

class CheckAvailability(BaseModel):
    action:  Literal[ActionType.check_availability] = ActionType.check_availability
    date:    Optional[str] = None   # ISO 8601, e.g. "2026-06-12"
    service: Optional[ServiceType] = None

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        if v is not None and not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError("date must be ISO 8601 (YYYY-MM-DD)")
        return v

class BookAppointment(BaseModel):
    action:  Literal[ActionType.book_appointment] = ActionType.book_appointment
    date:    str                   # required
    time:    str                   # HH:MM 24h, e.g. "15:00"
    service: ServiceType

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError("date must be ISO 8601 (YYYY-MM-DD)")
        return v

    @field_validator("time")
    @classmethod
    def validate_time(cls, v):
        if not re.match(r"^\d{2}:\d{2}$", v):
            raise ValueError("time must be HH:MM")
        return v

class CancelAppointment(BaseModel):
    action:         Literal[ActionType.cancel_appointment] = ActionType.cancel_appointment
    appointment_id: Optional[str] = None  # if caller provides a reference number
    date:           Optional[str] = None  # fallback identifier
    time:           Optional[str] = None

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        if v is not None and not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError("date must be ISO 8601 (YYYY-MM-DD)")
        return v

class Clarify(BaseModel):
    action:         Literal[ActionType.clarify] = ActionType.clarify
    missing_fields: list[str]  # e.g. ["date", "service"]

class OutOfScope(BaseModel):
    action: Literal[ActionType.out_of_scope] = ActionType.out_of_scope

class EndCall(BaseModel):
    action: Literal[ActionType.end_call] = ActionType.end_call

# Union type for outlines / lm-format-enforcer
# Pass this to outlines.generate.json() or build a JSON schema from it.
# Only CheckAvailability, BookAppointment, CancelAppointment are trained actions.
# Clarify and OutOfScope are safety valves.

from typing import Annotated, Union
from pydantic import Field

ActionOutput = Annotated[
    Union[
        CheckAvailability,
        BookAppointment,
        CancelAppointment,
        Clarify,
        OutOfScope,
        EndCall,
    ],
    Field(discriminator="action"),
]

# JSON schema export (for lm-format-enforcer)

def get_json_schema() -> dict:
    """
    Returns a JSON schema dict suitable for lm-format-enforcer's
    JsonSchemaParser or outlines.generate.json().
    """
    import json
    from pydantic import TypeAdapter
    adapter = TypeAdapter(ActionOutput)
    return adapter.json_schema()

# Template-based confirmation strings
# The LLM never generates these. Backend fills them from structured output.

CONFIRMATION_TEMPLATES = {
    ActionType.check_availability: (
        "Let me check availability"
        "{date_str}"
        "{service_str}."
    ),
    ActionType.book_appointment: (
        "I have booked your {service_label} for {date_str} at {time_str}. "
        "You will receive a confirmation shortly."
    ),
    ActionType.cancel_appointment: (
        "Your appointment has been cancelled. "
        "Is there anything else I can help you with?"
    ),
    ActionType.clarify: (
        "Could you please confirm {missing_str}?"
    ),
    ActionType.out_of_scope: (
        "I can help with appointment booking, cancellations, and availability. "
        "Is there something along those lines I can help with?"
    ),
    ActionType.end_call: (
        "Thank you for calling. Have a great day. Goodbye!"
    ),
}

SERVICE_LABELS = {
    ServiceType.general:      ("a general appointment",  "30 minutes"),
    ServiceType.consultation:  ("a consultation",         "60 minutes"),
    ServiceType.follow_up:    ("a follow-up",            "15 minutes"),
}

def render_confirmation(action_obj: BaseModel) -> str:
    """
    Fills a confirmation string from a validated action object.
    No LLM involved -- pure template logic.
    """
    from datetime import datetime

    a = action_obj.action

    if a == ActionType.check_availability:
        date_str = ""
        service_str = ""
        if action_obj.date:
            try:
                d = datetime.strptime(action_obj.date, "%Y-%m-%d")
                date_str = f" on {d.strftime('%A, %d %B')}"
            except ValueError:
                date_str = f" on {action_obj.date}"
        if action_obj.service:
            label, _ = SERVICE_LABELS[action_obj.service]
            service_str = f" for {label}"
        return (
            f"Let me check availability{date_str}{service_str}."
        )

    elif a == ActionType.book_appointment:
        d = datetime.strptime(action_obj.date, "%Y-%m-%d")
        t = datetime.strptime(action_obj.time, "%H:%M")
        label, duration = SERVICE_LABELS[action_obj.service]
        return (
            f"I have booked {label} for {d.strftime('%A, %d %B')} "
            f"at {t.strftime('%I:%M %p').lstrip('0')}. "
            f"The appointment is {duration}. "
            f"You will receive a confirmation shortly."
        )

    elif a == ActionType.cancel_appointment:
        return (
            "Your appointment has been cancelled. "
            "Is there anything else I can help you with?"
        )

    elif a == ActionType.clarify:
        field_hints = {
            "time":    "what time you would like - for example, '10am' or '2:30pm'",
            "date":    "which date you would like - for example, 'this Monday' or 'June 30th'",
            "service": "what type of appointment - general, consultation, or follow-up",
        }
        prompts = []
        for f in action_obj.missing_fields:
            prompts.append(field_hints.get(f, f))
        joined = " and ".join(prompts)
        return f"Of course - could you also let me know {joined}?"

    elif a == ActionType.out_of_scope:
        return (
            "I can help with appointment booking, cancellations, and availability. "
            "Is there something along those lines I can help with?"
        )

    elif a == ActionType.end_call:
        return "Thank you for calling. Have a great day. Goodbye!"

    return "I did not quite catch that. Could you repeat?"

# Quick smoke test

if __name__ == "__main__":
    import json

    # Simulate what Phi-3 mini would output (10-15 tokens of JSON)
    raw_llm_outputs = [
        '{"action": "book_appointment", "date": "2026-06-12", "time": "15:00", "service": "general"}',
        '{"action": "check_availability", "date": "2026-06-14", "service": "consultation"}',
        '{"action": "cancel_appointment", "appointment_id": "APT-0042"}',
        '{"action": "clarify", "missing_fields": ["date", "service"]}',
        '{"action": "out_of_scope"}',
    ]

    from pydantic import TypeAdapter
    adapter = TypeAdapter(ActionOutput)

    for raw in raw_llm_outputs:
        obj = adapter.validate_json(raw)
        print(f"Parsed: {obj}")
        print(f"Spoken: {render_confirmation(obj)}")
        print()

    print("JSON Schema (pass to lm-format-enforcer):")
    print(json.dumps(get_json_schema(), indent=2))
