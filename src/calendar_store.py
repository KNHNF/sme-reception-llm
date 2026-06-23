"""
Mock calendar store.

Reads data/calendar.json and provides slot lookup.
In production this would be replaced by a database query (SQLite / Supabase).
"""

import json
import os
from datetime import datetime
from typing import Optional

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "calendar.json")

_SERVICE_ALIASES = {
    "general":      "general",
    "consultation": "consultation",
    "consult":      "consultation",
    "follow_up":    "follow_up",
    "follow up":    "follow_up",
    "followup":     "follow_up",
}


def _load() -> list[dict]:
    with open(_DATA_PATH) as f:
        data = json.load(f)
    return data["slots"]


def _fmt_slot(slot: dict) -> str:
    """Human-readable slot description."""
    d = datetime.strptime(slot["date"], "%Y-%m-%d")
    t = datetime.strptime(slot["time"], "%H:%M")
    service_labels = {
        "general":      "a general appointment",
        "consultation": "a consultation",
        "follow_up":    "a follow-up",
    }
    label = service_labels.get(slot["service"], slot["service"])
    return (
        f"{label} on {d.strftime('%A %d %B')} "
        f"at {t.strftime('%I:%M %p').lstrip('0')}"
    )


def find_slots(
    service: Optional[str] = None,
    preferred_date: Optional[str] = None,
    skip: int = 0,
) -> list[dict]:
    """
    Return available slots matching the criteria, ordered by date/time.

    Args:
        service: ServiceType string ("general", "consultation", "follow_up") or None for any.
        preferred_date: ISO date string "YYYY-MM-DD" — if given, prefer this date first.
        skip: Skip the first N matching slots (for "suggest the next one").

    Returns:
        List of available slot dicts, up to 10.
    """
    slots = _load()
    today = datetime.today().strftime("%Y-%m-%d")

    # normalise service alias
    if service and service in _SERVICE_ALIASES:
        service = _SERVICE_ALIASES[service]

    available = [
        s for s in slots
        if s["available"]
        and s["date"] >= today
        and (service is None or s["service"] == service)
    ]

    # Sort: preferred date first, then chronological
    if preferred_date:
        available.sort(key=lambda s: (s["date"] != preferred_date, s["date"], s["time"]))
    else:
        available.sort(key=lambda s: (s["date"], s["time"]))

    return available[skip : skip + 10]


def get_next_slot(
    service: Optional[str] = None,
    preferred_date: Optional[str] = None,
    skip: int = 0,
) -> Optional[dict]:
    """Return the next available slot, or None if none found."""
    slots = find_slots(service=service, preferred_date=preferred_date, skip=skip)
    return slots[0] if slots else None


def describe_slot(slot: dict) -> str:
    """Return the spoken description of a slot."""
    return _fmt_slot(slot)


def book_slot(date: str, time: str, service: str) -> bool:
    """
    Mark a slot as booked (sets available=False in memory only — mock).
    In production this would write to a database.
    Returns True if the slot was found and booked.
    """
    slots = _load()
    for slot in slots:
        if slot["date"] == date and slot["time"] == time and slot["service"] == service:
            slot["available"] = False
            # Write back (mock persistence)
            data_path = os.path.join(os.path.dirname(__file__), "..", "data", "calendar.json")
            with open(data_path) as f:
                full = json.load(f)
            full["slots"] = slots
            with open(data_path, "w") as f:
                json.dump(full, f, indent=2)
            return True
    return False


def get_business_info() -> dict:
    with open(_DATA_PATH) as f:
        data = json.load(f)
    return {
        "name":   data.get("business_name", "the practice"),
        "hours":  data.get("opening_hours", "Monday to Friday, 9am to 5pm"),
    }
