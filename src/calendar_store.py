"""
Mock calendar store.

Reads data/calendar.json and provides slot lookup.
In production this would be replaced by a database query (SQLite / Supabase).
"""

import json
import os
from datetime import date, datetime, timedelta
from typing import Optional

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "calendar.json")

_SLOT_TIMES = [
    "09:00", "09:30", "10:00", "10:30", "11:00", "11:30",
    "14:00", "14:30", "15:00", "15:30", "16:00", "16:30",
]
_SERVICES = ["general", "consultation", "follow_up"]
_WEEKS = 3

_SERVICE_ALIASES = {
    "general":      "general",
    "consultation": "consultation",
    "consult":      "consultation",
    "follow_up":    "follow_up",
    "follow up":    "follow_up",
    "followup":     "follow_up",
}


def _regenerate() -> dict:
    """Build a fresh 3-week schedule from today and write it to disk."""
    slots = []
    day = date.today()
    weeks_done = 0
    while weeks_done < _WEEKS:
        if day.weekday() < 5:
            for t in _SLOT_TIMES:
                for svc in _SERVICES:
                    slots.append({"date": str(day), "time": t, "service": svc, "available": True})
            if day.weekday() == 4:
                weeks_done += 1
        day += timedelta(days=1)

    with open(_DATA_PATH) as f:
        full = json.load(f)
    full["slots"] = slots
    with open(_DATA_PATH, "w") as f:
        json.dump(full, f, indent=2)
    return full


def _load() -> list[dict]:
    with open(_DATA_PATH) as f:
        data = json.load(f)
    slots = data["slots"]
    today = str(date.today())
    if not any(s["date"] >= today for s in slots):
        slots = _regenerate()["slots"]
    return slots


def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]}"


def _fmt_slot(slot: dict) -> str:
    d = datetime.strptime(slot["date"], "%Y-%m-%d")
    t = datetime.strptime(slot["time"], "%H:%M")
    service_labels = {
        "general":      "a general appointment",
        "consultation": "a consultation",
        "follow_up":    "a follow-up",
    }
    label = service_labels.get(slot["service"], slot["service"])
    day_str = f"{d.strftime('%A')} {_ordinal(d.day)} {d.strftime('%B')}"
    return f"{label} on {day_str} at {t.strftime('%I:%M %p').lstrip('0')}"


def find_slots(
    service: Optional[str] = None,
    preferred_date: Optional[str] = None,
    skip: int = 0,
) -> list[dict]:
    slots = _load()
    now = datetime.today()
    today = now.strftime("%Y-%m-%d")
    now_time = now.strftime("%H:%M")

    if service and service in _SERVICE_ALIASES:
        service = _SERVICE_ALIASES[service]

    available = [
        s for s in slots
        if s["available"]
        and s["date"] >= today
        and (s["date"] > today or s["time"] > now_time)
        and datetime.strptime(s["date"], "%Y-%m-%d").weekday() < 5
        and (service is None or s["service"] == service)
    ]

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
    slots = find_slots(service=service, preferred_date=preferred_date, skip=skip)
    return slots[0] if slots else None


def describe_slot(slot: dict) -> str:
    return _fmt_slot(slot)


def book_slot(date: str, time: str, service: str) -> bool:
    slots = _load()
    for slot in slots:
        if slot["date"] == date and slot["time"] == time and slot["service"] == service:
            slot["available"] = False
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
        "name":  data.get("business_name", "the practice"),
        "hours": data.get("opening_hours", "Monday to Friday, 9am to 5pm"),
    }
