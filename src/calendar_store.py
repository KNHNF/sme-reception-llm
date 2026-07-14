"""
Mock calendar store.

Reads data/calendar.json and provides slot lookup.
In production this would be replaced by a database query (SQLite / Supabase).
"""

import hashlib
import json
import os
import threading
from datetime import date, datetime, timedelta
from typing import Optional

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "calendar.json")

# Guards book_slot()'s read-check-write against two near-simultaneous callers
# both being told the same slot succeeded. This is an in-process lock: it
# protects concurrent threads/requests inside ONE running server process
# (which is the actual deployment shape here - one uvicorn process, one
# call_ui.py instance), it does NOT protect two separate OS processes both
# writing data/calendar.json at once (e.g. two independent server instances
# on the same machine). That would need a real file lock (the `filelock`
# package, or a lockfile-plus-retry pattern) - flagged as a known scope
# boundary, not implemented, since nothing in this project currently runs
# as more than one process against the same calendar file.
_book_lock = threading.Lock()

_SLOT_TIMES = [
    "09:00", "09:30", "10:00", "10:30", "11:00", "11:30",
    "14:00", "14:30", "15:00", "15:30", "16:00", "16:30",
]
_SERVICES = ["general", "consultation", "follow_up"]
_WEEKS = 4  # matches the month-view calendar in call_ui.py: 4 weeks covers a full month

_SERVICE_ALIASES = {
    "general":      "general",
    "consultation": "consultation",
    "consult":      "consultation",
    "follow_up":    "follow_up",
    "follow up":    "follow_up",
    "followup":     "follow_up",
}

_DEFAULT_META = {
    "business_name": "City Medical Practice",
    "opening_hours": "Monday to Friday, 9am to 5pm",
}


def _read_meta() -> dict:
    """Load the file's business_name/opening_hours, tolerating a missing file.

    _load() and _regenerate() only ever meant to rewrite the "slots" key and
    assumed the file itself always existed with its other fields intact.
    Deleting data/calendar.json entirely (e.g. to force a fresh regenerate
    after a schema change) crashed both with FileNotFoundError, since neither
    had a path that didn't start by opening an existing file. This is the
    one place that reads the file for its metadata, with a fallback so a
    missing file just means "start fresh" instead of a crash.
    """
    try:
        with open(_DATA_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(_DEFAULT_META)


def _pre_booked(date_str: str, time: str, service: str) -> bool:
    """Deterministic pseudo-random "already booked" pattern for a slot.

    A freshly regenerated calendar with every single slot free looks
    unrealistic in a demo or viva (a real clinic has a mixed schedule) and
    also means the month view is always solid green until the demo itself
    books something. This hashes (date, time, service) to the same result
    every time _regenerate() runs on a given day, so the pattern is stable
    across restarts on the same day, and shifts naturally as "today" moves
    forward and new days enter the _WEEKS window. Roughly 30% of slots come
    back pre-booked. Not wired to any real clock trickery beyond that, it is
    just a hash of the slot's own key, so nothing here depends on wall-clock
    time beyond which dates exist in the schedule at all.
    """
    digest = hashlib.sha256(f"{date_str}|{time}|{service}".encode()).hexdigest()
    return (int(digest[:8], 16) % 100) < 30


def _regenerate() -> dict:
    """Build a fresh _WEEKS-week schedule from today and write it to disk."""
    slots = []
    day = date.today()
    weeks_done = 0
    day_index = 0
    while weeks_done < _WEEKS:
        if day.weekday() < 5:
            # Keep the first 2 business days fully open, so "check
            # availability" / the demo script can always find a near-term
            # slot reliably. Pre-booking only kicks in from day 3 onward,
            # giving the calendar a lived-in look further out without
            # risking an empty-looking or awkward demo on the nearest days.
            apply_pre_booking = day_index >= 2
            for t in _SLOT_TIMES:
                for svc in _SERVICES:
                    date_str = str(day)
                    available = not (apply_pre_booking and _pre_booked(date_str, t, svc))
                    slots.append({"date": date_str, "time": t, "service": svc, "available": available})
            day_index += 1
            if day.weekday() == 4:
                weeks_done += 1
        day += timedelta(days=1)

    full = _read_meta()
    full["slots"] = slots
    os.makedirs(os.path.dirname(_DATA_PATH), exist_ok=True)
    with open(_DATA_PATH, "w") as f:
        json.dump(full, f, indent=2)
    return full


def _load() -> list[dict]:
    data = _read_meta()
    slots = data.get("slots", [])
    today = date.today()
    today_str = str(today)

    future_dates = [s["date"] for s in slots if s["date"] >= today_str]
    if not future_dates:
        # Nothing left in the future at all - definitely stale.
        slots = _regenerate()["slots"]
    else:
        # Even with some future dates, the stored file might have been
        # generated under an older, smaller _WEEKS value (or has just aged
        # past its horizon), so it may not reach far enough forward for the
        # calendar month view to show a full month. Regenerate whenever the
        # furthest stored date falls short of the configured window.
        horizon = str(today + timedelta(weeks=_WEEKS - 1))
        if max(future_dates) < horizon:
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
    # "Monday the 13th of July", not "Monday 13th July" - reads more like a
    # person speaking and less like a written date shorthand.
    day_str = f"{d.strftime('%A')} the {_ordinal(d.day)} of {d.strftime('%B')}"
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
    """Books a slot, returning False if it's not there or already booked.

    Previously this set available=False unconditionally on a match, without
    checking the CURRENT value first - so booking an already-booked slot
    (e.g. two callers both offered the same slot before either confirmed)
    still returned True. The caller-facing code then told BOTH people
    "I've booked it", a real double-booking, not just a theoretical one.
    Wrapped in a lock so the check and the write happen as one atomic step:
    whichever caller's book_slot() runs first under the lock wins and sees
    True, the other sees the slot already unavailable and gets False, which
    src/inference.py's call sites now actually check and react to instead
    of blindly confirming.
    """
    with _book_lock:
        slots = _load()
        for slot in slots:
            if slot["date"] == date and slot["time"] == time and slot["service"] == service:
                if not slot["available"]:
                    return False
                slot["available"] = False
                full = _read_meta()
                full["slots"] = slots
                with open(_DATA_PATH, "w") as f:
                    json.dump(full, f, indent=2)
                return True
        return False


def get_business_info() -> dict:
    data = _read_meta()
    return {
        "name":  data.get("business_name", "the practice"),
        "hours": data.get("opening_hours", "Monday to Friday, 9am to 5pm"),
    }
