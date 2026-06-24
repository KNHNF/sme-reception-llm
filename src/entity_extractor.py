"""
Entity Extractor
Pulls date, time, and service type from a caller utterance and
normalises them to ISO 8601 / HH:MM before the LLM sees them.

pip install spacy python-dateutil
python -m spacy download en_core_web_sm
"""

import re
from datetime import date, datetime, timedelta
from typing import Optional

import spacy
from dateutil import parser as dparser

nlp = spacy.load("en_core_web_sm")

# Order matters: more specific terms checked before generic ones.
SERVICE_KEYWORDS = [
    ("consultation", ["consultation", "consult", "60-minute", "60 min"]),
    ("follow_up",    ["follow-up", "follow up", "followup", "15-minute", "15 min"]),
    ("general",      ["general appointment", "general", "slot", "come in", "appointment"]),
]

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

def extract(text: str, today: Optional[date] = None) -> dict:
    """
    Run spaCy NER on the utterance and return a dict of resolved entities.

    Returns:
        {
            "date_raw":      str or None   ("tomorrow", "Monday", "June 24th")
            "date_resolved": str or None   ("2026-06-19")  ISO 8601
            "time_raw":      str or None   ("3pm", "half past three")
            "time_resolved": str or None   ("15:00")       HH:MM 24h
            "service":       str or None   ("general" | "consultation" | "follow_up")
        }
    """
    today = today or date.today()
    doc = nlp(text)

    result = {
        "date_raw":      None,
        "date_resolved": None,
        "time_raw":      None,
        "time_resolved": None,
        "service":       None,
        "person":        None,   # PERSON entity (caller name)
    }

    for ent in doc.ents:
        if ent.label_ == "DATE" and result["date_raw"] is None:
            result["date_raw"]      = ent.text
            result["date_resolved"] = _resolve_date(ent.text, today)
        elif ent.label_ == "TIME" and result["time_raw"] is None:
            result["time_raw"]      = ent.text
            result["time_resolved"] = _resolve_time(ent.text)
        elif ent.label_ == "PERSON" and result["person"] is None:
            result["person"] = ent.text.title()

    text_lower = text.lower()
    for service, keywords in SERVICE_KEYWORDS:
        if any(kw in text_lower for kw in keywords):
            result["service"] = service
            break

    return result

def to_prompt_context(entities: dict) -> str:
    """
    Format extracted entities as a short context string to prepend to the
    LLM prompt, so the model does not have to resolve dates from natural language.

    Example output:
        "[Extracted: date=2026-06-19, time=15:00, service=consultation]"
    """
    parts = []
    if entities.get("date_resolved"):
        parts.append(f"date={entities['date_resolved']}")
    elif entities.get("date_raw"):
        parts.append(f"date_hint={entities['date_raw']}")

    if entities.get("time_resolved"):
        parts.append(f"time={entities['time_resolved']}")
    elif entities.get("time_raw"):
        parts.append(f"time_hint={entities['time_raw']}")

    if entities.get("service"):
        parts.append(f"service={entities['service']}")

    if not parts:
        return ""
    return "[Extracted: " + ", ".join(parts) + "]"

def _resolve_date(text: str, today: date) -> Optional[str]:
    t = text.lower().strip()

    if t in ("today", "now"):
        return today.isoformat()
    if t in ("tomorrow", "tmrw", "tmr"):
        return (today + timedelta(days=1)).isoformat()
    if t in ("yesterday",):
        return (today - timedelta(days=1)).isoformat()

    # "next Monday", "this Friday", "on Wednesday", plain weekday name
    for i, wd in enumerate(WEEKDAYS):
        if wd in t:
            days_ahead = (i - today.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            return (today + timedelta(days=days_ahead)).isoformat()

    # "in 3 days", "in two days"
    m = re.search(r"in (\d+|one|two|three|four|five) days?", t)
    if m:
        word_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        n_str = m.group(1)
        n = int(n_str) if n_str.isdigit() else word_map.get(n_str, 1)
        return (today + timedelta(days=n)).isoformat()

    # "next week"
    if "next week" in t:
        return (today + timedelta(weeks=1)).isoformat()

    # Absolute dates: "June 24th", "24th", "24/06", etc.
    try:
        default_dt = datetime(today.year, today.month, today.day)
        parsed = dparser.parse(text, default=default_dt)
        # If the resolved date is in the past, bump to next year.
        if parsed.date() < today:
            parsed = parsed.replace(year=today.year + 1)
        return parsed.date().isoformat()
    except (ValueError, OverflowError):
        return None

def _resolve_time(text: str) -> Optional[str]:
    try:
        default_dt = datetime(2000, 1, 1)
        t = dparser.parse(text.lower().strip(), default=default_dt)
        return t.strftime("%H:%M")
    except (ValueError, OverflowError):
        return None

if __name__ == "__main__":
    samples = [
        "I want to book a consultation for tomorrow at 3pm.",
        "Can I get a follow-up on Monday at half past nine?",
        "Book me in for a general appointment on June 24th at 14:30.",
        "I need to come in next week for a consult.",
        "Cancel my appointment on Wednesday at 10am.",
        "What times do you have available?",
    ]

    print(f"Today: {date.today()}\n")
    for s in samples:
        e = extract(s)
        ctx = to_prompt_context(e)
        print(f"Input:   {s}")
        print(f"Context: {ctx}")
        print(f"Entities: {e}")
        print()
