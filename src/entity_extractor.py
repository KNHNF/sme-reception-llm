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

_MONTHS_LIST = ["january", "february", "march", "april", "may", "june", "july",
                "august", "september", "october", "november", "december"]
_MONTH_ALT = "|".join(_MONTHS_LIST)

# Direct regex scan over the raw utterance for an explicit numeric date,
# independent of spaCy's DATE entity boundaries. Real spaCy turned out to be
# unreliable for exactly the phrasing the DATE-priority fix cares about: on
# "the 25th of December, Monday please" it sometimes tags ONLY "Monday" as
# DATE (missing the explicit date as an entity entirely), and on "the 1st of
# January, Monday please" it sometimes merges both into ONE entity
# ("the 1st of January, Monday"), which then let the weekday-shortcut in
# _resolve_date fire on the trailing "monday" substring instead of resolving
# the explicit date. Found via real-model runs of test_date_priority_battery.py
# (the sandboxed fake-spaCy stub used to develop the original fix never
# exhibited either failure mode, since its regex-based tagging always split
# entities cleanly).
_EXPLICIT_DATE_RE = re.compile(
    rf"\b(?:the\s+)?\d{{1,2}}(?:st|nd|rd|th)\s+of\s+(?:{_MONTH_ALT})\b"
    rf"|\b(?:{_MONTH_ALT})\s+\d{{1,2}}(?:st|nd|rd|th)?\b"
    rf"|\b\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?\b",
    re.IGNORECASE,
)

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

    # DATE entities get special handling: prefer an explicit day+month
    # ("the 13th of July") over a bare weekday name ("Monday"), instead of
    # just taking whichever DATE entity spaCy found first. Without this, a
    # caller correcting themselves - "No, I mean Monday, the 13th of July" -
    # had the correction silently ignored: "Monday" (containing no digit)
    # was found first, resolved to next week's Monday via the weekday
    # shortcut below, and the explicit "13th of July" right after it in the
    # same sentence was never even considered.
    #
    # Priority: a direct regex match on the raw text first (exact, clean
    # boundaries, independent of spaCy), then an explicit (digit-containing)
    # spaCy DATE entity, then whatever spaCy found first.
    date_ents = [ent for ent in doc.ents if ent.label_ == "DATE"]
    text_date_match = _EXPLICIT_DATE_RE.search(text)
    explicit_ents = [e.text for e in date_ents if re.search(r"\d", e.text)]

    if text_date_match:
        chosen_date_text = text_date_match.group()
    elif explicit_ents:
        chosen_date_text = explicit_ents[0]
    elif date_ents:
        chosen_date_text = date_ents[0].text
    else:
        chosen_date_text = None

    if chosen_date_text:
        result["date_raw"]      = chosen_date_text
        result["date_resolved"] = _resolve_date(chosen_date_text, today)

    for ent in doc.ents:
        if ent.label_ == "TIME" and result["time_raw"] is None:
            result["time_raw"]      = ent.text
            result["time_resolved"] = _resolve_time(ent.text)
        elif ent.label_ == "PERSON" and result["person"] is None:
            result["person"] = ent.text.title()

    # spaCy's TIME entity boundaries are unreliable for spoken phrasing - it
    # sometimes tags only a trailing fragment ("afternoon" instead of "half
    # past three in the afternoon"), or misses phrases like "quarter past
    # two" or "ten o'clock" as a TIME entity entirely. If nothing resolved
    # from spaCy's entities, scan the raw utterance directly.
    if result["time_resolved"] is None:
        text_time_match = _TIME_TEXT_RE.search(text)
        if text_time_match:
            result["time_raw"]      = text_time_match.group()
            result["time_resolved"] = _resolve_time(text_time_match.group())

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
    has_digit = bool(re.search(r"\d", t))

    if t in ("today", "now"):
        return today.isoformat()
    if t in ("tomorrow", "tmrw", "tmr"):
        return (today + timedelta(days=1)).isoformat()
    if t in ("yesterday",):
        return (today - timedelta(days=1)).isoformat()

    # "next Monday", "this Friday", "on Wednesday", plain weekday name.
    # Skipped when the text also has a digit - spaCy sometimes merges an
    # explicit date with a trailing weekday into one DATE entity ("the 1st
    # of January, Monday"), and without this guard the weekday substring
    # match would fire first and silently discard the explicit date.
    if not has_digit:
        for i, wd in enumerate(WEEKDAYS):
            if wd in t:
                days_ahead = (i - today.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                return (today + timedelta(days=days_ahead)).isoformat()

    # If a weekday name got merged in alongside an explicit date, strip it
    # out before handing the string to dateutil below - otherwise dateutil
    # can anchor on the weekday token instead of the explicit date.
    if has_digit:
        for wd in WEEKDAYS:
            t = re.sub(rf"\b{wd}\b", "", t).strip()
        text = t

    # "in 3 days", "in two days" - "in" is optional since spaCy sometimes
    # excludes it from the DATE entity span, tagging just "3 days".
    m = re.search(r"(?:in\s+)?(\d+|one|two|three|four|five)\s+days?\b", t)
    if m:
        word_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        n_str = m.group(1)
        n = int(n_str) if n_str.isdigit() else word_map.get(n_str, 1)
        return (today + timedelta(days=n)).isoformat()

    # "next week"
    if "next week" in t:
        return (today + timedelta(weeks=1)).isoformat()

    # Absolute dates: "June 24th", "24th", "the 13th of July", "24/06", etc.
    # fuzzy=True is required: dateutil's strict parser raises on filler words
    # like "the" and "of" ("the 13th of July" -> ParserError: Unknown string
    # format), so without it every "the Nth of Month" phrase silently
    # resolved to None - including the exact explicit-date correction the
    # DATE-priority fix above exists to capture. This went unnoticed because
    # only the entity *selection* logic (explicit-vs-weekday) was verified in
    # isolation, not the full extract() -> resolved-date output.
    # dayfirst=True matches UK spoken/written convention (24/06 -> 24 June,
    # not month 24) for the rare caller who gives a slash date.
    try:
        default_dt = datetime(today.year, today.month, today.day)
        parsed = dparser.parse(text, default=default_dt, fuzzy=True, dayfirst=True)
        # If the resolved date is in the past, bump to next year.
        if parsed.date() < today:
            parsed = parsed.replace(year=today.year + 1)
        return parsed.date().isoformat()
    except (ValueError, OverflowError):
        return None

_WORD_NUMS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_HOUR_WORD = r"(?:" + "|".join(_WORD_NUMS) + r"|\d{1,2})"

# Direct regex scan over the raw utterance for a spoken time phrase,
# independent of spaCy's TIME entity tagging (see the comment above
# extract()'s time-fallback block for why this is needed). Each pattern
# optionally captures a trailing "in the morning/afternoon/evening" so the
# period is resolved from context instead of the ambiguous-time default in
# _to_24h.
_PERIOD_SUFFIX = r"(?:\s+in the (?:morning|afternoon|evening))?"
_TIME_TEXT_RE = re.compile(
    r"\b(?:noon|midday|midnight)\b"
    rf"|\bhalf past {_HOUR_WORD}{_PERIOD_SUFFIX}\b"
    rf"|\bquarter (?:past|to) {_HOUR_WORD}{_PERIOD_SUFFIX}\b"
    rf"|\b{_HOUR_WORD}\s*(?:o'?\s?clock|oclock){_PERIOD_SUFFIX}\b"
    rf"|\b{_HOUR_WORD}\s*(?:am|pm)\b"
    rf"|\b{_HOUR_WORD}\s+in the (?:morning|afternoon|evening)\b"
    r"|\b\d{1,2}:\d{2}\b",
    re.IGNORECASE,
)


def _hour_val(w: str) -> Optional[int]:
    w = w.lower()
    if w in _WORD_NUMS:
        return _WORD_NUMS[w]
    if w.isdigit():
        n = int(w)
        return n if 1 <= n <= 12 else None
    return None


def _period(text: str) -> Optional[str]:
    """"am"/"pm"/None from context words. None = genuinely ambiguous."""
    t = text.lower()
    if "morning" in t:
        return "am"
    if any(w in t for w in ("afternoon", "evening", "tonight", "night")):
        return "pm"
    if re.search(r"\bam\b", t):
        return "am"
    if re.search(r"\bpm\b", t):
        return "pm"
    return None


def _to_24h(hour: int, period: Optional[str]) -> int:
    if period == "pm" and hour != 12:
        return hour + 12
    if period == "am" and hour == 12:
        return 0
    if period is None:
        # No am/pm cue at all ("half past nine" with nothing else). Default
        # to the practice's likely opening hours rather than returning
        # nothing: 8-11 -> morning, 12-7 -> afternoon/evening, since a GP
        # practice is not open at 1-6am. This is a guess, not a fact about
        # what the caller meant - but the slot is always read back to the
        # caller for confirmation before booking (see pending_suggestion
        # flow in inference.py), so a wrong guess here gets caught there
        # rather than silently booking the wrong appointment.
        if hour in (8, 9, 10, 11, 12):
            return hour if hour != 12 else 12
        return hour + 12
    return hour


def _resolve_time(text: str) -> Optional[str]:
    """Resolve a time phrase to 24h HH:MM.

    dateutil.parser cannot parse spoken time phrasing at all - no word
    numbers ("ten", "nine"), no "half past"/"quarter to", no "noon"/
    "midnight", no "o'clock", and no "in the morning" style qualifiers
    (confirmed: every one of those raises ParserError). Untested, this
    silently meant any caller who said a time in natural English instead
    of a bare digit ("3pm", "10:30") got no time_resolved at all, and the
    booking flow would treat it as if no time had been given. These
    phrasings are handled explicitly first; dateutil is only used as a
    fallback for clean digit-based input it can actually parse.
    """
    t = text.lower().strip()

    if re.search(r"\b(noon|midday)\b", t):
        return "12:00"
    if re.search(r"\bmidnight\b", t):
        return "00:00"

    m = re.search(rf"\bhalf past ({_HOUR_WORD})\b", t)
    if m:
        h = _hour_val(m.group(1))
        if h is not None:
            return f"{_to_24h(h, _period(t)):02d}:30"

    m = re.search(rf"\bquarter past ({_HOUR_WORD})\b", t)
    if m:
        h = _hour_val(m.group(1))
        if h is not None:
            return f"{_to_24h(h, _period(t)):02d}:15"

    m = re.search(rf"\bquarter to ({_HOUR_WORD})\b", t)
    if m:
        h = _hour_val(m.group(1))
        if h is not None:
            prev_hour = 12 if h == 1 else h - 1
            return f"{_to_24h(prev_hour, _period(t)):02d}:45"

    m = re.search(rf"\b({_HOUR_WORD})\s*(?:o'?\s?clock|oclock)\b", t)
    if m:
        h = _hour_val(m.group(1))
        if h is not None:
            return f"{_to_24h(h, _period(t)):02d}:00"

    m = re.search(rf"\b({_HOUR_WORD})\s*(am|pm)\b", t)
    if m:
        h = _hour_val(m.group(1))
        if h is not None:
            return f"{_to_24h(h, m.group(2)):02d}:00"

    # Word-number with a morning/afternoon/evening qualifier but no
    # o'clock/am/pm ("ten in the morning", "nine in the evening").
    m = re.search(rf"\b({_HOUR_WORD})\s+in the (morning|afternoon|evening)\b", t)
    if m:
        h = _hour_val(m.group(1))
        if h is not None:
            period = "am" if m.group(2) == "morning" else "pm"
            return f"{_to_24h(h, period):02d}:00"

    try:
        default_dt = datetime(2000, 1, 1)
        parsed = dparser.parse(t, default=default_dt, fuzzy=True)
        return parsed.strftime("%H:%M")
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
