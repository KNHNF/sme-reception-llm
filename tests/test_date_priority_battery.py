"""
Combinatorial regression battery for entity_extractor.py's date AND time
resolution, including the DATE-entity-priority fix.

Background bug (2026-07-13 real-call transcript): a caller said "Monday,
the 13th of July" and the explicit correction ("the 13th of July") was
silently ignored - the code only ever looked at the FIRST DATE entity
spaCy found, "Monday" resolved via the weekday shortcut to *next* Monday
(the shortcut always skips today, even said on the matching day), and the
explicit date right after it in the same sentence was never considered.

The fix: prefer any DATE entity containing a digit over a bare weekday
name, regardless of which one spaCy found first. This battery checks
that preference holds across every weekday x explicit-date x ordering
combination, not just the one transcript that surfaced it.

Also covers a second bug this battery caught while verifying the first
fix end-to-end: dateutil.parser cannot parse "the Nth of Month" phrasing
at all without fuzzy=True (raises ParserError, silently swallowed,
date_resolved just came back None), and cannot parse spoken time phrasing
("half past nine", "quarter to three", "noon", word-numbers like "ten")
under any settings. Both were fixed in entity_extractor.py's _resolve_date
and _resolve_time. Neither had been exercised end-to-end before - the
original fix for the DATE-priority bug was only verified via a FakeEnt
stub that checked entity *selection*, not what date/time it actually
resolved to.

Run: python test_date_priority_battery.py
Exit 0 = all passed, exit 1 = failures. Imports fake_spacy_stub first so
this runs even without the real en_core_web_sm model on disk.
"""
import re
import sys
from datetime import date
from pathlib import Path

# Moved into tests/ on 2026-07-14: needs both this dir (fake_spacy_stub.py
# lives here too) and repo root (for `from src.X import ...`) on the path.
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import fake_spacy_stub  # noqa: E402

from src.entity_extractor import extract, _resolve_time  # noqa: E402

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
results = []


def check(label, ok, detail=""):
    print(f"  {PASS if ok else FAIL} {label}")
    if not ok and detail:
        print(f"       {detail}")
    results.append(ok)


TODAY = date(2026, 7, 13)  # a Monday - matches the real transcript that found this bug

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# (explicit phrase, expected ISO date against TODAY)
EXPLICIT = [
    ("the 13th of July", "2026-07-13"),      # today itself
    ("the 14th of July", "2026-07-14"),
    ("the 1st of August", "2026-08-01"),
    ("the 25th of December", "2026-12-25"),
    ("the 1st of January", "2027-01-01"),    # already passed this year -> rolls to next year
    ("the 5th of March", "2027-03-05"),      # already passed this year -> rolls to next year
]

print("=== explicit date always wins over a weekday name, either order ===")
n = 0
for weekday in WEEKDAYS:
    for explicit, expected_iso in EXPLICIT:
        for utter in (f"I'd like {weekday}, {explicit} please",
                      f"I'd like {explicit}, {weekday} please"):
            n += 1
            ents = extract(utter, today=TODAY)
            has_digit = bool(ents["date_raw"] and re.search(r"\d", ents["date_raw"]))
            resolved_ok = ents["date_resolved"] == expected_iso
            ok = has_digit and resolved_ok
            check(f"[{n}] {utter!r} -> date_raw={ents['date_raw']!r} "
                  f"date_resolved={ents['date_resolved']!r}",
                  ok,
                  f"expected an explicit (digit) date resolving to {expected_iso}")
print(f"  ({n} combinations checked)")


print("\n=== weekday alone (no explicit date) still resolves, current documented behaviour ===")
# Known, unchanged limitation: the weekday shortcut always skips to the NEXT
# occurrence, even if today matches - so "Monday" said on a Monday resolves
# to next Monday, not today. That is a separate, pre-existing design choice
# (not the bug fixed above) - this test locks in the current behaviour so a
# future change to it is a deliberate decision, not an accidental regression.
n = 0
for i, weekday in enumerate(WEEKDAYS):
    utter = f"can I come in on {weekday}"
    ents = extract(utter, today=TODAY)
    days_ahead = (i - TODAY.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    from datetime import timedelta
    expected = (TODAY + timedelta(days=days_ahead)).isoformat()
    n += 1
    check(f"[{n}] {utter!r} -> {ents['date_resolved']!r} (expected {expected!r})",
          ents["date_resolved"] == expected)
print(f"  ({n} combinations checked)")


print("\n=== relative dates ===")
RELATIVE = [
    ("today", TODAY.isoformat()),
    ("I need it today", TODAY.isoformat()),
    ("tomorrow", "2026-07-14"),
    ("can you fit me in tomorrow", "2026-07-14"),
    ("next week", "2026-07-20"),
    ("in 3 days", "2026-07-16"),
    ("in two days", "2026-07-15"),
    ("in 1 day", "2026-07-14"),
]
n = 0
for utter, expected in RELATIVE:
    ents = extract(utter, today=TODAY)
    n += 1
    check(f"[{n}] {utter!r} -> {ents['date_resolved']!r} (expected {expected!r})",
          ents["date_resolved"] == expected)
print(f"  ({n} combinations checked)")


print("\n=== three-way: two explicit dates in one sentence, first one wins ===")
# Not the bug scenario (both have digits) but worth locking in: when there
# are multiple explicit candidates, extract() takes the first explicit one
# in document order, matching entity_extractor.py's `explicit[0]`.
THREE_WAY = [
    ("actually not the 1st of August, the 14th of July instead", "2026-08-01"),
    ("the 14th of July, or actually the 1st of August", "2026-07-14"),
]
n = 0
for utter, expected in THREE_WAY:
    ents = extract(utter, today=TODAY)
    n += 1
    check(f"[{n}] {utter!r} -> {ents['date_resolved']!r} (expected {expected!r})",
          ents["date_resolved"] == expected)
print(f"  ({n} combinations checked)")


print("\n=== spoken time phrasing (_resolve_time) ===")
TIME_CASES = [
    ("3pm", "15:00"), ("10:30", "10:30"), ("2:30", "02:30"),
    ("noon", "12:00"), ("midday", "12:00"), ("midnight", "00:00"),
    ("half past nine", "09:30"), ("half past three in the afternoon", "15:30"),
    ("quarter past two", "14:15"), ("quarter to three", "14:45"),
    ("ten in the morning", "10:00"), ("nine in the evening", "21:00"),
    ("ten am", "10:00"), ("ten pm", "22:00"),
    ("ten o'clock", "10:00"), ("ten oclock", "10:00"),
    ("eleven am", "11:00"), ("twelve pm", "12:00"), ("one pm", "13:00"),
]
n = 0
for phrase, expected in TIME_CASES:
    n += 1
    got = _resolve_time(phrase)
    check(f"[{n}] {phrase!r} -> {got!r} (expected {expected!r})", got == expected)
print(f"  ({n} cases checked)")

print("\n=== spoken time phrasing survives full extract() too, not just _resolve_time directly ===")
n = 0
for phrase, expected in TIME_CASES:
    n += 1
    ents = extract(f"I'd like an appointment at {phrase} please")
    got = ents["time_resolved"]
    check(f"[{n}] {phrase!r} in a sentence -> {got!r} (expected {expected!r})", got == expected,
          f"time_raw was {ents['time_raw']!r}")
print(f"  ({n} cases checked)")


total = len(results)
passed = sum(results)
print(f"\n{passed}/{total} passed")
sys.exit(0 if passed == total else 1)
