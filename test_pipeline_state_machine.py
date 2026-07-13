"""
Integration battery over Pipeline.run() (mock mode): the state-machine
branches that sit above the LLM - booking/cancel/transfer/distress/
message-taking, profanity escalation, out-of-scope ladder, and the
callback-logging dead-end fix from the 2026-07-13 real-call transcript.

Booking confirmations actually call book_slot(), which writes to
data/calendar.json on disk. This backs the file up before running and
restores it afterward so repeated test runs don't eat into the real
demo calendar's availability.

Run: python test_pipeline_state_machine.py
Exit 0 = all passed, exit 1 = failures.
"""
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import fake_spacy_stub  # noqa: E402

from src.inference import Pipeline  # noqa: E402
import src.session_manager as sm  # noqa: E402

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
results = []

CAL_PATH = Path(__file__).parent / "data" / "calendar.json"
CAL_BACKUP = CAL_PATH.read_text() if CAL_PATH.exists() else None


def check(label, ok, detail=""):
    print(f"  {PASS if ok else FAIL} {label}")
    if not ok and detail:
        print(f"       {detail}")
    results.append(ok)


def sid():
    return str(uuid.uuid4())


def named_session(p, name="Sam Brown"):
    s = sid()
    p.run("Hello", s)
    p.run(name, s)
    p.run("yes", s)
    return s


def safe_run(p, utterance, session_id):
    try:
        return p.run(utterance, session_id=session_id), None
    except Exception as e:
        return None, e


print("=== direct booking (explicit date + time) ===")
p = Pipeline(mode="mock")
n = 0
for phrase in [
    "book a general appointment on the 20th of July at 10am",
    "I'd like a consultation on the 21st of July at half past two",
    "can I get a follow-up on the 22nd of July at 9:30",
]:
    n += 1
    s = named_session(p)
    r, err = safe_run(p, phrase, s)
    check(f"[{n}] {phrase!r} does not crash", err is None, str(err))
    if r:
        check(f"[{n}] non-empty spoken response", bool(r.get("spoken")))
print(f"  ({n} cases checked)")


print("\n=== booking via slot suggestion, confirm yes ===")
n = 0
for phrase in ["do you have any general appointments free", "any slots this week",
               "what times do you have for a consultation"]:
    n += 1
    s = named_session(p)
    r1, err1 = safe_run(p, phrase, s)
    check(f"[{n}] {phrase!r} offers a slot without crashing", err1 is None, str(err1))
    if not r1:
        continue
    r2, err2 = safe_run(p, "yes please", s)
    check(f"[{n}] confirming with 'yes please' does not crash", err2 is None, str(err2))
    if r2:
        spoken = r2["spoken"]
        check(f"[{n}] confirmation does NOT promise an unbacked 'shortly'",
              "shortly" not in spoken.lower() or "outside scope" in spoken.lower(),
              spoken)
        check(f"[{n}] confirmation is honest about email/SMS being out of scope",
              "outside scope" in spoken.lower() or "full deployment" in spoken.lower(),
              spoken)
print(f"  ({n} cases checked)")


print("\n=== repeated decline exhausts slots -> logged callback, no dead-end number ask ===")
from datetime import datetime as _dt_check
_cal = json.loads(CAL_PATH.read_text())
_kept_one = False
for _slot in _cal["slots"]:
    if _slot["service"] == "general" and _slot["available"]:
        _slot_dt = _dt_check.strptime(_slot["date"], "%Y-%m-%d")
        _is_weekday = _slot_dt.weekday() < 5
        _now = _dt_check.now()
        _is_future = (_slot_dt.date() > _now.date()
                      or _slot["time"] > _now.strftime("%H:%M"))
        if not _kept_one and _is_weekday and _is_future:
            _kept_one = True
        else:
            _slot["available"] = False
CAL_PATH.write_text(json.dumps(_cal))

s = named_session(p)
r, err = safe_run(p, "any general appointments available", s)
check("initial offer does not crash", err is None, str(err))
crashed = False
saw_callback_wording = False
for i in range(5):
    r, err = safe_run(p, "no, another time", s)
    if err:
        crashed = True
        break
    if "call you back" in r["spoken"].lower():
        saw_callback_wording = True
        check("does NOT ask 'could I take your number' at the dead end",
              "take your number" not in r["spoken"].lower(), r["spoken"])
        break
check("never crashes while exhausting slots", not crashed)
check("eventually reaches the honest callback message, not a dead end", saw_callback_wording)

if CAL_BACKUP is not None:
    CAL_PATH.write_text(CAL_BACKUP)


print("\n=== explicit date correction mid-negotiation ===")
s = named_session(p)
safe_run(p, "any general appointments available", s)
r, err = safe_run(p, "actually, do you have the 25th of July instead", s)
check("date-correction turn does not crash", err is None, str(err))
if r:
    spoken = r["spoken"]
    check("mentions a properly formatted date (the Nth of Month)",
          " the " in spoken and " of " in spoken, spoken)


print("\n=== cancellation always routes to a human, never silently done ===")
n = 0
for phrase in ["I need to cancel my appointment", "please cancel my booking",
               "cancel my consultation on Friday"]:
    n += 1
    s = named_session(p)
    r, err = safe_run(p, phrase, s)
    check(f"[{n}] {phrase!r} does not crash", err is None, str(err))
    if r:
        spoken = r["spoken"].lower()
        check(f"[{n}] confirms it was passed to a human, not auto-cancelled",
              "passed" in spoken or "reception team" in spoken or "confirm it" in spoken,
              r["spoken"])
print(f"  ({n} cases checked)")


print("\n=== transfer requests end the call and say so clearly ===")
n = 0
for phrase in ["can I speak to a real person", "I want to talk to someone",
               "put me through to reception", "transfer me please"]:
    n += 1
    s = named_session(p)
    r, err = safe_run(p, phrase, s)
    check(f"[{n}] {phrase!r} does not crash", err is None, str(err))
    if r:
        check(f"[{n}] ends the call", r.get("end_call") is True)
print(f"  ({n} cases checked)")


print("\n=== reschedule intent (unsupported) offers cancel+rebook, doesn't crash ===")
s = named_session(p)
r, err = safe_run(p, "I'd like to reschedule my appointment", s)
check("does not crash", err is None, str(err))
if r:
    check("offers cancel-and-rebook rather than pretending to reschedule",
          "cancel" in r["spoken"].lower(), r["spoken"])


print("\n=== leave-a-message flow captures the very next utterance ===")
s = named_session(p)
r1, err1 = safe_run(p, "can you take a message for the team", s)
check("message-intent turn does not crash", err1 is None, str(err1))
r2, err2 = safe_run(p, "tell them my appointment card was lost, please post a new one", s)
check("message-content turn does not crash", err2 is None, str(err2))
if r2:
    check("thanks the caller and confirms the message was taken",
          "thank you" in r2["spoken"].lower() and "message" in r2["spoken"].lower(), r2["spoken"])


print("\n=== distress/emergency phrases get the 999 redirect, not the booking flow ===")
n = 0
for phrase in ["this is an emergency", "I'm in severe pain", "I can't breathe",
               "someone has collapsed"]:
    n += 1
    s = named_session(p)
    r, err = safe_run(p, phrase, s)
    check(f"[{n}] {phrase!r} does not crash", err is None, str(err))
    if r:
        check(f"[{n}] tells the caller to call 999", "999" in r["spoken"], r["spoken"])
print(f"  ({n} cases checked)")


print("\n=== profanity escalates over exactly three strikes ===")
s = sid()
p.run("Hello", s)
p.run("Jamie Fox", s)
p.run("yes", s)
r1, e1 = safe_run(p, "this is damn ridiculous", s)
check("strike 1 does not crash", e1 is None, str(e1))
if r1:
    check("strike 1 is a gentle redirect, call continues", r1.get("end_call") is False)
r2, e2 = safe_run(p, "for damn sake just book it", s)
check("strike 2 does not crash", e2 is None, str(e2))
if r2:
    check("strike 2 is a firmer reminder, call still continues", r2.get("end_call") is False)
r3, e3 = safe_run(p, "this is such bullshit", s)
check("strike 3 does not crash", e3 is None, str(e3))
if r3:
    check("strike 3 ends the call", r3.get("end_call") is True)


print("\n=== out-of-scope escalates over four turns then ends gracefully ===")
s = named_session(p)
prev_spoken = set()
crashed = False
ended_on_turn = None
for i in range(1, 6):
    r, err = safe_run(p, "what are your opening hours", s)
    if err:
        crashed = True
        break
    if r["end_call"]:
        ended_on_turn = i
        break
    prev_spoken.add(r["spoken"])
check("never crashes through the escalation ladder", not crashed)
check("eventually ends the call gracefully instead of looping forever",
      ended_on_turn is not None and ended_on_turn <= 4,
      f"ended on turn {ended_on_turn}")
check("gave more than one distinct escalation message (not repeating the same line)",
      len(prev_spoken) >= 2, str(prev_spoken))


print("\n=== goodbye phrases end the call with the standard sign-off ===")
n = 0
for phrase in ["bye", "that's all thanks", "no thanks", "goodbye", "nothing else"]:
    n += 1
    s = named_session(p)
    r, err = safe_run(p, phrase, s)
    check(f"[{n}] {phrase!r} does not crash", err is None, str(err))
    if r:
        check(f"[{n}] ends the call with the sign-off",
              r.get("end_call") is True and "goodbye" in r["spoken"].lower(), r["spoken"])
print(f"  ({n} cases checked)")


print("\n=== empty input never crashes ===")
s = sid()
r, err = safe_run(p, "", s)
check("empty utterance does not crash", err is None, str(err))
if r:
    check("asks the caller to repeat rather than erroring", bool(r["spoken"]))

r, err = safe_run(p, "   ", s)
check("whitespace-only utterance does not crash", err is None, str(err))


print("\n=== very long input is capped safely ===")
s = named_session(p)
long_text = "book an appointment " * 100
r, err = safe_run(p, long_text, s)
check("500+ char utterance does not crash", err is None, str(err))
if r:
    check("still returns a spoken response", bool(r["spoken"]))


total = len(results)
passed = sum(results)
print(f"\n{passed}/{total} passed")

if CAL_BACKUP is not None:
    CAL_PATH.write_text(CAL_BACKUP)
    print("(calendar.json restored to its pre-test state)")

sys.exit(0 if passed == total else 1)
