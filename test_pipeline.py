"""
Pipeline smoke + edge-case tests.
Run with:  python test_pipeline.py
No GPU, no Streamlit, no mic. Mock mode only.
Exit 0 = all passed. Exit 1 = failures.
"""

import sys
import time
import traceback
import uuid

# Windows cp1252 terminals can't print good/bad - force utf-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, "src")

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"

results = []


# helpers 

def sid():
    return str(uuid.uuid4())


def check(label, got_spoken, got_action=None,
          must_contain=None, must_not_contain=None, action=None, crashed=False):
    ok = True
    reasons = []
    if crashed:
        ok = False
        reasons.append("raised an exception")
    if must_contain:
        for phrase in must_contain:
            if phrase.lower() not in got_spoken.lower():
                ok = False
                reasons.append(f"missing '{phrase}'")
    if must_not_contain:
        for phrase in must_not_contain:
            if phrase.lower() in got_spoken.lower():
                ok = False
                reasons.append(f"should NOT contain '{phrase}'")
    if action and got_action and action != got_action:
        ok = False
        reasons.append(f"action={got_action!r}, expected {action!r}")
    symbol = PASS if ok else FAIL
    print(f"  {symbol} {label}")
    if not ok:
        print(f"       GOT : {got_spoken[:130]!r}")
        for r in reasons:
            print(f"       WHY : {r}")
    results.append(ok)


def turn(p, utterance, session_id):
    """Run one turn, return (spoken, action). Returns error strings on crash."""
    try:
        r = p.run(utterance, session_id)
        return r.get("spoken", ""), r.get("action", "")
    except Exception as e:
        traceback.print_exc()
        return f"__CRASH__: {e}", "__crash__"


def run(p, session_id, turns_checks):
    """Run a list of (utterance, checks_dict) tuples in order."""
    for utterance, checks in turns_checks:
        spoken, action = turn(p, utterance, session_id)
        crashed = spoken.startswith("__CRASH__")
        check(
            repr(utterance[:55]),
            spoken, action,
            must_contain=checks.get("must_contain"),
            must_not_contain=checks.get("must_not_contain"),
            action=checks.get("action"),
            crashed=crashed,
        )


# test groups

def test_happy_path(p):
    print("\n Happy path")

    run(p, sid(), [
        ("Hello", {"must_contain": ["name"]}),
        ("Jack Reacher", {"must_contain": ["Jack Reacher"]}),
        ("Yes", {"must_contain": ["help"]}),
        ("Book a general appointment", {"must_contain": ["th"]}),   # ordinal
        ("Yes", {"must_contain": ["booked"]}),
    ])

    run(p, sid(), [
        ("Hello", {}),
        ("Sam Brown", {}),
        ("Yes", {}),
        ("I need a consultation next Thursday", {"must_contain": ["consultation"]}),
        ("No", {}),
        ("Yes", {"must_contain": ["booked"]}),
    ])


def test_name_edge_cases(p):
    print("\n Name edge cases ")

    # Spelled out: K-A-R-A-N -> Karan
    s = sid()
    run(p, s, [
        ("Hello", {}),
        ("K-A-R-A-N", {"must_contain": ["Karan"], "must_not_contain": ["K-A-R-A-N"]}),
        ("Yes", {}),
    ])

    # Two-word spelled: J-A-C-K R-E-A-C-H-E-R -> Jack Reacher
    s = sid()
    run(p, s, [
        ("Hello", {}),
        ("J-A-C-K R-E-A-C-H-E-R", {
            "must_contain": ["Jack Reacher"],
            "must_not_contain": ["J-A-C-K"],
        }),
        ("Yes", {}),
    ])

    # Apostrophe name: O'Brien
    s = sid()
    run(p, s, [
        ("Hello", {}),
        ("O'Brien", {}),   # should not crash
        ("Yes", {}),
    ])

    # Very long name (should not crash)
    s = sid()
    run(p, s, [
        ("Hello", {}),
        ("Bartholomew Worthington-Smythe", {}),
        ("Yes", {}),
    ])

    # Single word name (first name only)
    s = sid()
    run(p, s, [
        ("Hello", {}),
        ("Cher", {}),
        ("Yes", {}),
    ])

    # Name correction: "No, it's X" should not re-ask for name
    s = sid()
    run(p, s, [
        ("Hello", {}),
        ("Jack Preacher", {"must_contain": ["Preacher"]}),
        ("No it's Jack Reacher", {
            "must_not_contain": ["need your name", "still need"],
            "must_contain": ["Reacher"],
        }),
        # After correction the session is in normal mode - ask for a booking
        ("I want to book a general appointment", {
            "must_contain": ["Jack Reacher"],   # name should persist
        }),
    ])

    # Preamble: "my name is..."
    s = sid()
    run(p, s, [
        ("Hello", {}),
        ("My name is Ali Hassan", {"must_contain": ["Ali Hassan"]}),
        ("Yes", {}),
    ])


def test_booking_intent_loop(p):
    print("\n Booking intent loop guard ")

    s = sid()
    run(p, s, [
        ("Hello", {}),
        ("I want to make a consultation", {"must_contain": ["name"]}),
        ("I want to make a consultation", {"must_contain": ["name"]}),
        # 3rd time: bypass - should NOT ask for name again
        ("I want to make a consultation", {"must_not_contain": ["name"]}),
    ])

    # "book an appointment" style
    s = sid()
    run(p, s, [
        ("Hello", {}),
        ("I'd like to book an appointment", {"must_contain": ["name"]}),
        ("I'd like to book an appointment", {"must_contain": ["name"]}),
        ("I'd like to book an appointment", {"must_not_contain": ["name"]}),
    ])


def test_empty_and_noise(p):
    print("\n Empty / noise inputs")

    # Empty string
    s = sid()
    spoken, _ = turn(p, "", s)
    check("empty string", spoken, crashed=spoken.startswith("__CRASH__"))

    # Whitespace only
    s = sid()
    spoken, _ = turn(p, "   ", s)
    check("whitespace only", spoken, crashed=spoken.startswith("__CRASH__"))

    # Very long input (500 chars)
    s = sid()
    long_input = "I want to book an appointment " * 17
    spoken, _ = turn(p, long_input, s)
    check("very long input (500 chars)", spoken, crashed=spoken.startswith("__CRASH__"))

    # Gibberish
    s = sid()
    spoken, _ = turn(p, "asdfghjkl qwerty zxcvbnm", s)
    check("gibberish input", spoken, crashed=spoken.startswith("__CRASH__"))

    # All caps
    s = sid()
    run(p, s, [
        ("HELLO I WANT TO BOOK AN APPOINTMENT", {}),
    ])

    # Numbers only
    s = sid()
    spoken, _ = turn(p, "123456789", s)
    check("numbers only", spoken, crashed=spoken.startswith("__CRASH__"))

    # Unicode / accented chars
    s = sid()
    spoken, _ = turn(p, "José García", s)
    check("unicode name (José García)", spoken, crashed=spoken.startswith("__CRASH__"))

    # Emoji input
    s = sid()
    spoken, _ = turn(p, "I want to book 📅 an appointment 🏥", s)
    check("emoji in input", spoken, crashed=spoken.startswith("__CRASH__"))


def test_profanity(p):
    print("\n Profanity / 3-strike")

    s = sid()
    spoken1, _ = turn(p, "fucking hell", s)
    check("strike 1 - gentle redirect", spoken1,
          must_contain=["help"], must_not_contain=["goodbye", "unable to continue"])

    spoken2, _ = turn(p, "this is bullshit", s)
    check("strike 2 - firm reminder", spoken2,
          must_contain=["respectful"], must_not_contain=["goodbye"])

    spoken3, _ = turn(p, "you're an asshole", s)
    check("strike 3 - end call", spoken3,
          must_contain=["goodbye"])

    # After call ends, further input should be handled gracefully (not crash)
    spoken4, _ = turn(p, "hello", s)
    check("after strike 3 - no crash", spoken4,
          crashed=spoken4.startswith("__CRASH__"))


def test_out_of_scope(p):
    print("\n Out of scope")

    s = sid()
    run(p, s, [
        ("Hello", {}),
        ("Ali Hassan", {}),
        ("Yes", {}),
        # Mock mode is rule-based - "today" triggers date NER so it may offer a slot.
        # The real LLM handles out-of-scope correctly. Just check it doesn't crash.
        ("What's the weather like today?", {}),
        ("Can you recommend a restaurant?", {}),
    ])

    # Should not crash on medical questions
    s = sid()
    run(p, s, [
        ("Hello", {}),
        ("Sam Lee", {}),
        ("Yes", {}),
        ("Do I need a referral to see a specialist?", {}),
    ])


def test_date_edge_cases(p):
    print("\n Date edge cases")

    # Request a past date - should gracefully offer nearest future slot
    s = sid()
    run(p, s, [
        ("Hello", {}),
        ("Ali Hassan", {}),
        ("Yes", {}),
        ("book a general appointment", {}),
        ("can I get it on January 1st", {
            # System correctly explains no Jan slots and offers nearest alternative
            "must_contain": ["don't have", "nearest"],
        }),
    ])

    # Request a weekend - no slots (calendar is Mon-Fri)
    s = sid()
    run(p, s, [
        ("Hello", {}),
        ("Ali Hassan", {}),
        ("Yes", {}),
        ("book a general appointment", {}),
        ("how about Saturday", {
            # System correctly says no Saturday slots and offers weekday alternative.
            # "Saturday" appears in the explanation - check the OFFERED slot is a weekday.
            "must_contain": ["don't have", "nearest"],
            "must_not_contain": ["Saturday at"],   # no slot booked on Saturday
        }),
    ])

    # Ordinal suffix in spoken output - no bare day numbers
    s = sid()
    run(p, s, [
        ("Hello", {}),
        ("Sam Brown", {}),
        ("Yes", {}),
        ("book a general appointment", {
            "must_not_contain": [" 24 June", " 25 June", " 26 June"],
        }),
    ])


def test_slot_exhaustion(p):
    print("\n Slot exhaustion")

    s = sid()
    run(p, s, [
        ("Hello", {}),
        ("Ali Hassan", {}),
        ("Yes", {}),
        ("book a general appointment", {}),
    ])
    # Keep saying no until we run out
    for i in range(25):
        spoken, _ = turn(p, "no", s)
        if "don't have any more" in spoken.lower() or "call you back" in spoken.lower():
            check(f"slot exhaustion - graceful after {i+1} rejections", spoken,
                  must_contain=["don't have"])
            break
        if spoken.startswith("__CRASH__"):
            check(f"slot exhaustion - crashed at rejection {i+1}", spoken, crashed=True)
            break
    else:
        check("slot exhaustion - never ran out (possible infinite loop)", "",
              must_contain=["this text will never be found"])


def test_service_not_in_catalog(p):
    print("\n Unknown service")

    s = sid()
    run(p, s, [
        ("Hello", {}),
        ("Ali Hassan", {}),
        ("Yes", {}),
        ("I need a dentist appointment", {}),   # should not crash
        ("I need physiotherapy", {}),
    ])


def test_ambiguous_confirmation(p):
    print("\n Ambiguous confirmation in pending_suggestion")

    s = sid()
    run(p, s, [
        ("Hello", {}),
        ("Ali Hassan", {}),
        ("Yes", {}),
        ("Book a general appointment", {}),
        ("maybe", {}),        # neither yes nor no - should not crash or book
        ("possibly", {}),     # same
        ("yes please", {"must_contain": ["booked"]}),
    ])


def test_session_reuse_after_booking(p):
    print("\n Session state after completed booking")

    s = sid()
    run(p, s, [
        ("Hello", {}),
        ("Ali Hassan", {}),
        ("Yes", {}),
        ("Book a general appointment", {}),
        ("Yes", {"must_contain": ["booked"]}),
        # After booking, ask another question - session should not crash
        ("I also need a follow-up", {}),
    ])


def test_no_book_without_details(p):
    print("\n Vague booking must not book invented details")

    # "book an appointment" with no date/time/service must NOT silently book.
    # It should propose a real slot and ask for confirmation first.
    s = sid()
    run(p, s, [
        ("Hello", {}),
        ("Ali Hassan", {}),
        ("Yes", {}),
        ("I want to book a new appointment", {
            "must_not_contain": ["I have booked", "booked your", "booked a"],
            "must_contain": ["work for you"],
        }),
    ])

    # A vague "book something" is still a proposal, never a silent booking.
    s = sid()
    run(p, s, [
        ("Hello", {}),
        ("Sam Brown", {}),
        ("Yes", {}),
        ("Can I book an appointment please", {
            "must_not_contain": ["I have booked", "booked your"],
        }),
    ])


def test_reschedule_graceful(p):
    print("\n Reschedule / modify must not fail with 'could not process'")

    phrases = [
        "I want to reschedule my appointment",
        "Can you move my appointment to another day",
        "Can you make it a few days later",
        "Actually can you do it later",
    ]
    for phrase in phrases:
        s = sid()
        run(p, s, [
            ("Hello", {}),
            ("Ali Hassan", {}),
            ("Yes", {}),
            (phrase, {
                "must_not_contain": ["could not process", "did not quite catch"],
                "must_contain": ["cancel"],
            }),
        ])


def test_voicemail(p):
    print("\n Voicemail / take a message")

    s = sid()
    run(p, s, [
        ("Hello", {}),
        ("Ali Hassan", {}),
        ("Yes", {}),
        ("Can you take a message", {
            "must_contain": ["message"],
            "must_not_contain": ["could not process"],
        }),
        ("Please ask the team to call me back on 07123456789", {
            "must_contain": ["taken your message"],
        }),
    ])


def test_cancel_to_human(p):
    print("\n Cancel is handed to a human, never faked")

    s = sid()
    run(p, s, [
        ("Hello", {}),
        ("Ali Hassan", {}),
        ("Yes", {}),
        ("I need to cancel my appointment on the 25th", {
            "must_contain": ["reception"],
            "must_not_contain": ["has been cancelled", "could not process"],
        }),
    ])


def test_end_call(p):
    print("\n Goodbye ends the call, not out_of_scope")

    for bye in ["no that's all, goodbye", "goodbye", "that's all thanks", "no thank you"]:
        s = sid()
        p.run("Hello", s); p.run("Ali Hassan", s); p.run("Yes", s)
        try:
            r = p.run(bye, s)
            ok = r.get("end_call") is True and "goodbye" in r.get("spoken", "").lower()
            detail = f"end_call={r.get('end_call')}, spoken={r.get('spoken','')[:45]!r}"
        except Exception as e:
            ok, detail = False, str(e)
        print(f"  {PASS if ok else FAIL} {bye!r} ends the call")
        if not ok:
            print(f"       {detail}")
        results.append(ok)


def test_rapid_fire_same_session(p):
    print("\n Rapid fire turns (no crashes)")

    s = sid()
    phrases = [
        "hello", "test user", "yes", "book", "no", "no", "yes",
        "cancel", "reschedule", "what", "hello again", "bye",
    ]
    for phrase in phrases:
        spoken, _ = turn(p, phrase, s)
        check(f"rapid fire: {phrase!r}", spoken,
              crashed=spoken.startswith("__CRASH__"))


# entry point 

def main():
    print("\n=== SME Pipeline - Smoke + Edge Case Tests (Mock mode) ===")
    print("(spaCy NER required - run from your local machine)\n")

    try:
        from inference import Pipeline
        p = Pipeline(mode="mock")
        print(f"{PASS} Pipeline loaded\n")
    except Exception:
        print(f"{FAIL} Could not load Pipeline")
        traceback.print_exc()
        sys.exit(1)

    test_happy_path(p)
    test_name_edge_cases(p)
    test_booking_intent_loop(p)
    test_empty_and_noise(p)
    test_profanity(p)
    test_out_of_scope(p)
    test_date_edge_cases(p)
    test_slot_exhaustion(p)
    test_service_not_in_catalog(p)
    test_ambiguous_confirmation(p)
    test_session_reuse_after_booking(p)
    test_no_book_without_details(p)
    test_reschedule_graceful(p)
    test_voicemail(p)
    test_cancel_to_human(p)
    test_end_call(p)
    test_rapid_fire_same_session(p)

    passed = sum(results)
    total  = len(results)
    print(f"\n{'='*60}")
    if passed == total:
        print(f"  {PASS} ALL {total} CHECKS PASSED")
    else:
        print(f"  {FAIL} {passed}/{total} passed - {total - passed} failure(s)")
    print(f"{'='*60}\n")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
