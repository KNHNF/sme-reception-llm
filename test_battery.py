"""
Scripted conversation battery against the pipeline (mock mode by default).
Runs ~105 short "calls" covering name-capture variants, name-rejection/
correction variants (the bug family that produced 'Not Script' and bare
'yes'/'no' as a caller name), booking/availability/cancel phrasing, goodbyes,
and adversarial/nonsense input. Flags crashes, empty responses, and garbage
caller-name captures.

Run: python test_battery.py
(Uses your real spaCy + en_core_web_sm install, unlike the sandboxed
 version used to develop this, which had to stub spaCy out.)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.inference import Pipeline
import src.session_manager as sm

BAD_NAME_TOKENS = {"not", "script", "wrong", "correct", "sure", "actually", "nope",
                    "quite", "yeah", "yes", "no"}

def run_case(pipeline, case_id, turns):
    sid = f"case-{case_id}"
    log = []
    sm._sessions.pop(sid, None)  # fresh session each case
    issues = []
    for i, t in enumerate(turns):
        try:
            r = pipeline.run(t, session_id=sid)
        except Exception as e:
            issues.append(f"CRASH on turn {i} ({t!r}): {type(e).__name__}: {e}")
            log.append((t, f"[CRASH] {e}"))
            break
        spoken = r.get("spoken", "")
        name = r.get("caller_name")
        log.append((t, spoken, name, r.get("action")))
        if not spoken:
            issues.append(f"Empty spoken response on turn {i} ({t!r})")
        if name and name.lower() in BAD_NAME_TOKENS:
            issues.append(f"Bad caller_name captured: {name!r} on turn {i} ({t!r})")
    return log, issues


CASES = []

# --- Name capture variants ---
NAME_STYLES = [
    ["Hi", "John Smith"],
    ["Hello", "My name is Sarah Johnson"],
    ["Hi there", "It's David"],
    ["Hello", "This is Emma Watson"],
    ["Hi", "I'm Michael"],
    ["Hello", "Name's Rob"],
    ["Hi", "Call me Kate"],
    ["Hello", "K-A-R-A-N"],
    ["Hi", "It's K A R A N"],
    ["Hello", "My name is J-A-C-K R-E-A-C-H-E-R"],
    ["Hi", "Priya Patel"],
    ["Hello", "Mohammed Al-Farsi"],
    ["Hi", "It's Anne-Marie"],
    ["Hello", "O'Brien, Sean O'Brien"],
    ["Hi", "uh, John, John Doe"],
    ["Hello", "yeah it's Tom"],
    ["Hi", "My name's uh Lisa"],
    ["Hello", "Dr. Patel"],
    ["Hi", "Mr. Johnson"],
    ["Hello", "Mrs. Williams"],
    ["Hi", "I would like to book an appointment"],
    ["Hi", "I need to cancel my appointment please"],
    ["Hi", "Do you have anything on Tuesday"],
    ["Hi", "hello"],
    ["Hi", "yes"],
    ["Hi", "no"],
    ["Hi", ""],
    ["Hi", "   "],
    ["Hi", "Xx Yy Zz"],
    ["Hi", "asdkjaslkdj"],
]
for i, turns in enumerate(NAME_STYLES):
    CASES.append((f"name-{i:02d}", turns))

# --- Name confirmation reject/correct variants (the bug family) ---
CONFIRM_STYLES = [
    ["Hi", "John", "no", "Actually it's James"],
    ["Hi", "John", "no that's wrong", "It's Karan"],
    ["Hi", "John", "that's not correct", "My name is Anna"],
    ["Hi", "John", "and it's not script", "My name is John"],
    ["Hi", "John", "no it's not right", "I'd like to book an appointment"],
    ["Hi", "John", "nope", "K-A-R-A-N"],
    ["Hi", "John", "wrong", "no"],
    ["Hi", "John", "not quite", "not quite either"],
    ["Hi", "John", "no it's Dave", "yes"],
    ["Hi", "John", "yes", "I'd like to book a general appointment"],
    ["Hi", "John", "yeah that's right", "cancel my appointment"],
    ["Hi", "John", "sure", "check availability Monday"],
    ["Hi", "John", "correct", "book me for a follow-up"],
    ["Hi", "John", "no", "I want to book a consultation"],
    ["Hi", "John", "no", "need an appointment"],
    ["Hi", "John", "actually no", "it's not John it's Jon"],
    ["Hi", "John", "no it's not", "hmm"],
    ["Hi", "John", "wrong person", "K A R A N"],
    ["Hi", "John", "not right at all", "sorry, David"],
    ["Hi", "John", "no way", "im David actually"],
]
for i, turns in enumerate(CONFIRM_STYLES):
    CASES.append((f"confirm-{i:02d}", turns))

# --- Booking / availability / cancel phrasing variety ---
BOOKING_STYLES = [
    ["Hi", "Amy Lee", "yes", "I'd like to book a consultation for next Monday at 2pm"],
    ["Hi", "Amy Lee", "yes", "Can I get a general appointment on Friday morning?"],
    ["Hi", "Amy Lee", "yes", "Do you have any slots this week?"],
    ["Hi", "Amy Lee", "yes", "I need to cancel my appointment on Wednesday"],
    ["Hi", "Amy Lee", "yes", "Book me in for a follow-up please"],
    ["Hi", "Amy Lee", "yes", "What times are free tomorrow?"],
    ["Hi", "Amy Lee", "yes", "I want an appointment as soon as possible"],
    ["Hi", "Amy Lee", "yes", "Can you check Tuesday the 14th?"],
    ["Hi", "Amy Lee", "yes", "I'd like to reschedule my appointment"],
    ["Hi", "Amy Lee", "yes", "I need to move my booking to later"],
    ["Hi", "Amy Lee", "yes", "Please call me back, my number is 07123456789"],
    ["Hi", "Amy Lee", "yes", "Can I leave a message for the doctor?"],
    ["Hi", "Amy Lee", "yes", "I want to speak to a real person"],
    ["Hi", "Amy Lee", "yes", "This is an emergency"],
    ["Hi", "Amy Lee", "yes", "I'm in a lot of pain, can you help"],
    ["Hi", "Amy Lee", "yes", "What are your opening hours?"],
    ["Hi", "Amy Lee", "yes", "What's your address?"],
    ["Hi", "Amy Lee", "yes", "Will I get a confirmation email?"],
    ["Hi", "Amy Lee", "yes", "book a general on the 3rd of August"],
    ["Hi", "Amy Lee", "yes", "consultation next Thursday 10am"],
    ["Hi", "Amy Lee", "yes", "follow up appointment please, any time"],
    ["Hi", "Amy Lee", "yes", "book appointment"],
    ["Hi", "Amy Lee", "yes", "I need to see someone urgently, chest pain"],
    ["Hi", "Amy Lee", "yes", "cancel please"],
    ["Hi", "Amy Lee", "yes", "book general appointment monday 9am", "no", "another day"],
    ["Hi", "Amy Lee", "yes", "check availability", "no", "later"],
    ["Hi", "Amy Lee", "yes", "check availability", "no", "tuesday"],
    ["Hi", "Amy Lee", "yes", "check availability", "yes"],
    ["Hi", "Amy Lee", "yes", "check availability", "that works"],
    ["Hi", "Amy Lee", "yes", "check availability", "not that one, something else"],
]
for i, turns in enumerate(BOOKING_STYLES):
    CASES.append((f"booking-{i:02d}", turns))

# --- End-call / politeness / goodbye variety ---
END_STYLES = [
    ["Hi", "Amy Lee", "yes", "that's all thanks"],
    ["Hi", "Amy Lee", "yes", "no thanks"],
    ["Hi", "Amy Lee", "yes", "goodbye"],
    ["Hi", "Amy Lee", "yes", "bye"],
    ["Hi", "Amy Lee", "yes", "nothing else, thank you"],
    ["Hi", "Amy Lee", "yes", "we're done here"],
    ["Hi", "Amy Lee", "yes", "that will be all"],
    ["Hi", "Amy Lee", "yes", "no that's all thanks"],
    ["Hi", "Amy Lee", "yes", "all done"],
    ["Hi", "Amy Lee", "yes", "thanks, bye now"],
]
for i, turns in enumerate(END_STYLES):
    CASES.append((f"end-{i:02d}", turns))

# --- Adversarial / nonsense / profanity / repeated confusion ---
ADVERSARIAL_STYLES = [
    ["Hi", "Amy Lee", "yes", "asdkjaslkdj"],
    ["Hi", "Amy Lee", "yes", "blah blah blah"],
    ["Hi", "Amy Lee", "yes", "this is fucking useless"],
    ["Hi", "Amy Lee", "yes", "you're an idiot"],
    ["Hi", "Amy Lee", "yes", "what?", "sorry?", "pardon?", "excuse me"],
    ["Hi", "Amy Lee", "yes", "hours", "directions", "email"],
    ["Hi", "Amy Lee", "yes", "xyz123"],
    ["Hi", "Amy Lee", "yes", ""],
    ["Hi", "Amy Lee", "yes", "   "],
    ["Hi", "Amy Lee", "yes", "a" * 600],
    ["Hi", "Amy Lee", "yes", "I NEED AN APPOINTMENT RIGHT NOW"],
    ["Hi", "Amy Lee", "yes", "book a general appointment", "book a general appointment", "book a general appointment"],
    ["Hi", "Amy Lee", "yes", "cancel", "cancel", "cancel"],
    ["Hi", "Amy Lee", "yes", "hello?", "anyone there?"],
    ["Hi", "Amy Lee", "yes", "operator", "human", "speak to someone"],
]
for i, turns in enumerate(ADVERSARIAL_STYLES):
    CASES.append((f"adversarial-{i:02d}", turns))

if __name__ == "__main__":
    print(f"Total cases: {len(CASES)}")
    print()

    pipeline = Pipeline(mode="mock")

    all_issues = []
    for case_id, turns in CASES:
        log, issues = run_case(pipeline, case_id, turns)
        if issues:
            all_issues.append((case_id, turns, log, issues))

    print(f"Cases with issues: {len(all_issues)} / {len(CASES)}")
    print("=" * 60)
    for case_id, turns, log, issues in all_issues:
        print(f"\n[{case_id}] turns={turns}")
        for entry in log:
            print(f"   {entry}")
        for issue in issues:
            print(f"   !! {issue}")
