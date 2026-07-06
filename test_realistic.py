"""
Realistic-phrasing robustness tests.

The main suite (test_pipeline.py) uses clean inputs, so it passes. Real callers do
not speak in clean inputs: they hesitate, use dialect, are misheard by the STT, say
"nah you're alright" instead of "no". This suite throws that at the pipeline and
asserts the invariants that must hold no matter what the caller says:

  1. it never crashes,
  2. it always returns a non-empty spoken reply,
  3. it never books an appointment after the caller declined,
  4. the caller's name, once captured, is not lost mid-call.

It does NOT assert the exact action (book vs check) in mock mode, because that is the
rule-based mock, not the fine-tuned model.

Two modes:
  python test_realistic.py          mock mode: invariants only, runs anywhere, fast
  python test_realistic.py --cpu    runs against the live Qwen server AND adds an action-
                                    intent check that catches the real model misclassifying
                                    realistic phrasing ("sort me out an appointment" etc).
                                    Start the server first:
                                    python scripts/03_cpu_server.py --model qwen0.5b --quant Q4_K_M
"""

import sys
import uuid
import traceback

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

from src.inference import Pipeline

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
results = []


def sid():
    return str(uuid.uuid4())


def named_session(p):
    """Return a session id that has already passed name capture."""
    s = sid()
    p.run("hello", s)
    p.run("Sam Brown", s)
    p.run("yes", s)
    return s


def check(label, ok, detail=""):
    print(f"  {PASS if ok else FAIL} {label}")
    if not ok and detail:
        print(f"       {detail}")
    results.append(ok)


def safe_run(p, utterance, s):
    """Return (result_dict, crash_message_or_None)."""
    try:
        return p.run(utterance, s), None
    except Exception as e:
        traceback.print_exc()
        return None, f"{type(e).__name__}: {e}"


def invariant(p, utterance, s, label, declined=False):
    """Assert the universal invariants for one turn."""
    r, crash = safe_run(p, utterance, s)
    if crash:
        check(label, False, f"CRASHED on {utterance!r}: {crash}")
        return None
    spoken = r.get("spoken", "")
    if not isinstance(spoken, str) or not spoken.strip():
        check(label, False, f"empty spoken reply on {utterance!r}")
        return r
    act = r.get("action")
    if act is not None and not (isinstance(act, dict) and "action" in act):
        check(label, False, f"malformed action {act!r} on {utterance!r}")
        return r
    if declined and isinstance(act, dict) and act.get("action") == "book_appointment":
        check(label, False, f"booked after a decline on {utterance!r}")
        return r
    check(label, True)
    return r


# 1. Messy, hesitant, dialect and STT-garbled inputs must not crash or go silent
def test_robustness(p):
    print("\n Robustness: messy real speech must not crash or go silent")
    messy = [
        "um yeah can I like book something",
        "uh I dunno, maybe Tuesday?",
        "sort me out an appointment please",
        "d'you have owt free thursday",
        "I need to see someone innit",
        "Sit no thanks.",                       # a real STT garble seen in testing
        "yeah so basically my situation is a bit complicated but anyway",
        "book book book book book",
        "cancel no wait actually book",
        "yes no yes no maybe",
        "12345",
        "!!!",
        "......",
        "\U0001F600 hi there",                   # emoji
        "CAN I BOOK RIGHT NOW",                 # shouting
        "appointment" * 80,                      # very long
        "     ",                                 # whitespace only
        "",                                      # empty
    ]
    for u in messy:
        s = named_session(p)
        invariant(p, u, s, f"no crash / non-empty: {u[:40]!r}")


# 2. Declines phrased naturally must never book
def test_declines_never_book(p):
    print("\n Declines must never book")
    declines = ["no thanks", "no that's all thanks", "nah you're alright",
                "nope not today", "no I'm good", "actually never mind"]
    for u in declines:
        s = named_session(p)
        invariant(p, u, s, f"decline does not book: {u!r}", declined=True)


# 3. Names that collide with vocabulary must still be captured
def test_name_collisions(p):
    print("\n Names that look like other words are still captured")
    for name in ["May", "June", "April", "Mark", "Bill", "Summer"]:
        s = sid()
        p.run("hello", s)
        r, crash = safe_run(p, name, s)
        if crash:
            check(f"name {name!r} no crash", False, crash)
            continue
        # after giving a name the system should not be silent and should move on
        ok = bool(r.get("spoken", "").strip())
        check(f"name {name!r} handled, non-empty reply", ok,
              f"reply was {r.get('spoken','')!r}")


# 4. Name is retained across a multi-turn call
def test_name_retained(p):
    print("\n Caller name survives the whole call")
    s = sid()
    p.run("hello", s)
    p.run("my name is Priya", s)
    p.run("yes", s)
    got = []
    for u in ["do you have anything tuesday", "what about wednesday", "ok book it", "yes"]:
        r, crash = safe_run(p, u, s)
        if crash:
            check("name retained (no crash mid-call)", False, crash)
            return
        got.append(r.get("caller_name"))
    # once set, the name should never silently become None again
    seen_name = [g for g in got if g]
    ok = len(seen_name) == len(got) and all(g == seen_name[0] for g in seen_name)
    check("caller name stable across turns", ok, f"names seen: {got}")


# 5. Natural confirmations book, natural declines do not (propose-then-confirm flow)
def _propose(p):
    """Fresh named session with a slot proposed and pending confirmation."""
    s = named_session(p)
    r = p.run("can I book a general appointment", s)
    return s, ("slot" in r["spoken"].lower() or "available" in r["spoken"].lower())


def test_confirmation_variety(p):
    print("\n Natural confirmations book, natural declines do not")
    affirmations = ["yes", "yeah", "that works", "perfect", "go on then",
                    "go ahead", "sounds good", "that'll do", "lovely",
                    "yes please", "book it", "works for me"]
    for word in affirmations:
        s, proposed = _propose(p)
        if not proposed:
            check(f"confirm {word!r} books", False, "no slot was proposed to confirm")
            continue
        r = p.run(word, s)
        booked = "booked" in r.get("spoken", "").lower()
        check(f"confirm {word!r} books", booked, f"reply: {r.get('spoken','')[:70]!r}")

    declines = ["no", "nope", "nah", "not now", "some other time", "different day"]
    for word in declines:
        s, proposed = _propose(p)
        if not proposed:
            continue
        r = p.run(word, s)
        booked = "booked" in r.get("spoken", "").lower()
        check(f"decline {word!r} does not book", not booked,
              f"reply: {r.get('spoken','')[:70]!r}")


# 6. Action intent on realistic phrasing (real model only, --cpu)
def _act(r):
    a = r.get("action")
    return a.get("action") if isinstance(a, dict) else None


def test_action_intent(p):
    print("\n Action intent on realistic phrasing (real model)")
    # (utterance, acceptable actions). These encode CORRECT behaviour, not one exact answer:
    # a vague booking that clarifies is right (over-eager booking was the earlier bug), and an
    # unsupported request the pipeline intercepts (reschedule, transfer) returns action None with
    # a helpful spoken reply, so None is acceptable there. A FAIL means a clearly wrong category,
    # e.g. a booking request classed as out_of_scope, which is the regression this guards against.
    cases = [
        ("I'd like to book an appointment please",
         {"book_appointment", "check_availability", "clarify"}),
        ("can I get a general appointment next tuesday at 2pm", {"book_appointment"}),
        ("sort me out an appointment", {"book_appointment", "check_availability", "clarify"}),
        ("yeah can I get booked in for a check up",
         {"book_appointment", "check_availability", "clarify"}),
        ("d'you have anything free on thursday", {"check_availability", "book_appointment"}),
        ("what times have you got on friday", {"check_availability", "book_appointment"}),
        ("I need to cancel my appointment", {"cancel_appointment", "out_of_scope", None}),
        ("can I move my appointment to another day",
         {"out_of_scope", "clarify", "cancel_appointment", None}),
        ("what are your opening hours", {"out_of_scope", "clarify"}),
        ("can I speak to a real person", {"out_of_scope", "clarify", None}),
        ("where are you based", {"out_of_scope", "clarify"}),
    ]
    for utt, acceptable in cases:
        s = named_session(p)
        r, crash = safe_run(p, utt, s)
        if crash:
            check(f"intent {utt[:34]!r}", False, crash)
            continue
        act = _act(r)
        ok = act in acceptable
        check(f"intent {utt[:34]!r} -> {act}", ok,
              f"got {act!r}, expected one of {sorted(map(str, acceptable))}")


# 7. Long polite booking phrasing works end to end without crashing
def test_polite_padding(p):
    print("\n Polite padded requests")
    s = named_session(p)
    invariant(p,
              "Hi there, I was wondering if I could possibly book a general "
              "appointment sometime next week please, thank you so much",
              s, "long polite booking handled")


def make_pipeline(use_cpu):
    if not use_cpu:
        return Pipeline(mode="mock"), "mock"
    import urllib.request
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=3) as resp:
            healthy = b"ok" in resp.read()
    except Exception:
        healthy = False
    if not healthy:
        print("No llama.cpp server at http://127.0.0.1:8080")
        print("Start it first, then re-run with --cpu:")
        print("  python scripts/03_cpu_server.py --model qwen0.5b --quant Q4_K_M")
        sys.exit(2)
    return Pipeline(mode="cpu", model_family="qwen0.5b",
                    cpu_url="http://127.0.0.1:8080"), "cpu, qwen0.5b"


if __name__ == "__main__":
    use_cpu = "--cpu" in sys.argv
    p, label = make_pipeline(use_cpu)
    print("=" * 60)
    print(f"  Realistic-phrasing tests ({label})")
    print("=" * 60)
    # Universal invariants run in both modes. Confirmation-variety tests the pipeline's
    # yes/no vocabulary (model-independent) and its setup relies on the rule-based mock
    # proposing a slot, so it is mock-only. Action-intent needs the real model, so cpu-only.
    suite = [test_robustness, test_declines_never_book, test_name_collisions,
             test_name_retained, test_polite_padding]
    suite.append(test_action_intent if use_cpu else test_confirmation_variety)
    for t in suite:
        try:
            t(p)
        except Exception:
            traceback.print_exc()
            results.append(False)

    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 60)
    tag = PASS if passed == total else FAIL
    print(f"  {tag} {passed}/{total} passed - {total - passed} failure(s)")
    print("=" * 60)
    sys.exit(0 if passed == total else 1)
