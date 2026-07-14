"""
Combinatorial regression battery for name-parsing logic in src/inference.py.

Covers, in isolation and via full Pipeline.run() name-capture turns:
  - _join_spelled_name()      hyphen/space/dot spelled-name joining
  - _extract_spelled_name()   spelled-run detection incl. the apostrophe-
                               lookbehind bug ("it's K-A-R-A-N" -> "Karan",
                               not "S Karan")
  - the awaiting_name / awaiting_name_confirm branches: bare greetings
    ("Hello?", "Hi") and yes/no fillers must never be captured as a name;
    real names (plain, spelled, with preambles) must be captured cleanly.

Run: python test_name_parsing_battery.py
Exit 0 = all passed, exit 1 = failures. Imports fake_spacy_stub first so
this runs even without the real en_core_web_sm model on disk (see that
file's docstring for why, and its limits).
"""
import itertools
import sys
import uuid
from pathlib import Path

# Moved into tests/ on 2026-07-14: needs both this dir (fake_spacy_stub.py
# lives here too) and repo root (for `from src.X import ...`) on the path.
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import fake_spacy_stub  # noqa: E402  (must run before src.inference is imported)

from src.inference import _join_spelled_name, _extract_spelled_name, Pipeline  # noqa: E402
import src.session_manager as sm  # noqa: E402

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
results = []


def check(label, ok, detail=""):
    print(f"  {PASS if ok else FAIL} {label}")
    if not ok and detail:
        print(f"       {detail}")
    results.append(ok)


def sid():
    return str(uuid.uuid4())


JOIN_CASES = [
    ("K-A-R-A-N", "Karan"),
    ("k-a-r-a-n", "Karan"),
    ("J-A-C-K R-E-A-C-H-E-R", "Jack Reacher"),
    ("my name is K-A-R-A-N", "my name is Karan"),
    ("K.A.R.A.N", "K.A.R.A.N"),
    ("K A R A N", "K A R A N"),
    ("no hyphenated words at all", "no hyphenated words at all"),
    ("A-B", "Ab"),
    ("O-L-U-W-A-S-E-U-N", "Oluwaseun"),
    ("D-A-V-I-D and S-M-I-T-H", "David and Smith"),
]

print("=== _join_spelled_name ===")
for raw, expected in JOIN_CASES:
    got = _join_spelled_name(raw)
    check(f"join({raw!r}) == {expected!r}", got == expected, f"got {got!r}")


EXTRACT_CASES = [
    ("it's K-A-R-A-N", "Karan"),
    ("it's K A R A N", "Karan"),
    ("that's J-O-H-N", "John"),
    ("let's see, K-A-R-A-N", "Karan"),
    ("no, K A R A N", "Karan"),
    ("K. A. R. A. N.", "Karan"),
    ("J-O-H-N, and V-I-C-K", "John Vick"),
    ("just a normal sentence", None),
    ("I said no", None),
    ("hello there", None),
    ("A B", None),
    ("it's Karan", None),
]

print("\n=== _extract_spelled_name ===")
for raw, expected in EXTRACT_CASES:
    got = _extract_spelled_name(raw)
    check(f"extract({raw!r}) == {expected!r}", got == expected, f"got {got!r}")


GREETINGS = ["Hi", "Hello", "Hi there", "Hello?", "Hiya", "Howdy", "Hey"]
BARE_FILLERS = ["yes", "yeah", "no", "nope", "ok", "sure", "correct", "sorry", "hello?"]

PLAIN_NAMES = [
    "John Smith", "Karan", "Priya Patel", "Mohammed Al-Farsi", "Li",
    "O", "Anna-Maria Jones",
]
PREAMBLE_TEMPLATES = [
    "It's {name}", "It's {name}.", "This is {name}", "My name is {name}",
    "Name's {name}", "Call me {name}", "I'm {name}",
]
SPELLED_NAMES = [
    ("K-A-R-A-N", "Karan"),
    ("it's K-A-R-A-N", "Karan"),
    ("J-A-C-K R-E-A-C-H-E-R", "Jack Reacher"),
    ("K A R A N", "Karan"),
]

print("\n=== Pipeline name-capture: bare greetings/fillers must NOT become the name ===")
n = 0
for greet, filler in itertools.product(GREETINGS, BARE_FILLERS):
    s = sid()
    p = Pipeline(mode="mock")
    p.run(greet, session_id=s)
    r = p.run(filler, session_id=s)
    n += 1
    bad = r.get("caller_name") is not None and r["caller_name"].lower() == filler.strip("?.! ").lower()
    check(f"[{n}] greet={greet!r} filler={filler!r} -> caller_name={r.get('caller_name')!r}",
          not bad)
print(f"  ({n} combinations checked)")

print("\n=== Pipeline name-capture: plain names ===")
n = 0
for greet, name in itertools.product(GREETINGS, PLAIN_NAMES):
    s = sid()
    p = Pipeline(mode="mock")
    p.run(greet, session_id=s)
    r = p.run(name, session_id=s)
    n += 1
    got = (r.get("caller_name") or "").strip()
    ok = got != "" and got.lower() not in ("there",)
    check(f"[{n}] greet={greet!r} name={name!r} -> caller_name={got!r}", ok)
print(f"  ({n} combinations checked)")

print("\n=== Pipeline name-capture: preamble + name ===")
n = 0
for tmpl, name in itertools.product(PREAMBLE_TEMPLATES, ["Karan", "Sarah Johnson", "David"]):
    s = sid()
    p = Pipeline(mode="mock")
    p.run("Hello", session_id=s)
    utter = tmpl.format(name=name)
    r = p.run(utter, session_id=s)
    n += 1
    got = (r.get("caller_name") or "")
    ok = name.split()[0].lower() in got.lower()
    check(f"[{n}] {utter!r} -> caller_name={got!r}", ok)
print(f"  ({n} combinations checked)")

print("\n=== Pipeline name-capture: spelled names ===")
n = 0
for greet in GREETINGS:
    for spoken, expected in SPELLED_NAMES:
        s = sid()
        p = Pipeline(mode="mock")
        p.run(greet, session_id=s)
        r = p.run(spoken, session_id=s)
        n += 1
        got = (r.get("caller_name") or "")
        ok = got == expected
        check(f"[{n}] greet={greet!r} spoken={spoken!r} -> {got!r} (expected {expected!r})", ok)
print(f"  ({n} combinations checked)")


total = len(results)
passed = sum(results)
print(f"\n{passed}/{total} passed")
sys.exit(0 if passed == total else 1)
