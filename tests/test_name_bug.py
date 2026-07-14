"""
Quick repro/verification for the name-confirmation bugs (2026-07-12).
Covers: 'Not Script' garbage-name capture on rejection, and bare
'yes'/'no' answers being accepted as a literal caller name.
Uses mode="mock" so no mic and no llama.cpp server are required.

Run: python test_name_bug.py
"""
import sys
from pathlib import Path
# repo root (for `from src.X import ...`), moved into tests/ on 2026-07-14.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import Pipeline

p = Pipeline(mode="mock")
sid = "test-name-bug"

turns = [
    "Dr. McKinna, appointment.",
    "Yes, my name is John.",
    "and it's not script.",
    "My name is John.",
]

for t in turns:
    r = p.run(t, session_id=sid)
    print(f"> {t}")
    print(f"  spoken: {r['spoken']}")
    print(f"  caller_name: {r['caller_name']}  action: {r['action']}")
    print()
