"""
SME Voice Assistant - v2 training data generator.

Purpose: fix the two weaknesses the real-audio eval exposed, at the data level,
without changing the action schema (so eval numbers stay comparable):
  1. book vs check_availability confusion -> add "hard" disambiguation pairs
     (explicit date+time = book; asking what is free = check).
  2. reschedule/modify -> out_of_scope, so the model learns to route unsupported
     requests to a safe action instead of producing garbage.
  3. real, messy phrasing (casual openers, filler, "dental checkup") from the
     real recordings, plus vague-time -> clarify so the model stops booking
     invented details.

It reuses the v1 builders and writes ONLY train+val:
  data/synthetic/sme_train_v2.jsonl
  data/synthetic/sme_val_v2.jsonl
data/synthetic/sme_test.jsonl is left untouched on purpose. That is the frozen
exam every reported number uses, so v1 vs v2 stays a fair comparison. Measure the
new behaviours on the real-audio eval (which has reschedule labels).

Run:  python scripts/generate_dataset_v2.py
Then train:  --train_file data/synthetic/sme_train_v2.jsonl
             --val_file   data/synthetic/sme_val_v2.jsonl
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_dataset import (  # reuse v1 building blocks
    SYSTEM_PROMPT, build_record, rand_date, rand_time, natural_time,
    weekday_name, book_sample, check_sample, cancel_with_id_sample,
    cancel_with_datetime_sample, clarify_sample, out_of_scope_sample,
)

random.seed(43)  # different pool from v1 train, test stays frozen
SERVICES = ["general", "consultation", "follow_up"]

_SVC_PHRASE = {
    "general": "an appointment", "consultation": "a consultation",
    "follow_up": "a follow-up",
}
_SVC_CASUAL = {
    "general": ["a checkup", "a dental checkup", "an appointment", "to be seen"],
    "consultation": ["a consultation", "to talk to someone", "a chat with the doctor"],
    "follow_up": ["a follow up", "a quick follow-up", "a review appointment"],
}


def hard_book_sample():
    """Explicit date AND time -> must be book_appointment."""
    d, t = rand_date(), rand_time()
    service = random.choice(SERVICES)
    wd, nt = weekday_name(d), natural_time(t)
    svc = _SVC_PHRASE[service]
    utt = random.choice([
        f"Book me in for {svc} on {wd} at {nt}, please.",
        f"Yes, book {svc} for {wd} at {nt}.",
        f"Go ahead and book {svc} on {wd} at {nt}.",
        f"Put me down for {svc} on {wd} at {nt}.",
        f"I'd like to confirm {svc} for {wd} at {nt}.",
    ])
    return utt, {"action": "book_appointment", "date": d, "time": t, "service": service}


def hard_check_sample():
    """Asking what is free, no specific time -> check_availability."""
    d = rand_date()
    service = random.choice(SERVICES)
    wd, svc = weekday_name(d), _SVC_PHRASE[service]
    utt = random.choice([
        f"Do you have anything free on {wd}?",
        f"What is open on {wd} for {svc}?",
        f"Is {wd} available at all?",
        f"Just checking what slots you have on {wd}.",
        f"Any availability {wd} for {svc}?",
    ])
    return utt, {"action": "check_availability", "date": d, "service": service}


def reschedule_oos_sample():
    """Reschedule/modify an existing booking -> out_of_scope (unsupported)."""
    wd = weekday_name(rand_date())
    utt = random.choice([
        "Can you move my appointment to another day?",
        f"I need to reschedule my appointment to {wd}.",
        "Can we make my appointment a bit later?",
        "I'd like to change my existing booking.",
        f"Please move my appointment to {wd}.",
        "Can you push my appointment back a few days?",
        "I want to reschedule.",
        "Change my appointment to a different time.",
    ])
    return utt, {"action": "out_of_scope"}


def real_book_sample():
    """Messy, real-caller phrasing with an explicit time -> book_appointment."""
    d, t = rand_date(), rand_time()
    service = random.choice(SERVICES)
    wd, nt = weekday_name(d), natural_time(t)
    svc = random.choice(_SVC_CASUAL[service])
    utt = random.choice([
        f"Hi, I just wanted to book {svc} for {wd} at {nt}, is that possible?",
        f"Hello, could I get {svc} on {wd} at {nt} please?",
        f"Yeah hi, I need {svc}, {wd} at {nt} would be good.",
        f"Um, can I book {svc} on {wd} at {nt}?",
        f"Morning, I'd like {svc} for {wd} at {nt} if you have it.",
    ])
    return utt, {"action": "book_appointment", "date": d, "time": t, "service": service}


def real_vague_sample():
    """Vague time -> clarify, so the model stops booking invented details."""
    wd = weekday_name(rand_date())
    utt = random.choice([
        f"Hi, I wanted to book something for {wd} afternoon.",
        "Can I get an appointment sometime next week?",
        "I'd like to come in, whenever is good.",
        f"Book me in for {wd} at some point.",
        "I need to see someone soon, not fussy on time.",
    ])
    return utt, {"action": "clarify", "missing_fields": ["time"]}


INTENT_COUNTS = {
    "book": 220, "check": 140, "cancel_id": 55, "cancel_datetime": 55,
    "clarify": 50, "out_of_scope": 30,
    "hard_book": 80, "hard_check": 60, "reschedule_oos": 80,
    "real_book": 60, "real_vague": 40,
}
GENERATORS = {
    "book": book_sample, "check": check_sample,
    "cancel_id": cancel_with_id_sample, "cancel_datetime": cancel_with_datetime_sample,
    "clarify": clarify_sample, "out_of_scope": out_of_scope_sample,
    "hard_book": hard_book_sample, "hard_check": hard_check_sample,
    "reschedule_oos": reschedule_oos_sample, "real_book": real_book_sample,
    "real_vague": real_vague_sample,
}


def main():
    records = []
    for intent, count in INTENT_COUNTS.items():
        gen = GENERATORS[intent]
        for _ in range(count):
            utt, out = gen()
            records.append(build_record(utt, out))
    random.shuffle(records)

    n = len(records)
    n_train = int(n * 0.9)
    out_dir = Path(__file__).parent.parent / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in [("sme_train_v2", records[:n_train]),
                         ("sme_val_v2", records[n_train:])]:
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for rec in subset:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(subset)} records to {path.name}")

    counts = {}
    for r in records:
        a = json.loads(r["output"]).get("action")
        counts[a] = counts.get(a, 0) + 1
    print(f"\nTotal train+val: {n}")
    print("Action distribution:")
    for a, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {a:<20} {c}")
    print("\nsme_test.jsonl left frozen (the comparison exam).")


if __name__ == "__main__":
    main()
