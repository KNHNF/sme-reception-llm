"""
SME Voice Assistant - Synthetic Training Dataset Generator
Produces instruction-tuning pairs for QLoRA fine-tuning of Phi-3 mini.

Format: Alpaca-style  {instruction, input, output}
  - instruction: system prompt (static, shared across all samples)
  - input:       the caller utterance after STT + entity extraction context
  - output:      the constrained JSON action (what the model must learn to output)

Run:  python generate_dataset.py
Output: sme_train.jsonl, sme_val.jsonl, sme_test.jsonl
"""

import json
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(42)

# System prompt
# Shared across all samples. Short -- keeps prompt tokens low.
SYSTEM_PROMPT = (
    "Appointment assistant. Output one JSON object only. "
    "Actions: book_appointment, check_availability, cancel_appointment, clarify, out_of_scope. "
    "Services: general, consultation, follow_up. "
    "Dates: YYYY-MM-DD. Times: HH:MM. "
    "If fields missing: {\"action\": \"clarify\", \"missing_fields\": [...]}."
)
    

# Helper: plausible future dates
TODAY = date.today()

def rand_date(min_days=1, max_days=30) -> str:
    d = TODAY + timedelta(days=random.randint(min_days, max_days))
    return d.strftime("%Y-%m-%d")

def rand_time() -> str:
    hour = random.choice([9, 10, 11, 14, 15, 16])
    minute = random.choice(["00", "30"])
    return f"{hour:02d}:{minute}"

def weekday_name(iso_date: str) -> str:
    d = date.fromisoformat(iso_date)
    return d.strftime("%A")

def month_day(iso_date: str) -> str:
    d = date.fromisoformat(iso_date)
    return d.strftime("%B %dth").replace(" 0", " ")

def natural_time(hhmm: str) -> str:
    from datetime import datetime
    t = datetime.strptime(hhmm, "%H:%M")
    return t.strftime("%I:%M %p").lstrip("0").lower()


# Utterance templates

def make_book_utterance(d: str, t: str, service: str) -> str:
    wd = weekday_name(d)
    md = month_day(d)
    nt = natural_time(t)
    service_phrases = {
        "general": random.choice([
            "an appointment", "a general appointment", "to book an appointment",
            "a slot", "to come in"
        ]),
        "consultation": random.choice([
            "a consultation", "a consult", "to speak with someone",
            "a 60-minute consultation", "an initial consultation"
        ]),
        "follow_up": random.choice([
            "a follow-up", "a follow up appointment", "my follow-up",
            "a short follow-up", "a 15-minute follow-up"
        ]),
    }
    svc = service_phrases[service]
    templates = [
        f"I'd like to book {svc} on {wd}.",
        f"Can I schedule {svc} for {md} at {nt}?",
        f"I need {svc} this {wd} at {nt} please.",
        f"Book me in for {svc} on {md}.",
        f"I want {svc} for {wd} at {nt}.",
        f"Could you put me down for {svc} on {md} at {nt}?",
        f"Hi, I'd like to make {svc} for {nt} on {wd}.",
        f"Is it possible to get {svc} on {md} at {nt}?",
        f"I need to come in for {svc}, {wd} at {nt} works for me.",
        f"Please book {svc} for {nt} on {md}.",
    ]
    return random.choice(templates)


def make_check_utterance(d: str, service: str) -> str:
    wd = weekday_name(d)
    md = month_day(d)
    service_phrases = {
        "general": "for an appointment",
        "consultation": "for a consultation",
        "follow_up": "for a follow-up",
    }
    svc = service_phrases[service]
    templates = [
        f"Do you have anything available {svc} on {wd}?",
        f"Are there any slots open on {md}?",
        f"What availability do you have {svc} this {wd}?",
        f"I was wondering if you have space {svc} on {md}.",
        f"Can you check if {wd} is free {svc}?",
        f"Is there any availability on {md} {svc}?",
        f"What times are free {svc} on {wd}?",
        f"Do you have a slot {svc} around {md}?",
    ]
    return random.choice(templates)


def make_cancel_utterance(d: str = None, t: str = None, appt_id: str = None) -> str:
    if appt_id:
        templates = [
            f"I need to cancel my appointment, reference {appt_id}.",
            f"Can you cancel booking {appt_id} please?",
            f"Please cancel my appointment, the reference is {appt_id}.",
            f"I want to cancel {appt_id}.",
        ]
    elif d and t:
        wd = weekday_name(d)
        md = month_day(d)
        nt = natural_time(t)
        templates = [
            f"I need to cancel my appointment on {wd} at {nt}.",
            f"Can I cancel the appointment on {md} at {nt}?",
            f"Please cancel my booking for {wd} at {nt}.",
            f"I want to cancel my {md} appointment at {nt}.",
            f"I can't make it on {wd} at {nt}, please cancel.",
        ]
    else:
        templates = [
            "I need to cancel my appointment.",
            "Can you cancel my upcoming appointment?",
            "I'd like to cancel please.",
            "Cancel my appointment.",
        ]
    return random.choice(templates)


def make_clarify_utterance(missing: list) -> str:
    """Utterances that are vague -- missing key fields."""
    if "date" in missing and "service" in missing:
        templates = [
            "I'd like to book an appointment please.",
            "Can I make a booking?",
            "I need to schedule something.",
            "Hi, I want to come in.",
            "Can I get an appointment?",
        ]
    elif "date" in missing:
        templates = [
            "I need a general appointment in the morning.",
            "Can I book a consultation sometime?",
            "I want to schedule a follow-up.",
            "Book me in for a general appointment please.",
        ]
    elif "time" in missing:
        templates = [
            "I want to book a general appointment on Monday.",
            "Can I get a consultation on the 15th?",
            "Book a follow-up for next Friday please.",
        ]
    elif "service" in missing:
        templates = [
            "I'd like to book something for Monday at 10.",
            "Can I come in on Wednesday at 2pm?",
            "I need an appointment on Friday at 3.",
        ]
    else:
        templates = ["I need to book something."]
    return random.choice(templates)


def make_out_of_scope_utterance() -> str:
    templates = [
        "What are your opening hours?",
        "How much does a consultation cost?",
        "Do you offer home visits?",
        "Can I speak to the manager?",
        "I have a question about my invoice.",
        "Where are you located?",
        "Do you have parking?",
        "What payment methods do you accept?",
        "Can you send me your brochure?",
        "I'd like to make a complaint.",
    ]
    return random.choice(templates)


# Sample builders

def book_sample():
    d = rand_date()
    t = rand_time()
    service = random.choice(["general", "consultation", "follow_up"])
    utterance = make_book_utterance(d, t, service)
    output = {
        "action": "book_appointment",
        "date": d,
        "time": t,
        "service": service,
    }
    return utterance, output


def check_sample():
    d = rand_date()
    service = random.choice(["general", "consultation", "follow_up"])
    utterance = make_check_utterance(d, service)
    output = {
        "action": "check_availability",
        "date": d,
        "service": service,
    }
    return utterance, output


def cancel_with_id_sample():
    appt_id = f"APT-{random.randint(1000, 9999)}"
    utterance = make_cancel_utterance(appt_id=appt_id)
    output = {
        "action": "cancel_appointment",
        "appointment_id": appt_id,
    }
    return utterance, output


def cancel_with_datetime_sample():
    d = rand_date()
    t = rand_time()
    utterance = make_cancel_utterance(d=d, t=t)
    output = {
        "action": "cancel_appointment",
        "date": d,
        "time": t,
    }
    return utterance, output


def clarify_sample():
    # Randomly choose which fields are missing
    options = [
        ["date", "service"],
        ["date"],
        ["time"],
        ["service"],
    ]
    missing = random.choice(options)
    utterance = make_clarify_utterance(missing)
    output = {
        "action": "clarify",
        "missing_fields": missing,
    }
    return utterance, output


def out_of_scope_sample():
    utterance = make_out_of_scope_utterance()
    output = {"action": "out_of_scope"}
    return utterance, output


# Dataset composition
# Target: ~600 training samples, balanced across intents.
# Over-sample book and check (most frequent real-world intents).

INTENT_COUNTS = {
    "book":            250,
    "check":           150,
    "cancel_id":        60,
    "cancel_datetime":  60,
    "clarify":          50,
    "out_of_scope":     30,
}
# Total: 600

GENERATORS = {
    "book":             book_sample,
    "check":            check_sample,
    "cancel_id":        cancel_with_id_sample,
    "cancel_datetime":  cancel_with_datetime_sample,
    "clarify":          clarify_sample,
    "out_of_scope":     out_of_scope_sample,
}


def build_record(utterance: str, output: dict) -> dict:
    """Alpaca-style instruction-tuning record."""
    return {
        "instruction": SYSTEM_PROMPT,
        "input":       utterance,
        "output":      json.dumps(output, ensure_ascii=False),
    }


def generate_dataset():
    all_records = []
    for intent, count in INTENT_COUNTS.items():
        gen = GENERATORS[intent]
        for _ in range(count):
            utterance, output = gen()
            all_records.append(build_record(utterance, output))

    random.shuffle(all_records)

    # 80 / 10 / 10 split
    n = len(all_records)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)

    splits = {
        "sme_train": all_records[:n_train],
        "sme_val":   all_records[n_train:n_train + n_val],
        "sme_test":  all_records[n_train + n_val:],
    }

    out_dir = Path(__file__).parent.parent / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(records)} records to {path}")

    with (out_dir / "sme_all.jsonl").open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nTotal: {n} records")
    print("Sample (first 3):")
    for rec in all_records[:3]:
        print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    generate_dataset()
