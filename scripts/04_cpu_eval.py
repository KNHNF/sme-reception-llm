"""
Step 4: Evaluate GGUF model via llama.cpp server.

Usage:
  1. Start server first:  python scripts/03_cpu_server.py --model phi3
  2. Run eval in second terminal:
       python scripts/04_cpu_eval.py --model phi3 [--port 8080] [--samples 30]

Results saved to evaluation/cpu_results/cpu_phi3.json
Summary printed at end and saved to evaluation/cpu_results/cpu_summary.json

Output format matches existing eval_results/ so charts can compare GPU vs CPU results.
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import urllib.request
import urllib.error

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "evaluation" / "cpu_results"

SYSTEM_PROMPT = (
    "Appointment assistant. Output one JSON object only. "
    "Actions: book_appointment, check_availability, cancel_appointment, clarify, out_of_scope. "
    "Services: general, consultation, follow_up. "
    "Dates: YYYY-MM-DD. Times: HH:MM. "
    "If fields missing: {\"action\": \"clarify\", \"missing_fields\": [...]}."
)

# 30 test utterances covering all action types, matched to expected outputs
# Format: (utterance, expected_dict)
TEST_CASES = [
    # book_appointment (8 cases)
    ("I'd like to book an appointment for next Monday at 10am",
     {"action": "book_appointment", "date": "2026-07-06", "time": "10:00", "service": "general"}),
    ("Can I schedule a consultation for Tuesday afternoon?",
     {"action": "book_appointment", "service": "consultation"}),
    ("I need to make an appointment for a follow up visit",
     {"action": "book_appointment", "service": "follow_up"}),
    ("Book me in for Wednesday morning please",
     {"action": "book_appointment"}),
    ("I want to set up a general appointment for Friday",
     {"action": "book_appointment", "service": "general"}),
    ("Can you book me an appointment for 2pm tomorrow?",
     {"action": "book_appointment", "time": "14:00"}),
    ("I need to see someone as soon as possible",
     {"action": "book_appointment"}),
    ("Schedule me for a consultation next week",
     {"action": "book_appointment", "service": "consultation"}),

    # check_availability (6 cases)
    ("Is there anything available on Thursday?",
     {"action": "check_availability"}),
    ("Do you have any slots free this week?",
     {"action": "check_availability"}),
    ("What times do you have available tomorrow?",
     {"action": "check_availability"}),
    ("Can I check if there's a morning slot on Monday?",
     {"action": "check_availability"}),
    ("Are you free next Friday afternoon?",
     {"action": "check_availability"}),
    ("Do you have anything on the 15th?",
     {"action": "check_availability"}),

    # cancel_appointment (5 cases)
    ("I need to cancel my appointment on Friday",
     {"action": "cancel_appointment"}),
    ("Please cancel my booking for tomorrow",
     {"action": "cancel_appointment"}),
    ("I want to cancel my consultation next Monday",
     {"action": "cancel_appointment", "service": "consultation"}),
    ("Can you remove my appointment please",
     {"action": "cancel_appointment"}),
    ("I need to cancel, something came up",
     {"action": "cancel_appointment"}),

    # clarify (4 cases)
    ("I need an appointment",
     {"action": "clarify"}),
    ("Yes",
     {"action": "clarify"}),
    ("Um, I'm not sure",
     {"action": "clarify"}),
    ("Maybe next week?",
     {"action": "clarify"}),

    # out_of_scope (4 cases)
    ("What's the weather like today?",
     {"action": "out_of_scope"}),
    ("Can you recommend a good restaurant nearby?",
     {"action": "out_of_scope"}),
    ("How much does a consultation cost?",
     {"action": "out_of_scope"}),
    ("Do you sell gift vouchers?",
     {"action": "out_of_scope"}),

    # edge cases (3 cases)
    ("I want to book a CONSULTATION for NEXT MONDAY at TWO PM",
     {"action": "book_appointment", "service": "consultation", "time": "14:00"}),
    ("cancel the friday one",
     {"action": "cancel_appointment"}),
    ("anything tomorrow morning",
     {"action": "check_availability"}),
]


def call_server(utterance: str, port: int, timeout: int = 30) -> tuple[str, float]:
    """Send utterance to llama.cpp server. Returns (raw_text, latency_ms)."""
    payload = json.dumps({
        "model": "local",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": utterance},
        ],
        "temperature": 0,
        "max_tokens": 80,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    latency_ms = (time.perf_counter() - t0) * 1000

    raw = body["choices"][0]["message"]["content"].strip()
    return raw, latency_ms


def extract_json(raw: str) -> dict | None:
    """Extract first JSON object from raw string."""
    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def evaluate_result(predicted: dict | None, expected: dict) -> dict:
    """Return score dict matching existing eval format."""
    json_valid = predicted is not None
    if not json_valid:
        return {"json_valid": False, "action_correct": False,
                "fields_correct": False, "exact_match": False}

    action_correct = predicted.get("action") == expected.get("action")

    # fields_correct: all expected keys present and matching (ignoring keys not in expected)
    fields_correct = all(
        str(predicted.get(k, "")).lower() == str(v).lower()
        for k, v in expected.items()
        if k != "action"
    ) if action_correct else False

    exact_match = action_correct and fields_correct and (
        set(predicted.keys()) == set(expected.keys())
    )

    return {
        "json_valid": json_valid,
        "action_correct": action_correct,
        "fields_correct": fields_correct,
        "exact_match": exact_match,
    }


def check_server(port: int) -> bool:
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   choices=["phi3", "llama3"], required=True)
    parser.add_argument("--quant",   default="Q3_K_M",
                        help="Quant level of the running server (e.g. Q4_K_M). For logging only.")
    parser.add_argument("--port",    type=int, default=8080)
    parser.add_argument("--samples", type=int, default=30,
                        help="Number of test cases to run (max 30)")
    args = parser.parse_args()
    quant = args.quant.upper()

    if not check_server(args.port):
        print(f"ERROR: No server running at port {args.port}.")
        print("Start it first: python scripts/03_cpu_server.py --model phi3")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    test_cases = TEST_CASES[:args.samples]
    results = []
    passed = 0

    print(f"Running {len(test_cases)} test cases against {args.model} GGUF server...")
    print(f"{'#':>3}  {'Action':20} {'JSON':4} {'Act':4} {'Fld':4} {'Exact':5} {'ms':>7}  Input")

    for i, (utterance, expected) in enumerate(test_cases, 1):
        try:
            raw, latency_ms = call_server(utterance, args.port)
            predicted = extract_json(raw)
            scores = evaluate_result(predicted, expected)
            j = "Y" if scores["json_valid"]    else "N"
            a = "Y" if scores["action_correct"] else "N"
            f = "Y" if scores["fields_correct"] else "N"
            e = "Y" if scores["exact_match"]    else "N"
            if scores["action_correct"]:
                passed += 1

            record = {
                **scores,
                "latency_ms":  round(latency_ms, 1),
                "expected":    expected,
                "predicted":   predicted or {},
                "raw":         raw,
                "utterance":   utterance,
            }
            results.append(record)
            action_label = (predicted or {}).get("action", "?")[:20]
            print(f"{i:>3}  {action_label:20} {j:4} {a:4} {f:4} {e:5} {latency_ms:>7.0f}  {utterance[:50]}")

        except urllib.error.URLError as exc:
            print(f"{i:>3}  ERROR: {exc}")
            results.append({"error": str(exc), "utterance": utterance, "latency_ms": 0})

    # Summary
    n = len(results)
    valid_results = [r for r in results if "error" not in r]
    n_valid = len(valid_results)

    action_acc  = sum(r["action_correct"] for r in valid_results) / n_valid if n_valid else 0
    json_valid  = sum(r["json_valid"]     for r in valid_results) / n_valid if n_valid else 0
    exact_match = sum(r["exact_match"]    for r in valid_results) / n_valid if n_valid else 0
    latencies   = [r["latency_ms"] for r in valid_results]
    p50 = sorted(latencies)[len(latencies)//2] if latencies else 0
    p95 = sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0

    summary = {
        "model":       args.model,
        "quant":       quant,
        "hardware":    "CPU",
        "timestamp":   datetime.now().isoformat(),
        "n_samples":   n,
        "action_acc":  round(action_acc * 100, 1),
        "json_valid":  round(json_valid  * 100, 1),
        "exact_match": round(exact_match * 100, 1),
        "latency_p50_ms": round(p50, 0),
        "latency_p95_ms": round(p95, 0),
    }

    print(f"\nResults ({n_valid}/{n} valid):")
    print(f"  Action accuracy : {summary['action_acc']}%")
    print(f"  JSON valid      : {summary['json_valid']}%")
    print(f"  Exact match     : {summary['exact_match']}%")
    print(f"  Latency P50     : {summary['latency_p50_ms']} ms")
    print(f"  Latency P95     : {summary['latency_p95_ms']} ms")

    out_path = RESULTS_DIR / f"cpu_{args.model}_{quant}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    summary_path = RESULTS_DIR / "cpu_summary.json"
    # Append to existing summaries if present; dedupe by model+quant pair
    all_summaries = []
    if summary_path.exists():
        with open(summary_path) as f:
            existing = json.load(f)
            all_summaries = existing if isinstance(existing, list) else [existing]
    all_summaries = [
        s for s in all_summaries
        if not (s.get("model") == args.model and s.get("quant") == quant)
    ]
    all_summaries.append(summary)
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nSaved: {out_path}")
    print(f"Summary: {summary_path}")
    print(f"Next: python scripts/05_compare_results.py  (to compare GPU vs CPU)")


if __name__ == "__main__":
    main()
