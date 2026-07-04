"""
Step 6: Aligned CPU eval on the SAME test set and prompts as the GPU eval.

Purpose: 04_cpu_eval.py used 30 hand-written cases with one short prompt, while
evaluate_model.py (GPU) used data/synthetic/sme_test.jsonl with per-record prompts.
That made GPU vs CPU numbers non-comparable. This script closes the gap: it runs the
exact sme_test.jsonl records through the llama.cpp server using the identical
format_prompt() strings from evaluate_model.py, via the /completion endpoint (raw
prompt, no chat-template guessing). Now GPU and CPU sit the same exam.

Usage:
  1. Start server:  python scripts/03_cpu_server.py --model llama3 --quant Q4_K_M
  2. Run:           python scripts/06_aligned_eval.py --model llama3 --quant Q4_K_M

Results: evaluation/cpu_results/aligned_<model>_<quant>.json (per sample)
         evaluation/cpu_results/aligned_summary.json        (one row per model+quant)
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import urllib.request
import urllib.error

try:
    from sklearn.metrics import precision_recall_fscore_support
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

ROOT = Path(__file__).parent.parent
TEST_FILE = ROOT / "data" / "synthetic" / "sme_test.jsonl"
RESULTS_DIR = ROOT / "evaluation" / "cpu_results"
ACTION_TYPES = ["check_availability", "book_appointment",
                "cancel_appointment", "clarify", "out_of_scope"]


def format_prompt(record: dict, model_family: str) -> tuple[str, str]:
    """Return (prompt, stop_token). Identical to evaluate_model.py format_prompt."""
    if model_family == "llama3":
        prompt = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{record['instruction']}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{record['input']}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        return prompt, "<|eot_id|>"
    prompt = (
        f"<|system|>\n{record['instruction']}<|end|>\n"
        f"<|user|>\n{record['input']}<|end|>\n"
        f"<|assistant|>\n"
    )
    return prompt, "<|end|>"


def call_server(prompt: str, stop: str, port: int, timeout: int = 60) -> tuple[str, float]:
    payload = json.dumps({
        "prompt": prompt,
        "n_predict": 80,
        "temperature": 0,
        "stop": [stop],
        "cache_prompt": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    latency_ms = (time.perf_counter() - t0) * 1000
    return body.get("content", "").strip(), latency_ms


def parse_output(text: str) -> dict | None:
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def compare(predicted: dict | None, expected: dict) -> dict:
    if predicted is None:
        return {"json_valid": False, "action_correct": False,
                "fields_correct": False, "exact_match": False}
    action_correct = predicted.get("action") == expected.get("action")
    expected_fields = {k: v for k, v in expected.items() if k != "action"}
    fields_correct = all(predicted.get(k) == v for k, v in expected_fields.items())
    exact_match = (action_correct and fields_correct
                   and set(predicted.keys()) == set(expected.keys()))
    return {"json_valid": True, "action_correct": action_correct,
            "fields_correct": fields_correct, "exact_match": exact_match}


def check_server(port: int) -> bool:
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["phi3", "llama3"], required=True)
    ap.add_argument("--quant", required=True, help="Quant of the running server, for labels (e.g. Q4_K_M, F16)")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    quant = args.quant.upper()

    if not check_server(args.port):
        print(f"ERROR: no server at port {args.port}. Start 03_cpu_server.py first.")
        return

    records = [json.loads(line) for line in TEST_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(records)} test records from {TEST_FILE.name}")
    print(f"Running {args.model} {quant} on the aligned exam...")

    results, latencies = [], []
    for i, rec in enumerate(records, 1):
        expected = json.loads(rec["output"])
        prompt, stop = format_prompt(rec, args.model)
        try:
            raw, latency_ms = call_server(prompt, stop, args.port)
        except urllib.error.URLError as exc:
            print(f"{i:>3}  ERROR: {exc}")
            results.append({"error": str(exc), "input": rec["input"], "latency_ms": 0})
            continue
        predicted = parse_output(raw)
        scores = compare(predicted, expected)
        latencies.append(latency_ms)
        results.append({**scores, "latency_ms": round(latency_ms, 1),
                        "input": rec["input"], "expected": expected,
                        "predicted": predicted or {}, "raw": raw})
        if i % 10 == 0:
            print(f"  {i}/{len(records)} done")

    valid = [r for r in results if "error" not in r]
    n = len(valid)
    action_acc = sum(r["action_correct"] for r in valid) / n if n else 0
    json_valid = sum(r["json_valid"] for r in valid) / n if n else 0
    fields_acc = sum(r["fields_correct"] for r in valid) / n if n else 0
    exact = sum(r["exact_match"] for r in valid) / n if n else 0
    lat_sorted = sorted(latencies)
    p50 = lat_sorted[len(lat_sorted) // 2] if lat_sorted else 0
    p95 = lat_sorted[int(len(lat_sorted) * 0.95)] if lat_sorted else 0

    f1_macro = None
    if _SKLEARN and n:
        y_true = [r["expected"].get("action") or "unparseable" for r in valid]
        y_pred = [(r["predicted"] or {}).get("action") or "unparseable" for r in valid]
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=ACTION_TYPES, average="macro", zero_division=0)
        f1_macro = round(float(f1) * 100, 1)

    summary = {
        "model": args.model, "quant": quant, "hardware": "CPU",
        "harness": "aligned_sme_test", "n_samples": len(results),
        "timestamp": datetime.now().isoformat(),
        "action_acc": round(action_acc * 100, 1),
        "json_valid": round(json_valid * 100, 1),
        "fields_acc": round(fields_acc * 100, 1),
        "exact_match": round(exact * 100, 1),
        "action_f1_macro": f1_macro,
        "latency_p50_ms": round(p50, 0),
        "latency_p95_ms": round(p95, 0),
    }

    print(f"\n{args.model} {quant} ({n} valid):")
    print(f"  Action acc : {summary['action_acc']}%   F1(macro): {f1_macro}")
    print(f"  JSON valid : {summary['json_valid']}%")
    print(f"  Fields acc : {summary['fields_acc']}%   Exact: {summary['exact_match']}%")
    print(f"  Latency    : P50 {summary['latency_p50_ms']}ms  P95 {summary['latency_p95_ms']}ms")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"aligned_{args.model}_{quant}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    summary_path = RESULTS_DIR / "aligned_summary.json"
    rows = []
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = existing if isinstance(existing, list) else [existing]
    rows = [s for s in rows if not (s.get("model") == args.model and s.get("quant") == quant)]
    rows.append(summary)
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"\nSaved: {out.name} and aligned_summary.json")


if __name__ == "__main__":
    main()
