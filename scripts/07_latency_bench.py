"""
Step 7: Clean latency benchmark, one server at a time.

Why this exists: 06_aligned_eval.py measures accuracy and latency in the same
pass, but its latency was taken while other models/servers were loading, so the
numbers are noisy (Q3 came out slower than Q4, which is backwards). Accuracy is
deterministic at temperature 0 and does not change, so it is NOT re-run here.
This script only re-measures latency, under clean conditions:

  - one llama.cpp server running at a time (this script starts and stops each)
  - a stray-server kill before each run so nothing competes for the CPU
  - warmup requests before timing so the first measured call is not cold
  - identical settings across every config (threads, ctx, n_predict)
  - the same 60 records and the same per-record prompts as 06_aligned_eval.py

For a trustworthy result: plug in the charger, set Windows power mode to Best
performance, close heavy apps, and do not touch the laptop while it runs.

Usage:
  python scripts/07_latency_bench.py                 # the three cited configs
  python scripts/07_latency_bench.py --all           # every GGUF present
  python scripts/07_latency_bench.py --configs llama3:Q4_K_M phi3:Q3_K_M

Outputs:
  evaluation/cpu_results/latency_bench.json   clean P50/P95 per config
  aligned_summary.json                        matching rows patched in place
"""

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import urllib.request
import urllib.error

ROOT = Path(__file__).parent.parent
TEST_FILE = ROOT / "data" / "synthetic" / "sme_test.jsonl"
RESULTS_DIR = ROOT / "evaluation" / "cpu_results"
GGUF_DIR = ROOT / "checkpoints" / "gguf"
TOOLS_DIR = ROOT / "tools" / "llama_cpp"

DEFAULT_CONFIGS = [("llama3", "Q4_K_M"), ("llama3", "Q3_K_M"), ("phi3", "Q3_K_M")]


def format_prompt(record: dict, model_family: str) -> tuple[str, str]:
    """Return (prompt, stop_token). Identical to 06_aligned_eval.py."""
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


def call_server(prompt: str, stop: str, port: int, n_predict: int, timeout: int = 120) -> float:
    payload = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
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
        resp.read()
    return (time.perf_counter() - t0) * 1000


def server_healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def find_server_binary() -> Path | None:
    for pattern in ["llama-server.exe", "server.exe", "llama-server"]:
        matches = list(TOOLS_DIR.rglob(pattern))
        if matches:
            return matches[0]
    return None


def kill_stray_servers():
    """Best-effort kill of any lingering llama-server so it does not steal CPU."""
    for name in ("llama-server.exe", "server.exe"):
        subprocess.run(["taskkill", "/F", "/IM", name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def start_server(gguf: Path, server_bin: Path, port: int, threads: int,
                 ctx: int, n_predict: int) -> subprocess.Popen:
    cmd = [
        str(server_bin), "-m", str(gguf),
        "-c", str(ctx), "-n", str(n_predict),
        "--host", "127.0.0.1", "--port", str(port),
        "-t", str(threads), "--log-disable",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_for_health(port: int, load_timeout: int) -> bool:
    deadline = time.time() + load_timeout
    while time.time() < deadline:
        if server_healthy(port):
            return True
        time.sleep(1)
    return False


def percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(int(len(sorted_vals) * q), len(sorted_vals) - 1)
    return sorted_vals[idx]


def bench_one(model: str, quant: str, records: list[dict], server_bin: Path,
              args) -> dict | None:
    gguf = GGUF_DIR / f"{model}-{quant}.gguf"
    if not gguf.exists():
        print(f"SKIP {model} {quant}: {gguf.name} not found")
        return None

    kill_stray_servers()
    time.sleep(2)
    print(f"\n=== {model} {quant} ===")
    print(f"  starting server ({gguf.name})...")
    proc = start_server(gguf, server_bin, args.port, args.threads, args.ctx, args.n_predict)
    try:
        if not wait_for_health(args.port, args.load_timeout):
            print(f"  ERROR: server did not become healthy within {args.load_timeout}s")
            return None

        print(f"  warmup ({args.warmup} requests)...")
        warm_rec = records[0]
        wprompt, wstop = format_prompt(warm_rec, model)
        for _ in range(args.warmup):
            call_server(wprompt, wstop, args.port, args.n_predict)

        print(f"  measuring {len(records)} records...")
        latencies = []
        for i, rec in enumerate(records, 1):
            prompt, stop = format_prompt(rec, model)
            latencies.append(call_server(prompt, stop, args.port, args.n_predict))
            if i % 20 == 0:
                print(f"    {i}/{len(records)}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        kill_stray_servers()

    s = sorted(latencies)
    row = {
        "model": model, "quant": quant, "hardware": "CPU",
        "n_samples": len(latencies),
        "threads": args.threads, "ctx": args.ctx, "n_predict": args.n_predict,
        "latency_p50_ms": round(percentile(s, 0.50), 0),
        "latency_p95_ms": round(percentile(s, 0.95), 0),
        "latency_mean_ms": round(sum(s) / len(s), 0) if s else 0,
        "latency_min_ms": round(s[0], 0) if s else 0,
        "latency_max_ms": round(s[-1], 0) if s else 0,
        "measured_at": datetime.now().isoformat(),
    }
    print(f"  P50 {row['latency_p50_ms']}ms   P95 {row['latency_p95_ms']}ms   "
          f"mean {row['latency_mean_ms']}ms")
    return row


def parse_configs(args) -> list[tuple[str, str]]:
    if args.configs:
        out = []
        for c in args.configs:
            model, quant = c.split(":")
            out.append((model, quant.upper()))
        return out
    if args.all:
        out = []
        for g in sorted(GGUF_DIR.glob("*.gguf")):
            stem = g.stem
            model, _, quant = stem.partition("-")
            out.append((model, quant.upper()))
        return out
    return DEFAULT_CONFIGS


def patch_summary(rows: list[dict]):
    """Overwrite latency fields in aligned_summary.json in place, keep accuracy."""
    path = RESULTS_DIR / "aligned_summary.json"
    if not path.exists():
        print("aligned_summary.json not found, skipping patch")
        return
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary, list):
        summary = [summary]
    by_key = {(r["model"], r["quant"]): r for r in rows}
    patched = 0
    for entry in summary:
        key = (entry.get("model"), entry.get("quant"))
        if key in by_key:
            clean = by_key[key]
            entry["latency_p50_ms"] = clean["latency_p50_ms"]
            entry["latency_p95_ms"] = clean["latency_p95_ms"]
            entry["latency_clean"] = True
            entry["latency_measured_at"] = clean["measured_at"]
            patched += 1
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Patched clean latency into {patched} aligned_summary.json rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", help="model:quant e.g. llama3:Q4_K_M phi3:Q3_K_M")
    ap.add_argument("--all", action="store_true", help="every GGUF in checkpoints/gguf")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--n-predict", type=int, default=80)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--load-timeout", type=int, default=240,
                    help="seconds to wait for a model to load (f16 is slow)")
    ap.add_argument("--no-patch", action="store_true",
                    help="do not touch aligned_summary.json")
    args = ap.parse_args()

    server_bin = find_server_binary()
    if server_bin is None:
        print("ERROR: llama-server.exe not found under tools/llama_cpp/")
        return

    records = [json.loads(line) for line in
               TEST_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    configs = parse_configs(args)
    print(f"Configs: {', '.join(f'{m} {q}' for m, q in configs)}")
    print(f"Settings: {args.threads} threads, ctx {args.ctx}, "
          f"n_predict {args.n_predict}, {len(records)} records, warmup {args.warmup}")

    rows = []
    for model, quant in configs:
        row = bench_one(model, quant, records, server_bin, args)
        if row:
            rows.append(row)

    if not rows:
        print("\nNo results produced.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "latency_bench.json"
    existing = []
    if out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
        if not isinstance(existing, list):
            existing = [existing]
    keys = {(r["model"], r["quant"]) for r in rows}
    existing = [r for r in existing if (r.get("model"), r.get("quant")) not in keys]
    existing.extend(rows)
    out.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nSaved clean latency to {out.name}")

    if not args.no_patch:
        patch_summary(rows)

    print("\nClean latency (P50 / P95 ms):")
    for r in rows:
        print(f"  {r['model']:<7} {r['quant']:<7}  "
              f"P50 {r['latency_p50_ms']:>6.0f}   P95 {r['latency_p95_ms']:>6.0f}")


if __name__ == "__main__":
    main()
