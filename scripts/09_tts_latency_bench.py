"""
TTS latency benchmark - measures actual numbers on YOUR machine instead of
guessing. No claims were made up for the "how much latency diff" question;
this script exists because that answer didn't exist anywhere in evaluation/
and shouldn't be invented.

What it measures:
  1. Current setup as deployed (src/tts.py's PIPER_MODEL, whatever that
     currently points to - "high" tier as of 2026-07-13).
  2. Optionally, a second voice file you point it at (e.g. download the
     "medium" tier voice and pass its path) for a direct side-by-side on
     the same machine, same phrases, same run.
  3. Reports the per-call subprocess+load overhead separately from
     synthesis-proper by timing a very short phrase vs a long one - the
     fixed cost (process spawn + model load) shows up as the difference
     between "short phrase time" and "expected linear scaling" from the
     long phrase.

Usage:
    python scripts/09_tts_latency_bench.py
    python scripts/09_tts_latency_bench.py --compare-model piper/en_US-lessac-medium.onnx

Requires piper.exe and at least one voice model already set up per
src/tts.py's docstring. Does not play audio - synthesizes to a temp file
and discards it, so this is safe to run repeatedly without a speaker.
"""
import argparse
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from src.tts import PIPER_EXE  # noqa: E402

PHRASES = {
    "short": "Yes.",
    "medium": "Thank you, Jack. Is there anything else I can help you with?",
    "long": (
        "Brilliant, Jack. I've booked a consultation on Tuesday the fourteenth "
        "of July at nine a.m. Is there anything else I can help you with?"
    ),
}

REPEATS = 5  # per phrase, per model - averages out one-off disk/OS noise


def bench_model(model_path: Path, label: str) -> dict:
    if not model_path.exists():
        print(f"  [skip] {label}: model file not found at {model_path}")
        return {}
    print(f"\n=== {label}  ({model_path.name}) ===")
    results = {}
    for name, text in PHRASES.items():
        times = []
        for _ in range(REPEATS):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                out_path = tmp.name
            t0 = time.perf_counter()
            cmd = [str(PIPER_EXE), "--model", str(model_path), "--output-file", out_path]
            subprocess.run(cmd, input=text, capture_output=True, text=True,
                            encoding="utf-8", timeout=15)
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)
            Path(out_path).unlink(missing_ok=True)
        results[name] = statistics.median(times)
        print(f"  {name:8s} ({len(text):3d} chars): "
              f"median {results[name]:6.0f}ms  "
              f"(min {min(times):.0f} / max {max(times):.0f} over {REPEATS} runs)")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-model", default=None,
                     help="Path to a second .onnx voice to benchmark alongside "
                          "the currently deployed one, e.g. a 'medium' tier voice.")
    args = ap.parse_args()

    if not PIPER_EXE.exists():
        print(f"[error] piper.exe not found at {PIPER_EXE}")
        sys.exit(1)

    from src.tts import PIPER_MODEL
    current = bench_model(PIPER_MODEL, "Currently deployed")

    if args.compare_model:
        alt_path = Path(args.compare_model)
        alt = bench_model(alt_path, "Comparison model")

        if current and alt:
            print("\n=== Difference (deployed - comparison), positive = comparison is faster ===")
            for name in PHRASES:
                if name in current and name in alt:
                    diff = current[name] - alt[name]
                    pct = (diff / current[name]) * 100 if current[name] else 0
                    print(f"  {name:8s}: {diff:+6.0f}ms  ({pct:+.0f}%)")

    print("\nThe 'short' row is the closest read on pure subprocess+model-load "
          "overhead, since there's almost no text to actually synthesize. If "
          "'short' isn't much cheaper than 'medium'/'long', most of the cost "
          "is fixed overhead (process spawn + ONNX model load), which is what "
          "a persistent/warm Piper process would remove - not synthesis compute "
          "itself. If 'short' scales down roughly with text length, the cost is "
          "mostly real synthesis time and a persistent process won't help much.")


if __name__ == "__main__":
    main()
