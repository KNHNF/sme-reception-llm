"""
Step 3: Start llama.cpp server for CPU inference.

Usage:
  python scripts/03_cpu_server.py --model phi3
  python scripts/03_cpu_server.py --model phi3 --port 8080 --threads 6 --ctx 2048

Runs in the foreground. Press Ctrl+C to stop.
Exposes OpenAI-compatible API at http://localhost:<port>

Endpoints:
  GET  /health
  POST /v1/chat/completions   (OpenAI format)
  POST /completion            (llama.cpp native)

After starting, run eval:
  python scripts/04_cpu_eval.py --model phi3
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
TOOLS_DIR = ROOT / "tools" / "llama_cpp"
GGUF_DIR = ROOT / "checkpoints" / "gguf"

def find_gguf(model: str, quant: str | None = None) -> Path | None:
    """Find best available GGUF for a model. If quant given, use that exactly."""
    if quant:
        p = GGUF_DIR / f"{model}-{quant.upper()}.gguf"
        return p if p.exists() else None
    # Prefer Q4_K_M > Q5_K_M > Q3_K_M > Q8_0 > Q2_K > anything else
    preference = ["Q4_K_M", "Q5_K_M", "Q3_K_M", "Q8_0", "Q2_K"]
    for q in preference:
        p = GGUF_DIR / f"{model}-{q}.gguf"
        if p.exists():
            return p
    # Fall back to any gguf for this model
    matches = list(GGUF_DIR.glob(f"{model}-*.gguf"))
    return matches[0] if matches else None


def find_server_binary():
    for pattern in ["llama-server.exe", "server.exe", "llama-server"]:
        matches = list(TOOLS_DIR.rglob(pattern))
        if matches:
            return matches[0]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   choices=["phi3", "llama3"], required=True)
    parser.add_argument("--quant",   default=None,
                        help="Force specific quant e.g. Q4_K_M. Auto-selects best if omitted.")
    parser.add_argument("--port",    type=int, default=8080)
    parser.add_argument("--threads", type=int, default=6,
                        help="CPU threads (6-8 recommended)")
    parser.add_argument("--ctx",     type=int, default=2048,
                        help="Context length (2048 is enough for our prompts)")
    parser.add_argument("--max-tokens", type=int, default=80,
                        help="Max tokens per response (~20 for JSON action output)")
    args = parser.parse_args()

    gguf_path = find_gguf(args.model, args.quant)

    if gguf_path is None:
        print(f"ERROR: No GGUF found for {args.model} in {GGUF_DIR}")
        print("Run scripts/02_convert_gguf.py --model", args.model)
        sys.exit(1)

    server_bin = find_server_binary()
    if server_bin is None:
        print("ERROR: llama-server.exe not found in tools/llama_cpp/")
        print("The bin/ zip should have included it alongside llama-quantize.exe.")
        print("Check: tools\\llama_cpp\\bin\\")
        sys.exit(1)

    cmd = [
        str(server_bin),
        "-m", str(gguf_path),
        "-c", str(args.ctx),
        "-n", str(args.max_tokens),
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "-t", str(args.threads),
        "--log-disable",  # cleaner output
    ]

    print(f"Starting llama.cpp server")
    print(f"  Model  : {gguf_path.name}")
    print(f"  Port   : {args.port}")
    print(f"  Threads: {args.threads}")
    print(f"  Ctx    : {args.ctx}")
    print(f"")
    print(f"API ready at: http://127.0.0.1:{args.port}")
    print(f"Health check: http://127.0.0.1:{args.port}/health")
    print(f"Press Ctrl+C to stop.\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except subprocess.CalledProcessError as e:
        print(f"\nServer exited with code {e.returncode}")
        # Try without --log-disable (older llama.cpp versions may not support it)
        print("Retrying without --log-disable flag...")
        cmd_retry = [c for c in cmd if c != "--log-disable"]
        try:
            subprocess.run(cmd_retry, check=True)
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
