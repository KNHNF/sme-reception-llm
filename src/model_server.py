"""
Auto-launches the llama.cpp CPU server so demo.py / call_ui.py can be one
command instead of two terminals (start scripts/03_cpu_server.py by hand,
then run the demo in a second window).

    from src.model_server import prompt_for_family, ensure_server

    family = args.family or prompt_for_family()
    server_proc = ensure_server(family, port=args.llm_port)
    try:
        ... run the demo ...
    finally:
        if server_proc:
            server_proc.terminate()

prompt_for_family() lists only models that actually have a GGUF built in
checkpoints/gguf/ (reads the same files scripts/03_cpu_server.py would use),
so the menu never offers a model that would just fail to start.

ensure_server() checks /health first and reuses an already-running server
untouched - it only starts a new one, and only returns a process handle
when it did, so callers know not to kill a server they didn't start.
"""

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GGUF_DIR = ROOT / "checkpoints" / "gguf"
SERVER_SCRIPT = ROOT / "scripts" / "03_cpu_server.py"

FAMILIES = ["phi3", "llama3", "llama1b", "qwen0.5b", "qwen1.5b", "smol360"]

_QUANT_PREFERENCE = ["Q4_K_M", "Q5_K_M", "Q3_K_M", "Q8_0", "Q2_K"]


def find_gguf(family: str, quant: str = None):
    """Same lookup scripts/03_cpu_server.py uses - duplicated here (not
    imported) because that file is a standalone script, and this is just a
    file-existence check, low risk of drifting out of sync."""
    if quant:
        p = GGUF_DIR / f"{family}-{quant.upper()}.gguf"
        return p if p.exists() else None
    for q in _QUANT_PREFERENCE:
        p = GGUF_DIR / f"{family}-{q}.gguf"
        if p.exists():
            return p
    matches = list(GGUF_DIR.glob(f"{family}-*.gguf"))
    return matches[0] if matches else None


def available_models() -> list:
    """Families that actually have a GGUF on disk right now, as
    (family, path) pairs, in FAMILIES order."""
    found = []
    for fam in FAMILIES:
        p = find_gguf(fam)
        if p:
            found.append((fam, p))
    return found


def prompt_for_family(default: str = "qwen0.5b") -> str:
    """Show a numbered menu of models with a GGUF already built and ask
    which one to run. Enter alone picks `default` (falling back to
    whatever is first available if `default` itself has no GGUF).
    """
    models = available_models()
    if not models:
        print(f"[model] no GGUF files found in {GGUF_DIR} -- nothing to choose from.")
        print(f"        build one first: python scripts/02_convert_gguf.py --model <family>")
        raise RuntimeError("no GGUF models available")

    fallback = default if any(f == default for f, _ in models) else models[0][0]

    print("\nAvailable models:")
    for i, (fam, path) in enumerate(models, 1):
        size_mb = path.stat().st_size / (1024 * 1024)
        tag = "  (default)" if fam == fallback else ""
        print(f"  {i}. {fam:10s} {path.name}  ({size_mb:.0f} MB){tag}")

    raw = input(f"\nChoose a model [1-{len(models)}, name, or Enter for {fallback}]: ").strip()
    if not raw:
        return fallback
    if raw.isdigit() and 1 <= int(raw) <= len(models):
        return models[int(raw) - 1][0]
    if raw in [f for f, _ in models]:
        return raw
    print(f"  [not recognised, using default: {fallback}]")
    return fallback


def _healthy(cpu_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{cpu_url}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_server(family: str, quant: str = None, port: int = 8080,
                  threads: int = 6, startup_timeout: float = 90.0):
    """Make sure a llama.cpp CPU server for `family` is reachable at
    127.0.0.1:port, launching scripts/03_cpu_server.py in the background if
    nothing answers /health there yet.

    Returns the Popen handle if this call started a new server - caller
    should terminate it when the demo exits (e.g. in a `finally` block).
    Returns None if a server was already running: reused as-is, and this
    call is not responsible for stopping it (don't kill a server someone
    else started).

    Raises RuntimeError if no GGUF/server binary is found, or the server
    doesn't answer /health within `startup_timeout` seconds (model loading
    on CPU can take a while for the larger families - the timeout is
    generous on purpose).
    """
    cpu_url = f"http://127.0.0.1:{port}"

    if _healthy(cpu_url):
        print(f"[model] server already running at {cpu_url}, reusing it.")
        return None

    if find_gguf(family, quant) is None:
        raise RuntimeError(
            f"No GGUF found for '{family}' in {GGUF_DIR}. "
            f"Build one first: python scripts/02_convert_gguf.py --model {family}")

    cmd = [sys.executable, str(SERVER_SCRIPT), "--model", family,
           "--port", str(port), "--threads", str(threads)]
    if quant:
        cmd += ["--quant", quant]

    print(f"[model] starting llama.cpp server for {family} on port {port}...")
    # Inherits this console so the caller still sees the server's own
    # startup output and any errors, rather than swallowing them silently.
    proc = subprocess.Popen(cmd)

    print("[model] waiting for it to come up", end="", flush=True)
    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if _healthy(cpu_url):
            print(" ready.")
            return proc
        if proc.poll() is not None:
            raise RuntimeError(
                f"llama.cpp server exited early (code {proc.returncode}) - see output above.")
        print(".", end="", flush=True)
        time.sleep(1.0)

    proc.terminate()
    raise RuntimeError(
        f"llama.cpp server did not become healthy within {startup_timeout:.0f}s.")
