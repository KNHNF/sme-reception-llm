"""
Step 2: Convert merged HuggingFace model to GGUF and quantize.

Usage:
  python scripts/02_convert_gguf.py --model phi3
  python scripts/02_convert_gguf.py --model phi3 --quant Q4_K_M
  python scripts/02_convert_gguf.py --model llama3 --quant Q4_K_M

Clones llama.cpp to tools/llama_cpp/ on first run (no build required for conversion).
Quantization needs llama-quantize binary -- auto-downloaded for Windows, or build manually.

Quant options (by size, Phi-3 3.8B):
  Q2_K   ~1.4 GB  ~80% quality  fastest
  Q3_K_M  2.0 GB  ~90% quality
  Q4_K_M  2.3 GB  ~95% quality  recommended
  Q5_K_M  2.7 GB  ~97% quality
  Q8_0    4.0 GB  ~99% quality  slowest
"""

import argparse
import json
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
TOOLS_DIR = ROOT / "tools" / "llama_cpp"
GGUF_DIR = ROOT / "checkpoints" / "gguf"

CONFIGS = {
    "phi3": {
        "merged_path": ROOT / "checkpoints" / "merged" / "phi3",
        "gguf_f16":    GGUF_DIR / "phi3-f16.gguf",
    },
    "llama3": {
        "merged_path": ROOT / "checkpoints" / "merged" / "llama3",
        "gguf_f16":    GGUF_DIR / "llama3-f16.gguf",
    },
    "llama1b": {
        "merged_path": ROOT / "checkpoints" / "merged" / "llama1b",
        "gguf_f16":    GGUF_DIR / "llama1b-f16.gguf",
    },
    "qwen0.5b": {
        "merged_path": ROOT / "checkpoints" / "merged" / "qwen0.5b",
        "gguf_f16":    GGUF_DIR / "qwen0.5b-f16.gguf",
    },
    "qwen1.5b": {
        "merged_path": ROOT / "checkpoints" / "merged" / "qwen1.5b",
        "gguf_f16":    GGUF_DIR / "qwen1.5b-f16.gguf",
    },
    "smol360": {
        "merged_path": ROOT / "checkpoints" / "merged" / "smol360",
        "gguf_f16":    GGUF_DIR / "smol360-f16.gguf",
    },
}


def run(cmd, **kwargs):
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def ensure_llama_cpp():
    convert_script = TOOLS_DIR / "convert_hf_to_gguf.py"
    if convert_script.exists():
        print("llama.cpp already present.", flush=True)
        return
    print("Cloning llama.cpp (shallow clone, ~200MB)...", flush=True)
    TOOLS_DIR.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1",
         "https://github.com/ggerganov/llama.cpp", str(TOOLS_DIR)])


def install_requirements():
    # Try new layout first, then fall back to root requirements
    candidates = [
        TOOLS_DIR / "requirements" / "requirements-convert_hf_to_gguf.txt",
        TOOLS_DIR / "requirements.txt",
    ]
    req_file = next((f for f in candidates if f.exists()), None)
    if req_file:
        print(f"Installing requirements from {req_file.name}...", flush=True)
        run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)])
    else:
        # Minimum needed
        run([sys.executable, "-m", "pip", "install", "-q", "gguf", "sentencepiece"])


def convert_to_gguf(merged_path: Path, out_f16: Path):
    if out_f16.exists():
        print(f"F16 GGUF already exists, skipping conversion: {out_f16}", flush=True)
        return
    out_f16.parent.mkdir(parents=True, exist_ok=True)
    convert_script = TOOLS_DIR / "convert_hf_to_gguf.py"
    print("Converting merged model to GGUF F16 (this takes a few minutes)...", flush=True)
    run([
        sys.executable, str(convert_script),
        str(merged_path),
        "--outfile", str(out_f16),
        "--outtype", "f16",
    ])
    size_gb = out_f16.stat().st_size / 1e9
    print(f"F16 GGUF saved: {out_f16} ({size_gb:.1f} GB)", flush=True)


def find_quantize_binary():
    # Search recursively under tools dir first, then PATH
    for match in TOOLS_DIR.rglob("llama-quantize.exe"):
        return match
    for match in TOOLS_DIR.rglob("llama-quantize"):
        return match
    found = shutil.which("llama-quantize") or shutil.which("llama-quantize.exe")
    return Path(found) if found else None


def download_quantize_binary():
    print("Fetching latest llama.cpp release from GitHub...", flush=True)
    api_url = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
    req = urllib.request.Request(api_url, headers={"User-Agent": "python"})
    with urllib.request.urlopen(req) as resp:
        release = json.loads(resp.read())

    assets = release.get("assets", [])
    # Exclude CUDA/Vulkan/cudart zips. Prefer AVX2 CPU build for Windows x64.
    def is_cpu_win_zip(name):
        n = name.lower()
        return (
            n.endswith(".zip")
            and "win" in n
            and "x64" in n
            and "cuda" not in n
            and "vulkan" not in n
            and "cudart" not in n
        )

    cpu_assets = [a for a in assets if is_cpu_win_zip(a["name"])]
    # Prefer avx2 over noavx over anything else
    win_zip = (
        next((a for a in cpu_assets if "avx2" in a["name"].lower()), None)
        or next((a for a in cpu_assets if "noavx" in a["name"].lower()), None)
        or (cpu_assets[0] if cpu_assets else None)
    )

    if not win_zip:
        print("No suitable CPU Windows zip found in release assets.", flush=True)
        return None

    zip_name = win_zip["name"]
    zip_url = win_zip["browser_download_url"]
    zip_path = TOOLS_DIR / zip_name
    print(f"Downloading {zip_name}...", flush=True)
    urllib.request.urlretrieve(zip_url, str(zip_path))

    # Extract all files into bin/ so DLLs are alongside the exe
    bin_dir = TOOLS_DIR / "bin"
    bin_dir.mkdir(exist_ok=True)
    print(f"Extracting all files to {bin_dir}...", flush=True)
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(bin_dir))

    # Close zip before deleting
    if zip_path.exists():
        zip_path.unlink()

    # Find llama-quantize.exe anywhere under bin/
    found = list(bin_dir.rglob("llama-quantize.exe"))
    if found:
        print(f"Binary ready: {found[0]}", flush=True)
        return found[0]

    return None


def quantize(f16_path: Path, q_path: Path, quant_type: str):
    if q_path.exists():
        print(f"Quantized GGUF already exists: {q_path}", flush=True)
        return

    quantize_bin = find_quantize_binary()

    if quantize_bin is None and platform.system() == "Windows":
        print("llama-quantize not found. Attempting to download pre-built binary...", flush=True)
        quantize_bin = download_quantize_binary()

    if quantize_bin is None:
        print("\nCould not find or download llama-quantize.", flush=True)
        print("To build manually:", flush=True)
        print(f"  cmake -B build {TOOLS_DIR} -DLLAMA_NATIVE=ON", flush=True)
        print("  cmake --build build --config Release --target llama-quantize", flush=True)
        print(f"\nThen run:", flush=True)
        print(f"  llama-quantize {f16_path} {q_path} {quant_type}", flush=True)
        return

    print(f"Quantizing to {quant_type} (~2GB output)...", flush=True)
    run([str(quantize_bin), str(f16_path), str(q_path), quant_type])

    size_gb = q_path.stat().st_size / 1e9
    print(f"Quantized GGUF: {q_path} ({size_gb:.1f} GB)", flush=True)

    # Delete intermediate F16 to recover ~7GB
    f16_path.unlink()
    print("Deleted F16 intermediate file (freed ~7GB).", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["phi3", "llama3", "llama1b", "qwen0.5b", "qwen1.5b", "smol360"], required=True)
    parser.add_argument("--quant", default="Q3_K_M",
                        help="Quantization type: Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q8_0 (default: Q3_K_M)")
    args = parser.parse_args()

    cfg = CONFIGS[args.model]
    quant = args.quant.upper()
    gguf_q = GGUF_DIR / f"{args.model}-{quant}.gguf"

    if not cfg["merged_path"].exists():
        print(f"ERROR: merged model not found at {cfg['merged_path']}")
        print("Run scripts/01_merge_adapter.py first.")
        sys.exit(1)

    ensure_llama_cpp()
    install_requirements()
    convert_to_gguf(cfg["merged_path"], cfg["gguf_f16"])
    quantize(cfg["gguf_f16"], gguf_q, quant)

    if gguf_q.exists():
        size_gb = gguf_q.stat().st_size / 1e9
        print(f"\nAll done.")
        print(f"  GGUF model : {gguf_q} ({size_gb:.1f} GB)")
        print(f"  Next       : python scripts/03_cpu_server.py --model {args.model}")
    else:
        print("\nConversion complete but quantization step needs manual llama-quantize binary.")


if __name__ == "__main__":
    main()
