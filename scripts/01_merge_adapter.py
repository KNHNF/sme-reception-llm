"""
Step 1: Merge QLoRA adapter into base model and save as full HuggingFace model.

Usage:
  python scripts/01_merge_adapter.py --model phi3
  python scripts/01_merge_adapter.py --model llama3

Output saved to checkpoints/merged/phi3/ or checkpoints/merged/llama3/
This merged model is the input for step 2 (GGUF conversion).
You can delete merged/ after GGUF is created to save disk space.
"""

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent.parent

CONFIGS = {
    "phi3": {
        "base_model": "microsoft/Phi-3-mini-4k-instruct",
        "adapter_path": ROOT / "checkpoints" / "sme-phi3-qlora",
        "output_path": ROOT / "checkpoints" / "merged" / "phi3",
    },
    "llama3": {
        "base_model": "meta-llama/Llama-3.2-3B-Instruct",
        "adapter_path": ROOT / "checkpoints" / "sme-llama3-qlora",
        "output_path": ROOT / "checkpoints" / "merged" / "llama3",
    },
    "llama1b": {
        "base_model": "meta-llama/Llama-3.2-1B-Instruct",
        "adapter_path": ROOT / "checkpoints" / "sme-llama1b-qlora",
        "output_path": ROOT / "checkpoints" / "merged" / "llama1b",
    },
    "qwen0.5b": {
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "adapter_path": ROOT / "checkpoints" / "sme-qwen0.5b-qlora",
        "output_path": ROOT / "checkpoints" / "merged" / "qwen0.5b",
    },
    "qwen1.5b": {
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "adapter_path": ROOT / "checkpoints" / "sme-qwen1.5b-qlora-v2",
        "output_path": ROOT / "checkpoints" / "merged" / "qwen1.5b",
    },
    "smol360": {
        "base_model": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "adapter_path": ROOT / "checkpoints" / "sme-smol360-qlora-v2",
        "output_path": ROOT / "checkpoints" / "merged" / "smol360",
    },
}


def merge(model_key: str):
    cfg = CONFIGS[model_key]
    print(f"\nMerging {model_key}")
    print(f"  Base model : {cfg['base_model']}")
    print(f"  Adapter    : {cfg['adapter_path']}")
    print(f"  Output     : {cfg['output_path']}")

    print("\nLoading base model in float16 (saves RAM)...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    print("Loading adapter and merging...")
    model = PeftModel.from_pretrained(model, str(cfg["adapter_path"]))
    model = model.merge_and_unload()

    print("Saving merged model...")
    cfg["output_path"].mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(cfg["output_path"]))

    print("Saving tokenizer...")
    # Load from base model so tokenizer.model (SentencePiece file) is included
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    tokenizer.save_pretrained(str(cfg["output_path"]))

    print(f"\nDone. Merged model saved to: {cfg['output_path']}")
    print("Next: run scripts/02_convert_gguf.py --model", model_key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["phi3", "llama3", "llama1b", "qwen0.5b", "qwen1.5b", "smol360"], required=True)
    args = parser.parse_args()
    merge(args.model)
