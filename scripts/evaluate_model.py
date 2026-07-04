"""
SME Voice Assistant - Model Evaluation Script
Compares three conditions:
  A) Fine-tuned Phi-3 mini + QLoRA adapter (primary)
  B) Vanilla Phi-3 mini, no fine-tuning (baseline)
  C) Ollama local model as convenience baseline (optional)

Metrics:
  - Action accuracy      correct action type / total samples
  - Field accuracy       correct field values given correct action
  - JSON validity rate   parseable valid JSON / total samples
  - Exact match rate     output == expected output exactly
  - Latency (ms)         time from first token request to last token received
  - Latency p50/p95/p99  percentile breakdown

Output:
  evaluation/results/eval_results.json   full per-sample results
  evaluation/results/eval_summary.json   aggregated metrics table

Usage:
  # Evaluate fine-tuned model only
  python evaluate_model.py --mode finetuned --adapter checkpoints/sme-phi3-qlora

  # Evaluate vanilla baseline
  python evaluate_model.py --mode vanilla

  # Evaluate both and compare
  python evaluate_model.py --mode both --adapter checkpoints/sme-phi3-qlora

  # Evaluate Ollama local model (phi3 or phi3:latest via Ollama API)
  python evaluate_model.py --mode ollama --ollama_model phi3
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

try:
    from sklearn.metrics import precision_recall_fscore_support
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

_SCRIPT_DIR  = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",           default="both",
                   choices=["finetuned", "vanilla", "both", "ollama"])
    p.add_argument("--model_family",   default="phi3", choices=["phi3", "llama3"])
    p.add_argument("--model_id",       default=None,
                   help="Override model ID (auto-set from --model_family if omitted)")
    p.add_argument("--adapter",        default=None,
                   help="Path to LoRA adapter directory (auto-set from --model_family if omitted)")
    p.add_argument("--test_file",      default=str(_PROJECT_DIR / "data/synthetic/sme_test.jsonl"))
    p.add_argument("--output_dir",     default=str(_PROJECT_DIR / "evaluation/results"))
    p.add_argument("--max_new_tokens", type=int, default=60)
    p.add_argument("--ollama_model",   default="phi3")
    p.add_argument("--ollama_url",     default="http://localhost:11434")
    p.add_argument("--use_4bit",       action="store_true", default=True)
    p.add_argument("--n_samples",      type=int, default=None)
    args = p.parse_args()

    MODEL_IDS = {
        "phi3":   "microsoft/Phi-3-mini-4k-instruct",
        "llama3": "meta-llama/Llama-3.2-3B-Instruct",
    }
    ADAPTER_PATHS = {
        "phi3":   str(_PROJECT_DIR / "checkpoints/sme-phi3-qlora"),
        "llama3": str(_PROJECT_DIR / "checkpoints/sme-llama3-qlora/checkpoints/sme-llama3-qlora"),
    }
    if args.model_id is None:
        args.model_id = MODEL_IDS[args.model_family]
    if args.adapter is None:
        args.adapter = ADAPTER_PATHS[args.model_family]

    if not torch.cuda.is_available():
        print("No GPU found. Running on CPU -- 4-bit disabled.")
        args.use_4bit = False

    return args

# Model loading

def load_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def load_vanilla_model(model_id: str, use_4bit: bool):
    bnb = None
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model

def load_finetuned_model(model_id: str, adapter_path: str, use_4bit: bool):
    from peft import PeftModel
    base = load_vanilla_model(model_id, use_4bit)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model

# Prompt formatting 
# Must match the format used in training exactly.

def format_prompt(record: dict, model_family: str = "phi3") -> str:
    if model_family == "llama3":
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{record['instruction']}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{record['input']}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    return (
        f"<|system|>\n{record['instruction']}<|end|>\n"
        f"<|user|>\n{record['input']}<|end|>\n"
        f"<|assistant|>\n"
    )

# Single inference

def run_inference_hf(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str = "cuda",
    model_family: str = "phi3",
) -> tuple[str, float]:
    """
    Returns (generated_text, latency_ms).
    Latency measured from tokenisation start to decode end.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    eos_token = "<|eot_id|>" if model_family == "llama3" else "<|end|>"
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids(eos_token),
        )
    t1 = time.perf_counter()

    latency_ms = (t1 - t0) * 1000
    new_tokens = outputs[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, latency_ms

def run_inference_ollama(
    model_name: str,
    prompt: str,
    base_url: str,
) -> tuple[str, float]:
    """
    Calls Ollama /api/generate endpoint.
    Returns (generated_text, latency_ms).
    """
    import urllib.request

    payload = json.dumps({
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 60},
    }).encode()

    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    t1 = time.perf_counter()

    latency_ms = (t1 - t0) * 1000
    text = data.get("response", "").strip()
    return text, latency_ms

# JSON parsing and field comparison 

def parse_output(text: str) -> Optional[dict]:
    """
    Attempt to parse JSON from model output.
    Handles minor formatting issues (trailing text, leading whitespace).
    """
    text = text.strip()
    # Find first { and last } in case model added extra text
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None

def compare_outputs(predicted: Optional[dict], expected: dict) -> dict:
    """
    Returns per-field comparison results.
    """
    if predicted is None:
        return {
            "json_valid":      False,
            "action_correct":  False,
            "fields_correct":  False,
            "exact_match":     False,
            "field_details":   {},
        }

    json_valid     = True
    action_correct = predicted.get("action") == expected.get("action")

    # Compare all fields except action
    field_details = {}
    expected_fields = {k: v for k, v in expected.items() if k != "action"}
    all_fields_correct = True

    for field, exp_val in expected_fields.items():
        pred_val = predicted.get(field)
        correct  = pred_val == exp_val
        field_details[field] = {
            "expected":  exp_val,
            "predicted": pred_val,
            "correct":   correct,
        }
        if not correct:
            all_fields_correct = False

    # Exact match: action + all fields correct, no extra fields
    exact_match = (
        action_correct
        and all_fields_correct
        and set(predicted.keys()) == set(expected.keys())
    )

    return {
        "json_valid":     json_valid,
        "action_correct": action_correct,
        "fields_correct": all_fields_correct,
        "exact_match":    exact_match,
        "field_details":  field_details,
    }

# Evaluation loop 

def evaluate(
    label: str,
    records: list[dict],
    infer_fn,
    max_new_tokens: int,
    model_family: str = "phi3",
) -> dict:
    """
    Runs evaluation over all records.
    infer_fn: callable(prompt) -> (text, latency_ms)
    Returns aggregated results dict.
    """
    per_sample = []
    latencies  = []

    for i, rec in enumerate(records):
        prompt   = format_prompt(rec, model_family)
        expected = json.loads(rec["output"])

        text, latency_ms = infer_fn(prompt)
        predicted = parse_output(text)
        comparison = compare_outputs(predicted, expected)

        per_sample.append({
            "index":        i,
            "input":        rec["input"],
            "expected":     expected,
            "predicted_raw": text,
            "predicted":    predicted,
            **comparison,
            "latency_ms":   latency_ms,
        })
        latencies.append(latency_ms)

        if (i + 1) % 10 == 0:
            print(f"  [{label}] {i+1}/{len(records)} done")

    # Aggregate
    n = len(per_sample)
    json_valid_rate  = sum(s["json_valid"]     for s in per_sample) / n
    action_acc       = sum(s["action_correct"] for s in per_sample) / n
    field_acc        = sum(s["fields_correct"] for s in per_sample) / n
    exact_match_rate = sum(s["exact_match"]    for s in per_sample) / n

    action_types = ["check_availability", "book_appointment",
                    "cancel_appointment", "clarify", "out_of_scope"]

    action_prf = {}
    if _SKLEARN_AVAILABLE:
        y_true = [s["expected"].get("action")        for s in per_sample]
        y_pred = [(s["predicted"] or {}).get("action") for s in per_sample]
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=action_types, average="macro", zero_division=0,
        )
        action_prf = {
            "action_precision_macro": round(float(p),  4),
            "action_recall_macro":    round(float(r),  4),
            "action_f1_macro":        round(float(f1), 4),
        }

    lat_arr = np.array(latencies)

    # Per-action breakdown
    action_breakdown = {}
    for action in action_types:
        subset = [s for s in per_sample if s["expected"].get("action") == action]
        if subset:
            action_breakdown[action] = {
                "n":               len(subset),
                "action_accuracy": sum(s["action_correct"] for s in subset) / len(subset),
                "exact_match":     sum(s["exact_match"]    for s in subset) / len(subset),
                "mean_latency_ms": float(np.mean([s["latency_ms"] for s in subset])),
            }

    summary = {
        "label":            label,
        "n_samples":        n,
        "json_valid_rate":  round(json_valid_rate,  4),
        "action_accuracy":  round(action_acc,        4),
        "field_accuracy":   round(field_acc,         4),
        "exact_match_rate": round(exact_match_rate,  4),
        "latency_ms": {
            "mean": round(float(lat_arr.mean()), 2),
            "p50":  round(float(np.percentile(lat_arr, 50)), 2),
            "p95":  round(float(np.percentile(lat_arr, 95)), 2),
            "p99":  round(float(np.percentile(lat_arr, 99)), 2),
            "min":  round(float(lat_arr.min()), 2),
            "max":  round(float(lat_arr.max()), 2),
        },
        "action_breakdown": action_breakdown,
        **action_prf,
    }

    return {"summary": summary, "per_sample": per_sample}

# MLflow logging

def log_run_to_mlflow(summary: dict, args, results_path: Path) -> None:
    """Log one eval run to MLflow if installed. No-op otherwise so the pipeline never breaks."""
    if not _MLFLOW_AVAILABLE:
        return
    mlflow.set_experiment("sme-reception-eval")
    with mlflow.start_run(run_name=summary["label"]):
        mlflow.log_params({
            "mode":           args.mode,
            "model_family":   args.model_family,
            "model_id":       args.model_id,
            "adapter":        args.adapter,
            "n_samples":      summary["n_samples"],
            "max_new_tokens": args.max_new_tokens,
        })
        lat = summary["latency_ms"]
        mlflow.log_metrics({
            "json_valid_rate":  summary["json_valid_rate"],
            "action_accuracy":  summary["action_accuracy"],
            "field_accuracy":   summary["field_accuracy"],
            "exact_match_rate": summary["exact_match_rate"],
            "latency_mean_ms":  lat["mean"],
            "latency_p50_ms":   lat["p50"],
            "latency_p95_ms":   lat["p95"],
            "latency_p99_ms":   lat["p99"],
            **{k: summary[k] for k in
               ("action_precision_macro", "action_recall_macro", "action_f1_macro")
               if k in summary},
        })
        if results_path.exists():
            mlflow.log_artifact(str(results_path))

# Main

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load test set
    ds = load_dataset("json", data_files=args.test_file, split="train")
    records = [dict(r) for r in ds]
    if args.n_samples:
        records = records[:args.n_samples]
    print(f"Evaluating on {len(records)} test samples")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mf     = args.model_family
    all_summaries = []

    if args.mode in ("vanilla", "both"):
        print(f"\nVanilla {mf} (no fine-tuning)")
        tok   = load_tokenizer(args.model_id)
        model = load_vanilla_model(args.model_id, args.use_4bit)

        def infer_vanilla(prompt):
            return run_inference_hf(model, tok, prompt, args.max_new_tokens, device, mf)

        results_v = evaluate(f"vanilla_{mf}", records, infer_vanilla, args.max_new_tokens, mf)
        with open(out_dir / f"eval_vanilla_{mf}.json", "w") as f:
            json.dump(results_v, f, indent=2)
        all_summaries.append(results_v["summary"])
        print_summary(results_v["summary"])
        log_run_to_mlflow(results_v["summary"], args, out_dir / f"eval_vanilla_{mf}.json")
        del model
        torch.cuda.empty_cache()

    if args.mode in ("finetuned", "both"):
        print(f"\nFine-tuned {mf} + QLoRA")
        tok   = load_tokenizer(args.model_id)
        model = load_finetuned_model(args.model_id, args.adapter, args.use_4bit)

        def infer_finetuned(prompt):
            return run_inference_hf(model, tok, prompt, args.max_new_tokens, device, mf)

        results_ft = evaluate(f"finetuned_{mf}_qlora", records, infer_finetuned, args.max_new_tokens, mf)
        with open(out_dir / f"eval_finetuned_{mf}.json", "w") as f:
            json.dump(results_ft, f, indent=2)
        all_summaries.append(results_ft["summary"])
        print_summary(results_ft["summary"])
        log_run_to_mlflow(results_ft["summary"], args, out_dir / f"eval_finetuned_{mf}.json")
        del model
        torch.cuda.empty_cache()

    if args.mode == "ollama":
        print(f"\nOllama {args.ollama_model}")

        def infer_ollama(prompt):
            clean = prompt.replace("<|system|>\n", "SYSTEM: ")
            clean = clean.replace("<|end|>\n<|user|>\n", "\nUSER: ")
            clean = clean.replace("<|end|>\n<|assistant|>\n", "\nASSISTANT: ")
            return run_inference_ollama(args.ollama_model, clean, args.ollama_url)

        results_ol = evaluate(f"ollama_{args.ollama_model}", records, infer_ollama, args.max_new_tokens, mf)
        with open(out_dir / f"eval_ollama_{args.ollama_model}.json", "w") as f:
            json.dump(results_ol, f, indent=2)
        all_summaries.append(results_ol["summary"])
        print_summary(results_ol["summary"])
        log_run_to_mlflow(results_ol["summary"], args, out_dir / f"eval_ollama_{args.ollama_model}.json")

    # Comparison table
    if len(all_summaries) > 1:
        print("\n=== Comparison ===")
        print_comparison_table(all_summaries)

    with open(out_dir / "eval_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nResults written to {out_dir}/")

# Pretty printing

def print_summary(s: dict):
    print(f"\n  Model:            {s['label']}")
    print(f"  Samples:          {s['n_samples']}")
    print(f"  JSON valid:       {s['json_valid_rate']:.1%}")
    print(f"  Action accuracy:  {s['action_accuracy']:.1%}")
    print(f"  Field accuracy:   {s['field_accuracy']:.1%}")
    print(f"  Exact match:      {s['exact_match_rate']:.1%}")
    if "action_f1_macro" in s:
        print(f"  Action F1 (macro):{s['action_f1_macro']:.1%}  "
              f"P={s['action_precision_macro']:.1%}  R={s['action_recall_macro']:.1%}")
    lat = s['latency_ms']
    print(f"  Latency (ms):     mean={lat['mean']}  p50={lat['p50']}  p95={lat['p95']}")
    print()
    for action, stats in s.get("action_breakdown", {}).items():
        print(f"  {action:<25} n={stats['n']:<4} "
              f"acc={stats['action_accuracy']:.0%}  "
              f"exact={stats['exact_match']:.0%}  "
              f"lat={stats['mean_latency_ms']:.0f}ms")

def print_comparison_table(summaries: list[dict]):
    col_w = 26
    header = f"{'Metric':<30}" + "".join(f"{s['label']:<{col_w}}" for s in summaries)
    print(header)
    print("-" * len(header))

    rows = [
        ("JSON valid rate",  lambda s: f"{s['json_valid_rate']:.1%}"),
        ("Action accuracy",  lambda s: f"{s['action_accuracy']:.1%}"),
        ("Field accuracy",   lambda s: f"{s['field_accuracy']:.1%}"),
        ("Exact match rate", lambda s: f"{s['exact_match_rate']:.1%}"),
        ("Latency p50 (ms)", lambda s: f"{s['latency_ms']['p50']}"),
        ("Latency p95 (ms)", lambda s: f"{s['latency_ms']['p95']}"),
    ]
    for label, fn in rows:
        row = f"{label:<30}" + "".join(f"{fn(s):<{col_w}}" for s in summaries)
        print(row)

if __name__ == "__main__":
    main()
