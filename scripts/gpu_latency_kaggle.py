"""
GPU latency benchmark for the offline-deployment story.

Runs the SAME GGUF you deploy on CPU, offloaded to a GPU via llama.cpp, over the
SAME 60-record test set and the SAME prompt format the CPU aligned harness used
(scripts/06_aligned_eval.py). That makes the GPU P50 directly comparable to the
CPU P50 in evaluation/cpu_results/aligned_summary.json (Qwen 2.5 0.5B Q4_K_M: 814ms).

The framing for the viva: same model, same prompts, both fully offline and on the
owner's own machine. The GPU is an on-premise speed upgrade, not a cloud service.
Nothing leaves the machine.

HOW TO RUN ON KAGGLE (GPU T4 enabled):
  1. New notebook, Settings -> Accelerator -> GPU T4 x1, Internet ON (for the pip install only).
  2. Add two datasets:
     - your GGUF (upload checkpoints/gguf/qwen0.5b-Q4_K_M.gguf as a dataset)
     - your test set (upload data/synthetic/sme_test.jsonl as a dataset)
  3. Cell 1 (install the prebuilt CUDA wheel, no compile):
       !pip install -q llama-cpp-python \
         --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
  4. Cell 2: paste this whole file, fix GGUF_PATH and TEST_PATH to your dataset
     paths (left panel shows the exact /kaggle/input/... path), run.
  5. Read the final line: "GPU latency ... P50 X ms". Send me X and I build the
     CPU-vs-GPU comparison graphic.

To benchmark Llama instead of Qwen, set MODEL_FAMILY = "llama3".
"""
import time
import json
from llama_cpp import Llama

# EDIT THESE TWO to match your uploaded Kaggle datasets:
GGUF_PATH = "/kaggle/input/datasets/karanhomayounfar1/qwen0-5b-q4-k-m-latency-test/qwen0.5b-Q4_K_M.gguf"
TEST_PATH = "/kaggle/input/datasets/karanhomayounfar1/qwen0-5b-q4-k-m-latency-test/sme_test.jsonl"

MODEL_FAMILY = "qwen0.5b"   # or "llama3"
N_PREDICT = 80              # matches the CPU aligned harness exactly
WARMUP = 3


def format_prompt(record, family):
    """Identical to scripts/06_aligned_eval.py so latency is comparable."""
    if family in ("llama3", "llama1b"):
        p = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
             f"{record['instruction']}<|eot_id|>"
             "<|start_header_id|>user<|end_header_id|>\n\n"
             f"{record['input']}<|eot_id|>"
             "<|start_header_id|>assistant<|end_header_id|>\n\n")
        return p, "<|eot_id|>"
    if family in ("qwen0.5b", "qwen1.5b", "smol360"):
        p = (f"<|im_start|>system\n{record['instruction']}<|im_end|>\n"
             f"<|im_start|>user\n{record['input']}<|im_end|>\n"
             f"<|im_start|>assistant\n")
        return p, "<|im_end|>"
    # phi3
    p = (f"<|system|>\n{record['instruction']}<|end|>\n"
         f"<|user|>\n{record['input']}<|end|>\n<|assistant|>\n")
    return p, "<|end|>"


records = [json.loads(line) for line in open(TEST_PATH) if line.strip()]
print(f"Loaded {len(records)} test records. Model family: {MODEL_FAMILY}")

llm = Llama(model_path=GGUF_PATH, n_gpu_layers=-1, n_ctx=2048, verbose=False)

# warmup so the first timed call is not cold
for i in range(WARMUP):
    pr, stop = format_prompt(records[i % len(records)], MODEL_FAMILY)
    llm(pr, max_tokens=N_PREDICT, temperature=0, stop=[stop])

lat = []
for rec in records:
    pr, stop = format_prompt(rec, MODEL_FAMILY)
    t0 = time.perf_counter()
    llm(pr, max_tokens=N_PREDICT, temperature=0, stop=[stop])
    lat.append((time.perf_counter() - t0) * 1000)

lat.sort()
p50 = lat[len(lat) // 2]
p95 = lat[min(int(len(lat) * 0.95), len(lat) - 1)]
print(f"\n{MODEL_FAMILY} GPU latency over {len(lat)} calls (1 per record, {N_PREDICT} tokens max):")
print(f"  P50 {p50:.0f}ms   P95 {p95:.0f}ms   mean {sum(lat)/len(lat):.0f}ms")
print(f"\nCompare against CPU (aligned_summary.json): Qwen 2.5 0.5B Q4_K_M P50 814ms.")
