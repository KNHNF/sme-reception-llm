"""
SME Voice Assistant - QLoRA Fine-tuning Script for Llama 3.2 3B
Model: meta-llama/Llama-3.2-3B-Instruct
Method: QLoRA (4-bit NF4 + LoRA adapters via PEFT)

IMPORTANT: Llama 3 is a gated model on HuggingFace.
Before running you must:
  1. Go to https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct and accept the licence.
  2. Generate a HF token at https://huggingface.co/settings/tokens
  3. Run: huggingface-cli login  (paste your token)
  OR set environment variable: export HF_TOKEN=your_token_here

On Kaggle:
  Add a secret named HF_TOKEN in Settings > Secrets (not environment variables).
  Then in the notebook: import os; os.environ["HF_TOKEN"] = ...

Key differences from Phi-3 training:
  - Chat template uses Llama 3 format (<|begin_of_text|> etc.)
  - Response template for loss masking is different
  - tf32 and bf16 behave slightly differently on T4

Run:
  python train_qlora_llama.py --train_file data/synthetic/sme_train.jsonl
  python train_qlora_llama.py --per_device_train_batch_size 1 --grad_accum 16  (T4)
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id",   default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--train_file", default="data/synthetic/sme_train.jsonl")
    p.add_argument("--val_file",   default="data/synthetic/sme_val.jsonl")
    p.add_argument("--output_dir", default="checkpoints/sme-llama3-qlora")
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int,   default=4)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--max_seq_len",type=int,   default=512)
    p.add_argument("--lora_r",     type=int,   default=16)
    p.add_argument("--lora_alpha", type=int,   default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--merge_and_save", action="store_true")
    return p.parse_args()


def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


# Llama 3 attention projection layers -- same names as Phi-3
LLAMA3_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def get_lora_config(args):
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LLAMA3_TARGET_MODULES,
        bias="none",
        inference_mode=False,
    )


def format_record_llama3(record: dict) -> str:
    """
    Llama 3 Instruct chat template.

    Format:
      <|begin_of_text|>
      <|start_header_id|>system<|end_header_id|>
      {instruction}<|eot_id|>
      <|start_header_id|>user<|end_header_id|>
      {input}<|eot_id|>
      <|start_header_id|>assistant<|end_header_id|>
      {output}<|eot_id|>

    The DataCollator will mask everything up to and including
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    so the model only gets loss on the JSON output.
    """
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{record['instruction']}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{record['input']}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{record['output']}<|eot_id|>"
    )


def load_and_format(path: str):
    ds = load_dataset("json", data_files=path, split="train")
    ds = ds.map(lambda x: {"text": format_record_llama3(x)})
    return ds


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not set. This will fail if the model is gated.")

    print(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        token=hf_token,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    ampere_plus = any(x in gpu_name for x in ["A100", "A10", "A30", "A40", "RTX 30", "RTX 40", "H100"])
    attn_impl = "flash_attention_2" if ampere_plus else "eager"
    print(f"GPU: {gpu_name or 'CPU'} -- attention: {attn_impl}")

    bnb_config = get_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=hf_token,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, get_lora_config(args))
    model.print_trainable_parameters()

    print("Loading datasets...")
    train_ds = load_and_format(args.train_file)
    val_ds   = load_and_format(args.val_file)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # The response template must match exactly what appears in format_record_llama3
    # after the final <|start_header_id|>assistant<|end_header_id|> token.
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,
        tf32=False,
        max_grad_norm=0.3,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print("Starting training...")
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    meta = {
        "model_id":       args.model_id,
        "lora_r":         args.lora_r,
        "lora_alpha":     args.lora_alpha,
        "lora_dropout":   args.lora_dropout,
        "target_modules": LLAMA3_TARGET_MODULES,
        "epochs":         args.epochs,
        "lr":             args.lr,
        "train_samples":  len(train_ds),
        "val_samples":    len(val_ds),
        "chat_template":  "llama3",
    }
    with open(f"{args.output_dir}/training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Adapter saved to {args.output_dir}")

    if args.merge_and_save:
        print("Merging LoRA into base model...")
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            args.model_id, token=hf_token,
            torch_dtype=torch.bfloat16, device_map="auto",
        )
        merged = PeftModel.from_pretrained(base, args.output_dir).merge_and_unload()
        merged_path = args.output_dir + "-merged"
        merged.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        print(f"Merged model saved to {merged_path}")


if __name__ == "__main__":
    main()
