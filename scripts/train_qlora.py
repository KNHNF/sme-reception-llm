"""
SME Voice Assistant - QLoRA Fine-tuning Script
Model: microsoft/Phi-3-mini-4k-instruct (3.8B)
Method: QLoRA (4-bit quantisation + LoRA adapters via PEFT)
Trainer: trl.SFTTrainer

Requirements:
    pip install transformers peft trl bitsandbytes accelerate datasets

Run on Kaggle / Colab A100:
    python train_qlora.py

Run on Kaggle / Colab T4 (16GB VRAM -- reduce batch size):
    python train_qlora.py --per_device_train_batch_size 1 --grad_accum 8

Outputs:
    checkpoints/sme-phi3-qlora/          LoRA adapter weights
    checkpoints/sme-phi3-qlora-merged/   Full merged model (optional, for inference)
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

# Args

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id",        default="microsoft/Phi-3-mini-4k-instruct")
    p.add_argument("--train_file",      default="data/synthetic/sme_train.jsonl")
    p.add_argument("--val_file",        default="data/synthetic/sme_val.jsonl")
    p.add_argument("--output_dir",      default="checkpoints/sme-phi3-qlora")
    p.add_argument("--epochs",          type=int,   default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--grad_accum",      type=int,   default=4)
    p.add_argument("--lr",              type=float, default=2e-4)
    p.add_argument("--max_seq_len",     type=int,   default=512)
    p.add_argument("--lora_r",          type=int,   default=16)
    p.add_argument("--lora_alpha",      type=int,   default=32)
    p.add_argument("--lora_dropout",    type=float, default=0.05)
    p.add_argument("--merge_and_save",  action="store_true",
                   help="Merge LoRA weights into base model after training")
    return p.parse_args()

# Quantisation config

def get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NormalFloat4 -- better than fp4 for LLMs
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,     # nested quantisation, saves ~0.4 GB
    )

# LoRA config
# Target modules for Phi-3 mini attention layers.
# These are the projection matrices QLoRA paper recommends targeting.

PHI3_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

def get_lora_config(args) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=PHI3_TARGET_MODULES,
        bias="none",
        inference_mode=False,
    )

# Dataset formatting
# Phi-3 uses a specific chat template. We format each record into it so the
# model learns to produce output in the same format used at inference time.
#
# Format:
#   <|system|>\n{instruction}<|end|>\n<|user|>\n{input}<|end|>\n<|assistant|>\n{output}<|end|>

def format_record(record: dict) -> str:
    return (
        f"<|system|>\n{record['instruction']}<|end|>\n"
        f"<|user|>\n{record['input']}<|end|>\n"
        f"<|assistant|>\n{record['output']}<|end|>"
    )

def load_and_format(path: str):
    ds = load_dataset("json", data_files=path, split="train")
    ds = ds.map(lambda x: {"text": format_record(x)})
    return ds

# Main

def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_id}")
    bnb_config = get_bnb_config()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        padding_side="right",   # required for SFT loss masking
    )
    # Phi-3 does not always set a pad token -- add one if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # flash_attention_2 requires Ampere+ GPU (A100, A10, RTX 30xx+).
    # T4 (Kaggle free tier) is Turing architecture and will crash with it.
    # Detect capability and fall back to eager attention on older GPUs.
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    ampere_plus = any(x in gpu_name for x in ["A100", "A10", "A30", "A40", "RTX 30", "RTX 40", "H100"])
    attn_impl = "flash_attention_2" if ampere_plus else "eager"
    print(f"GPU: {gpu_name or 'CPU'} -- attention: {attn_impl}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    model.config.use_cache = False          # required during training
    model.config.pretraining_tp = 1

    # Prepare model for QLoRA (freezes base weights, casts layer norms to fp32)
    model = prepare_model_for_kbit_training(model)

    lora_config = get_lora_config(args)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Expected output: ~1-2% of parameters trainable

    # Datasets
    print("Loading datasets...")
    train_ds = load_and_format(args.train_file)
    val_ds   = load_and_format(args.val_file)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # DataCollator masks the prompt tokens -- model only gets loss on the
    # assistant output (the JSON). This is critical: without this the model
    # would try to learn to reproduce the system prompt too.
    response_template = "<|assistant|>\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # Training args
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,        # trades compute for memory
        optim="paged_adamw_32bit",          # QLoRA paper recommendation
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
        tf32=True,
        max_grad_norm=0.3,                  # QLoRA paper recommendation
        report_to="none",                   # set to "wandb" if you use it
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

    print(f"Saving adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training metadata for evaluation traceability
    meta = {
        "model_id":        args.model_id,
        "lora_r":          args.lora_r,
        "lora_alpha":      args.lora_alpha,
        "lora_dropout":    args.lora_dropout,
        "target_modules":  PHI3_TARGET_MODULES,
        "epochs":          args.epochs,
        "lr":              args.lr,
        "train_samples":   len(train_ds),
        "val_samples":     len(val_ds),
        "max_seq_len":     args.max_seq_len,
    }
    with open(f"{args.output_dir}/training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved training_meta.json")

    # Optional: merge adapters into base model
    # Useful for deployment. Produces a full fp16 model (~7GB).
    # For the IGP demo you probably want to keep them separate (smaller).
    if args.merge_and_save:
        print("Merging LoRA weights into base model...")
        from peft import PeftModel
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        merged = PeftModel.from_pretrained(base_model, args.output_dir)
        merged = merged.merge_and_unload()
        merged_path = args.output_dir + "-merged"
        merged.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        print(f"Merged model saved to {merged_path}")

if __name__ == "__main__":
    main()
