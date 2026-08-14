#!/usr/bin/env python3
"""
train_lora_qwen.py — Fine-tune Qwen 2.5 Coder on Strudel live-coding datasets using QLoRA / PEFT.
"""

import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


SYSTEM_PROMPT = """You are Strudel AI, an expert live-coding music assistant and algorithmic composition master.
You generate concise, expressive, and syntactically valid Strudel (JavaScript) patterns using mini-notation, WebAudio synthesis, soundbanks, and 13-channel DAW MIDI mappings."""


def format_prompts(batch, tokenizer):
    texts = []
    for inst, inp, out in zip(batch["instruction"], batch["input"], batch["output"]):
        user_content = f"{inst}\n{inp}".strip() if inp else inst
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": out}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        texts.append(text)
    return {"text": texts}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen 2.5 Coder for Strudel.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Base model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to strudel_training_dataset.jsonl")
    parser.add_argument("--output-dir", type=str, default="./strudel-qwen-lora", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")

    args = parser.parse_args()

    print(f"🚀 Loading base model: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 4-bit Quantization Config for efficient memory usage
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        device_map=device_map,
        trust_remote_code=True,
    )

    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"📂 Loading dataset from {args.dataset}...")
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    dataset = dataset.map(lambda batch: format_prompts(batch, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        learning_rate=args.lr,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("🔥 Starting training...")
    trainer.train()

    print(f"💾 Saving LoRA adapter to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ Fine-tuning complete!")


if __name__ == "__main__":
    main()
