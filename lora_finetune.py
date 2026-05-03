import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for q->code generation.")
    parser.add_argument("--data_dir", type=str, default="data_collection_2", help="Folder with train/val/test.jsonl")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="Base HF model for LoRA fine-tuning",
    )
    parser.add_argument("--output_dir", type=str, default="data_collection_2/lora_runs", help="Run output root")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
        help="Comma-separated module names for LoRA injection",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantized base model")
    parser.add_argument("--save_samples", type=int, default=8, help="Number of test generations to save")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = str(row.get("q", "")).strip()
            a = str(row.get("a", "")).strip()
            if not q or not a:
                continue
            rows.append({"q": q, "a": a})
    if not rows:
        raise ValueError(f"No usable rows found in {path}")
    return rows


def format_train_text(example: Dict[str, str]) -> str:
    return f"Question:\n{example['q']}\n\nCode:\n{example['a']}"


def format_prompt(question: str) -> str:
    return f"Question:\n{question}\n\nCode:\n"


def build_dataset(rows: List[Dict[str, str]], tokenizer, max_length: int) -> Dataset:
    texts = [format_train_text(x) for x in rows]
    ds = Dataset.from_dict({"text": texts})

    def tok_fn(batch):
        tokens = tokenizer(batch["text"], truncation=True, max_length=max_length)
        return tokens

    return ds.map(tok_fn, batched=True, remove_columns=["text"])


def build_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = None
    model_kwargs = {}

    if args.load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quant_cfg
        model_kwargs["device_map"] = "auto"

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
    except Exception as e:
        if args.load_in_4bit:
            print(f"4-bit load failed, retrying without quantization: {e}", flush=True)
            model = AutoModelForCausalLM.from_pretrained(args.model_name)
            args.load_in_4bit = False
        else:
            raise

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


@torch.no_grad()
def sample_generations(model, tokenizer, test_rows: List[Dict[str, str]], max_new_tokens: int = 128, limit: int = 8):
    model.eval()
    out = []
    device = model.device
    for row in test_rows[:limit]:
        prompt = format_prompt(row["q"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        full = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        pred = full[len(prompt) :] if full.startswith(prompt) else full
        out.append({"q": row["q"], "target": row["a"], "pred": pred.strip()})
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    print(f"starting LoRA run with model={args.model_name}", flush=True)
    print(
        f"torch={torch.__version__} cuda_build={torch.version.cuda} cuda_available={torch.cuda.is_available()}",
        flush=True,
    )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this environment. Aborting to avoid very slow CPU training."
        )
    print(f"cuda_device_count={torch.cuda.device_count()}", flush=True)

    data_dir = Path(args.data_dir)
    train_rows = read_jsonl(data_dir / "train.jsonl")
    val_rows = read_jsonl(data_dir / "val.jsonl")
    test_rows = read_jsonl(data_dir / "test.jsonl")
    print(f"splits: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}", flush=True)

    run_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = Path(args.output_dir) / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_id={run_id} output_dir={run_dir}", flush=True)

    print("loading tokenizer/model...", flush=True)
    model, tokenizer = build_model_and_tokenizer(args)
    print("model/tokenizer loaded", flush=True)

    print("tokenizing datasets...", flush=True)
    train_ds = build_dataset(train_rows, tokenizer, args.max_length)
    val_ds = build_dataset(val_rows, tokenizer, args.max_length)
    test_ds = build_dataset(test_rows, tokenizer, args.max_length)
    print(
        f"tokenized dataset sizes: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}",
        flush=True,
    )

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=20,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        report_to="none",
        seed=args.seed,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("starting training...", flush=True)
    trainer.train()
    print("training complete; running eval...", flush=True)
    val_metrics = trainer.evaluate(eval_dataset=val_ds)
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")

    adapter_dir = run_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    samples = sample_generations(
        trainer.model,
        tokenizer,
        test_rows,
        max_new_tokens=128,
        limit=max(0, args.save_samples),
    )
    with (run_dir / "samples.json").open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    report = {
        "run_id": str(run_id),
        "base_model": args.model_name,
        "data_dir": str(data_dir),
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "test_size": len(test_rows),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "adapter_dir": str(adapter_dir),
        "samples_file": str(run_dir / "samples.json"),
        "config": vars(args),
    }
    with (run_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"saved adapter: {adapter_dir}", flush=True)
    print(f"saved report: {run_dir / 'report.json'}", flush=True)
    print(f"saved samples: {run_dir / 'samples.json'}", flush=True)


if __name__ == "__main__":
    main()
