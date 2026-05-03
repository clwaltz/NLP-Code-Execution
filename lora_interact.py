import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a LoRA adapter.")
    parser.add_argument(
        "--adapter_dir",
        type=str,
        required=True,
        help="Path to adapter directory (contains adapter_model.safetensors + adapter_config.json).",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="",
        help="Optional base model override. If omitted, use base_model_name_or_path from adapter_config.json.",
    )
    parser.add_argument("--input", type=str, default="", help="Optional JSONL input with {'q': ...} rows.")
    parser.add_argument("--output", type=str, default="lora_predictions.jsonl", help="Output JSONL path.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    return parser.parse_args()


def resolve_base_model(adapter_dir: Path, base_override: str) -> str:
    if base_override:
        return base_override

    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing adapter config: {cfg_path}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    base_model = cfg.get("base_model_name_or_path", "")
    if not base_model:
        raise ValueError("Could not determine base model from adapter_config.json; pass --base_model.")
    return base_model


def build_prompt(question: str) -> str:
    return f"Question:\n{question}\n\nCode:\n"


def load_model_and_tokenizer(adapter_dir: Path, base_model: str):
    device_map = "auto" if torch.cuda.is_available() else None
    base = AutoModelForCausalLM.from_pretrained(base_model, device_map=device_map)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, question: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    prompt = build_prompt(question)
    encoded = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

    out_ids = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    full = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    pred = full[len(prompt) :] if full.startswith(prompt) else full
    return pred.strip()


def run_batch(model, tokenizer, args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = str(row.get("q", "")).strip()
            if not q:
                continue

            pred = generate(
                model,
                tokenizer,
                q,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            fout.write(json.dumps({"q": q, "pred": pred}, ensure_ascii=False) + "\n")

    print(f"Saved predictions to: {output_path}")


def run_interactive(model, tokenizer, args):
    while True:
        prompt = input("Enter question (q to exit): ").strip()
        if prompt.lower() == "q":
            break
        pred = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"\nGENERATED CODE:\n{pred}\n")


def main():
    args = parse_args()
    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    base_model = resolve_base_model(adapter_dir, args.base_model)
    print(f"Loading base model: {base_model}")
    print(f"Loading adapter from: {adapter_dir}")

    model, tokenizer = load_model_and_tokenizer(adapter_dir, base_model)

    if args.input:
        run_batch(model, tokenizer, args)
    else:
        run_interactive(model, tokenizer, args)


if __name__ == "__main__":
    main()
