import argparse
import json
from pathlib import Path

import torch

from seq2seq import build_model, generate_code


def load_inference_objects(model_path, device):
    ckpt = torch.load(model_path, map_location=device)

    stoi = ckpt["stoi"]
    itos = ckpt["itos"]

    model = build_model(
        vocab_size=len(itos),
        emb_dim=ckpt["emb_dim"],
        hid_dim=ckpt["hid_dim"],
        device=device,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, stoi, itos


def run_batch(model, stoi, itos, device, input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
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

            pred = generate_code(model, q, stoi, itos, device)
            fout.write(json.dumps({"q": q, "pred": pred}, ensure_ascii=False) + "\n")


def run_interactive(model, stoi, itos, device):
    while True:
        prompt = input("Enter question/prompt (q to exit): ").strip()
        if prompt.lower() == "q":
            break

        pred = generate_code(model, prompt, stoi, itos, device)
        print(f"\nGENERATED CODE:\n{pred}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run seq2seq inference.")
    parser.add_argument("--model", default="seq2seq_prototype.pt", help="Path to model checkpoint.")
    parser.add_argument("--input", default="", help="Optional JSONL input file with {'q': ...} rows.")
    parser.add_argument("--output", default="predictions.jsonl", help="Output JSONL for batch mode.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, stoi, itos = load_inference_objects(args.model, device)

    if args.input:
        run_batch(model, stoi, itos, device, args.input, args.output)
        print(f"Saved predictions to: {args.output}")
    else:
        run_interactive(model, stoi, itos, device)


if __name__ == "__main__":
    main()