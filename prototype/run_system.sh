#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE="${1:-test.jsonl}"
OUTPUT_FILE="${2:-predictions.jsonl}"
MODEL_PATH="seq2seq_prototype.pt"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .

mkdir -p outputs
python -u interact.py --model "$MODEL_PATH" --input "$INPUT_FILE" --output "$OUTPUT_FILE"
echo "Predictions: $OUTPUT_FILE"