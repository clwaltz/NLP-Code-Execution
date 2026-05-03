# NLP Code Generation Prototype (Seq2Seq)

## Submission Contents
This submission zip includes:
- `seq2seq_prototype.pt` (pretrained model checkpoint)
- `seq2seq.py` (training and cross-validation)
- `interact.py` (inference script)
- `run_system.sh` (single-command runner)
- `train.jsonl` (training data)
- `test.jsonl` (test data)
- `seq2seq_cv_10fold.json` (cross-validation results)
- `pyproject.toml` (Python dependencies)

## Requirements
- Python 3.12

## To Run:

```bash
./run_system.sh test.jsonl predictions.jsonl
```

This command loads the pretrained model and writes predictions to `predictions.jsonl`.

## Input and Output Format
- Input file: JSONL, one object per line
- Required input field: `q`

Example input row:

```json
{"q": "sort dataframe by population column"}
```

Example output row:

```json
{"q": "sort dataframe by population column", "pred": "df . sort_values ( by = ... )"}
```

## How to Interpret Output
- `pred` is the generated code-like response from the seq2seq model for input prompt `q`.
- Output quality is baseline/prototype quality and may include formatting artifacts due to limited training data.

## Model Evaluation Included
- `reports/seq2seq_cv_10fold.json`

This file contains fold-level validation losses and aggregate statistics (`mean_val_loss`, `std_val_loss`).

Lower validation loss indicates better held-out performance.
