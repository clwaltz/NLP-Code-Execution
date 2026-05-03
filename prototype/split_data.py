import json
import random
from pathlib import Path

from data_loader import load_examples


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    seed = 42
    test_ratio = 0.15

    examples = load_examples("data")
    if not examples:
        raise ValueError("No examples found in data/")

    random.seed(seed)
    random.shuffle(examples)

    n_total = len(examples)
    n_test = int(n_total * test_ratio)

    test_rows = examples[:n_test]
    train_rows = examples[n_test:]

    out_dir = Path("data")
    train_path = out_dir / "train.jsonl"
    test_path = out_dir / "test.jsonl"

    write_jsonl(train_path, train_rows)
    write_jsonl(test_path, test_rows)

    print(f"Total: {n_total}")
    print(f"Train: {len(train_rows)} -> {train_path}")
    print(f"Test:  {len(test_rows)} -> {test_path}")
    print(f"Seed: {seed}, test_ratio: {test_ratio}")


if __name__ == "__main__":
    main()