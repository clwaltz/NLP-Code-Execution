#!/usr/bin/env python3
import json
import random
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PANDAS_PATH = BASE_DIR / "pandas.jsonl"
STACKOVERFLOW_PATH = BASE_DIR / "stackoverflow.jsonl"

MERGED_PATH = BASE_DIR / "all.jsonl"
TRAIN_PATH = BASE_DIR / "train.jsonl"
VAL_PATH = BASE_DIR / "val.jsonl"
TEST_PATH = BASE_DIR / "test.jsonl"

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
    return rows


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def add_source(rows, source_name: str):
    tagged = []
    for item in rows:
        q = item.get("q", "").strip()
        a = item.get("a", "").strip()
        if not q or not a:
            continue
        tagged.append({"q": q, "a": a, "source": source_name})
    return tagged


def split_rows(rows):
    n = len(rows)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val
    return rows[:n_train], rows[n_train : n_train + n_val], rows[n_train + n_val : n_train + n_val + n_test]


def main():
    random.seed(SEED)

    pandas_rows = add_source(read_jsonl(PANDAS_PATH), "pandas_docs")
    so_rows = add_source(read_jsonl(STACKOVERFLOW_PATH), "stackoverflow")

    random.shuffle(pandas_rows)
    random.shuffle(so_rows)

    p_train, p_val, p_test = split_rows(pandas_rows)
    s_train, s_val, s_test = split_rows(so_rows)

    train_rows = p_train + s_train
    val_rows = p_val + s_val
    test_rows = p_test + s_test
    all_rows = train_rows + val_rows + test_rows

    random.shuffle(train_rows)
    random.shuffle(val_rows)
    random.shuffle(test_rows)
    random.shuffle(all_rows)

    write_jsonl(MERGED_PATH, all_rows)
    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(VAL_PATH, val_rows)
    write_jsonl(TEST_PATH, test_rows)

    print(f"pandas rows: {len(pandas_rows)}")
    print(f"stackoverflow rows: {len(so_rows)}")
    print(f"all rows: {len(all_rows)}")
    print(f"train rows: {len(train_rows)}")
    print(f"val rows: {len(val_rows)}")
    print(f"test rows: {len(test_rows)}")


if __name__ == "__main__":
    main()
