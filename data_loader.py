import json
from pathlib import Path

def load_examples(data_dir):
    examples = []

    for path in Path(data_dir).glob("*.jsonl"):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                for row in obj:
                    q = str(row.get("q", "")).strip()
                    a = str(row.get("a", "")).strip()
                    if q and a:
                        examples.append({"q": q, "a": a})
                continue
        except json.JSONDecodeError:
            pass

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = str(row.get("q", "")).strip()
            a = str(row.get("a", "")).strip()
            if q and a:
                examples.append({"q": q, "a": a})

    return examples

if __name__ == "__main__":
    data = load_examples("data")
    print(f"Loaded {len(data)} examples.")
    if data:
        print("Sample question:", data[0]["q"][:120])
        print("Sample answer:", data[0]["a"][:120])