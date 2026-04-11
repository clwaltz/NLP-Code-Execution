import pickle
from datetime import datetime, timezone
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from data_loader import load_examples

def train_model(data_dir, model_path):
    examples = load_examples(data_dir)

    questions = [ex["q"] for ex in examples]
    answers = [ex["a"] for ex in examples]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1
    )
    x = vectorizer.fit_transform(questions)

    nn = NearestNeighbors(metric="cosine", n_neighbors=1)
    nn.fit(x)

    artifact = {
        "vectorizer": vectorizer,
        "nn": nn,
        "questions": questions,
        "answers": answers,
        "train_size": len(examples),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)

    with model_file.open("wb") as f:
        pickle.dump(artifact, f)

    print(f"Trained on {len(examples)} examples")
    print(f"Saved model to {model_file}")

if __name__ == "__main__":
    data_dir = "data"
    model_path = "models/prototype.pkl"
    train_model(data_dir, model_path)