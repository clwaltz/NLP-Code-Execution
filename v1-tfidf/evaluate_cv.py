import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

from data_loader import load_examples

def normalize(s):
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def token_f1(pred, gold):
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    gold_set = set(gold_tokens)
    overlap = len(pred_set & gold_set)

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_set)
    recall = overlap / len(gold_set)

    return 2 * precision * recall / (precision + recall)

def run_cv(n_splits, random_state: 42):
    examples = load_examples("data")
    questions = np.array([e["q"] for e in examples])
    answers = np.array([e["a"] for e in examples])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(questions), start=1):
        train_q = questions[train_idx].tolist()
        train_a = answers[train_idx].tolist()
        test_q = questions[test_idx].tolist()
        test_a = answers[test_idx].tolist()
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
        x_train = vectorizer.fit_transform(train_q)
        nn = NearestNeighbors(metric="cosine", n_neighbors=1)
        nn.fit(x_train)
        exact_matches = []
        f1_scores = []
        for q_gold, a_gold in zip(test_q, test_a):
            q_vec = vectorizer.transform([q_gold])
            distances, indices = nn.kneighbors(q_vec, n_neighbors=1)
            pred_idx = indices[0][0]
            a_pred = train_a[pred_idx]
            em = float(normalize(a_pred) == normalize(a_gold))
            f1 = token_f1(a_pred, a_gold)
            exact_matches.append(em)
            f1_scores.append(f1)
        fold_result = {
            "fold": fold_idx,
            "n_test": len(test_q),
            "exact_match": float(np.mean(exact_matches)),
            "token_f1": float(np.mean(f1_scores)),
        }
        fold_results.append(fold_result)
        print(
            f"Fold {fold_idx}: "
            f"EM={fold_result['exact_match']:.4f}, "
            f"F1={fold_result['token_f1']:.4f}, "
            f"n_test={fold_result['n_test']}"
        )
    summary = {
        "n_splits": n_splits,
        "n_examples": len(examples),
        "folds": fold_results,
        "mean_exact_match": float(np.mean([f["exact_match"] for f in fold_results])),
        "std_exact_match": float(np.std([f["exact_match"] for f in fold_results])),
        "mean_token_f1": float(np.mean([f["token_f1"] for f in fold_results])),
        "std_token_f1": float(np.std([f["token_f1"] for f in fold_results])),
    }
    print("\n=== Summary ===")
    print(f"Splits: {summary['n_splits']}")
    print(f"Examples: {summary['n_examples']}")
    print(f"Mean EM: {summary['mean_exact_match']:.4f} +/- {summary['std_exact_match']:.4f}")
    print(f"Mean F1: {summary['mean_token_f1']:.4f} +/- {summary['std_token_f1']:.4f}")
    return summary
if __name__ == "__main__":
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    summary_5 = run_cv(n_splits=5, random_state=42)
    (reports_dir / "cv_5fold.json").write_text(json.dumps(summary_5, indent=2), encoding="utf-8")
    summary_10 = run_cv(n_splits=10, random_state=42)
    (reports_dir / "cv_10fold.json").write_text(json.dumps(summary_10, indent=2), encoding="utf-8")
    print("\nSaved:")
    print(" - reports/cv_5fold.json")
    print(" - reports/cv_10fold.json")
