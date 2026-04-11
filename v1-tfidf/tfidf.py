import argparse
import pickle

def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)

def predict(model, query):
    vectorizer = model["vectorizer"]
    nn = model["nn"]
    questions = model["questions"]
    answers = model["answers"]

    query_vec = vectorizer.transform([query])
    distances, indices = nn.kneighbors(query_vec, n_neighbors=1)

    idx = indices[0][0]
    distance = float(distances[0][0])
    similarity = 1.0 - distance

    return {
        "query": query,
        "matched_question": questions[idx],
        "predicted_code": answers[idx],
        "similarity": similarity,
    }

def main():
    parser = argparse.ArgumentParser(description="NL -> Python code prototype")
    parser.add_argument(
        "--model",
        default="models/prototype.pkl",
        help="Path to trained model artifact",
    )
    parser.add_argument("--query", default=None, help="Single query to run")
    args = parser.parse_args()

    model = load_model(args.model)

    if args.query:
        result = predict(model, args.query)
        print("\n=== Result ===")
        print("Query:", result["query"])
        print("Matched question:", result["matched_question"])
        print(f"Similarity: {result['similarity']:.4f}")
        print("\nPredicted code:\n")
        print(result["predicted_code"])
        return

    print("Interactive mode. Type 'exit' to quit.\n")
    while True:
        query = input("Question> ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        
        result = predict(model, query)
        print("\nMatched question:", result["matched_question"])
        print(f"Similarity: {result['similarity']:.4f}")
        print("\nPredicted code:\n")
        print(result["predicted_code"])
        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    main()
