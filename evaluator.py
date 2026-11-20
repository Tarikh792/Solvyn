
import re
from collections import Counter
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sentence_transformers import util


# ==============================================
# SOLVYN - AI EVALUATION ENGINE
# ==============================================


# 1. Text normalization
def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


# 2. Exact Match
def exact_match(expected: str, actual: str) -> float:
    expected = normalize_text(expected)
    actual = normalize_text(actual)

    return 1.0 if expected == actual else 0.0


# 3. Token F1
def token_f1(expected: str, actual: str) -> float:
    expected_tokens = normalize_text(expected).split()
    actual_tokens = normalize_text(actual).split()

    if not expected_tokens and not actual_tokens:
        return 1.0

    if not expected_tokens or not actual_tokens:
        return 0.0

    expected_counts = Counter(expected_tokens)
    actual_counts = Counter(actual_tokens)

    common = sum(
        (expected_counts & actual_counts).values()
    )

    if common == 0:
        return 0.0

    precision = common / len(actual_tokens)
    recall = common / len(expected_tokens)

    return 2 * precision * recall / (precision + recall)


# 4. Keyword Coverage
def keyword_coverage(expected: str, actual: str) -> float:
    expected_words = set(
        normalize_text(expected).split()
    )

    actual_words = set(
        normalize_text(actual).split()
    )

    if not expected_words:
        return 1.0 if not actual_words else 0.0

    common = expected_words.intersection(actual_words)

    return len(common) / len(expected_words)


# 5. Load embedding model once
@lru_cache(maxsize=1)
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# 6. Semantic Similarity
def semantic_similarity(expected: str, actual: str) -> float:
    if not expected.strip() and not actual.strip():
        return 1.0

    if not expected.strip() or not actual.strip():
        return 0.0

    model = load_embedding_model()

    embeddings = model.encode(
        [expected, actual],
        convert_to_tensor=True,
    )

    similarity = util.cos_sim(
        embeddings[0],
        embeddings[1],
    ).item()

    # Normalize to display range 0 to 1
    similarity = max(0.0, min(1.0, similarity))

    return round(similarity, 4)


# 7. Complete evaluation
def evaluate_answer(expected: str, actual: str) -> dict:
    return {
        "exact_match": round(
            exact_match(expected, actual), 4
        ),
        "token_f1": round(
            token_f1(expected, actual), 4
        ),
        "keyword_coverage": round(
            keyword_coverage(expected, actual), 4
        ),
        "semantic_similarity": semantic_similarity(
            expected, actual
        ),
    }


# 8. Standalone testing
if __name__ == "__main__":
    expected = "Python is a popular programming language"
    actual = "Python is a programming language"

    results = evaluate_answer(expected, actual)

    print("\nSOLVYN - AI EVALUATION ENGINE")
    print("-" * 40)

    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")

    print("-" * 40)
    print("Evaluation completed successfully!")
