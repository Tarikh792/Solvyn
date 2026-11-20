
import json
import sqlite3

from datetime import datetime, timezone
from pathlib import Path


# ==========================================
# SOLVYN - SQLITE DATABASE
# ==========================================

DB_PATH = Path(__file__).parent / "solvyn.db"


# 1. Create database and table
def initialize_database():
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                created_at TEXT NOT NULL,
                results_json TEXT NOT NULL
            )
            """
        )


# 2. Save a completed evaluation
def save_evaluation_run(model: str, results: list) -> int:
    initialize_database()

    created_at = datetime.now(timezone.utc).isoformat()
    results_json = json.dumps(results, ensure_ascii=False)

    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute(
            """
            INSERT INTO evaluation_runs
                (model, created_at, results_json)
            VALUES (?, ?, ?)
            """,
            (model, created_at, results_json),
        )

        return cursor.lastrowid


# 3. Retrieve evaluation history
def get_evaluation_history() -> list:
    initialize_database()

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT id, model, created_at, results_json
            FROM evaluation_runs
            ORDER BY id DESC
            """
        ).fetchall()

    history = []

    for row in rows:
        results = json.loads(row["results_json"])

        count = len(results)

        avg_exact = (
            sum(item["exact_match"] for item in results) / count
            if count else 0.0
        )

        avg_f1 = (
            sum(item["token_f1"] for item in results) / count
            if count else 0.0
        )

        avg_semantic = (
            sum(
                item["semantic_similarity"]
                for item in results
            ) / count
            if count else 0.0
        )

        history.append(
            {
                "id": row["id"],
                "model": row["model"],
                "created_at": row["created_at"],
                "test_cases": count,
                "exact_match": round(avg_exact, 4),
                "token_f1": round(avg_f1, 4),
                "semantic_similarity": round(
                    avg_semantic, 4
                ),
            }
        )

    return history


# 4. Retrieve one evaluation run
def get_evaluation_run(run_id: int) -> dict | None:
    initialize_database()

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        row = connection.execute(
            """
            SELECT id, model, created_at, results_json
            FROM evaluation_runs
            WHERE id = ?
            """,
            (run_id,),
        ).fetchone()

    if row is None:
        return None

    return {
        "id": row["id"],
        "model": row["model"],
        "created_at": row["created_at"],
        "results": json.loads(row["results_json"]),
    }


# 5. Test database functionality
if __name__ == "__main__":

    initialize_database()

    print("\nSOLVYN - DATABASE TEST")
    print("-" * 40)

    sample_results = [
        {
            "question": "What is the capital of France?",
            "expected": "Paris",
            "actual": "Paris",
            "exact_match": 1.0,
            "token_f1": 1.0,
            "keyword_coverage": 1.0,
            "semantic_similarity": 1.0,
            "latency": 0.5,
            "tokens": 50,
        }
    ]

    run_id = save_evaluation_run(
        model="test-model",
        results=sample_results,
    )

    print(f"Saved evaluation run: #{run_id}")

    history = get_evaluation_history()

    print(f"Total saved runs: {len(history)}")

    saved_run = get_evaluation_run(run_id)

    print(f"Retrieved model: {saved_run['model']}")
    print(f"Retrieved test cases: {len(saved_run['results'])}")

    print("-" * 40)
    print("Database working successfully!")
