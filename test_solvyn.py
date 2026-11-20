
import gc
import tempfile
import time
import unittest
from pathlib import Path

import database

from evaluator import (
    normalize_text,
    exact_match,
    token_f1,
    keyword_coverage,
    semantic_similarity,
    evaluate_answer,
)


# ==============================================
# SOLVYN - AUTOMATED TEST SUITE
# ==============================================


class TestEvaluationMetrics(unittest.TestCase):
    """Test SOLVYN evaluation metrics."""

    def test_normalization(self):
        self.assertEqual(
            normalize_text("  HELLO,   World!  "),
            "hello world",
        )

    def test_exact_match(self):
        self.assertEqual(
            exact_match("Paris", "PARIS."),
            1.0,
        )

        self.assertEqual(
            exact_match("Paris", "London"),
            0.0,
        )

    def test_token_f1(self):
        self.assertEqual(
            token_f1("red blue", "red blue"),
            1.0,
        )

        self.assertEqual(
            token_f1("red blue", "green yellow"),
            0.0,
        )

        self.assertAlmostEqual(
            token_f1("red blue", "red green"),
            0.5,
        )

    def test_keyword_coverage(self):
        self.assertEqual(
            keyword_coverage(
                "Python programming language",
                "Python is a programming language",
            ),
            1.0,
        )

    def test_semantic_similarity(self):
        score = semantic_similarity(
            "A car is moving quickly",
            "An automobile is traveling fast",
        )

        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_complete_evaluation(self):
        results = evaluate_answer(
            "Paris",
            "Paris.",
        )

        self.assertEqual(
            results["exact_match"],
            1.0,
        )

        self.assertEqual(
            results["token_f1"],
            1.0,
        )

        self.assertIn(
            "keyword_coverage",
            results,
        )

        self.assertIn(
            "semantic_similarity",
            results,
        )


# ==============================================
# DATABASE TESTS
# ==============================================


class TestDatabase(unittest.TestCase):
    """Test SQLite using an isolated temporary database."""

    def setUp(self):
        self.original_path = database.DB_PATH

        self.temp_dir = tempfile.TemporaryDirectory()

        database.DB_PATH = (
            Path(self.temp_dir.name) / "test_solvyn.db"
        )

        database.initialize_database()

    def tearDown(self):
        database.DB_PATH = self.original_path

        gc.collect()

        # Retry temporary database cleanup on Windows.
        for attempt in range(10):
            try:
                self.temp_dir.cleanup()
                return

            except PermissionError:
                if attempt == 9:
                    raise

                gc.collect()
                time.sleep(0.2)

    @staticmethod
    def sample_results():
        return [
            {
                "question": "Capital of France?",
                "expected": "Paris",
                "actual": "Paris",
                "exact_match": 1.0,
                "token_f1": 1.0,
                "keyword_coverage": 1.0,
                "semantic_similarity": 1.0,
                "latency": 0.5,
                "tokens": 30,
            }
        ]

    def test_database_initialization(self):
        # Verify the database exists and its API works.
        self.assertTrue(database.DB_PATH.exists())

        history = database.get_evaluation_history()

        self.assertEqual(history, [])

    def test_save_and_load_run(self):
        run_id = database.save_evaluation_run(
            "test-model",
            self.sample_results(),
        )

        self.assertIsInstance(run_id, int)

        loaded = database.get_evaluation_run(run_id)

        self.assertIsNotNone(loaded)

        self.assertEqual(
            loaded["model"],
            "test-model",
        )

        self.assertEqual(
            loaded["results"][0]["actual"],
            "Paris",
        )

    def test_evaluation_history(self):
        database.save_evaluation_run(
            "test-model",
            self.sample_results(),
        )

        history = database.get_evaluation_history()

        self.assertEqual(len(history), 1)

        self.assertEqual(
            history[0]["test_cases"],
            1,
        )

        self.assertEqual(
            history[0]["exact_match"],
            1.0,
        )

        self.assertEqual(
            history[0]["semantic_similarity"],
            1.0,
        )

    def test_missing_run(self):
        result = database.get_evaluation_run(999999)

        self.assertIsNone(result)


# ==============================================
# RUN TESTS
# ==============================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
