import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROUTER = ROOT / "scripts/route.py"


def run(*args):
    result = subprocess.run([sys.executable, str(ROUTER), *args], text=True,
                            capture_output=True, check=False)
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


class RouterTests(unittest.TestCase):
    def test_known_task_uses_frozen_vector_but_excludes_its_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            support = Path(directory) / "support.json"
            support.write_text(json.dumps({
                "schema": "skilldelta-support-v1", "embedding_model": "test",
                "tasks": [
                    {"id": "a", "text": "a", "embedding": [1, 0], "d": 1},
                    {"id": "b", "text": "b", "embedding": [0, 1], "d": -1},
                ],
            }))
            value = run("route", "--support", str(support), "--task-id", "a", "--k", "1")
        self.assertEqual(value["query_source"], "frozen-index")
        self.assertFalse(value["enable_skill"])
        self.assertEqual(value["predicted_gain"], -1.0)
        self.assertEqual(value["neighbors"][0]["id"], "b")


    def test_threshold_and_query_vector(self):
        with tempfile.TemporaryDirectory() as directory:
            support = Path(directory) / "support.json"
            support.write_text(json.dumps({
                "schema": "skilldelta-support-v1",
                "tasks": [
                    {"id": "a", "embedding": [1, 0], "d": 1},
                    {"id": "b", "embedding": [0, 1], "d": -1},
                ],
            }))
            value = run("route", "--support", str(support), "--query-embedding", "[1,0]",
                        "--k", "2", "--threshold", "0.1")
        self.assertEqual(value["query_source"], "argument")
        self.assertEqual(value["predicted_gain"], 0.0)
        self.assertFalse(value["enable_skill"])

    def test_query_only_index_avoids_outcome_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            support = Path(directory) / "support.json"
            query = Path(directory) / "query.json"
            support.write_text(json.dumps({
                "schema": "skilldelta-support-v1",
                "tasks": [{"id": "a", "embedding": [1, 0], "d": 1},
                           {"id": "b", "embedding": [0, 1], "d": -1}],
            }))
            query.write_text(json.dumps({"schema": "skilldelta-query-index-v1",
                                         "embeddings": {"heldout": [1, 0]}}))
            value = run("route", "--support", str(support), "--query-index", str(query),
                        "--task-id", "heldout", "--k", "1")
        self.assertEqual(value["query_source"], "frozen-query-index")
        self.assertEqual(value["neighbors"][0]["id"], "a")


    def test_status_hashes_support_and_skill(self):
        with tempfile.TemporaryDirectory() as directory:
            support = Path(directory) / "support.json"
            skill = Path(directory) / "skill.md"
            support.write_text(json.dumps({"schema": "skilldelta-support-v1",
                                           "tasks": [{"id": "a", "embedding": [1], "d": 0}]}))
            skill.write_text("skill")
            value = run("status", "--support", str(support), "--skill", str(skill))
        self.assertEqual(value["support"]["count"], 1)
        self.assertTrue(value["skill"]["sha256"])

    def test_equality_at_threshold_skips(self):
        value = run("route", "--support", str(ROOT / "examples/support.json"),
                    "--task-id", "demo-help", "--query-index", str(ROOT / "examples/queries.json"),
                    "--k", "2", "--threshold", "1")
        self.assertFalse(value["enable_skill"])

    def test_invalid_inputs_return_structured_errors(self):
        for extra in [("--k", "0"), ("--threshold", "nan"),
                      ("--query-embedding", "[0,0]"), ("--query-embedding", "[1]")]:
            with self.subTest(extra=extra):
                result = subprocess.run([sys.executable, str(ROUTER), "route", "--support",
                    str(ROOT / "examples/support.json"), "--query-embedding", "[1,0]", *extra],
                    text=True, capture_output=True)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stderr)["error"], "ValueError")

    def test_duplicate_ids_and_no_remaining_neighbors_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            support = Path(directory) / "support.json"
            row = {"id": "target", "embedding": [1, 0], "d": 1}
            for rows in [[row, row], [row]]:
                with self.subTest(count=len(rows)):
                    support.write_text(json.dumps({"tasks": rows}))
                    result = subprocess.run([sys.executable, str(ROUTER), "route", "--support",
                        str(support), "--task-id", "target"], text=True, capture_output=True)
                    self.assertEqual(result.returncode, 2)
                    self.assertEqual(json.loads(result.stderr)["error"], "ValueError")
