"""The public CLI evaluates fresh local inputs without archived paper outputs."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class LocalEvaluationTests(unittest.TestCase):
    def test_encoder_receives_only_questions_and_keeps_id_order(self):
        spec = importlib.util.spec_from_file_location("encode_tasks", ROOT / "scripts/encode_tasks.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        calls = {}

        class LocalEncoder:
            def __init__(self, path, **kwargs):
                calls["constructor"] = kwargs

            def encode(self, texts, **kwargs):
                calls["texts"] = texts
                return np.asarray([[1., 0.], [0., 1.]])

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "model").mkdir()
            (root / "questions.json").write_text(json.dumps([
                {"id": "toy-b", "question": "A synthetic query", "y0": 1},
                {"id": "toy-a", "question": "Another synthetic query", "y1": 0}]))
            argv = ["encode_tasks", "--input", str(root / "questions.json"),
                    "--model", str(root / "model"), "--output", str(root / "vectors.npz")]
            with patch.object(sys, "argv", argv), patch.dict(sys.modules, {
                "sentence_transformers": SimpleNamespace(SentenceTransformer=LocalEncoder)}):
                module.main()
            self.assertEqual(calls["texts"], ["A synthetic query", "Another synthetic query"])
            self.assertTrue(calls["constructor"]["local_files_only"])
            self.assertFalse(calls["constructor"]["trust_remote_code"])
            with np.load(root / "vectors.npz", allow_pickle=False) as data:
                self.assertEqual(data["ids"].tolist(), ["toy-b", "toy-a"])

    def test_evaluate_new_panel_and_reject_wrong_embedding_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [{"id": "toy-a", "family": "toy", "y0": 0, "y1": 1,
                     "tokens0": 10, "tokens1": 20},
                    {"id": "toy-b", "family": "toy", "y0": 1, "y1": 0,
                     "tokens0": 10, "tokens1": 20}]
            (root / "panel.json").write_text(json.dumps({"rows": rows}))
            args = [sys.executable, str(ROOT / "scripts/evaluate.py"),
                    "--panel", str(root / "panel.json"), "--embeddings", str(root / "e.npz"),
                    "--benchmark", "ToolQA", "--output", str(root / "metrics.json")]
            np.savez(root / "e.npz", ids=["toy-a", "toy-b"], vectors=[[1, 0], [1, 0]])
            result = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            metrics = json.loads((root / "metrics.json").read_text())
            # Each held-out task sees only the other task's opposite gain.
            self.assertEqual([p["score"] for p in metrics["predictions"]], [-1, 1])
            self.assertEqual(metrics["policy"]["success"], 0)
            np.savez(root / "e.npz", ids=["toy-b", "toy-a"], vectors=[[1, 0], [1, 0]])
            result = subprocess.run(args, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("match embedding order", result.stderr)


if __name__ == "__main__":
    unittest.main()
