import unittest
import numpy as np
from skilldelta import Protocol, predict_gain, predict_skill_success
from skilldelta.metrics import auroc, matched_rate_advantage, policy


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.kw = dict(query_vectors=[[1., 0.]], support_vectors=[[1., 0.], [1., 1.], [0., 1.]],
                       query_ids=["target"], support_ids=["target", "a", "b"],
                       query_families=["f"], support_families=["f", "f", "f"],
                       protocol=Protocol(k=2, threshold="support_prevalence"))

    def test_target_outcome_changes_neither_score_nor_threshold(self):
        a = predict_gain(**self.kw, skip_outcomes=[0, 0, 1], use_outcomes=[1, 1, 0])
        b = predict_gain(**self.kw, skip_outcomes=[1, 0, 1], use_outcomes=[0, 1, 0])
        np.testing.assert_array_equal(a["scores"], b["scores"])
        np.testing.assert_array_equal(a["thresholds"], b["thresholds"])
        self.assertEqual(a["neighbors"], [[1, 2]])
        self.assertEqual(a["thresholds"].tolist(), [0.5])

    def test_unseen_target_needs_no_outcome(self):
        kw = {**self.kw, "query_ids": ["new"]}
        result = predict_gain(**kw, skip_outcomes=[0, 0, 0], use_outcomes=[1, 1, 1])
        self.assertAlmostEqual(result["scores"][0], 1.)
        self.assertFalse(result["use_skill"][0])  # Strict > at equality.

    def test_empty_family_skips(self):
        kw = {**self.kw, "query_families": ["unseen"]}
        result = predict_gain(**kw, skip_outcomes=[0, 0, 0], use_outcomes=[1, 1, 1])
        self.assertEqual(result["scores"].tolist(), [0.])
        self.assertEqual(result["use_skill"].tolist(), [False])

    def test_zero_similarity_falls_back_to_uniform(self):
        kw = {**self.kw, "query_vectors": [[-1., -1.]], "protocol": Protocol(k=2)}
        result = predict_gain(**kw, skip_outcomes=[0, 0, 1], use_outcomes=[1, 1, 0])
        self.assertEqual(result["scores"].tolist(), [0.])

    def test_on_only_api_uses_only_use_outcomes(self):
        a = predict_skill_success(**self.kw, use_outcomes=[0, 1, 0])
        b = predict_skill_success(**self.kw, use_outcomes=[1, 1, 0])
        np.testing.assert_array_equal(a, b)

    def test_duplicate_support_ids_are_rejected(self):
        kw = {**self.kw, "support_ids": ["a", "a", "b"]}
        with self.assertRaises(ValueError):
            predict_gain(**kw, skip_outcomes=[0, 0, 0], use_outcomes=[1, 1, 1])

    def test_auc_ties_and_missing_class(self):
        self.assertEqual(auroc([0, 1], [.2, .2]), .5)
        self.assertEqual(auroc([0, 1], [.2, .3]), 1.)
        self.assertIsNone(auroc([1, 1], [.2, .3]))

    def test_selection_and_router_cost(self):
        m = matched_rate_advantage([0, 1], [1, 0], [1, 0])
        self.assertEqual(m["random_success"], .5)
        self.assertEqual(m["selection_advantage_pp"], 50.)
        p = policy([0, 1], [1, 0], [10, 10], [30, 30], [1, 0], mean_router_tokens=10)
        self.assertEqual(p["success"], 1.)
        self.assertEqual(p["token_saving"], 0.)


if __name__ == "__main__":
    unittest.main()
