# tests/test_contrastive_pipeline.py
import unittest
import torch

from utils.contrastive_utils import (
    compute_delta_dlig_site_value,
    update_site_score_accumulator,
    finalize_site_scores,
)


class TestContrastiveUtils(unittest.TestCase):
    def test_delta_dlig_site_value_shape(self):
        B, L, H = 2, 7, 11
        harm = torch.randn(B, L, H)
        benign = torch.randn(B, L, H)
        v = compute_delta_dlig_site_value(harm, benign, norm_type="l2")
        self.assertEqual(tuple(v.shape), (B,))

    def test_site_score_accumulator_and_finalize(self):
        accum = {}
        update_site_score_accumulator(accum, "14", 3, torch.tensor([1.0, 3.0]))
        update_site_score_accumulator(accum, "14", 3, torch.tensor([2.0]))
        scores = finalize_site_scores(accum)
        self.assertIn("14", scores)
        self.assertIn("3", scores["14"])
        self.assertAlmostEqual(scores["14"]["3"], 2.0, places=6)

    def test_delta_dlig_mismatch_raises(self):
        harm = torch.randn(1, 5, 7)
        benign = torch.randn(1, 6, 7)
        with self.assertRaises(ValueError):
            _ = compute_delta_dlig_site_value(harm, benign)


class FakeRecorder:
    def __init__(self):
        self.x_by_step = {}

    def hook(self, step, x, logits):
        if step is None:
            return logits
        self.x_by_step[int(step)] = x.detach().clone()
        return logits


class TestTrajectoryDeterminism(unittest.TestCase):
    def test_hook_records_and_determinism(self):
        # Deterministic simulation (no HF model dependency).
        def simulate(seed: int):
            torch.manual_seed(seed)
            rec = FakeRecorder()
            x = torch.randint(0, 100, (1, 10))
            for step in range(5):
                x = (x + step) % 100
                logits = torch.randn(1, 10, 50)
                rec.hook(step, x, logits)
            return rec.x_by_step

        a = simulate(0)
        b = simulate(0)
        self.assertEqual(sorted(a.keys()), [0, 1, 2, 3, 4])
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]))

    def test_hook_differs_with_different_seed(self):
        def simulate(seed: int):
            torch.manual_seed(seed)
            rec = FakeRecorder()
            x = torch.randint(0, 100, (1, 10))
            for step in range(5):
                x = (x + step) % 100
                logits = torch.randn(1, 10, 50)
                rec.hook(step, x, logits)
            return rec.x_by_step

        a = simulate(0)
        b = simulate(1)
        # At least one step should differ
        any_diff = any(not torch.equal(a[k], b[k]) for k in a.keys())
        self.assertTrue(any_diff)


if __name__ == "__main__":
    unittest.main()