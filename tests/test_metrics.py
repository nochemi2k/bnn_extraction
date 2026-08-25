"""Fold-mean ± SD aggregation used by the manuscript (ddof=1)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uq_remake.metrics.cv_protocol import fold_mean_sd


class TestMetrics(unittest.TestCase):
    def test_fold_mean_sd_ddof1(self):
        stats = fold_mean_sd(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        self.assertAlmostEqual(stats["fold_mean"], 3.0)
        self.assertEqual(stats["n_valid_folds"], 5)
        self.assertAlmostEqual(stats["fold_sd"], float(np.std([1, 2, 3, 4, 5], ddof=1)))


if __name__ == "__main__":
    unittest.main()
