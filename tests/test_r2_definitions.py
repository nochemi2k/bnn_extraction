"""R² estimands are distinct: row-level and MoP."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uq_remake.metrics.cv_protocol import (
    FORBIDDEN_PUBLISHED_METRIC_NAMES,
    PUBLISHED_POINT_METRIC_NAMES,
    assert_published_metric_names,
)
from uq_remake.metrics.fold_metrics import mop_block, row_point_metrics


def _frame(pairs: list[tuple[str, list[float], list[float]]]) -> pd.DataFrame:
    rows = []
    for pair, yt, yp in pairs:
        for i, (y, p) in enumerate(zip(yt, yp)):
            rows.append(
                {
                    "fold_id": "fold0",
                    "evaluation_partition": "heldout_fold",
                    "y_true": float(y),
                    "y_pred": float(p),
                    "metal_species_pair": pair,
                    "ligand_smiles": f"{pair}_{i}",
                }
            )
    return pd.DataFrame(rows)


class TestR2Definitions(unittest.TestCase):
    def test_row_and_mop_r2_differ_on_synthetic_pairs(self):
        df = _frame(
            [
                ("A:1.0--B:1.0", [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]),
                ("C:1.0--D:1.0", [10.0, 11.0, 12.0], [12.0, 11.0, 10.0]),
                ("E:1.0--F:1.0", [5.0, 6.0, 7.0, 8.0], [5.2, 5.8, 7.1, 8.4]),
            ]
        )
        row = row_point_metrics(df["y_true"], df["y_pred"])
        mop = mop_block(df)
        self.assertTrue(np.isfinite(row["r2"]))
        self.assertTrue(np.isfinite(mop["mop_r2"]))
        self.assertGreater(abs(row["r2"] - mop["mop_r2"]), 0.05)
        self.assertTrue(np.isfinite(row["rmse"]))
        self.assertTrue(np.isfinite(mop["mop_rmse"]))

    def test_mop_r2_is_mean_of_within_pair_r2_not_pair_mean_r2(self):
        df = _frame(
            [
                ("A:1.0--B:1.0", [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]),
                ("C:1.0--D:1.0", [10.0, 11.0, 12.0], [10.0, 10.0, 10.0]),
            ]
        )
        mop = mop_block(df)
        self.assertAlmostEqual(mop["mop_r2"], (1.0 + -1.5) / 2.0, places=12)
        ybar_p = np.array([1.0, 11.0])
        yhat_p = np.array([1.0, 10.0])
        pair_mean_r2 = 1.0 - np.sum((ybar_p - yhat_p) ** 2) / np.sum((ybar_p - ybar_p.mean()) ** 2)
        self.assertNotAlmostEqual(mop["mop_r2"], float(pair_mean_r2), places=6)

    def test_constant_target_pair_invalidates_only_mop_r2(self):
        df = _frame(
            [
                ("A:1.0--B:1.0", [1.0, 1.0, 1.0], [0.8, 1.1, 0.9]),
                ("C:1.0--D:1.0", [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]),
            ]
        )
        mop = mop_block(df)
        self.assertEqual(mop["n_pairs_invalid"]["r2"], 1)
        self.assertEqual(mop["n_pairs_valid"]["r2"], 1)
        self.assertAlmostEqual(mop["mop_r2"], 1.0, places=12)
        self.assertTrue(np.isfinite(mop["mop_rmse"]))
        row = row_point_metrics(df["y_true"], df["y_pred"])
        self.assertTrue(np.isfinite(row["r2"]))
        self.assertTrue(np.isfinite(row["rmse"]))

    def test_published_metric_allowlist(self):
        assert_published_metric_names(PUBLISHED_POINT_METRIC_NAMES)
        self.assertIn("rmse", PUBLISHED_POINT_METRIC_NAMES)
        self.assertIn("mop_rmse", PUBLISHED_POINT_METRIC_NAMES)
        for name in ("mae", "mop_mae", "pair_balanced_mae", "pair_balanced_global_r2", "mape"):
            self.assertIn(name, FORBIDDEN_PUBLISHED_METRIC_NAMES)
        with self.assertRaises(ValueError):
            assert_published_metric_names(["mae"])
        with self.assertRaises(ValueError):
            assert_published_metric_names(["pair_balanced_global_r2"])


if __name__ == "__main__":
    unittest.main()
