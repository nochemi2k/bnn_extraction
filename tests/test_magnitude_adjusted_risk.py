"""Fold-local magnitude-adjusted risk diagnostics."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uq_remake.downstream.magnitude_adjusted_risk import (
    fold_correlations,
    partial_spearman,
    spearman,
)
from uq_remake.metrics.cv_protocol import fold_mean_sd


def _partial_from_pairwise(rxy, rxz, ryz):
    return (rxy - rxz * ryz) / np.sqrt((1.0 - rxz * rxz) * (1.0 - ryz * ryz))


class TestPartialSpearman(unittest.TestCase):
    def test_formula_matches_rank_pearson(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=80)
        x = z + 0.4 * rng.normal(size=80)
        y = 0.5 * z + 0.8 * x + 0.3 * rng.normal(size=80)
        got = partial_spearman(x, y, z)
        rxy = spearman(x, y)
        rxz = spearman(x, z)
        ryz = spearman(y, z)
        expected = _partial_from_pairwise(rxy, rxz, ryz)
        self.assertTrue(np.isfinite(got))
        self.assertAlmostEqual(got, expected, places=10)

    def test_recovers_direct_association_after_confounder(self):
        rng = np.random.default_rng(1)
        z = rng.normal(size=200)
        x = z + rng.normal(size=200)
        y = x + 0.05 * rng.normal(size=200)
        raw = spearman(x, y)
        partial = partial_spearman(x, y, z)
        self.assertGreater(raw, 0.7)
        self.assertGreater(partial, 0.5)

    def test_collinear_control_is_undefined(self):
        x = np.arange(10.0)
        y = np.arange(10.0) ** 2
        self.assertTrue(np.isnan(partial_spearman(x, y, x)))

    def test_fold_local_not_pooled_headline(self):
        path = REPO / "results" / "downstream_cv" / "pair_acquisition" / "magnitude_adjusted_risk_summary.csv"
        fold_path = REPO / "results" / "downstream_cv" / "pair_acquisition" / "magnitude_adjusted_risk_fold.csv"
        if not path.exists() or not fold_path.exists():
            self.skipTest("magnitude-adjusted artifacts missing")
        summary = pd.read_csv(path)
        fold = pd.read_csv(fold_path)
        self.assertNotIn("pooled", " ".join(summary.metric_name.astype(str)).lower())
        self.assertNotIn("pooled", " ".join(fold.metric_name.astype(str)).lower())
        part = fold[fold.metric_name.astype(str) == "rho_partial_sigma_error_given_abs_yhat"]
        self.assertEqual(len(part), 5)
        stats = fold_mean_sd(part.value)
        hit = summary[summary.metric_name.astype(str) == "rho_partial_sigma_error_given_abs_yhat"].iloc[0]
        self.assertEqual(int(hit.n_valid_folds), 5)
        self.assertAlmostEqual(float(hit.fold_mean), stats["fold_mean"], places=12)
        self.assertAlmostEqual(float(hit.fold_sd), stats["fold_sd"], places=12)
        self.assertEqual(str(hit.aggregation), "fold_mean_plus_sd")

    def test_does_not_overwrite_canonical_retention(self):
        canon = REPO / "results" / "downstream_cv" / "ad_retention" / "fold_retention.csv"
        if not canon.exists():
            self.skipTest("canonical retention missing")
        text = canon.read_text(encoding="utf-8")
        self.assertNotIn("abs_yhat", text.split("\n", 1)[0])

    def test_fold_correlations_use_heldout_columns(self):
        df = pd.DataFrame(
            {
                "y_true": [0.0, 1.0, 2.0, 3.0, 4.0],
                "y_pred": [0.1, 1.2, 2.4, 2.5, 5.0],
                "sigma_total": [0.2, 0.3, 0.9, 0.4, 1.5],
            }
        )
        rec = fold_correlations(df)
        self.assertEqual(rec["n_rows"], 5)
        self.assertTrue(np.isfinite(rec["rho_partial_sigma_error_given_abs_yhat"]))
        self.assertAlmostEqual(
            rec["rho_partial_sigma_error_given_abs_yhat"],
            partial_spearman(
                df.sigma_total,
                (df.y_true - df.y_pred).abs(),
                df.y_pred.abs(),
            ),
        )


if __name__ == "__main__":
    unittest.main()
