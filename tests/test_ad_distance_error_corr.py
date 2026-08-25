"""AD-distance versus row-level |error| association is an SI diagnostic only."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uq_remake.downstream.ad_distance_error_corr import fold_distance_error_corr
from uq_remake.downstream.protocol import DISTANCE_ERROR_CORR_SCORES, FEATURE_SPACE_SI_SCORES


def _toy_scores(n: int = 40, monotone: bool = True, constant_score: bool = False) -> pd.DataFrame:
    x = np.linspace(0.1, 2.0, n)
    if constant_score:
        score = np.full(n, 0.5)
        err = np.linspace(0.2, 1.5, n)
    elif monotone:
        score = x
        err = 0.3 + 1.2 * x
    else:
        score = x
        err = np.full(n, 0.4)
    y_true = np.zeros(n)
    y_pred = -err
    return pd.DataFrame(
        {
            "record_id": [f"r{i}" for i in range(n)],
            "y_true": y_true,
            "y_pred": y_pred,
            "tanimoto_nn_distance": score,
            "mahalanobis_feature_centroid": score,
            "tanimoto_median_distance": score,
            "tanimoto_knn_mean_5": score,
            "tanimoto_knn_mean_10": score,
            "metal_species_pair": ["Nd:3.0--Dy:3.0"] * n,
            "is_ree_pair": True,
        }
    )


class TestAdDistanceErrorCorr(unittest.TestCase):
    def test_linear_monotone_is_one(self):
        rows = fold_distance_error_corr(_toy_scores(monotone=True), "fold0")
        all_pair = [r for r in rows if r["scope"] == "all_pairs" and r["score_name"] == "tanimoto_nn_distance"]
        pearson = next(r["value"] for r in all_pair if r["metric_name"] == "pearson_abs_residual")
        spearman = next(r["value"] for r in all_pair if r["metric_name"] == "spearman_abs_residual")
        self.assertAlmostEqual(pearson, 1.0, places=6)
        self.assertAlmostEqual(spearman, 1.0, places=6)

    def test_constant_score_is_undefined(self):
        rows = fold_distance_error_corr(_toy_scores(constant_score=True), "fold0")
        vals = [
            r["value"]
            for r in rows
            if r["scope"] == "all_pairs" and r["score_name"] == "tanimoto_nn_distance"
        ]
        self.assertTrue(all(not np.isfinite(v) for v in vals))
        self.assertTrue(any(r["metric_status"] == "undefined_zero_variance" for r in rows))

    def test_ree_scope_uses_flag(self):
        df = _toy_scores(n=20)
        df.loc[:9, "is_ree_pair"] = False
        rows = fold_distance_error_corr(df, "fold0")
        ree = next(
            r
            for r in rows
            if r["scope"] == "ree_pairs"
            and r["score_name"] == "tanimoto_nn_distance"
            and r["metric_name"] == "pearson_abs_residual"
        )
        self.assertEqual(ree["n_rows_used"], 10)

    def test_missing_score_is_nan(self):
        df = _toy_scores()
        df = df.drop(columns=["tanimoto_median_distance"])
        rows = fold_distance_error_corr(df, "fold0")
        miss = [
            r
            for r in rows
            if r["score_name"] == "tanimoto_median_distance" and r["scope"] == "all_pairs"
        ]
        self.assertTrue(all(r["metric_status"] == "missing_score" for r in miss))
        self.assertTrue(all(not np.isfinite(r["value"]) for r in miss))

    def test_score_list_is_distances_only(self):
        self.assertIn("tanimoto_nn_distance", DISTANCE_ERROR_CORR_SCORES)
        self.assertIn("mahalanobis_feature_centroid", DISTANCE_ERROR_CORR_SCORES)
        for name in FEATURE_SPACE_SI_SCORES:
            self.assertIn(name, DISTANCE_ERROR_CORR_SCORES)
        self.assertNotIn("sigma_total", DISTANCE_ERROR_CORR_SCORES)
        self.assertNotIn("pair_density_risk", DISTANCE_ERROR_CORR_SCORES)

    def test_not_a_selection_metric(self):
        rows = fold_distance_error_corr(_toy_scores(), "fold0")
        self.assertTrue(all(r["experiment_id"] == "ad_distance_error_corr" for r in rows))
        self.assertNotIn("mop_spearman", {r["metric_name"] for r in rows})


if __name__ == "__main__":
    unittest.main()
