"""RDKit feature-space AD aggregations. SI Spearman extras, not retention figures."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uq_remake.downstream.ad_scores import feature_distance_bundle
from uq_remake.downstream.protocol import FEATURE_SPACE_SI_SCORES, WORKHORSE_SENSITIVITY_SCORES


class TestFeatureSpaceDistances(unittest.TestCase):
    def test_knn_means_are_monotone(self):
        rng = np.random.default_rng(0)
        x_train = rng.normal(size=(80, 6))
        x_query = rng.normal(size=(12, 6))
        bundle = feature_distance_bundle(x_train, x_query, ks=(5, 10))
        nn = bundle["mahalanobis_feature_nn"]
        k5 = bundle["mahalanobis_feature_knn_mean_5"]
        k10 = bundle["mahalanobis_feature_knn_mean_10"]
        self.assertTrue(np.all(nn <= k5 + 1e-12))
        self.assertTrue(np.all(k5 <= k10 + 1e-12))
        self.assertTrue(np.all(bundle["euclidean_feature_nn"] <= bundle["euclidean_feature_median"] + 1e-12))

    def test_centroid_matches_published_name(self):
        rng = np.random.default_rng(1)
        x_train = rng.normal(size=(40, 4))
        x_query = rng.normal(size=(8, 4))
        bundle = feature_distance_bundle(x_train, x_query)
        self.assertEqual(list(bundle["mahalanobis_feature_centroid"]), list(bundle["mahalanobis_joint"]))
        self.assertTrue(np.all(np.isfinite(bundle["mahalanobis_feature_centroid"])))

    def test_si_scores_are_not_retention_sensitivity(self):
        for name in FEATURE_SPACE_SI_SCORES:
            self.assertNotIn(name, WORKHORSE_SENSITIVITY_SCORES)


if __name__ == "__main__":
    unittest.main()
