"""Regression: fold summaries reproduce from fold-level rows (ddof=1)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from uq_remake.metrics.cv_protocol import fold_mean_sd


class TestSummaryReproduces(unittest.TestCase):
    def test_fold_metric_summary_matches_recompute(self):
        fm_p = REPO / "results" / "cv_metrics" / "fold_metrics.csv"
        sm_p = REPO / "results" / "cv_metrics" / "fold_metric_summary.csv"
        if not fm_p.exists():
            self.skipTest("fold metrics missing")
        fm = pd.read_csv(fm_p)
        sm = pd.read_csv(sm_p)
        self.assertNotIn("pooled_oof_value", sm.columns)
        sample = sm.dropna(subset=["fold_mean"]).head(25)
        for _, row in sample.iterrows():
            g = fm[
                (fm["dataset"] == row["dataset"])
                & (fm["model_id"] == row["model_id"])
                & (fm["scope"] == row["scope"])
                & (fm["metric_name"] == row["metric_name"])
            ]
            if len(g) == 0:
                continue
            stats = fold_mean_sd(pd.to_numeric(g["value"], errors="coerce"))
            if np.isfinite(row["fold_mean"]):
                self.assertAlmostEqual(float(row["fold_mean"]), stats["fold_mean"], places=8)
            if np.isfinite(row["fold_sd"]) and stats["n_valid_folds"] >= 2:
                self.assertAlmostEqual(float(row["fold_sd"]), stats["fold_sd"], places=8)


if __name__ == "__main__":
    unittest.main()
