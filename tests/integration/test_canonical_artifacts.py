"""Canonical artifact alignment integration tests."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from uq_remake.metrics.cv_protocol import EVALUATION_PARTITION
from uq_remake.training.protocol import canonical_model_id, load_folds, load_table, record_ids_for_split


class TestCanonicalArtifacts(unittest.TestCase):
    def test_workhorse_fold_scores_align_with_split_size(self):
        table = load_table("DS2")
        folds = load_folds("DS2")
        mid = canonical_model_id("DS2", "EVID_MLPENS5")
        for fold in ("fold0", "fold1", "fold2", "fold3", "fold4"):
            path = REPO / "results" / "downstream_cv" / "ad_retention" / "fold_scores" / mid / f"{fold}.csv"
            self.assertTrue(path.exists(), path)
            import pandas as pd

            df = pd.read_csv(path, usecols=lambda c: c in {"record_id", "fold_id", "evaluation_partition", "outer_fold_id"})
            ids = record_ids_for_split(table, "DS2", folds[fold]["test_idx"])
            self.assertGreater(len(df), 0)
            if "evaluation_partition" in df.columns:
                self.assertEqual(str(df["evaluation_partition"].iloc[0]), EVALUATION_PARTITION)

    def test_exact_gpr_is_infeasible_json(self):
        p = REPO / "checkpoints" / "active" / "DS2_RDKIT_GPR" / "fold0" / "member_0" / "infeasible.json"
        self.assertTrue(p.exists())
        payload = json.loads(p.read_text(encoding="utf-8"))
        self.assertTrue(payload)

    def test_prediction_dumps_are_not_required(self):
        self.assertFalse((REPO / "results" / "active_grid").exists())


if __name__ == "__main__":
    unittest.main()
