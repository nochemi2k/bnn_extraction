"""Registry and allowlist unit tests."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from uq_remake.metrics.cv_protocol import EVALUATION_PARTITION, archived_model_blocked
from uq_remake.training.protocol import canonical_model_id


class TestRegistryContract(unittest.TestCase):
    def test_exact_gpr_is_excluded_identity(self):
        self.assertEqual(canonical_model_id("DS2", "GPR"), "DS2_RDKIT_GPR")
        self.assertEqual(canonical_model_id("DS2", "GPR_SPARSE"), "DS2_RDKIT_GPR_SPARSE")
        self.assertNotEqual(canonical_model_id("DS1", "BAYES_FULL"), canonical_model_id("DS1", "MCDROPOUT_MLP"))

    def test_archived_ids_blocked(self):
        self.assertTrue(archived_model_blocked("DS1_RDKIT_BAYES_LASTLAYER"))
        self.assertTrue(archived_model_blocked("DS2_GRAPH_DMPNN_BAYES"))
        self.assertFalse(archived_model_blocked("DS2_RDKIT_MLPENS5_EVID"))

    def test_deleted_trees_are_absent(self):
        self.assertFalse((REPO / "archive").exists())
        self.assertFalse((REPO / "src" / "uq_remake" / "stage1").exists())
        self.assertFalse((REPO / "src" / "uq_remake" / "runmode.py").exists())
        self.assertFalse((REPO / "src" / "uq_remake" / "audit").exists())
        self.assertTrue((REPO / "src" / "uq_remake" / "data" / "prepare_data.py").exists())

    def test_evaluation_partition(self):
        self.assertEqual(EVALUATION_PARTITION, "heldout_fold")

    def test_registry_file_if_present(self):
        path = REPO / "configs" / "experiments" / "active_registry.json"
        if not path.exists():
            self.skipTest("registry not written yet")
        reg = json.loads(path.read_text(encoding="utf-8"))
        ids = {m["model_id"] for m in reg["models"]}
        self.assertIn("DS2_RDKIT_MLPENS5_EVID", ids)
        gpr = [m for m in reg["models"] if m["model_family"] == "GPR"]
        self.assertTrue(gpr)
        self.assertTrue(all(m["overall_status"] == "excluded" for m in gpr))
        chem = [m for m in reg["models"] if m["model_id"] == "DS2_GRAPH_DMPNN_EVID"][0]
        self.assertEqual(chem["native_uq_status"], "partial_native_invalid_rows")
        self.assertEqual(chem["ranking_status"], "complete_5_of_5")
        self.assertEqual(chem["model_modification"], "none")
        self.assertEqual(reg["workhorse_model_id"], "DS2_RDKIT_MLPENS5_HETERO")
        self.assertNotIn("oof", json.dumps(reg).lower().replace("proof", ""))

    def test_allowlist_required_paths_exist(self):
        path = REPO / "configs" / "experiments" / "manuscript_artifact_allowlist.yaml"
        self.assertTrue(path.exists(), path)
        text = path.read_text(encoding="utf-8")
        self.assertIn("results/cv_metrics/", text)
        self.assertIn("chemprop_evidential_status_v2.csv", text)
        for rel in (
            "configs/experiments/active_registry.yaml",
            "configs/experiments/active_registry.json",
            "data/splits/project_folds.json",
            "data/splits/ds2_project_folds.json",
            "results/cv_metrics/fold_metrics.csv",
            "results/cv_metrics/fold_metric_summary.csv",
            "checkpoints/active",
        ):
            self.assertTrue((REPO / rel).exists(), rel)


if __name__ == "__main__":
    unittest.main()
