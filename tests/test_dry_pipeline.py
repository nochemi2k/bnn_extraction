"""Canonical dry-rebuild pipeline tests. No retraining, no torch."""
from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uq_remake.downstream.context_diagnostics import bin_pair_count, pair_density_fold
from uq_remake.downstream.run_downstream_cv import canonical_artifacts
from uq_remake.run_full_pipeline import STEPS


class TestDryPipelineWiring(unittest.TestCase):
    def test_canonical_steps_include_manuscript_and_pair_acq_path(self):
        names = [s[0] for s in STEPS]
        self.assertEqual(
            names,
            [
                "prepare_data",
                "cmisplit",
                "cmi_repro",
                "hplus",
                "finalize_ds1",
                "active_grid",
                "cv_metrics",
                "downstream_cv",
                "ad_hoc_foldmatched",
                "manuscript",
                "validate",
            ],
        )
        by_name = {s[0]: s for s in STEPS}
        self.assertIn("--verify", by_name["prepare_data"][3])
        self.assertIn("--audit-only", by_name["active_grid"][3])
        self.assertIn("--stored-only", by_name["cv_metrics"][3])
        self.assertIn("--from-existing-scores", by_name["downstream_cv"][3])
        self.assertIn("--rescore", by_name["downstream_cv"][2])
        self.assertTrue(by_name["manuscript"][1].endswith("run_manuscript_rebuild.py"))
        self.assertTrue(by_name["validate"][1].endswith("validate_release.py"))

    def test_pipeline_help(self):
        from uq_remake.run_full_pipeline import main as pipe_main

        buf = io.StringIO()
        with redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
            pipe_main(["--help"])
        self.assertEqual(cm.exception.code, 0)
        text = buf.getvalue()
        self.assertIn("--dry", text)
        self.assertIn("--reuse-downstream", text)

    def test_downstream_help(self):
        from uq_remake.downstream.run_downstream_cv import main as ds_main

        buf = io.StringIO()
        with redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
            ds_main(["--help"])
        self.assertEqual(cm.exception.code, 0)
        text = buf.getvalue()
        self.assertIn("--from-existing-scores", text)
        self.assertIn("--skip-pair-acq", text)
        self.assertIn("--skip-context", text)
        self.assertIn("--skip-if-complete", text)

    def test_canonical_artifacts_on_this_repo(self):
        present = canonical_artifacts()
        self.assertTrue(present["ad"], present)
        self.assertTrue(present["pair_acq"], present)
        self.assertTrue(present["pair_acq_sens"], present)
        self.assertTrue(present["si_diagnostics"], present)
        self.assertTrue(present["context"], present)
        self.assertTrue(present["ad_hoc"], present)


class TestContextDiagnostics(unittest.TestCase):
    def test_bin_pair_count(self):
        self.assertEqual(bin_pair_count(0), "0")
        self.assertEqual(bin_pair_count(3), "1-5")
        self.assertEqual(bin_pair_count(8), "6-10")
        self.assertEqual(bin_pair_count(20), "11-25")
        self.assertEqual(bin_pair_count(40), "26-50")
        self.assertEqual(bin_pair_count(80), ">50")

    def test_pair_density_fold_emits_rmse_not_rank_correlation(self):
        n = 12
        df = pd.DataFrame(
            {
                "record_id": [f"r{i}" for i in range(n)],
                "fold_id": "fold0",
                "outer_fold_id": "fold0",
                "evaluation_partition": "heldout_fold",
                "y_true": np.linspace(-1.0, 1.5, n),
                "y_pred": np.linspace(-0.8, 1.2, n),
                "sigma_total": np.linspace(0.2, 0.9, n),
                "metal_species_pair": ["Nd:3.0--Dy:3.0"] * 6 + ["La:3.0--Gd:3.0"] * 6,
                "is_ree_pair": True,
                "n_training_ligands_for_pair": [0, 2, 8, 20, 40, 80] * 2,
            }
        )
        rows = pair_density_fold(df, "fold0")
        self.assertGreater(len(rows), 0)
        metrics = {r["metric_name"] for r in rows}
        self.assertIn("rmse", metrics)
        self.assertIn("mop_rmse", metrics)
        self.assertNotIn("spearman", metrics)
        self.assertNotIn("mop_spearman", metrics)
        joined = " ".join(str(v) for r in rows for v in r.values())
        self.assertNotIn("spearman", joined.lower())


if __name__ == "__main__":
    unittest.main()
