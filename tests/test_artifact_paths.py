"""Artifact-path tests for the reproduction checkout."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uq_remake.manuscript.figure_specs import FIGURE_SPECS


REQUIRED = [
    "data/processed/ds1.csv",
    "data/processed/ds2.csv",
    "data/splits/project_folds.json",
    "data/splits/ds2_project_folds.json",
    "results/cv_metrics/fold_metric_summary.csv",
    "results/downstream_cv/ad_retention/retention_50pct.csv",
    "results/downstream_cv/ad_hoc_selectivity/ad_hoc_selectivity_summary.csv",
    "results/downstream_cv/pair_acquisition/pair_acquisition_summary.csv",
    "results/downstream_cv/pair_acquisition_tau_2_0p5/pair_acquisition_summary.csv",
    "src/uq_remake/downstream/run_ad_hoc_selectivity.py",
    "src/uq_remake/downstream/run_pair_acquisition.py",
    "results/canonical/ds1_composition.json",
    "manuscripts/final/main_manuscript_iz.md",
    "manuscripts/final/supplementary_information.md",
    "scripts/validate_release.py",
    "scripts/evaluate.py",
    "src/uq_remake/downstream/run_ad_study.py",
    "src/uq_remake/manuscript/generate_figures.py",
    "scripts/run_pipeline.py",
    "scripts/train.py",
]

ABSENT = [
    "manuscripts/archive",
    "manuscripts/full_manuscript",
    "results/active_grid",
    "results/audit",
    "src/uq_remake/audit",
    "embeddings",
]


class TestArtifactPaths(unittest.TestCase):
    def test_required_release_paths_exist(self):
        for rel in REQUIRED:
            self.assertTrue((REPO / rel).exists(), rel)

    def test_removed_trees_are_absent(self):
        for rel in ABSENT:
            self.assertFalse((REPO / rel).exists(), rel)

    def test_figure_sources_exist(self):
        for spec in FIGURE_SPECS:
            self.assertTrue((REPO / spec["source_artifact"]).exists(), spec["source_artifact"])

    def test_exact_gpr_infeasible_stub(self):
        p = REPO / "checkpoints" / "active" / "DS2_RDKIT_GPR" / "fold0" / "member_0" / "infeasible.json"
        self.assertTrue(p.exists(), p)


if __name__ == "__main__":
    unittest.main()
