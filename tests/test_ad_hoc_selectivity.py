"""Ad hoc DS1-difference: alignment, subtraction, orientation, and uncertainty."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uq_remake.downstream.ad_hoc_selectivity import (
    AdHocAlignmentError,
    SELECTIVITY_AD_HOC,
    SELECTIVITY_DIRECT,
    ad_hoc_comparison_phrase,
    ad_hoc_difference,
    assemble_ad_hoc_pairs,
    assert_no_ligand_overlap,
    evaluate_ad_hoc_fold,
    propagate_difference_sigma,
    reverse_orientation,
    summarize_fold_metrics,
)
from uq_remake.metrics.cv_protocol import EVALUATION_PARTITION, MIN_PAIR_N, PAIR_COLUMN


def _ds1_preds():
    return pd.DataFrame(
        {
            "record_id": ["Rx1", "Ry1", "Rx2", "Ry2", "Rx3", "Ry3"],
            "y_pred": [3.0, 1.0, 4.0, 2.5, 0.5, 1.5],
            "sigma_total": [0.4, 0.3, 0.2, 0.6, 0.5, 0.1],
            "ligand_smiles": ["L1", "L1", "L2", "L2", "L3", "L3"],
            "source_fold": ["fold0"] * 6,
        }
    )


def _ds2_pairs():
    return pd.DataFrame(
        {
            "pair_id": ["p1", "p2", "p3"],
            "record_id_x": ["Rx1", "Rx2", "Rx3"],
            "record_id_y": ["Ry1", "Ry2", "Ry3"],
            "ligand_smiles": ["L1", "L2", "L3"],
            PAIR_COLUMN: ["Nd:3.0--Dy:3.0"] * 3,
            "is_ree_pair": [True, True, True],
            "selectivity_logK": [2.1, 1.4, -0.8],
        }
    )


class TestSubtractionAndOrientation(unittest.TestCase):
    def test_ad_hoc_difference_is_x_minus_y(self):
        np.testing.assert_allclose(ad_hoc_difference([3.0, 4.0], [1.0, 2.5]), [2.0, 1.5])

    def test_difference_does_not_take_abs(self):
        np.testing.assert_allclose(ad_hoc_difference([0.5], [1.5]), [-1.0])

    def test_reverse_orientation_negates(self):
        delta = ad_hoc_difference([3.0, 0.5], [1.0, 1.5])
        np.testing.assert_allclose(reverse_orientation(delta), -delta)


class TestUncertaintyPropagation(unittest.TestCase):
    def test_independent_quadrature(self):
        np.testing.assert_allclose(propagate_difference_sigma([0.4], [0.3]), [np.sqrt(0.16 + 0.09)])

    def test_propagated_sigma_is_nonnegative(self):
        out = propagate_difference_sigma([0.0, 1.2], [0.5, 0.0])
        self.assertTrue(np.all(out >= 0))

    def test_no_invented_covariance_term(self):
        src = Path(__file__).resolve().parents[1] / "src" / "uq_remake" / "downstream" / "ad_hoc_selectivity.py"
        text = src.read_text(encoding="utf-8")
        self.assertIn("sqrt", text)
        self.assertNotIn("2*sigma_x*sigma_y", text.replace(" ", ""))


class TestAlignment(unittest.TestCase):
    def test_no_ligand_overlap_passes(self):
        assert_no_ligand_overlap(["trainL"], ["heldL"])

    def test_ligand_overlap_raises(self):
        with self.assertRaises(AdHocAlignmentError):
            assert_no_ligand_overlap(["same"], ["same"])

    def test_assemble_subtracts_aligned_endpoints(self):
        out = assemble_ad_hoc_pairs(_ds2_pairs(), _ds1_preds(), "DS1_RDKIT_MLPENS5_EVID", "fold0")
        np.testing.assert_allclose(out["y_pred"], [2.0, 1.5, -1.0])
        np.testing.assert_allclose(out["y_pred_x"] - out["y_pred_y"], out["y_pred"])
        self.assertTrue((out["evaluation_partition"] == EVALUATION_PARTITION).all())
        self.assertEqual(out["selectivity_method"].iloc[0], SELECTIVITY_AD_HOC)

    def test_missing_endpoint_raises(self):
        pairs = _ds2_pairs()
        pairs.loc[0, "record_id_x"] = "missing"
        with self.assertRaises(AdHocAlignmentError):
            assemble_ad_hoc_pairs(pairs, _ds1_preds(), "DS1_RDKIT_MLPENS5_EVID", "fold0")


class TestPairSupportAndAggregation(unittest.TestCase):
    def test_pair_support_is_at_least_three(self):
        self.assertEqual(MIN_PAIR_N, 3)
        recs_x, recs_y, ligs, ytrue, pred_rows = [], [], [], [], []
        for i in range(4):
            recs_x.append(f"Rx{i}")
            recs_y.append(f"Ry{i}")
            ligs.append(f"L{i}")
            ytrue.append(1.0 + 0.1 * i)
            pred_rows.append({"record_id": f"Rx{i}", "y_pred": 2.0 + 0.1 * i, "sigma_total": 0.2, "ligand_smiles": f"L{i}", "source_fold": "fold0"})
            pred_rows.append({"record_id": f"Ry{i}", "y_pred": 1.0, "sigma_total": 0.1, "ligand_smiles": f"L{i}", "source_fold": "fold0"})
        pairs = pd.DataFrame(
            {
                "pair_id": [f"p{i}" for i in range(4)],
                "record_id_x": recs_x,
                "record_id_y": recs_y,
                "ligand_smiles": ligs,
                PAIR_COLUMN: ["Nd:3.0--Dy:3.0"] * 4,
                "is_ree_pair": [True] * 4,
                "selectivity_logK": ytrue,
            }
        )
        assembled = assemble_ad_hoc_pairs(pairs, pd.DataFrame(pred_rows), "DS1_RDKIT_MLPENS5_EVID", "fold0")
        rec = evaluate_ad_hoc_fold(assembled, "DS1_RDKIT_MLPENS5_EVID", "ree_pairs")
        self.assertGreaterEqual(rec["n_pairs"], 1)
        self.assertEqual(rec["min_pair_size"], 3)
        self.assertTrue(np.isfinite(rec["row_r2"]))
        self.assertTrue(np.isfinite(rec["mop_picp_90"]))
        self.assertTrue(np.isfinite(rec["mop_ace"]))

    def test_point_ad_hoc_has_no_interval_metrics(self):
        recs_x, recs_y, ligs, ytrue, pred_rows = [], [], [], [], []
        for i in range(4):
            recs_x.append(f"Rx{i}")
            recs_y.append(f"Ry{i}")
            ligs.append(f"L{i}")
            ytrue.append(1.0 + 0.1 * i)
            pred_rows.append({"record_id": f"Rx{i}", "y_pred": 2.0 + 0.1 * i, "ligand_smiles": f"L{i}", "source_fold": "fold0"})
            pred_rows.append({"record_id": f"Ry{i}", "y_pred": 1.0, "ligand_smiles": f"L{i}", "source_fold": "fold0"})
        pairs = pd.DataFrame(
            {
                "pair_id": [f"p{i}" for i in range(4)],
                "record_id_x": recs_x,
                "record_id_y": recs_y,
                "ligand_smiles": ligs,
                PAIR_COLUMN: ["Nd:3.0--Dy:3.0"] * 4,
                "is_ree_pair": [True] * 4,
                "selectivity_logK": ytrue,
            }
        )
        assembled = assemble_ad_hoc_pairs(pairs, pd.DataFrame(pred_rows), "DS1_RDKIT_MLP_POINT", "fold0")
        rec = evaluate_ad_hoc_fold(assembled, "DS1_RDKIT_MLP_POINT", "ree_pairs")
        self.assertTrue(np.isfinite(rec["row_r2"]))
        self.assertFalse(np.isfinite(rec["mop_picp_90"]))
        self.assertFalse(np.isfinite(rec["picp_90"]))

    def test_fold_aggregation_is_mean_plus_sd(self):
        rows = []
        for fold, rmse in zip(["fold0", "fold1", "fold2", "fold3", "fold4"], [0.5, 0.4, 0.6, 0.45, 0.55]):
            rows.append(
                {
                    "dataset": "DS2",
                    "outer_fold": fold,
                    "source_model": "Evidential MLP ensemble",
                    "source_model_id": "DS1_RDKIT_MLPENS5_EVID",
                    "selectivity_method": SELECTIVITY_AD_HOC,
                    "scope": "ree_pairs",
                    "mop_rmse": rmse,
                    "mop_r2": 0.5,
                    "sign_accuracy": 0.7,
                    "row_rmse": 0.8,
                    "picp_90": 0.9,
                    "ace": 0.05,
                    "n_rows": 10,
                    "n_pairs": 4,
                    "uncertainty_status": "propagated_independent_quadrature",
                    "propagation_method": "independent_quadrature_sigma_total",
                }
            )
        summary = summarize_fold_metrics(pd.DataFrame(rows))
        self.assertEqual(int(summary.iloc[0]["n_valid"]), 5)
        self.assertAlmostEqual(float(summary.iloc[0]["mop_rmse"]), float(np.mean([0.5, 0.4, 0.6, 0.45, 0.55])))
        self.assertTrue(np.isfinite(summary.iloc[0]["mop_rmse_sd"]))

    def test_direct_and_ad_hoc_are_separate_methods(self):
        self.assertNotEqual(SELECTIVITY_DIRECT, SELECTIVITY_AD_HOC)

    def test_comparison_phrase_is_restrained(self):
        df = pd.DataFrame(
            [
                {
                    "scope": "ree_pairs",
                    "source_model_id": "DS2_RDKIT_MLPENS5_HETERO",
                    "selectivity_method": SELECTIVITY_DIRECT,
                    "mop_rmse": 0.42,
                },
                {
                    "scope": "ree_pairs",
                    "source_model_id": "DS1_RDKIT_MLPENS5_EVID",
                    "selectivity_method": SELECTIVITY_AD_HOC,
                    "mop_rmse": 0.71,
                },
                {
                    "scope": "ree_pairs",
                    "source_model_id": "DS1_RDKIT_MLP_HETERO",
                    "selectivity_method": SELECTIVITY_AD_HOC,
                    "mop_rmse": 0.80,
                },
            ]
        )
        self.assertEqual(ad_hoc_comparison_phrase(df), "performed worse than")


class TestNoHeldOutTargets(unittest.TestCase):
    def test_assemble_uses_predictions_not_targets_for_difference(self):
        preds = _ds1_preds()
        preds["y_true"] = [99.0] * len(preds)
        out = assemble_ad_hoc_pairs(_ds2_pairs(), preds, "DS1_RDKIT_MLPENS5_EVID", "fold0")
        np.testing.assert_allclose(out["y_pred"], [2.0, 1.5, -1.0])
        self.assertFalse(np.allclose(out["y_pred"], out["y_true"]))


if __name__ == "__main__":
    unittest.main()
