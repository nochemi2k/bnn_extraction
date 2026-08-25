"""Operational accuracy–calibration workhorse selection."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uq_remake.metrics.cv_protocol import OOFTerminologyError
from uq_remake.metrics.fold_metrics import per_pair_metric
from uq_remake.metrics.workhorse import (
    ACE_UPPER,
    CHEMPROP_DS2,
    PICP_LOWER,
    PICP_UPPER,
    select_workhorse_accuracy_calibration,
)


def _metric_row(mid, scope, metric, mean, n_valid=5, sd=0.01):
    return {
        "dataset": "DS2",
        "model_id": mid,
        "scope": scope,
        "metric_name": metric,
        "fold_mean": mean,
        "fold_sd": sd,
        "n_folds": 5,
        "n_valid_folds": n_valid,
    }


def _complete_model(mid, rmse, picp, ace, r2=0.1, sign=0.7, n_valid=5, all_pair_rmse=None):
    if all_pair_rmse is None:
        all_pair_rmse = float(rmse) + 1.0
    rows = [
        _metric_row(mid, "all_pairs", "mop_rmse", all_pair_rmse, n_valid),
        _metric_row(mid, "all_pairs", "mop_r2", r2, n_valid),
        _metric_row(mid, "all_pairs", "sign_accuracy_all", sign, n_valid),
        _metric_row(mid, "ree_pairs", "mop_rmse", rmse, n_valid),
        _metric_row(mid, "ree_pairs", "mop_r2", r2, n_valid),
        _metric_row(mid, "ree_pairs", "sign_accuracy_all", sign, n_valid),
        _metric_row(mid, "all_pairs", "marginal_picp_90", picp, n_valid),
        _metric_row(mid, "all_pairs", "marginal_ace", ace, n_valid),
        _metric_row(mid, "all_pairs", "mop_picp_90", picp, n_valid),
        _metric_row(mid, "all_pairs", "mop_ace", ace, n_valid),
        _metric_row(mid, "all_pairs", "rmse", all_pair_rmse - 0.1, n_valid),
        _metric_row(mid, "all_pairs", "r2", 0.9, n_valid),
        _metric_row(mid, "ree_pairs", "rmse", rmse - 0.1, n_valid),
        _metric_row(mid, "ree_pairs", "r2", 0.9, n_valid),
    ]
    return rows


class TestOperationalWorkhorse(unittest.TestCase):
    def test_pareto_picks_closest_to_ideal(self):
        rows = []
        rows += _complete_model("DS2_RDKIT_MLP_MCDROPOUT", rmse=0.40, picp=0.86, ace=0.09, all_pair_rmse=1.0)
        rows += _complete_model("DS2_RDKIT_MLP_HETERO", rmse=0.50, picp=0.90, ace=0.02, all_pair_rmse=1.5)
        rows += _complete_model("DS2_RDKIT_MLP_POINT", rmse=0.30, picp=float("nan"), ace=float("nan"))
        rows += _complete_model("DS2_RDKIT_MLP_EVID", rmse=0.20, picp=0.70, ace=0.04, all_pair_rmse=0.8)
        choice = select_workhorse_accuracy_calibration(pd.DataFrame(rows))
        self.assertEqual(choice["selected"], "DS2_RDKIT_MLP_HETERO")
        self.assertEqual(choice["primary_accuracy_metric"], "all_pair_mop_rmse")
        self.assertEqual(set(choice["pareto_front"]), {"DS2_RDKIT_MLP_MCDROPOUT", "DS2_RDKIT_MLP_HETERO"})
        self.assertFalse(choice["row_level_metrics_used_for_selection"])
        self.assertFalse(choice["selection_nested"])
        self.assertFalse(choice["retraining_performed"])
        self.assertFalse(choice["composite_calibration_score_used"])
        by_id = {r["model_id"]: r for r in choice["rows"]}
        self.assertTrue(by_id["DS2_RDKIT_MLP_HETERO"]["pareto_member"])
        self.assertTrue(by_id["DS2_RDKIT_MLP_MCDROPOUT"]["pareto_member"])
        self.assertFalse(by_id["DS2_RDKIT_MLP_EVID"]["calibration_usable"])
        self.assertFalse(by_id["DS2_RDKIT_MLP_POINT"]["calibration_usable"])
        self.assertLess(by_id["DS2_RDKIT_MLP_HETERO"]["pareto_distance"], by_id["DS2_RDKIT_MLP_MCDROPOUT"]["pareto_distance"])

    def test_dominated_screen_passer_not_ranked(self):
        rows = []
        rows += _complete_model("DS2_RDKIT_MLP_MCDROPOUT", rmse=0.40, picp=0.90, ace=0.02, all_pair_rmse=1.0)
        rows += _complete_model("DS2_RDKIT_MLP_HETERO", rmse=0.50, picp=0.90, ace=0.05, all_pair_rmse=1.5)
        choice = select_workhorse_accuracy_calibration(pd.DataFrame(rows))
        self.assertEqual(choice["selected"], "DS2_RDKIT_MLP_MCDROPOUT")
        self.assertEqual(choice["pareto_front"], ["DS2_RDKIT_MLP_MCDROPOUT"])
        by_id = {r["model_id"]: r for r in choice["rows"]}
        self.assertEqual(by_id["DS2_RDKIT_MLP_HETERO"]["selection_status"], "dominated_screen_passer")
        self.assertTrue(math.isnan(by_id["DS2_RDKIT_MLP_HETERO"]["operational_rank"]))

    def test_selects_all_pair_rmse_not_ree_rmse(self):
        rows = []
        rows += _complete_model(
            "DS2_RDKIT_MLPENS5_EVID", rmse=0.30, picp=0.90, ace=0.05, all_pair_rmse=1.570
        )
        rows += _complete_model(
            "DS2_RDKIT_MLP_MCDROPOUT", rmse=0.46, picp=0.90, ace=0.05, all_pair_rmse=1.543
        )
        choice = select_workhorse_accuracy_calibration(pd.DataFrame(rows))
        self.assertEqual(choice["selected"], "DS2_RDKIT_MLP_MCDROPOUT")
        self.assertEqual(choice["primary_accuracy_metric"], "all_pair_mop_rmse")
        self.assertEqual(choice["secondary_usage_metric"], "ree_mop_rmse")

    def test_uses_unrounded_all_pair_rmse(self):
        rows = []
        rows += _complete_model(
            "DS2_RDKIT_MLPENS5_EVID", rmse=0.42, picp=0.90, ace=0.05, all_pair_rmse=1.5434
        )
        rows += _complete_model(
            "DS2_RDKIT_MLP_MCDROPOUT", rmse=0.46, picp=0.90, ace=0.05, all_pair_rmse=1.5431
        )
        choice = select_workhorse_accuracy_calibration(pd.DataFrame(rows))
        self.assertEqual(choice["selected"], "DS2_RDKIT_MLP_MCDROPOUT")

    def test_r2_and_sign_are_not_screen_keys(self):
        rows = []
        rows += _complete_model("DS2_RDKIT_MLPENS5_EVID", rmse=0.42, picp=0.90, ace=0.05, r2=0.05, sign=0.50)
        rows += _complete_model("DS2_RDKIT_MLP_HETERO", rmse=0.50, picp=0.90, ace=0.05, r2=0.90, sign=0.95)
        choice = select_workhorse_accuracy_calibration(pd.DataFrame(rows))
        self.assertEqual(choice["selected"], "DS2_RDKIT_MLPENS5_EVID")
        self.assertEqual(choice["r2_role"], "auxiliary_diagnostic")
        self.assertEqual(choice["sign_accuracy_role"], "directional_diagnostic")

    def test_row_level_metrics_cannot_enter_selection(self):
        rows = []
        rows += _complete_model("DS2_RDKIT_MLPENS5_EVID", rmse=0.50, picp=0.90, ace=0.05)
        rows += _complete_model("DS2_RDKIT_MLP_HETERO", rmse=0.40, picp=0.90, ace=0.05)
        df = pd.DataFrame(rows)
        df.loc[
            (df.model_id == "DS2_RDKIT_MLPENS5_EVID") & (df.metric_name == "rmse"),
            "fold_mean",
        ] = 0.01
        df.loc[
            (df.model_id == "DS2_RDKIT_MLP_HETERO") & (df.metric_name == "rmse"),
            "fold_mean",
        ] = 0.99
        choice = select_workhorse_accuracy_calibration(df)
        self.assertEqual(choice["selected"], "DS2_RDKIT_MLP_HETERO")
        self.assertFalse(choice["row_level_metrics_used_for_selection"])
        for r in choice["rows"]:
            self.assertNotIn("row_rmse_mean", r)

    def test_calibration_screen_uses_all_pair_picp_and_ace(self):
        rows = []
        rows += _complete_model("DS2_RDKIT_MLPENS5_EVID", rmse=0.42, picp=0.96, ace=0.02)
        rows += _complete_model("DS2_RDKIT_MLP_HETERO", rmse=0.50, picp=0.90, ace=0.02)
        choice = select_workhorse_accuracy_calibration(pd.DataFrame(rows))
        by_id = {r["model_id"]: r for r in choice["rows"]}
        self.assertFalse(by_id["DS2_RDKIT_MLPENS5_EVID"]["calibration_usable"])
        self.assertIn("picp_outside_operational_band", by_id["DS2_RDKIT_MLPENS5_EVID"]["calibration_exclusion_reason"])
        self.assertEqual(choice["selected"], "DS2_RDKIT_MLP_HETERO")
        self.assertEqual(choice["calibration_screen"]["picp_lower"], PICP_LOWER)
        self.assertEqual(choice["calibration_screen"]["picp_upper"], PICP_UPPER)
        self.assertEqual(choice["calibration_screen"]["ace_upper"], ACE_UPPER)

        rows2 = _complete_model("DS2_RDKIT_MLPENS5_MCDROPOUT", rmse=0.40, picp=0.89, ace=0.15)
        rows2 += _complete_model("DS2_RDKIT_MLP_HETERO", rmse=0.50, picp=0.90, ace=0.02)
        choice2 = select_workhorse_accuracy_calibration(pd.DataFrame(rows2))
        by2 = {r["model_id"]: r for r in choice2["rows"]}
        self.assertFalse(by2["DS2_RDKIT_MLPENS5_MCDROPOUT"]["calibration_usable"])
        self.assertIn("ace_above_operational_band", by2["DS2_RDKIT_MLPENS5_MCDROPOUT"]["calibration_exclusion_reason"])

    def test_incomplete_folds_fail_screen(self):
        rows = _complete_model("DS2_RDKIT_MLPENS5_EVID", rmse=0.42, picp=0.86, ace=0.05, n_valid=4)
        rows += _complete_model("DS2_RDKIT_MLP_HETERO", rmse=0.50, picp=0.90, ace=0.02)
        choice = select_workhorse_accuracy_calibration(pd.DataFrame(rows))
        by_id = {r["model_id"]: r for r in choice["rows"]}
        self.assertFalse(by_id["DS2_RDKIT_MLPENS5_EVID"]["calibration_usable"])
        self.assertEqual(choice["selected"], "DS2_RDKIT_MLP_HETERO")

    def test_chemprop_partial_uq_not_operational_candidate(self):
        rows = _complete_model(CHEMPROP_DS2, rmse=0.30, picp=0.90, ace=0.02)
        rows += _complete_model("DS2_RDKIT_MLPENS5_EVID", rmse=0.42, picp=0.86, ace=0.05)
        choice = select_workhorse_accuracy_calibration(pd.DataFrame(rows), chemprop_invalid_nig_rows=13)
        by_id = {r["model_id"]: r for r in choice["rows"]}
        self.assertEqual(by_id[CHEMPROP_DS2]["uq_status"], "partial_native_invalid_rows")
        self.assertFalse(by_id[CHEMPROP_DS2]["calibration_usable"])
        self.assertIn("incomplete_native_uq", by_id[CHEMPROP_DS2]["calibration_exclusion_reason"])
        self.assertEqual(choice["selected"], "DS2_RDKIT_MLPENS5_EVID")
        self.assertFalse(choice["chemprop_native_evidential_operational_candidate"])
        self.assertTrue(choice["chemprop_native_evidential_scientific_comparator"])

    def test_constant_target_pair_r2_invalid(self):
        g = pd.DataFrame({"y_true": [1.0, 1.0, 1.0], "y_pred": [0.8, 1.1, 0.9]})
        self.assertTrue(math.isnan(per_pair_metric(g, "r2")))
        g2 = pd.DataFrame({"y_true": [1.0, 2.0, 3.0], "y_pred": [1.0, 2.0, 3.0]})
        self.assertAlmostEqual(per_pair_metric(g2, "r2"), 1.0)

    def test_mop_rmse_is_mean_of_pairs_not_row_then_fold(self):
        from uq_remake.metrics.fold_metrics import mop_block, row_point_metrics

        df2 = pd.DataFrame(
            {
                "fold_id": ["fold0"] * 7,
                "evaluation_partition": ["heldout_fold"] * 7,
                "y_true": [0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0],
                "y_pred": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "metal_species_pair": ["A:1.0--B:1.0"] * 3 + ["C:1.0--D:1.0"] * 4,
                "ligand_smiles": [f"L{i}" for i in range(7)],
            }
        )
        mop2 = mop_block(df2)
        row = row_point_metrics(df2["y_true"], df2["y_pred"])
        self.assertAlmostEqual(mop2["mop_rmse"], 5.0)  # mean(0, 10)
        self.assertGreater(row["rmse"], mop2["mop_rmse"])
        self.assertNotAlmostEqual(mop2["mop_rmse"], row["rmse"])

    def test_composite_calibration_not_a_selection_key(self):
        rows = _complete_model("DS2_RDKIT_MLPENS5_EVID", rmse=0.42, picp=0.86, ace=0.05, all_pair_rmse=1.0)
        rows += _complete_model("DS2_RDKIT_MLP_HETERO", rmse=0.44, picp=0.90, ace=0.02, all_pair_rmse=1.5)
        choice = select_workhorse_accuracy_calibration(pd.DataFrame(rows))
        self.assertFalse(choice["composite_calibration_score_used"])
        self.assertTrue(choice["calibration_deviation_plot_is_visualization_aid_only"])
        self.assertIn("DS2_RDKIT_MLP_HETERO", choice["pareto_front"])
        self.assertIn("DS2_RDKIT_MLPENS5_EVID", choice["pareto_front"])

    def test_oof_column_rejected(self):
        df = pd.DataFrame(
            [
                {
                    "dataset": "DS2",
                    "model_id": "DS2_RDKIT_MLP_EVID",
                    "scope": "ree_pairs",
                    "metric_name": "mop_rmse",
                    "fold_mean": 0.4,
                    "n_valid_folds": 5,
                    "pooled_oof_value": 0.1,
                }
            ]
        )
        with self.assertRaises(OOFTerminologyError):
            select_workhorse_accuracy_calibration(df)

    def test_json_contract_fields(self):
        rows = _complete_model("DS2_RDKIT_MLPENS5_EVID", rmse=0.42, picp=0.86, ace=0.05)
        choice = select_workhorse_accuracy_calibration(pd.DataFrame(rows))
        for key in (
            "selection_type",
            "primary_accuracy_metric",
            "calibration_metrics",
            "calibration_screen",
            "pareto_objectives",
            "r2_role",
            "sign_accuracy_role",
            "row_level_metrics_used_for_selection",
            "retraining_performed",
            "selection_nested",
        ):
            self.assertIn(key, choice)
        self.assertEqual(choice["selection_type"], "operational_accuracy_calibration_compromise")
        self.assertIn("all_pair_picp_90", choice["calibration_metrics"])
        self.assertIn("all_pair_ace", choice["calibration_metrics"])
        row = choice["rows"][0]
        for key in (
            "model_id",
            "display_name",
            "ranking_eligibility",
            "uq_status",
            "n_valid_folds_accuracy",
            "n_valid_folds_picp",
            "n_valid_folds_ace",
            "all_pair_mop_rmse_mean",
            "all_pair_mop_rmse_sd",
            "all_pair_mop_r2_mean",
            "all_pair_mop_r2_sd",
            "all_pair_sign_accuracy_mean",
            "all_pair_sign_accuracy_sd",
            "ree_mop_rmse_mean",
            "ree_mop_rmse_sd",
            "ree_mop_r2_mean",
            "ree_mop_r2_sd",
            "ree_sign_accuracy_mean",
            "ree_sign_accuracy_sd",
            "all_pair_picp_90_mean",
            "all_pair_picp_90_sd",
            "all_pair_ace_mean",
            "all_pair_ace_sd",
            "all_pair_mop_picp_90_mean",
            "all_pair_mop_ace_mean",
            "pareto_member",
            "pareto_distance",
            "calibration_usable",
            "calibration_exclusion_reason",
            "operational_rank",
            "selection_status",
        ):
            self.assertIn(key, row)


if __name__ == "__main__":
    unittest.main()
