"""Unit tests for fold-local CV metrics. Pooled OOF is forbidden."""
from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uq_remake.metrics.bootstrap_fold import ligand_cluster_bootstrap, paired_ligand_bootstrap
from uq_remake.metrics.ligand_splits import nested_calibration_split
from uq_remake.metrics.cv_protocol import (
    ACE_LEVELS,
    EVALUATION_PARTITION,
    InvalidPairIdentifierError,
    OOFTerminologyError,
    PooledFoldError,
    archived_model_blocked,
    assert_no_oof_terminology,
    contains_oof_terminology,
    fold_mean_sd,
    scan_active_outputs_for_oof,
)
from uq_remake.metrics.fold_metrics import (
    apply_scope,
    evaluate_one_fold,
    filter_finite_rows,
    high_selectivity_frames,
    interval_metrics_fold,
    mop_block,
    mop_interval_block,
    per_pair_metric,
    row_point_metrics,
    sign_metrics,
)
from uq_remake.metrics.workhorse import select_workhorse_lexicographic

# Retired lexicographic keys used only to test archived-model exclusion.
_HISTORICAL_RANKING_KEYS = [
    ("ree_pairs", "mop_spearman", True),
    ("ree_pairs", "sign_accuracy_all", True),
    ("ree_pairs", "mop_mae", False),
    ("all_pairs", "mop_spearman", True),
    ("high_selectivity_row", "mop_spearman", True),
    ("high_selectivity_row", "mop_mae", False),
    ("ree_pairs", "marginal_ace", False),
    ("ree_pairs", "mae_retained", False),
]


def _fold_df(n_pairs=4, n_per=5, fold="fold0", seed=0, ox="2.0"):
    rng = np.random.default_rng(seed)
    rows = []
    metals = [("Fe", "Co"), ("Nd", "Dy"), ("La", "Gd"), ("Ni", "Zn")]
    for i, (a, b) in enumerate(metals[:n_pairs]):
        pair = f"{a}:{ox}--{b}:{ox}"
        ligand = f"L{i}"
        for j in range(n_per):
            y = float(i + 0.2 * j)
            rows.append(
                {
                    "fold_id": fold,
                    "evaluation_partition": EVALUATION_PARTITION,
                    "record_id": f"{i}_{j}",
                    "row_id": i * n_per + j,
                    "y_true": y,
                    "y_pred": y + 0.1 * rng.normal(),
                    "metal_species_pair": pair,
                    "ligand_smiles": ligand,
                    "is_ree_pair": a in {"Nd", "Dy", "La", "Gd"} or b in {"Nd", "Dy", "La", "Gd"},
                    "model_id": "DS2_RDKIT_MLP_EVID",
                    "dataset": "DS2",
                    "member_id": 0,
                    "uncertainty_status": "valid",
                    "sigma_total": 0.3 + 0.05 * j,
                    "gamma": y,
                    "nu": 8.0,
                    "alpha": 3.0,
                    "beta": 0.4,
                }
            )
    return pd.DataFrame(rows)


class TestGuards(unittest.TestCase):
    def test_pooled_frame_rejected(self):
        a = _fold_df(fold="fold0")
        b = _fold_df(fold="fold1", seed=1)
        pooled = pd.concat([a, b], ignore_index=True)
        with self.assertRaises(PooledFoldError):
            mop_block(pooled)

    def test_oof_partition_rejected(self):
        df = _fold_df()
        df["evaluation_partition"] = "OOF"
        with self.assertRaises((OOFTerminologyError, PooledFoldError)):
            mop_block(df)

    def test_element_only_pair_rejected(self):
        df = _fold_df()
        df["metal_species_pair"] = "Fe--Co"
        with self.assertRaises(InvalidPairIdentifierError):
            mop_block(df)

    def test_archived_models_blocked(self):
        self.assertTrue(archived_model_blocked("DS2_RDKIT_GPR_BAYES"))
        self.assertTrue(archived_model_blocked("DS2_RDKIT_SNGP_INCOMPLETE"))
        self.assertTrue(archived_model_blocked("DS1_GRAPH_DMPNN_BAYES"))
        self.assertTrue(archived_model_blocked("DS2_RDKIT_BAYES_LASTLAYER"))
        self.assertFalse(archived_model_blocked("DS2_GRAPH_DMPNN_EVID"))
        self.assertFalse(archived_model_blocked("DS2_RDKIT_MLP_MCDROPOUT"))
        self.assertFalse(archived_model_blocked("DS2_RDKIT_MLP_BAYES_FULL"))
        self.assertFalse(archived_model_blocked("DS2_RDKIT_SNGP"))
        self.assertFalse(archived_model_blocked("DS2_RDKIT_GPR_SPARSE"))


class TestFoldAggregation(unittest.TestCase):
    def test_fold_mean_sd_matches_formula(self):
        vals = [0.2, 0.4, 0.6, 0.8, 1.0]
        out = fold_mean_sd(vals)
        self.assertAlmostEqual(out["fold_mean"], float(np.mean(vals)))
        self.assertAlmostEqual(out["fold_sd"], float(np.std(vals, ddof=1)))
        self.assertEqual(out["n_folds"], 5)
        self.assertEqual(out["n_valid_folds"], 5)
        self.assertNotIn("pooled", out)


class TestMoP(unittest.TestCase):
    def test_mop_rmse_differs_from_row_rmse_on_unequal_pairs(self):
        df = pd.DataFrame(
            {
                "fold_id": ["fold0"] * 7,
                "evaluation_partition": [EVALUATION_PARTITION] * 7,
                "y_true": [0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0],
                "y_pred": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "metal_species_pair": ["A:1.0--B:1.0"] * 3 + ["C:1.0--D:1.0"] * 4,
                "ligand_smiles": [f"L{i}" for i in range(7)],
            }
        )
        mop = mop_block(df)
        row = row_point_metrics(df["y_true"], df["y_pred"])
        self.assertAlmostEqual(mop["mop_rmse"], 5.0)
        self.assertGreater(row["rmse"], mop["mop_rmse"])

    def test_oxidation_states_kept_separate(self):
        a = _fold_df(n_pairs=1, ox="2.0")
        b = _fold_df(n_pairs=1, ox="3.0", seed=2)
        b["fold_id"] = "fold0"
        df = pd.concat([a, b], ignore_index=True)
        mop = mop_block(df)
        self.assertEqual(mop["n_pairs_total"], 2)
        self.assertIn("Fe:2.0--Co:2.0", set(df["metal_species_pair"]))
        self.assertIn("Fe:3.0--Co:3.0", set(df["metal_species_pair"]))

    def test_constant_pair_r2_nan(self):
        df = _fold_df(n_pairs=1, n_per=4)
        df["y_true"] = 1.0
        df["y_pred"] = [1.0, 1.1, 0.9, 1.2]
        g = df
        self.assertTrue(np.isnan(per_pair_metric(g, "r2")))
        mop = mop_block(df)
        self.assertTrue(np.isnan(mop["mop_r2"]))
        self.assertEqual(mop["n_pairs_invalid"]["r2"], 1)
        self.assertEqual(mop["n_pairs_valid"]["rmse"], 1)


class TestSignAndFinite(unittest.TestCase):
    def test_exact_zero_sign_policy(self):
        yt = np.array([1.0, -1.0, 0.0, 0.0])
        yp = np.array([0.5, -0.2, 0.0, 0.3])
        s = sign_metrics(yt, yp)
        self.assertEqual(s["n_exact_zero_observed"], 2)
        # zeros: sign 0==0 True; 0==sign(0.3)=0? np.sign(0.3)=1 → False
        self.assertAlmostEqual(s["sign_accuracy_all"], 0.75)
        self.assertAlmostEqual(s["sign_accuracy_nonzero_observed"], 1.0)

    def test_nonfinite_counted(self):
        df = _fold_df(n_pairs=1, n_per=4)
        df.loc[0, "y_pred"] = np.nan
        used, diag = filter_finite_rows(df)
        self.assertEqual(diag["n_rows_total"], 4)
        self.assertEqual(diag["n_rows_invalid"], 1)
        self.assertEqual(diag["n_rows_used"], 3)
        self.assertEqual(len(used), 3)


class TestHighSelectivity(unittest.TestCase):
    def test_named_estimands_differ(self):
        df = _fold_df(n_pairs=2, n_per=6)
        df["y_true"] = [0.1, 0.2, 2.0, 2.1, 2.2, 2.3, 0.1, 0.2, 0.3, 0.4, 2.0, 2.1]
        hs = high_selectivity_frames(df, tau=1.0, min_n=3)
        self.assertIn("high_selectivity_row", hs)
        self.assertIn("high_selectivity_all_pair_membership", hs)
        self.assertNotEqual(len(hs["high_selectivity_row"]), len(hs["high_selectivity_all_pair_membership"]))


class TestIntervals(unittest.TestCase):
    def test_ace_nine_level_grid(self):
        self.assertEqual(ACE_LEVELS, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))

    def test_evidential_student_t_and_invalid_counts(self):
        df = _fold_df(n_pairs=2, n_per=4)
        df.loc[0, "alpha"] = 0.5  # invalid
        im = interval_metrics_fold(df, "DS2_RDKIT_MLP_EVID")
        self.assertEqual(im["interval_method"], "student_t_nig_predictive")
        self.assertGreater(im["n_invalid_intervals"], 0)
        self.assertEqual(im["interval_status"], "partial_invalid_rows")
        self.assertTrue(np.isfinite(im["marginal_picp_90"]))
        self.assertEqual(len(im["coverages"]), 9)

    def test_point_model_intervals_unavailable(self):
        df = _fold_df(n_pairs=1)
        df["uncertainty_status"] = "unavailable"
        im = interval_metrics_fold(df, "DS2_RDKIT_MLP_POINT")
        self.assertEqual(im["interval_method"], "unavailable")
        self.assertTrue(np.isnan(im["marginal_picp_90"]))

    def test_mop_interval_is_mean_of_pairs(self):
        df = _fold_df(n_pairs=2, n_per=4)
        mop = mop_interval_block(df, "DS2_RDKIT_MLPENS5_HETERO")
        self.assertEqual(mop["n_pairs_valid"]["picp_90"], 2)
        self.assertTrue(np.isfinite(mop["mop_picp_90"]))
        self.assertTrue(np.isfinite(mop["mop_ace"]))
        out = evaluate_one_fold(df, "DS2", "DS2_RDKIT_MLPENS5_HETERO")
        names = {r["metric_name"] for r in out["fold_metrics"] if r["scope"] == "all_pairs"}
        self.assertIn("mop_picp_90", names)
        self.assertIn("mop_ace", names)
        self.assertIn("marginal_picp_90", {r["metric_name"] for r in out["interval_metrics"]})


class TestLigandSplits(unittest.TestCase):
    def test_nested_split_disjoint(self):
        idx = np.arange(20)
        groups = np.repeat(np.arange(10), 2)
        inner, cal = nested_calibration_split(idx, groups, seed=0)
        self.assertEqual(len(set(inner) & set(cal)), 0)


class TestBootstrap(unittest.TestCase):
    def test_preserves_pair_ids_and_single_fold(self):
        df = _fold_df()

        def fn(part):
            self.assertEqual(part["fold_id"].nunique(), 1)
            self.assertIn("metal_species_pair", part.columns)
            return mop_block(part)["mop_rmse"]

        out = ligand_cluster_bootstrap(df, fn, n_boot=20, seed=1)
        self.assertEqual(out["bootstrap_unit"], "ligand")
        self.assertEqual(out["n_bootstrap_requested"], 20)
        self.assertNotIn("pooled", out["ci_method"])

    def test_paired_uses_same_clusters(self):
        a = _fold_df(seed=0)
        b = _fold_df(seed=1)
        b["y_pred"] = a["y_pred"] + 0.5
        seen = []

        def fn(part):
            seen.append(tuple(sorted(part["ligand_smiles"].astype(str).unique())))
            return float(np.mean(np.abs(part["y_true"] - part["y_pred"])))

        paired_ligand_bootstrap(a, b, fn, n_boot=5, seed=4)
        # fn called twice per replicate (a then b); ligand sets should match pairwise
        self.assertGreaterEqual(len(seen), 2)
        for i in range(0, len(seen) - 1, 2):
            self.assertEqual(seen[i], seen[i + 1])


class TestWorkhorse(unittest.TestCase):
    def test_archived_cannot_enter(self):
        rows = []
        for mid, elig in [("DS2_RDKIT_MLP_EVID", True), ("DS2_RDKIT_GPR_BAYES", True)]:
            for fold in range(5):
                rows.append(
                    {
                        "dataset": "DS2",
                        "model_id": mid,
                        "scope": "ree_pairs",
                        "metric_name": "mop_mae",
                        "fold_mean": 0.9 if "EVID" in mid else 0.99,
                        "fold_sd": 0.01,
                        "n_folds": 5,
                        "n_valid_folds": 5,
                    }
                )
        summary = pd.DataFrame(rows)
        # Need all lookup metrics; fill remaining with NaN rows for EVID only completeness
        for scope, metric in [
            ("ree_pairs", "sign_accuracy_all"),
            ("ree_pairs", "mop_mae"),
            ("all_pairs", "mop_mae"),
            ("high_selectivity_row", "mop_mae"),
            ("high_selectivity_row", "mop_mae"),
            ("ree_pairs", "marginal_ace"),
            ("ree_pairs", "mae_retained"),
        ]:
            summary = pd.concat(
                [
                    summary,
                    pd.DataFrame(
                        [
                            {
                                "dataset": "DS2",
                                "model_id": "DS2_RDKIT_MLP_EVID",
                                "scope": scope,
                                "metric_name": metric,
                                "fold_mean": 0.5,
                                "fold_sd": 0.01,
                                "n_folds": 5,
                                "n_valid_folds": 5,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        choice = select_workhorse_lexicographic(summary, _HISTORICAL_RANKING_KEYS)
        self.assertEqual(choice["selected"], "DS2_RDKIT_MLP_EVID")
        self.assertTrue(any("GPR" in s["model_id"] for s in choice["skipped"]))

    def test_oof_column_rejected(self):
        summary = pd.DataFrame(
            [
                {
                    "dataset": "DS2",
                    "model_id": "DS2_RDKIT_MLP_EVID",
                    "scope": "ree_pairs",
                    "metric_name": "mop_mae",
                    "fold_mean": 0.5,
                    "n_valid_folds": 5,
                    "pooled_oof_value": 0.9,
                }
            ]
        )
        with self.assertRaises(OOFTerminologyError):
            select_workhorse_lexicographic(summary, _HISTORICAL_RANKING_KEYS)


class TestEvaluateOneFoldNoPool(unittest.TestCase):
    def test_evaluate_one_fold_emits_heldout_partition(self):
        df = _fold_df()
        out = evaluate_one_fold(df, "DS2", "DS2_RDKIT_MLP_EVID")
        self.assertTrue(out["fold_metrics"])
        self.assertTrue(all(r["evaluation_partition"] == EVALUATION_PARTITION for r in out["fold_metrics"]))
        self.assertTrue(all(r["fold_id"] == "fold0" for r in out["fold_metrics"]))
        names = {r["metric_name"] for r in out["fold_metrics"]}
        self.assertIn("mop_rmse", names)
        self.assertIn("rmse", names)
        self.assertIn("r2", names)
        self.assertIn("sign_accuracy_all", names)
        self.assertNotIn("mae", names)
        self.assertNotIn("mop_mae", names)
        self.assertNotIn("pair_balanced_global_r2", names)
        self.assertNotIn("mop_spearman", names)
        self.assertNotIn("sign_accuracy_nonzero_observed", names)

    def test_oof_word_detected(self):
        self.assertTrue(contains_oof_terminology("pooled_oof_metrics"))
        self.assertTrue(contains_oof_terminology("OOF"))
        self.assertFalse(contains_oof_terminology("fold_test_predictions"))


class TestOOFFileScan(unittest.TestCase):
    def test_scan_finds_oof_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bad = root / "fold_metrics.csv"
            bad.write_text("dataset,metric\nDS2,oof_mae\n", encoding="utf-8")
            hits = scan_active_outputs_for_oof((root,))
            self.assertTrue(hits)


if __name__ == "__main__":
    unittest.main()
