"""Downstream AD protocol tests. No pooled OOF headline metrics."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch  # noqa: F401  — before rdkit on Windows
except OSError as exc:  # WinError 1114 if RDKit already loaded in this process
    torch = None  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uq_remake.downstream.ad_scores import (  # noqa: E402
    density_score_from_count,
    fingerprint_library,
    fit_mahalanobis_ledoit,
    mahalanobis_distance,
    medium_indicators,
    pair_training_ligand_counts,
    tanimoto_nn_and_median,
)
from uq_remake.downstream.analyses import (  # noqa: E402
    deterministic_rank,
    retained_metrics,
    risk_coverage_fold,
    summarize_fold_metrics,
)
from uq_remake.downstream.protocol import (  # noqa: E402
    AD_RETENTION_DIR,
    AD_SCORES,
    CHEMPROP_ONLY_SCORES,
    DOWNSTREAM,
    FOLDS,
    RETENTION_FRACTIONS,
    WORKHORSE_PRIMARY_SCORES,
)
from uq_remake.metrics.cv_protocol import (  # noqa: E402
    EVALUATION_PARTITION,
    PooledFoldError,
    contains_oof_terminology,
    fold_mean_sd,
)
from uq_remake.metrics.fold_metrics import mop_block  # noqa: E402
from uq_remake.metrics.ligand_splits import nested_three_way_ligand_split  # noqa: E402


def _toy_fold_df(n=12, fold="fold0", seed=0):
    rng = np.random.default_rng(seed)
    pairs = ["Nd:3.0--Dy:3.0"] * 6 + ["La:3.0--Gd:3.0"] * 6
    return pd.DataFrame(
        {
            "record_id": [f"r{i}" for i in range(n)],
            "fold_id": fold,
            "outer_fold_id": fold,
            "evaluation_partition": EVALUATION_PARTITION,
            "y_true": rng.normal(0.2, 0.4, n),
            "y_pred": rng.normal(0.1, 0.4, n),
            "sigma_total": np.linspace(0.1, 1.2, n),
            "metal_species_pair": pairs[:n],
            "is_ree_pair": True,
            "ligand_smiles": [f"C{'C' * (i % 4)}" for i in range(n)],
        }
    )


class TestFoldLocalSplits(unittest.TestCase):
    def test_inner_outer_ligand_groups_disjoint(self):
        idx = np.arange(40)
        groups = np.repeat([f"L{i}" for i in range(20)], 2)
        inner, val, cal = nested_three_way_ligand_split(idx, groups, seed=3)
        gi = set(groups[inner])
        gv = set(groups[val])
        gc = set(groups[cal])
        self.assertEqual(len(gi & gv), 0)
        self.assertEqual(len(gi & gc), 0)
        self.assertEqual(len(gv & gc), 0)

    def test_summary_is_fold_mean_sd_not_pooled(self):
        vals = np.array([0.40, 0.42, 0.44, 0.46, 0.48])
        stats = fold_mean_sd(vals)
        self.assertAlmostEqual(stats["fold_mean"], float(np.mean(vals)))
        self.assertAlmostEqual(stats["fold_sd"], float(np.std(vals, ddof=1)))
        pooled = float(np.mean(np.repeat(vals, [10, 20, 5, 40, 3])))
        self.assertNotAlmostEqual(stats["fold_mean"], pooled)


class TestADDefinitions(unittest.TestCase):
    def test_tanimoto_distance_is_one_minus_nn_similarity_train_only(self):
        train = ["CCO", "CCC", "CCN"]
        test = ["CCO", "c1ccccc1"]
        ref, n_fail = fingerprint_library(train)
        self.assertEqual(n_fail, 0)
        d_nn, d_med, n_q = tanimoto_nn_and_median(np.array(test), ref, 2, 2048)
        self.assertEqual(n_q, 0)
        self.assertAlmostEqual(d_nn[0], 0.0, places=6)
        self.assertGreater(d_nn[1], d_nn[0])
        self.assertTrue(np.all(d_nn >= 0) and np.all(d_nn <= 1))
        self.assertTrue(np.all(d_med >= d_nn - 1e-12))
        self.assertNotIn("c1ccccc1", ref)

    def test_mahalanobis_outer_train_only_and_orientation(self):
        rng = np.random.default_rng(1)
        Xtr = rng.normal(0, 1, size=(80, 6))
        fit = fit_mahalanobis_ledoit(Xtr)
        self.assertEqual(fit["reference_row_count"], 80)
        self.assertEqual(fit["covariance_estimator"], "ledoit_wolf")
        d_center = mahalanobis_distance(fit, fit["mean"].reshape(1, -1))[0]
        d_far = mahalanobis_distance(fit, (fit["mean"] + 8).reshape(1, -1))[0]
        self.assertLess(d_center, d_far)
        self.assertEqual(fit["stability_flag"], "stable")
        Xte = rng.normal(5, 1, size=(10, 6))
        d_te = mahalanobis_distance(fit, Xte)
        fit_te = fit_mahalanobis_ledoit(Xte)
        d_wrong = mahalanobis_distance(fit_te, Xte)
        self.assertFalse(np.allclose(d_te, d_wrong))

    def test_pair_density_uses_training_counts_and_higher_is_risk(self):
        train = pd.DataFrame(
            {
                "metal_species_pair": ["Nd:3.0--Dy:3.0"] * 4 + ["Fe:2.0--Co:2.0"] * 2,
                "ligand_smiles": ["CCO", "CCO", "CCC", "CCN", "c1ccccc1", "c1ccccc1"],
            }
        )
        counts = pair_training_ligand_counts(train)
        self.assertEqual(counts["Nd:3.0--Dy:3.0"], 3)
        self.assertEqual(counts["Fe:2.0--Co:2.0"], 1)
        s = density_score_from_count(np.array([0, 1, 3, 50]))
        self.assertTrue(np.all(np.diff(s) < 0))
        self.assertAlmostEqual(s[0], 1.0)

    def test_empty_medium_stratum_is_non_identifiable_not_negative_result(self):
        train = pd.DataFrame(
            {
                "ligand_smiles": ["CCO", "CCC"],
                "electrolyte": ["NaNO3", "NaNO3"],
                "metal_species_pair": ["Nd:3.0--Dy:3.0"] * 2,
            }
        )
        test = pd.DataFrame(
            {
                "ligand_smiles": ["c1ccccc1", "CCN"],
                "electrolyte": ["NaNO3", "NaNO3"],
                "metal_species_pair": ["Nd:3.0--Dy:3.0"] * 2,
            }
        )
        med = medium_indicators(train, test)
        self.assertFalse(bool(med["ligand_medium_seen"].any()))
        n_seen = int(med["ligand_medium_seen"].sum())
        n_unseen = int((~med["ligand_medium_seen"].astype(bool)).sum())
        status = "identifiable" if n_seen > 0 and n_unseen > 0 else "non_identifiable_under_current_split"
        self.assertEqual(status, "non_identifiable_under_current_split")

    def test_species_pair_ids_remain_distinct(self):
        from uq_remake.downstream.ad_scores import assert_species_pairs

        self.assertEqual(assert_species_pairs(["Nd:3.0--Dy:3.0", "Fe:2.0--Co:2.0"]), 0)
        self.assertGreater(assert_species_pairs(["Nd--Dy", "Fe:2.0--Co:2.0"]), 0)

    def test_chemprop_scores_are_marked_chemprop_only(self):
        self.assertTrue(all(s.startswith("chemprop_") for s in CHEMPROP_ONLY_SCORES))
        self.assertTrue(set(WORKHORSE_PRIMARY_SCORES).isdisjoint(CHEMPROP_ONLY_SCORES))


class TestRiskCoverage(unittest.TestCase):
    def test_curves_are_fold_local_with_fixed_retention_grid(self):
        a = _toy_fold_df(fold="fold0", seed=0)
        b = _toy_fold_df(fold="fold1", seed=1)
        rows_a = risk_coverage_fold(a, "sigma_total", "fold0", "ad_retention")
        rows_b = risk_coverage_fold(b, "sigma_total", "fold1", "ad_retention")
        rets_a = sorted({r["retention_fraction"] for r in rows_a if r["metric_name"] == "rmse"})
        rets_b = sorted({r["retention_fraction"] for r in rows_b if r["metric_name"] == "rmse"})
        self.assertEqual(set(rets_a), set(RETENTION_FRACTIONS))
        self.assertEqual(set(rets_a), set(rets_b))
        self.assertTrue(all(r["outer_fold_id"] == "fold0" for r in rows_a))
        pooled = pd.concat([a, b], ignore_index=True)
        with self.assertRaises(PooledFoldError):
            mop_block(pooled)

    def test_tie_break_is_deterministic(self):
        df = _toy_fold_df()
        df["sigma_total"] = 0.5
        r1 = deterministic_rank(df, "sigma_total")
        r2 = deterministic_rank(df, "sigma_total")
        self.assertEqual(list(r1["record_id"]), list(r2["record_id"]))
        self.assertEqual(list(r1["record_id"]), sorted(df["record_id"].astype(str)))

    def test_mop_requires_pair_coverage(self):
        df = _toy_fold_df(n=4)
        df["metal_species_pair"] = ["Nd:3.0--Dy:3.0"] * 2 + ["La:3.0--Gd:3.0"] * 2
        met = retained_metrics(df)
        self.assertEqual(met["metric_status"], "insufficient_pair_coverage")
        self.assertTrue(np.isnan(met["mop_rmse"]))

    def test_aurc_is_per_fold_before_aggregation(self):
        rows = []
        for i, fold in enumerate(FOLDS):
            df = _toy_fold_df(fold=fold, seed=i)
            rows.extend(risk_coverage_fold(df, "sigma_total", fold, "ad_retention"))
        fold_df = pd.DataFrame(rows)
        aurc = fold_df[
            (fold_df["metric_name"] == "retention_curve_auc_rmse") & (fold_df["scope"] == "all_pairs")
        ]
        self.assertEqual(aurc["outer_fold_id"].nunique(), 5)
        summary = summarize_fold_metrics(
            aurc, ["dataset", "experiment_id", "scope", "score_name", "metric_name"]
        )
        self.assertEqual(int(summary.iloc[0]["n_folds_attempted"]), 5)
        self.assertIn("fold_mean", summary.columns)
        self.assertIn("fold_sd", summary.columns)

    def test_workhorse_primary_scores_are_frozen(self):
        self.assertEqual(
            WORKHORSE_PRIMARY_SCORES,
            ("sigma_total", "tanimoto_nn_distance", "mahalanobis_feature_centroid"),
        )
        self.assertIn("sigma_total", AD_SCORES)
        self.assertIn("tanimoto_nn_distance", AD_SCORES)
        self.assertIn(0.10, RETENTION_FRACTIONS)
        self.assertIn(0.50, RETENTION_FRACTIONS)
        self.assertIn(1.00, RETENTION_FRACTIONS)


class TestAggregation(unittest.TestCase):
    def test_ddof_one_and_missing_folds(self):
        stats = fold_mean_sd([1.0, 2.0, np.nan, 3.0])
        self.assertEqual(stats["n_valid_folds"], 3)
        self.assertAlmostEqual(stats["fold_sd"], float(np.std([1.0, 2.0, 3.0], ddof=1)))
        stats2 = fold_mean_sd([1.0])
        self.assertTrue(np.isnan(stats2["fold_sd"]))
        self.assertEqual(stats2["n_valid_folds"], 1)

    def test_summary_traces_to_fold_rows(self):
        fold_df = pd.DataFrame(
            {
                "dataset": "DS2",
                "experiment_id": "ad_retention",
                "outer_fold_id": FOLDS,
                "scope": "ree_pairs",
                "score_name": "sigma_total",
                "retention_fraction": 0.5,
                "metric_name": "rmse",
                "value": [0.41, 0.42, 0.43, 0.44, 0.45],
            }
        )
        summary = summarize_fold_metrics(
            fold_df,
            ["dataset", "experiment_id", "scope", "score_name", "retention_fraction", "metric_name"],
        )
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary.iloc[0]["fold_mean"], 0.43)
        self.assertEqual(int(summary.iloc[0]["n_folds_attempted"]), 5)
        self.assertEqual(int(summary.iloc[0]["n_valid_folds"]), 5)

    def test_no_oof_in_downstream_protocol_constants(self):
        self.assertFalse(contains_oof_terminology(EVALUATION_PARTITION))
        self.assertEqual(EVALUATION_PARTITION, "heldout_fold")


@unittest.skipUnless((AD_RETENTION_DIR / "retention_summary.csv").exists(), "AD retention artifacts not yet written")
class TestADRetentionArtifacts(unittest.TestCase):
    def test_required_files_and_no_oof(self):
        for rel in (
            "retention_summary.csv",
            "fold_retention.csv",
            "random_retention_summary.csv",
            "retention_50pct.csv",
            "enrichment_summary.csv",
            "leakage_audit.csv",
            "score_definitions.csv",
        ):
            self.assertTrue((AD_RETENTION_DIR / rel).exists(), rel)
        for path in AD_RETENTION_DIR.rglob("*.csv"):
            self.assertFalse(contains_oof_terminology(path.name), path.name)

    def test_distances_are_fold_local(self):
        leak = pd.read_csv(AD_RETENTION_DIR / "leakage_audit.csv")
        self.assertTrue((leak["fit_partition"] == "outer_train").all())
        self.assertTrue((~leak["tanimoto_heldout_in_reference"].astype(bool)).all())
        self.assertTrue((leak["status"] == "pass").all())

    def test_chemprop_scores_not_on_workhorse(self):
        fold = pd.read_csv(AD_RETENTION_DIR / "fold_retention.csv")
        work = fold[fold.family == "workhorse"]
        for score in CHEMPROP_ONLY_SCORES:
            self.assertFalse((work.score_name == score).any(), score)

    def test_every_curve_has_five_folds(self):
        fold = pd.read_csv(AD_RETENTION_DIR / "fold_retention.csv")
        rmse = fold[(fold.metric_name == "rmse") & (pd.to_numeric(fold.retention_fraction, errors="coerce") == 0.5)]
        for keys, g in rmse.groupby(["dataset", "model_id", "scope", "score_name"]):
            self.assertEqual(g["outer_fold_id"].nunique(), 5, keys)


if __name__ == "__main__":
    unittest.main()
