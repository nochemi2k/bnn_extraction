"""Sigma-only pair acquisition: eligibility, ranking, and no outcome leakage."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uq_remake.downstream.pair_acquisition import (
    ORIENT_CANONICAL,
    ORIENT_NEAR_ZERO,
    ORIENT_REVERSED,
    TASK_HIGH,
    TASK_LOW,
    assert_fold_hit_monotone,
    eligible_tasks,
    evaluate_regret,
    evaluate_topk,
    event_eligible,
    expected_random_metrics,
    hit_and_precision_wide,
    observed_event,
    pair_event_counts,
    rank_corr,
    rank_rows,
    ranking_score,
    target_observed,
)


def _rows(items: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(items)


class TestEligibility(unittest.TestCase):
    def test_event_count_threshold_is_strictly_greater_than_min_exclusive(self):
        self.assertFalse(event_eligible(1))
        self.assertTrue(event_eligible(2))

    def test_high_and_low_use_the_same_event_count_rule(self):
        y = np.array([1.2] * 11 + [-1.2] * 11 + [0.2] * 11 + [0.0] * 2)
        df = _rows(
            [
                {
                    "metal_species_pair": "A:3.0--B:3.0",
                    "y_true": v,
                    "ligand_smiles": f"L{i}",
                    "is_ree_pair": True,
                }
                for i, v in enumerate(y)
            ]
        )
        counts = pair_event_counts(df)
        self.assertTrue(bool(counts.iloc[0]["high_canonical_eligible"]))
        self.assertTrue(bool(counts.iloc[0]["high_reversed_eligible"]))
        self.assertTrue(bool(counts.iloc[0]["low_eligible"]))
        tasks = eligible_tasks(df)
        self.assertEqual(sorted(t["orientation"] for t in tasks), ["canonical", "near_zero", "reversed"])

    def test_boundary_equals_one_is_not_an_event(self):
        y = np.array([1.0, -1.0, 1.0000001, -1.0000001, 0.999])
        self.assertEqual(observed_event(y, TASK_HIGH, ORIENT_CANONICAL).tolist(), [False, False, True, False, False])
        self.assertEqual(observed_event(y, TASK_HIGH, ORIENT_REVERSED).tolist(), [False, False, False, True, False])
        self.assertEqual(observed_event(y, TASK_LOW, ORIENT_NEAR_ZERO).tolist(), [False, False, False, False, True])


class TestRanking(unittest.TestCase):
    def test_high_lcb_and_ei_are_not_identical_to_yhat(self):
        yhat = np.array([5.0, 3.0, 1.2])
        sigma = np.array([2.5, 0.2, 0.01])
        labels = np.array(["A", "B", "C"])
        orders = {}
        for strategy in ("y_pred", "lcb_1sigma", "lcb_2sigma", "lcb_3sigma", "expected_improvement"):
            q = ranking_score(yhat, sigma, strategy, TASK_HIGH, ORIENT_CANONICAL)
            orders[strategy] = labels[np.argsort(-q, kind="mergesort")].tolist()
        self.assertEqual(orders["y_pred"], ["A", "B", "C"])
        self.assertEqual(orders["lcb_1sigma"][0], "B")
        self.assertEqual(orders["expected_improvement"][0], "C")
        self.assertGreaterEqual(len({tuple(v) for v in orders.values()}), 3)

    def test_reverse_flips_high_sel_scores(self):
        yhat = np.array([2.0, -2.0])
        sigma = np.array([0.2, 0.2])
        q_c = ranking_score(yhat, sigma, "y_pred", TASK_HIGH, ORIENT_CANONICAL)
        q_r = ranking_score(yhat, sigma, "y_pred", TASK_HIGH, ORIENT_REVERSED)
        np.testing.assert_allclose(q_r, -yhat)
        self.assertGreater(q_c[0], q_c[1])
        self.assertGreater(q_r[1], q_r[0])
        ei_r = ranking_score(yhat, sigma, "expected_improvement", TASK_HIGH, ORIENT_REVERSED)
        np.testing.assert_allclose(ei_r, (-yhat - 1.0) / sigma)

    def test_low_sel_is_the_dual(self):
        yhat = np.array([0.2, 1.5, 0.0])
        sigma = np.array([0.1, 0.1, 0.4])
        q = ranking_score(yhat, sigma, "y_pred", TASK_LOW, ORIENT_NEAR_ZERO)
        np.testing.assert_allclose(q, -np.abs(yhat))
        lcb = ranking_score(yhat, sigma, "lcb_2sigma", TASK_LOW, ORIENT_NEAR_ZERO)
        np.testing.assert_allclose(lcb, -(np.abs(yhat) + 2.0 * sigma))
        ei = ranking_score(yhat, sigma, "expected_improvement", TASK_LOW, ORIENT_NEAR_ZERO)
        np.testing.assert_allclose(ei, (1.0 - np.abs(yhat)) / sigma)

    def test_ei_sigma_zero_has_no_epsilon_floor(self):
        ei = ranking_score(np.array([2.0]), np.array([0.0]), "expected_improvement", TASK_HIGH, ORIENT_CANONICAL)
        ei_fail = ranking_score(np.array([0.5]), np.array([0.0]), "expected_improvement", TASK_HIGH, ORIENT_CANONICAL)
        self.assertTrue(np.isposinf(ei[0]))
        self.assertTrue(np.isneginf(ei_fail[0]))

    def test_topk_uses_score_not_label_and_na_when_n_lt_k(self):
        df = _rows(
            [
                {"record_id": "a", "ligand_smiles": "A", "y_pred": 5.0, "sigma_total": 2.0, "y_true": 0.0},
                {"record_id": "b", "ligand_smiles": "B", "y_pred": 4.0, "sigma_total": 0.1, "y_true": 3.0},
                {"record_id": "c", "ligand_smiles": "C", "y_pred": 3.0, "sigma_total": 0.1, "y_true": 2.0},
            ]
        )
        ranked = rank_rows(df, "y_pred", TASK_HIGH, ORIENT_CANONICAL)
        self.assertEqual(ranked["ligand_smiles"].tolist()[:2], ["A", "B"])
        metrics = evaluate_topk(ranked, TASK_HIGH, ORIENT_CANONICAL, n_event_pool=2)
        self.assertEqual(metrics["n_hits_at_1"], 0)
        self.assertFalse(np.isfinite(metrics["n_hits_at_5"]))
        self.assertFalse(np.isfinite(metrics["n_hits_at_10"]))


class TestHitVersusPrecision(unittest.TestCase):
    def _ten_rows(self) -> pd.DataFrame:
        # Ranked by y_pred: a non-event first, then nine events.
        rows = [{"record_id": "a", "ligand_smiles": "A", "y_pred": 9.0, "sigma_total": 0.2, "y_true": 0.2}]
        for i in range(9):
            rows.append(
                {
                    "record_id": f"e{i}",
                    "ligand_smiles": f"E{i}",
                    "y_pred": 8.0 - i,
                    "sigma_total": 0.2,
                    "y_true": 1.5,
                }
            )
        return _rows(rows)

    def test_precision_is_fraction_and_hit_is_any(self):
        ranked = rank_rows(self._ten_rows(), "y_pred", TASK_HIGH, ORIENT_CANONICAL)
        self.assertEqual(ranked["ligand_smiles"].tolist()[0], "A")
        metrics = evaluate_topk(ranked, TASK_HIGH, ORIENT_CANONICAL, n_event_pool=9)
        self.assertAlmostEqual(metrics["hit_rate_at_1"], 0.0)
        self.assertAlmostEqual(metrics["event_precision_at_1"], 0.0)
        self.assertAlmostEqual(metrics["any_hit_at_1"], 0.0)
        self.assertAlmostEqual(metrics["hit_at_1"], 0.0)
        self.assertAlmostEqual(metrics["hit_rate_at_5"], 4 / 5)
        self.assertAlmostEqual(metrics["event_precision_at_5"], 4 / 5)
        self.assertAlmostEqual(metrics["any_hit_at_5"], 1.0)
        self.assertAlmostEqual(metrics["hit_at_5"], 1.0)
        self.assertAlmostEqual(metrics["hit_rate_at_10"], 9 / 10)
        self.assertAlmostEqual(metrics["any_hit_at_10"], 1.0)
        self.assertNotAlmostEqual(metrics["event_precision_at_5"], metrics["hit_at_5"])
        self.assertLessEqual(metrics["hit_at_1"], metrics["hit_at_5"])
        self.assertLessEqual(metrics["hit_at_5"], metrics["hit_at_10"])

    def test_hit_monotone_when_first_is_event(self):
        rows = []
        for i in range(10):
            rows.append(
                {
                    "record_id": f"r{i}",
                    "ligand_smiles": f"L{i}",
                    "y_pred": 10.0 - i,
                    "sigma_total": 0.2,
                    "y_true": 1.5 if i == 0 else 0.1,
                }
            )
        ranked = rank_rows(_rows(rows), "y_pred", TASK_HIGH, ORIENT_CANONICAL)
        metrics = evaluate_topk(ranked, TASK_HIGH, ORIENT_CANONICAL, n_event_pool=1)
        self.assertAlmostEqual(metrics["hit_at_1"], 1.0)
        self.assertAlmostEqual(metrics["hit_at_5"], 1.0)
        self.assertAlmostEqual(metrics["hit_at_10"], 1.0)
        self.assertAlmostEqual(metrics["event_precision_at_1"], 1.0)
        self.assertAlmostEqual(metrics["event_precision_at_5"], 1 / 5)

    def test_ranking_uses_score_not_heldout_label(self):
        df = _rows(
            [
                {"record_id": "a", "ligand_smiles": "A", "y_pred": 0.1, "sigma_total": 0.2, "y_true": 4.0},
                {"record_id": "b", "ligand_smiles": "B", "y_pred": 3.0, "sigma_total": 0.2, "y_true": 0.0},
            ]
        )
        ranked = rank_rows(df, "y_pred", TASK_HIGH, ORIENT_CANONICAL)
        self.assertEqual(ranked["ligand_smiles"].tolist(), ["B", "A"])

    def test_canonical_fold_metrics_are_monotone_if_present(self):
        path = REPO / "results" / "downstream_cv" / "pair_acquisition" / "pair_acquisition_fold_metrics.csv"
        if not path.exists():
            self.skipTest("canonical fold metrics missing")
        fold = pd.read_csv(path)
        assert_fold_hit_monotone(fold)

    def test_wide_table_maps_legacy_hit_rate_to_precision(self):
        summary = pd.DataFrame(
            [
                {
                    "task": "high_selectivity",
                    "orientation": "both",
                    "strategy": "y_pred",
                    "metric_name": "hit_rate",
                    "k": 1,
                    "fold_mean": 0.5,
                    "fold_sd": 0.1,
                    "n_tasks_fold_mean": 12.0,
                    "n_tasks_fold_sd": 1.0,
                    "n_valid_folds": 5,
                },
                {
                    "task": "high_selectivity",
                    "orientation": "both",
                    "strategy": "y_pred",
                    "metric_name": "any_hit",
                    "k": 1,
                    "fold_mean": 0.9,
                    "fold_sd": 0.05,
                    "n_tasks_fold_mean": 12.0,
                    "n_tasks_fold_sd": 1.0,
                    "n_valid_folds": 5,
                },
                {
                    "task": "high_selectivity",
                    "orientation": "both",
                    "strategy": "y_pred",
                    "metric_name": "n_tasks_defined",
                    "k": 1,
                    "fold_mean": 12.0,
                    "fold_sd": 1.0,
                    "n_tasks_fold_mean": 12.0,
                    "n_tasks_fold_sd": 1.0,
                    "n_valid_folds": 5,
                },
                {
                    "task": "high_selectivity",
                    "orientation": "both",
                    "strategy": "y_pred",
                    "metric_name": "n_tasks_defined",
                    "k": 5,
                    "fold_mean": 8.0,
                    "fold_sd": 0.5,
                    "n_tasks_fold_mean": 12.0,
                    "n_tasks_fold_sd": 1.0,
                    "n_valid_folds": 5,
                },
            ]
        )
        wide = hit_and_precision_wide(summary)
        self.assertEqual(len(wide), 1)
        self.assertAlmostEqual(wide.iloc[0]["event_precision_at_1"], 0.5)
        self.assertAlmostEqual(wide.iloc[0]["hit_at_1"], 0.9)
        self.assertNotAlmostEqual(wide.iloc[0]["event_precision_at_1"], wide.iloc[0]["hit_at_1"])
        self.assertAlmostEqual(wide.iloc[0]["n_tasks_defined_at_1"], 12.0)
        self.assertAlmostEqual(wide.iloc[0]["n_tasks_defined_at_5"], 8.0)


class TestRegret(unittest.TestCase):
    def test_high_selectivity_zero_when_best_selected(self):
        df = _rows(
            [
                {"record_id": "best", "ligand_smiles": "A", "y_pred": 5.0, "sigma_total": 0.2, "y_true": 4.0},
                {"record_id": "mid", "ligand_smiles": "B", "y_pred": 4.0, "sigma_total": 0.2, "y_true": 2.0},
                {"record_id": "low", "ligand_smiles": "C", "y_pred": 3.0, "sigma_total": 0.2, "y_true": 0.1},
            ]
        )
        ranked = rank_rows(df, "y_pred", TASK_HIGH, ORIENT_CANONICAL)
        metrics = evaluate_topk(ranked, TASK_HIGH, ORIENT_CANONICAL, n_event_pool=2)
        self.assertAlmostEqual(metrics["regret_at_1"], 0.0)
        self.assertAlmostEqual(metrics["selected_contains_optimum_at_1"], 1.0)
        self.assertAlmostEqual(metrics["pool_optimum"], 4.0)
        self.assertAlmostEqual(metrics["hit_rate_at_1"], 1.0)

    def test_high_selectivity_positive_when_best_missed(self):
        df = _rows(
            [
                {"record_id": "a", "ligand_smiles": "A", "y_pred": 9.0, "sigma_total": 0.2, "y_true": 0.2},
                {"record_id": "b", "ligand_smiles": "B", "y_pred": 8.0, "sigma_total": 0.2, "y_true": 3.0},
                {"record_id": "c", "ligand_smiles": "C", "y_pred": 7.0, "sigma_total": 0.2, "y_true": 1.5},
                {"record_id": "d", "ligand_smiles": "D", "y_pred": 6.0, "sigma_total": 0.2, "y_true": 0.4},
                {"record_id": "e", "ligand_smiles": "E", "y_pred": 5.0, "sigma_total": 0.2, "y_true": 0.3},
            ]
        )
        ranked = rank_rows(df, "y_pred", TASK_HIGH, ORIENT_CANONICAL)
        metrics = evaluate_topk(ranked, TASK_HIGH, ORIENT_CANONICAL, n_event_pool=2)
        self.assertAlmostEqual(metrics["regret_at_1"], 3.0 - 0.2)
        self.assertAlmostEqual(metrics["regret_at_5"], 0.0)
        self.assertGreaterEqual(metrics["regret_at_1"], -1e-12)
        self.assertAlmostEqual(metrics["any_hit_at_1"], 0.0)
        self.assertAlmostEqual(metrics["event_precision_at_1"], 0.0)

    def test_tied_optima_zero_if_either_selected(self):
        df = _rows(
            [
                {"record_id": "a", "ligand_smiles": "A", "y_pred": 1.0, "sigma_total": 0.2, "y_true": 4.0},
                {"record_id": "b", "ligand_smiles": "B", "y_pred": 5.0, "sigma_total": 0.2, "y_true": 4.0},
                {"record_id": "c", "ligand_smiles": "C", "y_pred": 4.0, "sigma_total": 0.2, "y_true": 1.0},
            ]
        )
        ranked = rank_rows(df, "y_pred", TASK_HIGH, ORIENT_CANONICAL)
        self.assertEqual(ranked["record_id"].tolist()[0], "b")
        metrics = evaluate_topk(ranked, TASK_HIGH, ORIENT_CANONICAL, n_event_pool=2)
        self.assertAlmostEqual(metrics["pool_optimum_tied"], 1.0)
        self.assertAlmostEqual(metrics["n_pool_optima"], 2.0)
        self.assertAlmostEqual(metrics["regret_at_1"], 0.0)
        self.assertAlmostEqual(metrics["selected_contains_optimum_at_1"], 1.0)

    def test_near_zero_uses_absolute_selectivity(self):
        df = _rows(
            [
                {"record_id": "a", "ligand_smiles": "A", "y_pred": 0.05, "sigma_total": 0.2, "y_true": 2.0},
                {"record_id": "b", "ligand_smiles": "B", "y_pred": 0.8, "sigma_total": 0.2, "y_true": 0.1},
                {"record_id": "c", "ligand_smiles": "C", "y_pred": 1.2, "sigma_total": 0.2, "y_true": 0.4},
            ]
        )
        ranked = rank_rows(df, "y_pred", TASK_LOW, ORIENT_NEAR_ZERO)
        self.assertEqual(ranked["record_id"].tolist()[0], "a")
        metrics = evaluate_topk(ranked, TASK_LOW, ORIENT_NEAR_ZERO, n_event_pool=2)
        self.assertAlmostEqual(metrics["pool_optimum"], 0.1)
        self.assertAlmostEqual(metrics["regret_at_1"], 2.0 - 0.1)
        self.assertFalse(np.isfinite(metrics["regret_at_5"]))
        rec = evaluate_regret(ranked, TASK_LOW, ORIENT_NEAR_ZERO)
        self.assertGreaterEqual(rec["regret_at_1"], -1e-12)

    def test_reversed_uses_target_direction(self):
        df = _rows(
            [
                {"record_id": "neg", "ligand_smiles": "N", "y_pred": -4.0, "sigma_total": 0.2, "y_true": -5.0},
                {"record_id": "pos", "ligand_smiles": "P", "y_pred": 3.0, "sigma_total": 0.2, "y_true": 4.0},
            ]
        )
        ranked = rank_rows(df, "y_pred", TASK_HIGH, ORIENT_REVERSED)
        self.assertEqual(ranked["record_id"].tolist()[0], "neg")
        metrics = evaluate_topk(ranked, TASK_HIGH, ORIENT_REVERSED, n_event_pool=1)
        self.assertAlmostEqual(metrics["pool_optimum"], 5.0)
        self.assertAlmostEqual(metrics["regret_at_1"], 0.0)

    def test_batch_is_subset_of_ranked_pool(self):
        df = _rows(
            [
                {"record_id": "a", "ligand_smiles": "A", "y_pred": 2.0, "sigma_total": 0.2, "y_true": 1.2},
                {"record_id": "b", "ligand_smiles": "B", "y_pred": 1.0, "sigma_total": 0.2, "y_true": 0.2},
            ]
        )
        ranked = rank_rows(df, "y_pred", TASK_HIGH, ORIENT_CANONICAL)
        evaluate_regret(ranked, TASK_HIGH, ORIENT_CANONICAL)


class TestDiagnostics(unittest.TestCase):
    def test_random_expectation_is_hypergeometric_rate(self):
        exp = expected_random_metrics(n_event=10, n_samples=40)
        self.assertAlmostEqual(exp["random_expected_hit_rate_at_5"], 0.25)
        self.assertAlmostEqual(exp["random_expected_recall_at_5"], 5 / 40)

    def test_rank_corr_monotone_and_undefined_for_k1(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(rank_corr(y, y), 1.0)
        self.assertAlmostEqual(rank_corr(-y, y), -1.0)
        self.assertTrue(np.isnan(rank_corr(np.array([1.0]), np.array([2.0]))))
        self.assertEqual(target_observed(np.array([-1.2]), ORIENT_REVERSED)[0], 1.2)


class TestEventThresholdSensitivity(unittest.TestCase):
    def test_high_threshold_two_excludes_values_between_one_and_two(self):
        y = np.array([1.2, 2.0, 2.0001, -2.0001])
        self.assertEqual(
            observed_event(y, TASK_HIGH, ORIENT_CANONICAL, high_threshold=2.0, low_threshold=0.5).tolist(),
            [False, False, True, False],
        )
        self.assertEqual(
            observed_event(y, TASK_HIGH, ORIENT_REVERSED, high_threshold=2.0, low_threshold=0.5).tolist(),
            [False, False, False, True],
        )

    def test_low_threshold_half_excludes_values_between_half_and_one(self):
        y = np.array([0.0, 0.49, 0.5, 0.9, 1.2])
        self.assertEqual(
            observed_event(y, TASK_LOW, ORIENT_NEAR_ZERO, high_threshold=2.0, low_threshold=0.5).tolist(),
            [True, True, False, False, False],
        )

    def test_default_threshold_is_unchanged(self):
        y = np.array([1.0000001, 0.999])
        self.assertEqual(observed_event(y, TASK_HIGH, ORIENT_CANONICAL).tolist(), [True, False])
        self.assertEqual(observed_event(y, TASK_LOW, ORIENT_NEAR_ZERO).tolist(), [False, True])

    def test_threshold_margin_uses_the_new_high_cutoff(self):
        yhat = np.array([1.5, 3.0])
        sigma = np.array([0.5, 0.5])
        q1 = ranking_score(yhat, sigma, "expected_improvement", TASK_HIGH, ORIENT_CANONICAL)
        q2 = ranking_score(
            yhat,
            sigma,
            "expected_improvement",
            TASK_HIGH,
            ORIENT_CANONICAL,
            high_threshold=2.0,
            low_threshold=0.5,
        )
        self.assertGreater(q1[0], 0.0)
        self.assertLess(q2[0], 0.0)
        self.assertGreater(q2[1], q2[0])

    def test_eligibility_counts_use_separate_high_and_low_cutoffs(self):
        y = np.array([2.2, 2.3, 1.2, -2.2, -2.3, 0.1, 0.2, 0.9])
        df = _rows(
            [
                {
                    "metal_species_pair": "A:3.0--B:3.0",
                    "y_true": v,
                    "ligand_smiles": f"L{i}",
                    "is_ree_pair": False,
                }
                for i, v in enumerate(y)
            ]
        )
        counts = pair_event_counts(df, high_threshold=2.0, low_threshold=0.5)
        self.assertEqual(int(counts.iloc[0]["n_high_canonical"]), 2)
        self.assertEqual(int(counts.iloc[0]["n_high_reversed"]), 2)
        self.assertEqual(int(counts.iloc[0]["n_low"]), 2)


if __name__ == "__main__":
    unittest.main()
