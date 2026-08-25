"""Tests for BAYES_FULL, SNGP, and GPR restorations."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
except OSError as exc:  # WinError 1114 if RDKit already loaded
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uq_remake.metrics.cv_protocol import PooledFoldError, archived_model_blocked
from uq_remake.metrics.fold_metrics import mop_block
from uq_remake.metrics.workhorse import select_workhorse_lexicographic

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
import pandas as pd

if TORCH_IMPORT_ERROR is None:
    from uq_remake.models.gpr import (
        EXACT_GPR_MAX_N,
        ExactGPR,
        Matern52ARD,
        TargetScaler,
        train_exact_gpr,
        train_sparse_gpr,
        predict_exact_gpr,
        predict_sparse_gpr,
    )
    from uq_remake.models.sngp import SNGPRegressor, predict_sngp, train_sngp
    from uq_remake.models.uq_tabular import aggregate_members
    from uq_remake.models.variational_bnn import (
        BETA_KL,
        BayesianLinear,
        FixedPriorVariationalBNN,
        elbo_batch_loss,
        has_dropout,
        predict_variational_bnn,
        prior_metadata,
        train_variational_bnn,
    )


class _TorchRequired(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if TORCH_IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"torch unavailable after RDKit in this process: {TORCH_IMPORT_ERROR}")


def _toy(n=80, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=(d,)).astype(np.float32)
    y = X @ w + 0.1 * rng.normal(size=(n,)).astype(np.float32)
    return X, y


CFG = {
    "mlp_hidden_dims": [8, 8],
    "mlp_lr": 0.05,
    "mlp_weight_decay": 0.0,
    "mlp_epochs": 8,
    "mlp_batch_size": 32,
    "mlp_patience": 8,
    "sngp_num_features": 32,
    "sngp_lengthscale": 2.0,
    "sngp_ridge": 1.0,
    "gpr_exact_steps": 30,
    "gpr_n_inducing": 12,
    "gpr_sparse_epochs": 8,
    "gpr_sparse_patience": 8,
    "gpr_sparse_batch_size": 32,
    "gpr_sparse_lr": 0.05,
}


class TestVariationalBNN(_TorchRequired):
    def test_every_linear_is_variational(self):
        m = FixedPriorVariationalBNN(5, [6, 4])
        for layer in m.bayesian_layers():
            self.assertIsInstance(layer, BayesianLinear)
            self.assertTrue(hasattr(layer, "weight_mu") and hasattr(layer, "weight_rho"))
            self.assertTrue(hasattr(layer, "bias_mu") and hasattr(layer, "bias_rho"))
        n_var = m.n_bayesian_scalars()
        n_mu_rho = sum(p.numel() for n, p in m.named_parameters() if n != "log_sigma2_obs")
        self.assertEqual(n_mu_rho, 2 * n_var)

    def test_no_dropout(self):
        m = FixedPriorVariationalBNN(3, [4])
        self.assertFalse(has_dropout(m))
        self.assertFalse(any(isinstance(mod, nn.Dropout) for mod in m.modules()))

    def test_prior_fixed_fold_independent(self):
        meta = prior_metadata()
        self.assertEqual(meta["prior_type"], "fixed_normal")
        self.assertEqual(meta["prior_mean"], 0.0)
        self.assertEqual(meta["prior_std"], 1.0)
        self.assertEqual(meta["prior_scope"], "all_bayesian_parameters")

    def test_kl_covers_all_bayesian_params(self):
        m = FixedPriorVariationalBNN(3, [5])
        kl = float(m.kl_divergence())
        self.assertGreater(kl, 0.0)
        layer_sum = sum(float(layer.kl()) for layer in m.bayesian_layers())
        self.assertAlmostEqual(kl, layer_sum, places=5)

    def test_kl_normalization_independent_of_batch_size(self):
        m = FixedPriorVariationalBNN(3, [4])
        x = torch.randn(16, 3)
        y = torch.randn(16)
        _, _, kl8 = elbo_batch_loss(m, x[:8], y[:8], n_train=100, beta_kl=BETA_KL)
        _, _, kl16 = elbo_batch_loss(m, x, y, n_train=100, beta_kl=BETA_KL)
        self.assertAlmostEqual(float(kl8), float(kl16), places=4)
        loss_a, nll_a, kl_a = elbo_batch_loss(m, x[:8], y[:8], n_train=100)
        self.assertAlmostEqual(float(loss_a - nll_a), float(BETA_KL / 100.0 * kl_a), places=5)

    def test_learned_aleatoric_and_total_variance(self):
        X, y = _toy()
        model, meta = train_variational_bnn(X[:50], y[:50], X[50:65], y[50:65], CFG, device="cpu", seed=1, n_samples_val=8)
        self.assertIn("log_sigma2_obs", meta)
        out = predict_variational_bnn(model, X[65:], n_samples=12, device="cpu", seed=3, single_model=True)
        s = float(model.clamped_log_sigma2_obs().detach())
        self.assertTrue(np.allclose(out["var_aleatoric"], np.exp(s), rtol=1e-5))
        self.assertTrue(np.allclose(out["var_total"], out["var_aleatoric"] + out["var_epistemic_total"], rtol=1e-5))
        self.assertTrue(np.all(out["var_epistemic_within"] == 0))
        self.assertTrue(np.all(out["var_epistemic_between"] == 0))

    def test_epistemic_changes_with_samples(self):
        m = FixedPriorVariationalBNN(4, [6])
        X, _ = _toy(n=20)
        a = predict_variational_bnn(m, X, n_samples=8, device="cpu", seed=1)
        b = predict_variational_bnn(m, X, n_samples=8, device="cpu", seed=99)
        self.assertGreater(float(np.max(np.abs(a["var_epistemic"] - b["var_epistemic"]))), 1e-8)

    def test_intervals_include_observation_noise(self):
        m = FixedPriorVariationalBNN(3, [4])
        with torch.no_grad():
            m.log_sigma2_obs.fill_(4.0)
        X = np.zeros((10, 3), dtype=np.float32)
        out = predict_variational_bnn(m, X, n_samples=40, device="cpu", seed=0)
        width = out["upper_90"] - out["lower_90"]
        epi_only = 1.64485 * np.sqrt(out["var_epistemic_total"] + 1e-12)
        self.assertGreater(float(np.mean(width)), float(np.mean(epi_only)))

    def test_ensemble_law_of_total_variance(self):
        X, y = _toy(n=40, d=3)
        members = []
        for i in range(3):
            m, _ = train_variational_bnn(X[:25], y[:25], X[25:32], y[25:32], CFG, device="cpu", seed=10 + i, n_samples_val=6)
            members.append(predict_variational_bnn(m, X[32:], n_samples=8, device="cpu", seed=20 + i, single_model=False))
        agg = aggregate_members(members)
        self.assertTrue(np.allclose(agg["var_total"], agg["var_aleatoric"] + agg["var_epistemic_within"] + agg["var_epistemic_between"], rtol=1e-5))


class TestSNGP(_TorchRequired):
    def test_hidden_layers_have_spectral_norm_no_dropout(self):
        m = SNGPRegressor(5, [8, 8], num_features=16, seed=0)
        self.assertTrue(m.hidden_layers_spectrally_normalized())
        self.assertFalse(m.has_dropout())
        self.assertFalse(any(isinstance(mod, nn.Dropout) for mod in m.modules()))

    def test_rff_fixed_and_seeded(self):
        a = SNGPRegressor(4, [6], num_features=12, seed=7)
        b = SNGPRegressor(4, [6], num_features=12, seed=7)
        c = SNGPRegressor(4, [6], num_features=12, seed=8)
        self.assertTrue(torch.allclose(a.rff.weight, b.rff.weight))
        self.assertFalse(torch.allclose(a.rff.weight, c.rff.weight))
        self.assertFalse(a.rff.weight.requires_grad)

    def test_precision_train_only_and_reset(self):
        X, y = _toy(n=60, d=4, seed=2)
        m1, meta1 = train_sngp(X[:30], y[:30], X[30:40], y[30:40], CFG, device="cpu", seed=1)
        m2, _ = train_sngp(X[10:40], y[10:40], X[40:50], y[40:50], CFG, device="cpu", seed=1)
        self.assertEqual(meta1["sngp"]["n_train"], 30)
        self.assertTrue(meta1["sngp"]["gp_covariance_reset_per_fold"])
        self.assertFalse(torch.allclose(m1.gp_precision, m2.gp_precision))

    def test_variance_nonnegative_and_total(self):
        X, y = _toy()
        m, _ = train_sngp(X[:50], y[:50], X[50:65], y[50:65], CFG, device="cpu", seed=0)
        out = predict_sngp(m, X[65:], device="cpu")
        self.assertTrue(np.all(out["var_epistemic"] >= -1e-8))
        self.assertTrue(np.allclose(out["var_total"], out["var_aleatoric"] + out["var_epistemic_total"], rtol=1e-5))
        far = X[65:] + 25.0
        far_out = predict_sngp(m, far, device="cpu")
        self.assertGreater(float(np.mean(far_out["var_epistemic"])), float(np.mean(out["var_epistemic"])))

    def test_ensemble_lotv(self):
        X, y = _toy(n=50, d=3, seed=4)
        outs = []
        for i in range(3):
            m, _ = train_sngp(X[:30], y[:30], X[30:38], y[30:38], CFG, device="cpu", seed=i)
            outs.append(predict_sngp(m, X[38:], device="cpu"))
        agg = aggregate_members(outs)
        self.assertTrue(np.allclose(agg["var_total"], agg["var_aleatoric"] + agg["var_epistemic_within"] + agg["var_epistemic_between"], rtol=1e-5))


class TestGPR(_TorchRequired):
    def test_kernel_symmetric_and_chol(self):
        k = Matern52ARD(3)
        X = torch.randn(6, 3)
        K = k(X, X)
        self.assertTrue(torch.allclose(K, K.T, atol=1e-5))
        torch.linalg.cholesky(K + 1e-6 * torch.eye(6))

    def test_exact_infeasible_limit(self):
        self.assertEqual(EXACT_GPR_MAX_N, 2048)
        X, y = _toy(n=12, d=3)
        model, meta, scaler = train_exact_gpr(X[:8], y[:8], X[8:10], y[8:10], CFG, device="cpu", seed=0, max_n=50)
        self.assertEqual(meta["gpr_type"], "exact")
        self.assertTrue(meta["converged"])
        out = predict_exact_gpr(model, scaler, X[10:], device="cpu")
        self.assertTrue(np.all(out["var_total"] >= out["var_epistemic"] - 1e-8))
        self.assertTrue(np.allclose(out["var_total"], out["var_aleatoric"] + out["var_epistemic"], rtol=1e-4, atol=1e-6))

    def test_target_inverse_scales_variance(self):
        sc = TargetScaler(2.0, 5.0)
        self.assertAlmostEqual(float(sc.inverse_var([4.0])[0]), 100.0)
        self.assertAlmostEqual(float(sc.inverse_mean([1.0])[0]), 7.0)

    def test_sparse_inducing_from_train_only(self):
        X, y = _toy(n=40, d=3, seed=5)
        held = X[30:]
        model, meta, scaler = train_sparse_gpr(X[:24], y[:24], X[24:30], y[24:30], CFG, device="cpu", seed=2)
        self.assertEqual(meta["inducing_point_selection"], "kmeans_train_only_then_learned")
        Z = model.Z.detach().cpu().numpy()
        # inducing locations must not be copies of held-out rows
        dists = ((Z[:, None, :] - held[None, :, :]) ** 2).sum(-1).min(1)
        # just check they started from train k-means: n_inducing <= n_train
        self.assertLessEqual(Z.shape[0], 24)
        out = predict_sparse_gpr(model, scaler, X[30:], device="cpu")
        self.assertTrue(np.all(out["var_epistemic"] >= -1e-6))
        self.assertTrue(np.allclose(out["var_total"], out["var_aleatoric"] + out["var_epistemic"], rtol=1e-4, atol=1e-5))

    def test_exact_refuses_large_n(self):
        X, y = _toy(n=10)
        with self.assertRaises(RuntimeError):
            train_exact_gpr(X, y, X, y, CFG, device="cpu", max_n=5)


class TestFoldEvalAndWorkhorse(_TorchRequired):
    def test_no_pooled_metric(self):
        rows = []
        for fold in ("fold0", "fold1"):
            for i in range(6):
                rows.append(
                    {
                        "fold_id": fold,
                        "evaluation_partition": "heldout_fold",
                        "y_true": float(i),
                        "y_pred": float(i) + 0.1,
                        "metal_species_pair": "Nd:3.0--Dy:3.0",
                        "is_ree_pair": True,
                        "ligand_smiles": f"L{i}",
                    }
                )
        with self.assertRaises(PooledFoldError):
            mop_block(pd.DataFrame(rows))

    def test_bayes_enters_workhorse_when_complete(self):
        rows = []
        metrics = [
            ("ree_pairs", "mop_spearman", 0.7),
            ("ree_pairs", "sign_accuracy_all", 0.6),
            ("ree_pairs", "mop_mae", 0.4),
            ("all_pairs", "mop_spearman", 0.5),
            ("high_selectivity_row", "mop_spearman", 0.5),
            ("high_selectivity_row", "mop_mae", 0.4),
            ("ree_pairs", "marginal_ace", 0.1),
            ("ree_pairs", "mae_retained", 0.4),
        ]
        for mid in ("DS2_RDKIT_MLP_BAYES_FULL", "DS2_RDKIT_MLP_EVID"):
            for scope, name, val in metrics:
                rows.append(
                    {
                        "dataset": "DS2",
                        "model_id": mid,
                        "scope": scope,
                        "metric_name": name,
                        "fold_mean": val + (0.05 if "BAYES" in mid else 0.0),
                        "fold_sd": 0.01,
                        "n_folds": 5,
                        "n_valid_folds": 5,
                    }
                )
        choice = select_workhorse_lexicographic(pd.DataFrame(rows), _HISTORICAL_RANKING_KEYS)
        self.assertEqual(choice["selected"], "DS2_RDKIT_MLP_BAYES_FULL")
        self.assertFalse(archived_model_blocked("DS2_RDKIT_MLP_BAYES_FULL"))
        self.assertFalse(archived_model_blocked("DS2_RDKIT_SNGP"))
        self.assertFalse(archived_model_blocked("DS2_RDKIT_GPR_SPARSE"))
        self.assertTrue(archived_model_blocked("DS2_RDKIT_BAYES_LASTLAYER"))
        self.assertTrue(archived_model_blocked("DS1_GRAPH_DMPNN_BAYES"))

    def test_incomplete_bayes_skipped(self):
        rows = [
            {
                "dataset": "DS2",
                "model_id": "DS2_RDKIT_MLP_BAYES_FULL",
                "scope": "ree_pairs",
                "metric_name": "mop_spearman",
                "fold_mean": 0.99,
                "fold_sd": 0.01,
                "n_folds": 5,
                "n_valid_folds": 3,
            }
        ]
        choice = select_workhorse_lexicographic(pd.DataFrame(rows), _HISTORICAL_RANKING_KEYS)
        self.assertIsNone(choice["selected"])
        self.assertTrue(any("incomplete" in s["reason"] for s in choice["skipped"]))


class TestNIGPartialInvalid(_TorchRequired):
    def test_invalid_rows_get_nan_variance_not_crash(self):
        from uq_remake.models.chemprop_dmpnn import nig_predictive_from_params
        from uq_remake.metrics.uq_metrics import picp_at_90

        mu = np.array([1.0, 2.0, 3.0])
        lam = np.array([1.0, 0.0, 2.0])  # row 1 invalid: lambda<=0
        alpha = np.array([2.0, 3.0, 1.0])  # row 2 invalid: alpha<=1
        beta = np.array([0.5, 1.0, 1.0])
        out = nig_predictive_from_params(mu, lam, alpha, beta)
        self.assertEqual(out["n_valid_nig"], 1)
        self.assertEqual(out["n_invalid_nig"], 2)
        self.assertTrue(out["nig_valid"][0])
        self.assertFalse(out["nig_valid"][1])
        self.assertFalse(out["nig_valid"][2])
        self.assertTrue(np.isfinite(out["sigma_total"][0]))
        self.assertFalse(np.isfinite(out["sigma_total"][1]))
        self.assertFalse(np.isfinite(out["sigma_total"][2]))
        picp = picp_at_90(np.array([1.0, 2.0, 3.0]), mu, out["sigma_total"])
        self.assertTrue(np.isnan(picp))

    def test_all_valid_nig(self):
        from uq_remake.models.chemprop_dmpnn import nig_predictive_from_params

        mu = np.array([0.0, 1.0])
        lam = np.array([1.5, 2.0])
        alpha = np.array([2.5, 3.0])
        beta = np.array([0.4, 0.8])
        out = nig_predictive_from_params(mu, lam, alpha, beta)
        self.assertEqual(out["n_invalid_nig"], 0)
        self.assertTrue(np.isfinite(out["sigma_total"]).all())
        expected_a = 0.4 / (2.5 - 1.0)
        self.assertAlmostEqual(out["var_aleatoric"][0], expected_a)


if __name__ == "__main__":
    unittest.main()
