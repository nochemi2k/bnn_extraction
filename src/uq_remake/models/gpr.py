"""Exact and sparse variational Gaussian-process regression.

GPR is a kernel GP with no neural feature-learning backbone.
Exact GPR uses Cholesky of Ky. If N_train exceeds the predeclared limit,
train GPR_SPARSE (SVGP) instead of silently swapping IDs.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from uq_remake.models.uq_tabular import VAR_FLOOR, VARIANCE_DDOF

LOG_2PI = math.log(2.0 * math.pi)
SOFTPLUS_EPS = 1e-6
GPR_JITTER = 1e-6
EXACT_GPR_MAX_N = 2048
SQRT5 = math.sqrt(5.0)


def _softplus_pos(raw: torch.Tensor) -> torch.Tensor:
    return F.softplus(raw) + SOFTPLUS_EPS


class TargetScaler:
    def __init__(self, mean: float, scale: float):
        self.mean = float(mean)
        self.scale = float(scale) if abs(scale) > 1e-12 else 1.0

    @classmethod
    def fit(cls, y) -> "TargetScaler":
        y = np.asarray(y, dtype=float).reshape(-1)
        return cls(float(np.mean(y)), float(np.std(y, ddof=0) or 1.0))

    def transform(self, y) -> np.ndarray:
        return (np.asarray(y, dtype=float).reshape(-1) - self.mean) / self.scale

    def inverse_mean(self, mu) -> np.ndarray:
        return np.asarray(mu, dtype=float) * self.scale + self.mean

    def inverse_var(self, var) -> np.ndarray:
        return np.asarray(var, dtype=float) * (self.scale ** 2)

    def hash_payload(self) -> dict:
        return {
            "target_scaler_mean": self.mean,
            "target_scaler_scale": self.scale,
            "uncertainty_units": "original_target_units",
        }


class Matern52ARD(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.rho_lengthscale = nn.Parameter(torch.zeros(int(n_features)))
        self.rho_outputscale = nn.Parameter(torch.zeros(()))

    def lengthscale(self) -> torch.Tensor:
        return _softplus_pos(self.rho_lengthscale)

    def outputscale(self) -> torch.Tensor:
        return _softplus_pos(self.rho_outputscale)

    def scaled_dist(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        ell = self.lengthscale()
        xs = X / ell
        ys = Y / ell
        xx = (xs * xs).sum(-1, keepdim=True)
        yy = (ys * ys).sum(-1, keepdim=True).transpose(-2, -1)
        d2 = (xx + yy - 2.0 * xs @ ys.transpose(-2, -1)).clamp_min(0.0)
        return torch.sqrt(d2 + 1e-12)

    def forward(self, X: torch.Tensor, Y: torch.Tensor | None = None) -> torch.Tensor:
        Y = X if Y is None else Y
        r = self.scaled_dist(X, Y)
        sf2 = self.outputscale()
        return sf2 * (1.0 + SQRT5 * r + (5.0 / 3.0) * r.square()) * torch.exp(-SQRT5 * r)

    def diag(self, X: torch.Tensor) -> torch.Tensor:
        return self.outputscale().expand(X.shape[0])


def _uq_from_latent(mu, var_e, var_n, *, z=1.6448536269514722) -> dict:
    mu = np.asarray(mu, dtype=float).reshape(-1)
    var_e = np.maximum(np.asarray(var_e, dtype=float).reshape(-1), 0.0)
    var_a = np.maximum(np.asarray(var_n, dtype=float).reshape(-1), VAR_FLOOR)
    if var_a.size == 1:
        var_a = np.full(mu.shape, float(var_a[0]))
    n = len(mu)
    var_t = var_a + var_e
    sig = np.sqrt(np.maximum(var_t, 0.0))
    return {
        "mu": mu,
        "var_aleatoric": var_a,
        "var_epistemic": var_e,
        "var_epistemic_within": var_e,
        "var_epistemic_between": np.zeros(n, dtype=float),
        "var_epistemic_total": var_e,
        "var_total": var_t,
        "sigma_aleatoric": np.sqrt(var_a),
        "sigma_epistemic": np.sqrt(np.maximum(var_e, 0.0)),
        "sigma_epistemic_within": np.sqrt(np.maximum(var_e, 0.0)),
        "sigma_epistemic_between": np.zeros(n, dtype=float),
        "sigma_epistemic_total": np.sqrt(np.maximum(var_e, 0.0)),
        "sigma_total": sig,
        "std_total": sig,
        "lower_90": mu - z * sig,
        "upper_90": mu + z * sig,
        "variance_ddof": VARIANCE_DDOF,
        "interval_method": "gaussian_predictive_moment",
        "n_posterior_samples": 0,
    }


class ExactGPR(nn.Module):
    """Exact GP with ARD Matérn-5/2 kernel. O(N^3); refuse large N in the trainer."""

    def __init__(self, n_features: int, jitter: float = GPR_JITTER):
        super().__init__()
        self.kernel = Matern52ARD(n_features)
        self.rho_noise = nn.Parameter(torch.tensor(-1.0))
        self.jitter = float(jitter)
        self.register_buffer("X_train", torch.zeros(1, n_features))
        self.register_buffer("y_train", torch.zeros(1))
        self._fitted = False

    def noise_var(self) -> torch.Tensor:
        return _softplus_pos(self.rho_noise)

    def kernel_matrix(self, X: torch.Tensor, Y: torch.Tensor | None = None) -> torch.Tensor:
        return self.kernel(X, Y)

    def nll(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = X.shape[0]
        K = self.kernel(X, X) + (self.noise_var() + self.jitter) * torch.eye(n, device=X.device, dtype=X.dtype)
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze(-1)
        logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
        return 0.5 * (y @ alpha + logdet + n * LOG_2PI)

    def set_train_data(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.X_train = X.detach().contiguous()
        self.y_train = y.detach().contiguous()
        self._fitted = True

    def predict_latent(self, Xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        X = self.X_train
        y = self.y_train
        n = X.shape[0]
        K = self.kernel(X, X) + (self.noise_var() + self.jitter) * torch.eye(n, device=X.device, dtype=X.dtype)
        L = torch.linalg.cholesky(K)
        Ks = self.kernel(X, Xs)
        alpha = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze(-1)
        mu = Ks.transpose(0, 1) @ alpha
        v = torch.cholesky_solve(Ks, L)
        kss = self.kernel.diag(Xs)
        var_e = (kss - (Ks * v).sum(0)).clamp_min(0.0)
        return mu, var_e


def train_exact_gpr(Xtr, ytr, Xva, yva, cfg, device=None, seed: int = 0, max_n: int = EXACT_GPR_MAX_N):
    n = int(len(Xtr))
    if n > int(max_n):
        raise RuntimeError(
            f"exact GPR is computationally infeasible for n_train={n} > exact_gpr_max_n={max_n}. "
            "Use train_sparse_gpr / GPR_SPARSE."
        )
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(seed))
    scaler = TargetScaler.fit(ytr)
    ytr_s = torch.tensor(scaler.transform(ytr), dtype=torch.float32, device=device)
    yva_s = torch.tensor(scaler.transform(yva), dtype=torch.float32, device=device)
    Xtr_t = torch.tensor(np.asarray(Xtr), dtype=torch.float32, device=device)
    Xva_t = torch.tensor(np.asarray(Xva), dtype=torch.float32, device=device)
    model = ExactGPR(Xtr_t.shape[1], jitter=float(cfg.get("gpr_jitter", GPR_JITTER))).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    best = None
    best_val = float("inf")
    t0 = time.perf_counter()
    steps = int(cfg.get("gpr_exact_steps", 200))
    last = 0
    for step in range(steps):
        last = step + 1
        model.train()
        opt.zero_grad()
        loss = model.nll(Xtr_t, ytr_s)
        loss.backward()
        opt.step()
        with torch.no_grad():
            model.set_train_data(Xtr_t, ytr_s)
            mu_v, ve_v = model.predict_latent(Xva_t)
            sn = model.noise_var()
            val = float((0.5 * (torch.log(2.0 * math.pi * sn) + ((yva_s - mu_v).square() + ve_v) / sn)).mean().item())
        if val < best_val:
            best_val = val
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best:
        model.load_state_dict(best)
    model.set_train_data(Xtr_t, ytr_s)
    with torch.no_grad():
        converged = True
        try:
            _ = torch.linalg.cholesky(
                model.kernel(Xtr_t, Xtr_t)
                + (model.noise_var() + model.jitter) * torch.eye(n, device=device, dtype=Xtr_t.dtype)
            )
        except RuntimeError:
            converged = False
    meta = {
        "gpr_type": "exact",
        "kernel_name": "matern52_ard",
        "noise_variance_scaled": float(model.noise_var().detach().cpu()),
        "jitter": model.jitter,
        "optimization_steps": last,
        "converged": converged,
        "best_val": best_val,
        "train_seconds": time.perf_counter() - t0,
        "n_train": n,
        "seed": int(seed),
        **scaler.hash_payload(),
        "lengthscale": model.kernel.lengthscale().detach().cpu().tolist(),
        "output_scale": float(model.kernel.outputscale().detach().cpu()),
        "pca": False,
        "imputation": False,
    }
    return model, meta, scaler


@torch.no_grad()
def predict_exact_gpr(model: ExactGPR, scaler: TargetScaler, X, device=None) -> dict:
    device = device or next(model.parameters()).device
    model = model.to(device).eval()
    xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    mu_s, var_e_s = model.predict_latent(xt)
    var_n_s = float(model.noise_var().item())
    mu = scaler.inverse_mean(mu_s.cpu().numpy())
    var_e = scaler.inverse_var(var_e_s.cpu().numpy())
    var_n = scaler.inverse_var(np.array([var_n_s]))
    out = _uq_from_latent(mu, var_e, var_n)
    out["uncertainty_method"] = "exact_gp_posterior"
    out["kernel_name"] = "matern52_ard"
    return out


def _kmeans_inducing(X: np.ndarray, m: int, seed: int) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans

    n = len(X)
    m = min(int(m), n)
    km = MiniBatchKMeans(n_clusters=m, random_state=int(seed), n_init=1, batch_size=min(2048, n))
    km.fit(np.asarray(X, dtype=np.float64))
    return km.cluster_centers_.astype(np.float32)


class SparseGPR(nn.Module):
    """SVGP with Matérn-5/2 ARD. Inducing points initialized from training k-means only."""

    def __init__(self, Z: torch.Tensor, jitter: float = GPR_JITTER):
        super().__init__()
        m, d = Z.shape
        self.kernel = Matern52ARD(d)
        self.rho_noise = nn.Parameter(torch.tensor(-1.0))
        self.Z = nn.Parameter(Z.clone())
        self.m_u = nn.Parameter(torch.zeros(m))
        raw = torch.zeros(m, m)
        raw.fill_diagonal_(math.log(math.expm1(0.1)))
        self.raw_L = nn.Parameter(raw)
        self.jitter = float(jitter)
        self.n_inducing = int(m)

    def noise_var(self) -> torch.Tensor:
        return _softplus_pos(self.rho_noise)

    def tril(self) -> torch.Tensor:
        L = torch.tril(self.raw_L)
        diag = _softplus_pos(torch.diagonal(L, 0))
        return L - torch.diag_embed(torch.diagonal(L, 0)) + torch.diag_embed(diag)

    def _kuu(self) -> tuple[torch.Tensor, torch.Tensor]:
        Z = self.Z
        m = Z.shape[0]
        Kuu = self.kernel(Z, Z) + self.jitter * torch.eye(m, device=Z.device, dtype=Z.dtype)
        Luu = torch.linalg.cholesky(Kuu)
        return Kuu, Luu

    def predictive(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        Kuu, Luu = self._kuu()
        Kuf = self.kernel(self.Z, X)
        Lq = self.tril()
        S = Lq @ Lq.transpose(-2, -1)
        alpha = torch.cholesky_solve(self.m_u.unsqueeze(-1), Luu).squeeze(-1)
        mu = Kuf.transpose(0, 1) @ alpha
        W = torch.cholesky_solve(Kuf, Luu)
        quad_k = (Kuf * W).sum(0)
        quad_s = (W * (S @ W)).sum(0)
        kff = self.kernel.diag(X)
        var = (kff - quad_k + quad_s).clamp_min(0.0)
        return mu, var

    def elbo_batch(self, X: torch.Tensor, y: torch.Tensor, n_train: int) -> torch.Tensor:
        mu, var_f = self.predictive(X)
        sn = self.noise_var()
        expected_nll = 0.5 * (torch.log(2.0 * math.pi * sn) + ((y - mu).square() + var_f) / sn)
        # KL(q(u)||p(u)), p=N(0,Kuu)
        Kuu, Luu = self._kuu()
        Lq = self.tril()
        m = self.m_u
        k = m.shape[0]
        # log|Kuu| = 2 sum log diag Luu; log|S| = 2 sum log diag Lq
        logdet_p = 2.0 * torch.log(torch.diagonal(Luu)).sum()
        logdet_q = 2.0 * torch.log(torch.diagonal(Lq)).sum()
        invK_m = torch.cholesky_solve(m.unsqueeze(-1), Luu).squeeze(-1)
        quad = m @ invK_m
        W = torch.linalg.solve_triangular(Luu, Lq, upper=False)
        tr = (W * W).sum()
        kl = 0.5 * (tr + quad - k + logdet_p - logdet_q)
        b = X.shape[0]
        return (n_train / b) * expected_nll.sum() + kl


def train_sparse_gpr(Xtr, ytr, Xva, yva, cfg, device=None, seed: int = 0):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    scaler = TargetScaler.fit(ytr)
    ytr_s = scaler.transform(ytr)
    yva_s = scaler.transform(yva)
    m = int(cfg.get("gpr_n_inducing", 512))
    Z = _kmeans_inducing(np.asarray(Xtr, dtype=np.float32), m, seed)
    model = SparseGPR(torch.tensor(Z, dtype=torch.float32), jitter=float(cfg.get("gpr_jitter", GPR_JITTER))).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get("gpr_sparse_lr", 0.01)))
    n_train = int(len(Xtr))
    loader = DataLoader(
        TensorDataset(torch.tensor(np.asarray(Xtr), dtype=torch.float32), torch.tensor(ytr_s, dtype=torch.float32)),
        batch_size=int(cfg.get("gpr_sparse_batch_size", 1024)),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    Xv = torch.tensor(np.asarray(Xva), dtype=torch.float32, device=device)
    yv = torch.tensor(yva_s, dtype=torch.float32, device=device)
    best, best_val, stale = None, float("inf"), 0
    t0 = time.perf_counter()
    epochs = int(cfg.get("gpr_sparse_epochs", 80))
    patience = int(cfg.get("gpr_sparse_patience", 15))
    last = 0
    for epoch in range(epochs):
        last = epoch + 1
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            model.elbo_batch(xb, yb, n_train=n_train).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            mu, var_f = model.predictive(Xv)
            sn = model.noise_var()
            val = float((0.5 * (torch.log(2.0 * math.pi * sn) + ((yv - mu).square() + var_f) / sn)).mean().item())
        if val < best_val - 1e-6:
            best_val = val
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best:
        model.load_state_dict(best)
    meta = {
        "gpr_type": "sparse_variational",
        "kernel_name": "matern52_ard",
        "n_inducing": int(model.n_inducing),
        "inducing_point_selection": "kmeans_train_only_then_learned",
        "inducing_seed": int(seed),
        "variational_family": "univariate_gaussian_full_rank_tril",
        "noise_variance_scaled": float(model.noise_var().detach().cpu()),
        "jitter": model.jitter,
        "optimization_steps": last,
        "converged": stale < patience,
        "best_val": best_val,
        "train_seconds": time.perf_counter() - t0,
        "n_train": n_train,
        "seed": int(seed),
        **scaler.hash_payload(),
        "lengthscale": model.kernel.lengthscale().detach().cpu().tolist(),
        "output_scale": float(model.kernel.outputscale().detach().cpu()),
        "pca": False,
        "imputation": False,
    }
    return model, meta, scaler


@torch.no_grad()
def predict_sparse_gpr(model: SparseGPR, scaler: TargetScaler, X, device=None) -> dict:
    device = device or next(model.parameters()).device
    model = model.to(device).eval()
    xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    mus, ves = [], []
    bs = 1024
    for i in range(0, len(xt), bs):
        mu, ve = model.predictive(xt[i : i + bs])
        mus.append(mu.cpu())
        ves.append(ve.cpu())
    mu_s = torch.cat(mus).numpy()
    var_e_s = torch.cat(ves).numpy()
    var_n_s = float(model.noise_var().item())
    out = _uq_from_latent(scaler.inverse_mean(mu_s), scaler.inverse_var(var_e_s), scaler.inverse_var(np.array([var_n_s])))
    out["uncertainty_method"] = "sparse_variational_gp"
    out["kernel_name"] = "matern52_ard"
    return out


def save_gpr(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def load_gpr(path: Path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return torch.load(path, map_location=device, weights_only=False)
