"""Spectral-normalized Neural Gaussian Process (SNGP) for tabular regression.

Not a full Bayesian neural network: the backbone is point-estimated under
spectral normalization; uncertainty comes from a GP posterior over random
Fourier feature readout weights plus learned homoscedastic observation noise.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.data import DataLoader, TensorDataset

from uq_remake.models.uq_tabular import VAR_FLOOR, VARIANCE_DDOF

LOG_2PI = math.log(2.0 * math.pi)
LOG_SIGMA2_OBS_MIN = -12.0
LOG_SIGMA2_OBS_MAX = 8.0
SPECTRAL_NORM_POWER_ITERATIONS = 1
GP_PRECISION_JITTER = 1e-6


def _sn_linear(in_f: int, out_f: int) -> nn.Module:
    return spectral_norm(nn.Linear(in_f, out_f), n_power_iterations=SPECTRAL_NORM_POWER_ITERATIONS)


def _gaussian_obs_nll(y: torch.Tensor, mu: torch.Tensor, log_sigma2: torch.Tensor) -> torch.Tensor:
    s = log_sigma2.expand_as(y) if log_sigma2.ndim == 0 else log_sigma2
    return 0.5 * (torch.exp(-s) * (y - mu).square() + s + LOG_2PI)


class SpectralMLPTrunk(nn.Module):
    """MLP trunk with spectral-normalized Linear layers. Dropout is forbidden."""

    def __init__(self, in_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        self.sn_layer_names: list[str] = []
        for i, h in enumerate(hidden_dims):
            layers += [_sn_linear(prev, h), nn.ReLU()]
            self.sn_layer_names.append(f"hidden_{i}")
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def hidden_linears(self) -> list[nn.Module]:
        return [m for m in self.net.modules() if isinstance(m, nn.Linear)]


class RandomFourierFeatures(nn.Module):
    """Fixed random Fourier features approximating an RBF kernel. Frequencies are not trained."""

    def __init__(self, in_dim: int, num_features: int = 512, lengthscale: float = 2.0, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        w = torch.randn(in_dim, num_features, generator=g) / float(lengthscale)
        b = 2 * math.pi * torch.rand(num_features, generator=g)
        self.register_buffer("weight", w)
        self.register_buffer("bias", b)
        self.num_features = int(num_features)
        self.lengthscale = float(lengthscale)
        self.seed = int(seed)
        self.scale = float(math.sqrt(2.0 / num_features))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.cos(h @ self.weight + self.bias)


class SNGPRegressor(nn.Module):
    """
    Spectral-normalized trunk + frozen RFF map + Bayesian linear GP head.

    Training uses a learnable linear readout on RFFs (mean network) and Gaussian NLL.
    After training, ``fit_gp_posterior`` uses training rows only.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        num_features: int = 512,
        lengthscale: float = 2.0,
        ridge: float = 1.0,
        seed: int = 0,
    ):
        super().__init__()
        self.trunk = SpectralMLPTrunk(in_dim, list(hidden_dims))
        self.rff = RandomFourierFeatures(self.trunk.out_dim, num_features, lengthscale, seed=seed)
        self.beta = nn.Linear(num_features, 1, bias=False)
        self.log_sigma2_obs = nn.Parameter(torch.zeros(()))
        self.ridge = float(ridge)
        self.num_features = int(num_features)
        self.in_dim = int(in_dim)
        self.hidden_dims = list(hidden_dims)
        self.register_buffer("gp_mean", torch.zeros(num_features))
        self.register_buffer("gp_precision", torch.eye(num_features))
        self.register_buffer("gp_cov", torch.eye(num_features))
        self.register_buffer("gp_fitted", torch.tensor(0, dtype=torch.int8))

    def clamped_log_sigma2(self) -> torch.Tensor:
        return self.log_sigma2_obs.clamp(LOG_SIGMA2_OBS_MIN, LOG_SIGMA2_OBS_MAX)

    def obs_var(self) -> torch.Tensor:
        return torch.exp(self.clamped_log_sigma2())

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)

    def phi(self, x: torch.Tensor) -> torch.Tensor:
        return self.rff(self.trunk(x))

    def forward_train(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        phi = self.phi(x)
        mu = self.beta(phi).squeeze(-1)
        return mu, self.clamped_log_sigma2()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phi = self.phi(x)
        if int(self.gp_fitted.item()) == 1:
            mu = (phi @ self.gp_mean).squeeze(-1)
            tmp = phi @ self.gp_cov
            var_epi = (tmp * phi).sum(dim=1).clamp_min(0.0)
        else:
            mu = self.beta(phi).squeeze(-1)
            var_epi = torch.zeros_like(mu)
        var_ale = self.obs_var().expand_as(mu)
        return mu, var_ale, var_epi

    def has_dropout(self) -> bool:
        return any(isinstance(m, nn.Dropout) for m in self.modules())

    def hidden_layers_spectrally_normalized(self) -> bool:
        linears = self.trunk.hidden_linears()
        if not linears:
            return False
        for lin in linears:
            if not hasattr(lin, "parametrizations") or "weight" not in lin.parametrizations:
                return False
        return True

    @torch.no_grad()
    def fit_gp_posterior(self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor, device=None):
        """Exact ridge-GP posterior on RFF features of the trained representation. Training rows only."""
        device = device or next(self.parameters()).device
        self.eval()
        xt = torch.as_tensor(X, dtype=torch.float32, device=device)
        yt = torch.as_tensor(y, dtype=torch.float32, device=device).reshape(-1)
        phis = []
        bs = 2048
        for i in range(0, len(xt), bs):
            phis.append(self.phi(xt[i : i + bs]))
        phi = torch.cat(phis, dim=0)
        d = phi.shape[1]
        noise_var = float(self.obs_var().item())
        jitter = GP_PRECISION_JITTER
        precision = self.ridge * torch.eye(d, device=device, dtype=phi.dtype)
        precision = precision + (phi.T @ phi) / noise_var
        precision = precision + jitter * torch.eye(d, device=device, dtype=phi.dtype)
        rhs = (phi.T @ yt) / noise_var
        try:
            scale_tril = torch.linalg.cholesky(precision)
            cov = torch.cholesky_inverse(scale_tril)
            mean = torch.cholesky_solve(rhs.unsqueeze(-1), scale_tril).squeeze(-1)
        except RuntimeError:
            cov = torch.linalg.pinv(precision)
            mean = cov @ rhs
        self.gp_mean.copy_(mean)
        self.gp_precision.copy_(precision)
        self.gp_cov.copy_(cov)
        self.gp_fitted.fill_(1)
        self.beta.weight.copy_(mean.view(1, -1))
        return {
            "noise_var": noise_var,
            "ridge": self.ridge,
            "n_train": int(len(yt)),
            "gp_precision_jitter": jitter,
            "gp_covariance_update_mode": "full_train_recompute",
            "gp_covariance_reset_per_fold": True,
        }


def train_sngp(Xtr, ytr, Xva, yva, cfg, device=None, seed: int = 0):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    hidden = list(cfg.get("sngp_hidden_dims") or cfg["mlp_hidden_dims"])
    model = SNGPRegressor(
        in_dim=Xtr.shape[1],
        hidden_dims=hidden,
        num_features=int(cfg.get("sngp_num_features", 512)),
        lengthscale=float(cfg.get("sngp_lengthscale", 2.0)),
        ridge=float(cfg.get("sngp_ridge", 1.0)),
        seed=seed,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["mlp_lr"]), weight_decay=float(cfg["mlp_weight_decay"]))
    loader = DataLoader(
        TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)),
        batch_size=int(cfg["mlp_batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    Xv = torch.tensor(Xva, dtype=torch.float32, device=device)
    yv = torch.tensor(yva, dtype=torch.float32, device=device)
    best_state, best_val, stale = None, float("inf"), 0
    t0 = time.perf_counter()
    epoch = 0
    for epoch in range(int(cfg["mlp_epochs"])):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            mu, s = model.forward_train(xb)
            _gaussian_obs_nll(yb, mu, s).mean().backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            mu_v, s_v = model.forward_train(Xv)
            val_loss = float(_gaussian_obs_nll(yv, mu_v, s_v).mean().item())
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg["mlp_patience"]):
                break
    if best_state:
        model.load_state_dict(best_state)
    gp_meta = model.fit_gp_posterior(Xtr, ytr, device=device)
    meta = {
        "best_val": best_val,
        "train_seconds": time.perf_counter() - t0,
        "epochs": epoch + 1,
        "sngp": gp_meta,
        "num_features": model.num_features,
        "in_dim": int(Xtr.shape[1]),
        "hidden_dims": hidden,
        "ridge": model.ridge,
        "seed": int(seed),
        "spectral_normalization": True,
        "spectral_norm_layers": model.trunk.sn_layer_names,
        "spectral_norm_power_iterations": SPECTRAL_NORM_POWER_ITERATIONS,
        "rff_kernel": "rbf",
        "rff_lengthscale": model.rff.lengthscale,
        "rff_output_dim": model.num_features,
        "rff_seed": model.rff.seed,
        "rff_frequency_distribution": "normal_0_inv_lengthscale",
        "gp_prior_precision": model.ridge,
        "sngp_objective": "gaussian_nll",
        "log_sigma2_obs": float(model.clamped_log_sigma2().detach().cpu()),
        "log_sigma2_obs_min": LOG_SIGMA2_OBS_MIN,
        "log_sigma2_obs_max": LOG_SIGMA2_OBS_MAX,
        "dropout": 0.0,
        "pca": False,
        "imputation": False,
    }
    return model, meta


@torch.no_grad()
def predict_sngp(model: SNGPRegressor, X, device=None, single_model: bool = True) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    mus, ales, epis = [], [], []
    bs = 2048
    for i in range(0, len(xt), bs):
        mu, ale, epi = model(xt[i : i + bs])
        mus.append(mu.cpu())
        ales.append(ale.cpu())
        epis.append(epi.cpu())
    mu = torch.cat(mus).numpy()
    var_a = np.maximum(torch.cat(ales).numpy(), VAR_FLOOR)
    var_e = np.maximum(torch.cat(epis).numpy(), 0.0)
    n = len(mu)
    between = np.zeros(n, dtype=float)
    within = var_e
    e_tot = var_e
    var_t = var_a + e_tot
    z = 1.6448536269514722
    sig = np.sqrt(np.maximum(var_t, 0.0))
    return {
        "mu": mu,
        "var_aleatoric": var_a,
        "var_epistemic": var_e,
        "var_epistemic_within": within,
        "var_epistemic_between": between,
        "var_epistemic_total": e_tot,
        "var_total": var_t,
        "sigma_aleatoric": np.sqrt(var_a),
        "sigma_epistemic": np.sqrt(np.maximum(var_e, 0.0)),
        "sigma_epistemic_within": np.sqrt(np.maximum(within, 0.0)),
        "sigma_epistemic_between": np.zeros(n, dtype=float),
        "sigma_epistemic_total": np.sqrt(np.maximum(e_tot, 0.0)),
        "sigma_total": sig,
        "std_total": sig,
        "lower_90": mu - z * sig,
        "upper_90": mu + z * sig,
        "uncertainty_method": "sngp_gp_posterior",
        "interval_method": "gaussian_predictive_moment",
        "variance_ddof": VARIANCE_DDOF,
        "n_posterior_samples": 0,
    }


def save_sngp(model: SNGPRegressor, path: Path, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": meta,
            "class": "SNGPRegressor",
            "in_dim": meta.get("in_dim", model.in_dim),
            "hidden_dims": meta.get("hidden_dims", model.hidden_dims),
            "num_features": model.num_features,
            "ridge": model.ridge,
            "lengthscale": model.rff.lengthscale,
            "rff_seed": model.rff.seed,
        },
        path,
    )


def load_sngp(path: Path, device=None) -> tuple[SNGPRegressor, dict]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(path, map_location=device, weights_only=False)
    meta = blob.get("meta", {})
    model = SNGPRegressor(
        int(blob.get("in_dim") or meta["in_dim"]),
        list(blob.get("hidden_dims") or meta["hidden_dims"]),
        num_features=int(blob.get("num_features") or meta.get("num_features", 512)),
        lengthscale=float(blob.get("lengthscale") or meta.get("rff_lengthscale", 2.0)),
        ridge=float(blob.get("ridge") or meta.get("ridge", 1.0)),
        seed=int(blob.get("rff_seed") or meta.get("rff_seed", 0)),
    ).to(device)
    model.load_state_dict(blob["state_dict"])
    return model, meta
