"""Fixed-prior full variational BNN (BAYES_FULL).

This is not MC-dropout, not last-layer Bayes, not a deep ensemble, and not evidential.
Every hidden and output weight/bias has a diagonal Gaussian variational posterior.
log_sigma2_obs is a deterministic likelihood parameter (excluded from the KL).
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

from uq_remake.models.uq_tabular import VAR_FLOOR, VARIANCE_DDOF, gaussian_mixture_nll

PRIOR_MEAN = 0.0
PRIOR_STD = 1.0
SIGMA_EPS = 1e-6
LOG_SIGMA2_OBS_MIN = -12.0
LOG_SIGMA2_OBS_MAX = 8.0
BETA_KL = 1.0
KL_SCHEDULE = "none"
DROPOUT = 0.0
LOG_2PI = math.log(2.0 * math.pi)


def _softplus_sigma(rho: torch.Tensor) -> torch.Tensor:
    return F.softplus(rho) + SIGMA_EPS


def kl_diagonal_gaussian(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    mu_p: float = PRIOR_MEAN,
    sigma_p: float = PRIOR_STD,
) -> torch.Tensor:
    """Sum of analytic KL(q||p) over every scalar Gaussian factor."""
    sp = torch.as_tensor(sigma_p, dtype=mu_q.dtype, device=mu_q.device)
    mp = torch.as_tensor(mu_p, dtype=mu_q.dtype, device=mu_q.device)
    return (
        torch.log(sp / sigma_q)
        + (sigma_q.square() + (mu_q - mp).square()) / (2.0 * sp.square())
        - 0.5
    ).sum()


def gaussian_obs_nll(y: torch.Tensor, mu: torch.Tensor, log_sigma2: torch.Tensor) -> torch.Tensor:
    """Per-row NLL of y ~ N(mu, exp(s)). Mean reduction is applied by the caller."""
    s = log_sigma2.expand_as(y) if log_sigma2.ndim == 0 else log_sigma2
    return 0.5 * (torch.exp(-s) * (y - mu).square() + s + LOG_2PI)


class BayesianLinear(nn.Module):
    """Linear layer with independent Normal variational posteriors on weight and bias."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_mean: float = PRIOR_MEAN,
        prior_std: float = PRIOR_STD,
        rho_init: float = -3.0,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.prior_mean = float(prior_mean)
        self.prior_std = float(prior_std)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_rho, rho_init)
        nn.init.constant_(self.bias_rho, rho_init)

    def weight_sigma(self) -> torch.Tensor:
        return _softplus_sigma(self.weight_rho)

    def bias_sigma(self) -> torch.Tensor:
        return _softplus_sigma(self.bias_rho)

    def sample_weight_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        w_sigma = self.weight_sigma()
        b_sigma = self.bias_sigma()
        w = self.weight_mu + w_sigma * torch.randn_like(self.weight_mu)
        b = self.bias_mu + b_sigma * torch.randn_like(self.bias_mu)
        return w, b

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            w, b = self.sample_weight_bias()
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)

    def kl(self) -> torch.Tensor:
        return kl_diagonal_gaussian(self.weight_mu, self.weight_sigma(), self.prior_mean, self.prior_std) + kl_diagonal_gaussian(
            self.bias_mu, self.bias_sigma(), self.prior_mean, self.prior_std
        )

    def variational_stds(self) -> torch.Tensor:
        return torch.cat([self.weight_sigma().reshape(-1), self.bias_sigma().reshape(-1)])


class FixedPriorVariationalBNN(nn.Module):
    """input → BayesianLinear → SiLU → … → BayesianLinear(out=1). Dropout is forbidden."""

    def __init__(self, in_dim: int, hidden_dims: list[int], prior_mean: float = PRIOR_MEAN, prior_std: float = PRIOR_STD):
        super().__init__()
        if any(h <= 0 for h in hidden_dims):
            raise ValueError("hidden_dims must be positive")
        dims = [int(in_dim), *[int(h) for h in hidden_dims], 1]
        self.layers = nn.ModuleList(
            [BayesianLinear(dims[i], dims[i + 1], prior_mean=prior_mean, prior_std=prior_std) for i in range(len(dims) - 1)]
        )
        self.log_sigma2_obs = nn.Parameter(torch.zeros(()))
        self.prior_mean = float(prior_mean)
        self.prior_std = float(prior_std)
        self.hidden_dims = list(hidden_dims)
        self.in_dim = int(in_dim)

    def clamped_log_sigma2_obs(self) -> torch.Tensor:
        return self.log_sigma2_obs.clamp(LOG_SIGMA2_OBS_MIN, LOG_SIGMA2_OBS_MAX)

    def bayesian_layers(self) -> list[BayesianLinear]:
        return list(self.layers)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        h = x
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            h = layer(h, sample=sample)
            if i != last:
                h = F.silu(h)
        return h.squeeze(-1)

    def kl_divergence(self) -> torch.Tensor:
        kl = self.layers[0].kl()
        for layer in self.layers[1:]:
            kl = kl + layer.kl()
        return kl

    def n_bayesian_scalars(self) -> int:
        n = 0
        for layer in self.layers:
            n += layer.weight_mu.numel() + layer.bias_mu.numel()
        return n

    def deterministic_parameter_names(self) -> list[str]:
        return ["log_sigma2_obs"]

    def posterior_std_vector(self) -> torch.Tensor:
        return torch.cat([layer.variational_stds() for layer in self.layers])

    def layer_posterior_std_means(self) -> dict[str, float]:
        out = {}
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            name = "output" if i == last else f"hidden_{i}"
            std = layer.variational_stds()
            out[f"{name}_mean_posterior_std"] = float(std.mean().detach().cpu())
            out[f"{name}_median_posterior_std"] = float(std.median().detach().cpu())
        return out


def has_dropout(model: nn.Module) -> bool:
    return any(isinstance(m, nn.Dropout) for m in model.modules())


def elbo_batch_loss(
    model: FixedPriorVariationalBNN,
    x: torch.Tensor,
    y: torch.Tensor,
    n_train: int,
    beta_kl: float = BETA_KL,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean NLL + (beta_KL / N_train) * KL. KL scale does not depend on batch size."""
    mu = model(x, sample=True)
    nll = gaussian_obs_nll(y, mu, model.clamped_log_sigma2_obs()).mean()
    kl = model.kl_divergence()
    loss = nll + (float(beta_kl) / float(n_train)) * kl
    return loss, nll, kl


def prior_metadata() -> dict:
    return {
        "prior_type": "fixed_normal",
        "prior_mean": PRIOR_MEAN,
        "prior_std": PRIOR_STD,
        "prior_scope": "all_bayesian_parameters",
        "hidden_weight_prior_std": PRIOR_STD,
        "hidden_bias_prior_std": PRIOR_STD,
        "output_weight_prior_std": PRIOR_STD,
        "output_bias_prior_std": PRIOR_STD,
        "beta_KL": BETA_KL,
        "kl_schedule": KL_SCHEDULE,
        "dropout": DROPOUT,
        "activation": "SiLU",
        "batch_norm": False,
        "log_sigma2_obs_min": LOG_SIGMA2_OBS_MIN,
        "log_sigma2_obs_max": LOG_SIGMA2_OBS_MAX,
        "implementation": "FixedPriorVariationalBNN",
    }


def _empty_uq(n: int, mu: np.ndarray, var_a: np.ndarray, var_e: np.ndarray, *, single: bool) -> dict:
    var_a = np.maximum(np.asarray(var_a, dtype=float), VAR_FLOOR)
    var_e = np.maximum(np.asarray(var_e, dtype=float), 0.0)
    if single:
        within = np.zeros(n, dtype=float)
        between = np.zeros(n, dtype=float)
        e_tot = var_e
    else:
        within = var_e
        between = np.zeros(n, dtype=float)
        e_tot = var_e
    var_t = var_a + e_tot
    return {
        "mu": np.asarray(mu, dtype=float),
        "var_aleatoric": var_a,
        "var_epistemic": var_e,
        "var_epistemic_within": within,
        "var_epistemic_between": between,
        "var_epistemic_total": e_tot,
        "var_total": var_t,
        "sigma_aleatoric": np.sqrt(var_a),
        "sigma_epistemic": np.sqrt(np.maximum(var_e, 0.0)),
        "sigma_epistemic_within": np.sqrt(np.maximum(within, 0.0)),
        "sigma_epistemic_between": np.sqrt(np.maximum(between, 0.0)),
        "sigma_epistemic_total": np.sqrt(np.maximum(e_tot, 0.0)),
        "sigma_total": np.sqrt(np.maximum(var_t, 0.0)),
        "std_total": np.sqrt(np.maximum(var_t, 0.0)),
    }


@torch.no_grad()
def predict_variational_bnn(
    model: FixedPriorVariationalBNN,
    X,
    n_samples: int = 30,
    device=None,
    seed: int = 0,
    single_model: bool = True,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    xt = torch.tensor(np.asarray(X), dtype=torch.float32, device=device)
    mus = []
    for _ in range(int(n_samples)):
        mus.append(model(xt, sample=True))
    stack = torch.stack(mus, 0)
    mu = stack.mean(0)
    var_e = stack.var(0, unbiased=False)
    log_s = model.clamped_log_sigma2_obs()
    var_a = torch.exp(log_s).expand_as(mu)
    sigma_obs = torch.sqrt(var_a)
    z = torch.randn_like(stack)
    y_tilde = stack + sigma_obs.unsqueeze(0) * z
    lo90 = torch.quantile(y_tilde, 0.05, dim=0)
    hi90 = torch.quantile(y_tilde, 0.95, dim=0)
    n = int(xt.shape[0])
    out = _empty_uq(
        n,
        mu.cpu().numpy(),
        var_a.cpu().numpy(),
        var_e.cpu().numpy(),
        single=single_model,
    )
    out.update(
        {
            "lower_90": lo90.cpu().numpy(),
            "upper_90": hi90.cpu().numpy(),
            "n_posterior_samples": int(n_samples),
            "prediction_seed": int(seed),
            "log_sigma2_obs": float(log_s.detach().cpu()),
            "uncertainty_method": "variational_weight_posterior",
            "interval_method": "empirical_posterior_predictive",
            "variance_ddof": VARIANCE_DDOF,
        }
    )
    finite = np.isfinite(out["var_total"])
    out["fraction_nonfinite_variance"] = float(1.0 - finite.mean()) if n else 0.0
    s = float(log_s.detach().cpu())
    out["fraction_variance_clamped"] = float(abs(s - LOG_SIGMA2_OBS_MIN) < 1e-8 or abs(s - LOG_SIGMA2_OBS_MAX) < 1e-8)
    return out


def train_variational_bnn(
    Xtr,
    ytr,
    Xva,
    yva,
    cfg,
    device=None,
    seed: int = 42,
    n_samples_val: int = 30,
    beta_kl: float = BETA_KL,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = FixedPriorVariationalBNN(Xtr.shape[1], list(cfg["mlp_hidden_dims"])).to(device)
    n_train = int(len(Xtr))
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["mlp_lr"]),
        weight_decay=0.0,
    )
    loader = DataLoader(
        TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)),
        batch_size=int(cfg["mlp_batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    yv = np.asarray(yva, dtype=float)
    best_state, best_val, stale = None, float("inf"), 0
    last_kl = last_nll = last_elbo = float("nan")
    t0 = time.perf_counter()
    last_epoch = 0
    for epoch in range(int(cfg["mlp_epochs"])):
        last_epoch = epoch + 1
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss, nll, kl = elbo_batch_loss(model, xb, yb, n_train=n_train, beta_kl=beta_kl)
            loss.backward()
            opt.step()
            last_kl = float(kl.detach().cpu())
            last_nll = float(nll.detach().cpu())
            last_elbo = float(loss.detach().cpu())
        pred = predict_variational_bnn(model, Xva, n_samples=n_samples_val, device=device, seed=seed + 7, single_model=True)
        val_nll = gaussian_mixture_nll(yv, pred["mu"], pred["var_total"])
        if val_nll < best_val - 1e-6:
            best_val = val_nll
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg["mlp_patience"]):
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    stds = model.posterior_std_vector().detach().cpu()
    tr_pred = predict_variational_bnn(model, Xtr, n_samples=n_samples_val, device=device, seed=seed + 5, single_model=True)
    meta = {
        "best_val": float(best_val),
        "train_seconds": time.perf_counter() - t0,
        "best_epoch": last_epoch - stale,
        "epochs": last_epoch,
        "checkpoint_rule": "validation_predictive_nll",
        "mean_kl": last_kl,
        "final_elbo": last_elbo,
        "training_nll": float(gaussian_mixture_nll(np.asarray(ytr, dtype=float), tr_pred["mu"], tr_pred["var_total"])),
        "validation_nll": float(best_val),
        "mean_posterior_std": float(stds.mean()),
        "median_posterior_std": float(stds.median()),
        "n_bayesian_scalars": model.n_bayesian_scalars(),
        "log_sigma2_obs": float(model.clamped_log_sigma2_obs().detach().cpu()),
        "pca": False,
        "imputation": False,
        "variance_ddof": VARIANCE_DDOF,
        **prior_metadata(),
        **model.layer_posterior_std_means(),
    }
    return model, meta


def save_bnn(model: FixedPriorVariationalBNN, path: Path, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": meta,
            "class": "FixedPriorVariationalBNN",
            "in_dim": model.in_dim,
            "hidden_dims": model.hidden_dims,
        },
        path,
    )


def load_bnn(path: Path, device=None) -> tuple[FixedPriorVariationalBNN, dict]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(path, map_location=device, weights_only=False)
    model = FixedPriorVariationalBNN(int(blob["in_dim"]), list(blob["hidden_dims"])).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob.get("meta", {})
