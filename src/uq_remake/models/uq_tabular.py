"""Tabular UQ heads for the shared DS1/DS2 active grid.

Families:
  MCDROPOUT — MLP mean + learned global log observation variance; epistemic = MC-dropout.
  HETERO    — MLP mean + per-row log aleatoric variance; production prediction
              uses MC dropout (`predict_hetero_mc`, 30 passes in the active grid)
              for the within-member epistemic term.
  EVID      — NIG evidential (gamma, nu, alpha, beta). Not a weight posterior.

Variance convention: ddof=0 (equal-weight empirical mixture). Not an unbiased
sample variance from five members.

PCA is not used.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

LOG_VAR_CLAMP = (-8.0, 8.0)
VAR_FLOOR = 1e-12
VARIANCE_DDOF = 0


def _mlp_trunk(in_dim: int, hidden_dims: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    return nn.Sequential(*layers)


def enable_mc_dropout(model: nn.Module) -> None:
    """Eval mode except Dropout modules, which stay stochastic."""
    model.eval()
    for mod in model.modules():
        if isinstance(mod, nn.Dropout):
            mod.train()


def restore_eval(model: nn.Module) -> None:
    model.eval()


class MCDropoutMLP(nn.Module):
    """Mean head + learned global log observation variance. Not a BNN."""

    def __init__(self, in_dim: int, hidden_dims: list[int], dropout: float = 0.2):
        super().__init__()
        self.trunk = _mlp_trunk(in_dim, hidden_dims, dropout)
        self.head = nn.Linear(hidden_dims[-1], 1)
        self.log_var_obs = nn.Parameter(torch.zeros(()))
        self.dropout_p = dropout

    def clamped_log_var(self) -> torch.Tensor:
        lo, hi = LOG_VAR_CLAMP
        return self.log_var_obs.clamp(lo, hi)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.head(self.trunk(x)).squeeze(-1)
        return mu, self.clamped_log_var()


class HeteroMLP(nn.Module):
    """Mean + per-row log aleatoric variance."""

    def __init__(self, in_dim: int, hidden_dims: list[int], dropout: float = 0.2):
        super().__init__()
        self.trunk = _mlp_trunk(in_dim, hidden_dims, dropout)
        self.mu = nn.Linear(hidden_dims[-1], 1)
        self.log_var = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        mu = self.mu(h).squeeze(-1)
        lo, hi = LOG_VAR_CLAMP
        s = self.log_var(h).squeeze(-1).clamp(lo, hi)
        return mu, s


class EvidMLP(nn.Module):
    """NIG evidential head. Epistemic term is NIG, not a weight posterior."""

    def __init__(self, in_dim: int, hidden_dims: list[int], dropout: float = 0.1):
        super().__init__()
        self.trunk = _mlp_trunk(in_dim, hidden_dims, dropout)
        self.out = nn.Linear(hidden_dims[-1], 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.out(self.trunk(x))
        gamma = raw[:, 0]
        nu = torch.nn.functional.softplus(raw[:, 1]) + 1e-4
        alpha = torch.nn.functional.softplus(raw[:, 2]) + 1.0 + 1e-4
        beta = torch.nn.functional.softplus(raw[:, 3]) + 1e-4
        return gamma, nu, alpha, beta


def gaussian_nll_from_log_var(y: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """0.5 * exp(-s) (y-mu)^2 + 0.5 s  (up to a constant)."""
    s = log_var.expand_as(y) if log_var.ndim == 0 else log_var
    return (0.5 * torch.exp(-s) * (y - mu) ** 2 + 0.5 * s).mean()


gaussian_nll = gaussian_nll_from_log_var


def evidential_nig_nll(y, gamma, nu, alpha, beta) -> torch.Tensor:
    two_b = 2.0 * beta
    return (
        0.5 * torch.log(np.pi / nu)
        - alpha * torch.log(two_b)
        + (alpha + 0.5) * torch.log(nu * (y - gamma) ** 2 + two_b)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )


def evidential_nig_loss(y, gamma, nu, alpha, beta, lambda_reg: float = 0.01) -> torch.Tensor:
    nll = evidential_nig_nll(y, gamma, nu, alpha, beta)
    reg = torch.abs(y - gamma) * (2.0 * nu + alpha)
    return (nll + lambda_reg * reg).mean()


def gaussian_mixture_nll(y: np.ndarray, mu: np.ndarray, var_total: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    vt = np.maximum(np.asarray(var_total, dtype=float), VAR_FLOOR)
    nll = 0.5 * np.log(2.0 * np.pi * vt) + 0.5 * (y - mu) ** 2 / vt
    return float(np.mean(nll[np.isfinite(nll)]))


def _train_loop(model, loss_fn, val_fn, Xtr, ytr, Xva, yva, cfg, device, seed: int):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["mlp_lr"]),
        weight_decay=float(cfg["mlp_weight_decay"]),
    )
    loader = DataLoader(
        TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)),
        batch_size=int(cfg["mlp_batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    best_state, best_val, stale = None, float("inf"), 0
    t0 = time.perf_counter()
    last_epoch = 0
    for epoch in range(int(cfg["mlp_epochs"])):
        last_epoch = epoch + 1
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model, xb, yb).backward()
            opt.step()
        with torch.no_grad():
            val_loss = float(val_fn(model))
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
    restore_eval(model)
    return model, {
        "best_val": best_val,
        "train_seconds": time.perf_counter() - t0,
        "best_epoch": last_epoch - stale,
        "epochs": last_epoch,
        "checkpoint_rule": "validation_predictive_nll",
        "pca": False,
        "variance_ddof": VARIANCE_DDOF,
    }


def train_mcdropout_mlp(Xtr, ytr, Xva, yva, cfg, device=None, seed: int = 42, n_mc_val: int = 30):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = MCDropoutMLP(Xtr.shape[1], list(cfg["mlp_hidden_dims"]), dropout=0.2).to(device)

    def loss_fn(m, x, y):
        mu, s = m(x)
        return gaussian_nll_from_log_var(y, mu, s)

    Xv = torch.tensor(Xva, dtype=torch.float32, device=device)
    yv = np.asarray(yva, dtype=float)

    def val_fn(m):
        out = predict_mcdropout_mc(m, Xva, n_mc=n_mc_val, device=device, seed=seed + 7)
        return gaussian_mixture_nll(yv, out["mu"], out["var_total"])

    return _train_loop(model, loss_fn, val_fn, Xtr, ytr, Xva, yva, cfg, device, seed)


def train_hetero_mlp(Xtr, ytr, Xva, yva, cfg, device=None, seed: int = 42, n_mc_val: int = 30):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = HeteroMLP(Xtr.shape[1], list(cfg["mlp_hidden_dims"]), dropout=0.2).to(device)

    def loss_fn(m, x, y):
        mu, s = m(x)
        return gaussian_nll_from_log_var(y, mu, s)

    yv = np.asarray(yva, dtype=float)

    def val_fn(m):
        out = predict_hetero_mc(m, Xva, n_mc=n_mc_val, device=device, seed=seed + 7)
        return gaussian_mixture_nll(yv, out["mu"], out["var_total"])

    return _train_loop(model, loss_fn, val_fn, Xtr, ytr, Xva, yva, cfg, device, seed)


def train_evid_mlp(Xtr, ytr, Xva, yva, cfg, device=None, seed: int = 42, lambda_reg: float = 0.01):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = EvidMLP(Xtr.shape[1], list(cfg["mlp_hidden_dims"]), dropout=0.1).to(device)
    Xv = torch.tensor(Xva, dtype=torch.float32, device=device)
    yv = torch.tensor(yva, dtype=torch.float32, device=device)

    def loss_fn(m, x, y):
        g, nu, a, b = m(x)
        return evidential_nig_loss(y, g, nu, a, b, lambda_reg=lambda_reg)

    def val_fn(m):
        restore_eval(m)
        g, nu, a, b = m(Xv)
        return float(evidential_nig_nll(yv, g, nu, a, b).mean().item())

    model, meta = _train_loop(model, loss_fn, val_fn, Xtr, ytr, Xva, yva, cfg, device, seed)
    meta["lambda_reg"] = float(lambda_reg)
    meta["interval_method"] = "normal_moment_approximation"
    return model, meta


def select_evidential_lambda(Xtr, ytr, Xva, yva, cfg, grid: list[float], device=None, seed: int = 42):
    """Pick evidence regularizer on validation NLL only (never test)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    best_lam, best_val, best_model, best_meta = None, float("inf"), None, None
    rows = []
    for lam in grid:
        model, meta = train_evid_mlp(Xtr, ytr, Xva, yva, cfg, device=device, seed=seed, lambda_reg=float(lam))
        val = float(meta["best_val"])
        rows.append({"lambda_reg": float(lam), "val_nll": val, "best_epoch": meta.get("best_epoch")})
        if val < best_val:
            best_val, best_lam, best_model, best_meta = val, float(lam), model, meta
    assert best_model is not None and best_meta is not None
    best_meta["lambda_reg"] = float(best_lam)
    best_meta["lambda_grid"] = [float(x) for x in grid]
    best_meta["lambda_selection"] = rows
    return best_model, best_meta, float(best_lam)


@torch.no_grad()
def predict_mcdropout_mc(model: MCDropoutMLP, X, n_mc: int = 30, device=None, seed: int = 0) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    prev = model.training
    try:
        enable_mc_dropout(model)
        torch.manual_seed(int(seed))
        xt = torch.tensor(np.asarray(X), dtype=torch.float32, device=device)
        mus = []
        s = None
        for _ in range(int(n_mc)):
            mu, s = model(xt)
            mus.append(mu)
        stack = torch.stack(mus, 0)
        mu = stack.mean(0).cpu().numpy()
        var_e = stack.var(0, unbiased=False).cpu().numpy()
        var_a = torch.exp(s).expand_as(stack[0]).cpu().numpy()
        var_a = np.maximum(var_a, VAR_FLOOR)
        var_t = var_a + var_e
        return {
            "mu": mu,
            "var_aleatoric": var_a,
            "var_epistemic": var_e,
            "var_epistemic_within": var_e,
            "var_epistemic_between": np.zeros_like(var_e),
            "var_epistemic_total": var_e,
            "var_total": var_t,
            "sigma_aleatoric": np.sqrt(var_a),
            "sigma_epistemic": np.sqrt(var_e),
            "sigma_epistemic_within": np.sqrt(var_e),
            "sigma_epistemic_between": np.zeros_like(var_e),
            "sigma_epistemic_total": np.sqrt(var_e),
            "sigma_total": np.sqrt(var_t),
            "log_var_obs": float(s.detach().cpu()) if s is not None else None,
            "n_mc": int(n_mc),
            "variance_ddof": VARIANCE_DDOF,
            "uncertainty_method": "mcdropout_global_observation_noise",
        }
    finally:
        model.train(prev)
        restore_eval(model)


@torch.no_grad()
def predict_hetero_mc(model: HeteroMLP, X, n_mc: int = 30, device=None, seed: int = 0) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    prev = model.training
    try:
        enable_mc_dropout(model)
        torch.manual_seed(int(seed))
        xt = torch.tensor(np.asarray(X), dtype=torch.float32, device=device)
        mus, logvars = [], []
        for _ in range(int(n_mc)):
            mu, s = model(xt)
            mus.append(mu)
            logvars.append(s)
        mus = torch.stack(mus, 0)
        logvars = torch.stack(logvars, 0)
        vars_a = torch.exp(logvars)
        mu = mus.mean(0).cpu().numpy()
        var_a = vars_a.mean(0).cpu().numpy()
        var_e = mus.var(0, unbiased=False).cpu().numpy()
        var_a = np.maximum(var_a, VAR_FLOOR)
        var_t = var_a + var_e
        s_np = logvars.mean(0).cpu().numpy()
        lo, hi = LOG_VAR_CLAMP
        at_lo = float(np.mean(np.abs(s_np - lo) < 1e-5))
        at_hi = float(np.mean(np.abs(s_np - hi) < 1e-5))
        return {
            "mu": mu,
            "var_aleatoric": var_a,
            "var_epistemic": var_e,
            "var_epistemic_within": var_e,
            "var_epistemic_between": np.zeros_like(var_e),
            "var_epistemic_total": var_e,
            "var_total": var_t,
            "sigma_aleatoric": np.sqrt(var_a),
            "sigma_epistemic": np.sqrt(var_e),
            "sigma_epistemic_within": np.sqrt(var_e),
            "sigma_epistemic_between": np.zeros_like(var_e),
            "sigma_epistemic_total": np.sqrt(var_e),
            "sigma_total": np.sqrt(var_t),
            "log_var": s_np,
            "log_var_quantiles": {q: float(np.quantile(s_np, q)) for q in (0.0, 0.05, 0.5, 0.95, 1.0)},
            "frac_at_clamp_lo": at_lo,
            "frac_at_clamp_hi": at_hi,
            "n_mc": int(n_mc),
            "variance_ddof": VARIANCE_DDOF,
            "uncertainty_method": "heteroscedastic_log_var_mc",
        }
    finally:
        model.train(prev)
        restore_eval(model)


@torch.no_grad()
def predict_evid(model: EvidMLP, X, device=None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    restore_eval(model)
    xt = torch.tensor(np.asarray(X), dtype=torch.float32, device=device)
    gamma, nu, alpha, beta = model(xt)
    g = gamma.cpu().numpy()
    nu_np = nu.cpu().numpy()
    a = alpha.cpu().numpy()
    b = beta.cpu().numpy()
    from uq_remake.models.chemprop_dmpnn import nig_predictive_from_params

    moments = nig_predictive_from_params(g, nu_np, a, b, var_floor=VAR_FLOOR)
    if moments["n_invalid_nig"]:
        print(
            f"  [tabular_evidential_invalid_nig] {moments['n_invalid_nig']}/{len(g)} rows "
            f"(nu>0, alpha>1, beta>0); variances set to NaN for those rows",
            flush=True,
        )
    var_a = moments["var_aleatoric"]
    var_e = moments["var_epistemic"]
    var_t = moments["var_total"]
    evidence = 2.0 * nu_np + a
    return {
        "mu": g,
        "gamma": g,
        "nu": nu_np,
        "alpha": a,
        "beta": b,
        "var_aleatoric": var_a,
        "var_epistemic": var_e,
        "var_epistemic_within": var_e,
        "var_epistemic_between": moments["var_epistemic_between"],
        "var_epistemic_total": var_e,
        "var_total": var_t,
        "nig_valid": moments["nig_valid"],
        "n_invalid_nig": moments["n_invalid_nig"],
        "n_valid_nig": moments["n_valid_nig"],
        "sigma_aleatoric": moments["sigma_aleatoric"],
        "sigma_epistemic": moments["sigma_epistemic"],
        "sigma_epistemic_within": np.sqrt(var_e),
        "sigma_epistemic_between": np.zeros_like(var_e),
        "sigma_epistemic_total": np.sqrt(var_e),
        "sigma_total": np.sqrt(var_t),
        "evidence_2nu_alpha": evidence,
        "nu_quantiles": {q: float(np.quantile(nu_np, q)) for q in (0.05, 0.5, 0.95)},
        "alpha_quantiles": {q: float(np.quantile(a, q)) for q in (0.05, 0.5, 0.95)},
        "beta_quantiles": {q: float(np.quantile(b, q)) for q in (0.05, 0.5, 0.95)},
        "uncertainty_method": "nig_evidential",
        "interval_method": "normal_moment_approximation",
    }


def aggregate_members(member_outs: list[dict]) -> dict:
    """Law of total variance. Between-member disagreement is not automatically 'total epistemic'."""
    mus = np.stack([o["mu"] for o in member_outs], 0)
    var_a_m = np.stack([o["var_aleatoric"] for o in member_outs], 0)
    var_w_m = np.stack([o.get("var_epistemic_within", o["var_epistemic"]) for o in member_outs], 0)
    mu = mus.mean(0)
    var_a = var_a_m.mean(0)
    var_w = var_w_m.mean(0)
    var_b = np.mean((mus - mu) ** 2, axis=0)  # ddof=0
    var_e_tot = var_w + var_b
    var_t = var_a + var_e_tot
    out = {
        "mu": mu,
        "var_aleatoric": var_a,
        "var_epistemic_within": var_w,
        "var_epistemic_between": var_b,
        "var_epistemic_total": var_e_tot,
        "var_epistemic": var_e_tot,
        "var_total": var_t,
        "sigma_aleatoric": np.sqrt(np.maximum(var_a, 0.0)),
        "sigma_epistemic_within": np.sqrt(np.maximum(var_w, 0.0)),
        "sigma_epistemic_between": np.sqrt(np.maximum(var_b, 0.0)),
        "sigma_epistemic_total": np.sqrt(np.maximum(var_e_tot, 0.0)),
        "sigma_epistemic": np.sqrt(np.maximum(var_e_tot, 0.0)),
        "sigma_total": np.sqrt(np.maximum(var_t, 0.0)),
        "n_members": len(member_outs),
        "variance_ddof": VARIANCE_DDOF,
        "uncertainty_method": "ensemble_law_of_total_variance",
    }
    if all("gamma" in o for o in member_outs):
        for key in ("gamma", "nu", "alpha", "beta"):
            out[key] = np.mean(np.stack([o[key] for o in member_outs], 0), 0)
        out["uncertainty_method"] = "nig_evidential_ensemble"
        out["interval_method"] = "normal_moment_approximation"
    return out


def save_torch(model, path: Path, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta, "class": model.__class__.__name__}, path)


def load_torch(model: nn.Module, path: Path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    restore_eval(model)
    return model, blob.get("meta", {})


def train_gpr(*_a, **_k):
    raise RuntimeError("Use uq_remake.models.gpr.train_exact_gpr or train_sparse_gpr.")


def train_ensemble(*_a, **_k):
    raise RuntimeError("train_ensemble is removed. Train members individually and call aggregate_members.")


predict_ensemble = aggregate_members
BayesMLP = MCDropoutMLP
