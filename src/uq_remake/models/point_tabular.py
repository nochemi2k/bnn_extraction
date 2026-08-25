"""Point-prediction MLP and XGBoost (canonical POINT heads)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class PointMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_point_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
    device: str | None = None,
    seed: int = 42,
) -> tuple[PointMLP, dict]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = PointMLP(
        in_dim=X_train.shape[1],
        hidden_dims=list(cfg["mlp_hidden_dims"]),
        dropout=float(cfg["mlp_dropout_point"]),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["mlp_lr"]),
        weight_decay=float(cfg["mlp_weight_decay"]),
    )
    loss_fn = nn.MSELoss()
    tr_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=int(cfg["mlp_batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)

    best_state = None
    best_val = float("inf")
    patience = int(cfg["mlp_patience"])
    stale = 0
    t0 = time.perf_counter()
    for epoch in range(int(cfg["mlp_epochs"])):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xv), yv).item())
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    train_s = time.perf_counter() - t0
    meta = {"best_val_mse": best_val, "train_seconds": train_s, "device": device, "epochs_ran": epoch + 1}
    return model, meta


@torch.no_grad()
def predict_mlp(model: PointMLP, X: np.ndarray, device: str | None = None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    return model(xt).cpu().numpy()


def train_point_xgb(X_train, y_train, X_val, y_val, cfg: dict, seed: int | None = None):
    import xgboost as xgb

    t0 = time.perf_counter()
    model = xgb.XGBRegressor(
        n_estimators=int(cfg["xgb_n_estimators"]),
        max_depth=int(cfg["xgb_max_depth"]),
        learning_rate=float(cfg["xgb_learning_rate"]),
        subsample=float(cfg["xgb_subsample"]),
        colsample_bytree=float(cfg["xgb_colsample_bytree"]),
        min_child_weight=int(cfg["xgb_min_child_weight"]),
        objective="reg:squarederror",
        n_jobs=4,
        random_state=int(seed if seed is not None else cfg["random_seed"]),
        early_stopping_rounds=40,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    meta = {"train_seconds": time.perf_counter() - t0, "best_iteration": int(getattr(model, "best_iteration", -1))}
    return model, meta


def save_mlp_checkpoint(model: PointMLP, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta, "in_dim": model.net[0].in_features}, path)
