"""Save and restore complete HeteroMLP member checkpoints for external models."""
from __future__ import annotations

from pathlib import Path
import sys

import torch

from common import REPOSITORY_ROOT

sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from uq_remake.models.uq_tabular import HeteroMLP  # noqa: E402
import uq_remake.models.uq_tabular as uq_mod  # noqa: E402

FORMAT_VERSION = 1


def save_weight_checkpoint(
    path: Path,
    *,
    model: HeteroMLP,
    fold: str,
    member: int,
    seed: int,
    target: str,
    feature_names: list[str],
    hidden_dims: list[int],
    log_var_clamp: tuple[float, float] | list[float],
    scaler_mean,
    scaler_scale,
    training_meta: dict,
) -> None:
    """Write a portable CPU checkpoint with everything needed for preprocessing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "architecture": "HeteroMLP",
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "in_dim": int(model.trunk[0].in_features),
        "hidden_dims": [int(value) for value in hidden_dims],
        "dropout": 0.2,
        "log_var_clamp": [float(value) for value in log_var_clamp],
        "fold": fold,
        "member": int(member),
        "seed": int(seed),
        "mc_test_passes": 30,
        "target": target,
        "feature_names": list(feature_names),
        "scaler_mean": torch.as_tensor(scaler_mean, dtype=torch.float32),
        "scaler_scale": torch.as_tensor(scaler_scale, dtype=torch.float32),
        "training_meta": training_meta,
    }
    if len(payload["feature_names"]) != payload["in_dim"]:
        raise ValueError(f"{path}: feature schema does not match model input width")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_weight_checkpoint(path: Path, device: str = "cpu") -> tuple[HeteroMLP, dict]:
    """Restore one member without requiring pickled model classes or numpy objects."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != FORMAT_VERSION or checkpoint.get("architecture") != "HeteroMLP":
        raise ValueError(f"{path}: unsupported checkpoint format")
    uq_mod.LOG_VAR_CLAMP = tuple(float(value) for value in checkpoint["log_var_clamp"])
    model = HeteroMLP(
        int(checkpoint["in_dim"]),
        [int(value) for value in checkpoint["hidden_dims"]],
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint
