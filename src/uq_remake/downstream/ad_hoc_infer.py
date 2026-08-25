"""Load held-out DS1 absolute-binding predictions from stored scores or checkpoints."""
from __future__ import annotations

from pathlib import Path

import torch  # noqa: I001  # import before RDKit on Windows
import numpy as np
import pandas as pd

from uq_remake import REMAKE_ROOT
from uq_remake.downstream.protocol import AD_RETENTION_DIR, FOLDS
from uq_remake.training.protocol import checkpoint_dir, load_active_grid_cfg

STORED_FOLD_SCORE_MODELS = (
    "DS1_RDKIT_MLPENS5_EVID",
    "DS1_GRAPH_DMPNN_EVID",
)

UQ_MODELS = (
    "DS1_RDKIT_MLP_EVID",
    "DS1_RDKIT_MLPENS5_EVID",
    "DS1_RDKIT_MLP_HETERO",
    "DS1_RDKIT_MLPENS5_HETERO",
    "DS1_RDKIT_MLP_MCDROPOUT",
    "DS1_RDKIT_MLPENS5_MCDROPOUT",
    "DS1_RDKIT_SNGP",
    "DS1_RDKIT_SNGPENS5",
    "DS1_RDKIT_GPR_SPARSE",
    "DS1_RDKIT_MLP_BAYES_FULL",
    "DS1_RDKIT_MLPENS5_BAYES_FULL",
    "DS1_GRAPH_DMPNN_EVID",
)

POINT_MODELS = (
    "DS1_RDKIT_MLP_POINT",
    "DS1_RDKIT_XGB_POINT",
)


def load_stored_ds1_fold_scores(model_id: str) -> pd.DataFrame | None:
    root = AD_RETENTION_DIR / "fold_scores" / model_id
    if not root.exists():
        return None
    frames = []
    for fold in FOLDS:
        path = root / f"{fold}.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        df["source_fold"] = fold
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["model_id"] = model_id
    return out


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _hidden_dims(meta: dict) -> list[int]:
    hps = meta.get("selected_hps") or {}
    dims = hps.get("mlp_hidden_dims") or [512, 512, 256]
    return [int(x) for x in dims]


def _load_ds1_heldout_features(fold: str) -> dict:
    """Load stored fold-local arrays without re-running the RDKit schema pin."""
    from uq_remake.training.protocol import load_folds, load_table

    table = load_table("DS1")
    split = load_folds("DS1")[fold]
    te = np.asarray(split["test_idx"], dtype=int)
    feat_dir = REMAKE_ROOT / "data" / "processed" / "features" / "rdkit_ds1" / fold
    arrays = np.load(feat_dir / "arrays.npz")
    x_test = np.asarray(arrays["X_test"])
    if len(x_test) != len(te):
        raise RuntimeError(f"DS1 {fold}: stored X_test {len(x_test)} != split {len(te)}")
    sub = table.iloc[te]
    return {
        "X_test": x_test,
        "y_true": sub["logK"].to_numpy(dtype=float),
        "record_ids": sub["record_id"].astype(str).to_numpy(),
        "ligand_smiles": sub["ligand_smiles"].astype(str).to_numpy(),
        "family": sub["family"].astype(str).to_numpy() if "family" in sub.columns else np.array([""] * len(te)),
    }


def infer_ds1_model(model_id: str) -> pd.DataFrame:
    stored = load_stored_ds1_fold_scores(model_id)
    if stored is not None:
        return stored
    frames = []
    for fold in FOLDS:
        bundle = _load_ds1_heldout_features(fold)
        X = bundle["X_test"]
        rec = pd.DataFrame(
            {
                "record_id": np.asarray(bundle["record_ids"]).astype(str),
                "y_true": np.asarray(bundle["y_true"], dtype=float),
                "ligand_smiles": np.asarray(bundle["ligand_smiles"]).astype(str),
                "family": np.asarray(bundle["family"]).astype(str),
                "fold_id": fold,
                "source_fold": fold,
                "model_id": model_id,
                "evaluation_partition": "heldout_fold",
            }
        )
        uq = _predict_model(model_id, fold, X, bundle)
        rec["y_pred"] = np.asarray(uq["mu"], dtype=float)
        rec["sigma_total"] = np.asarray(uq.get("sigma_total", np.full(len(rec), np.nan)), dtype=float)
        frames.append(rec)
    return pd.concat(frames, ignore_index=True)


def _predict_model(model_id: str, fold: str, X, bundle: dict) -> dict:
    mid = str(model_id)
    if mid.endswith("MLPENS5_EVID") or mid.endswith("MLP_EVID"):
        return _predict_mlp_family(mid, fold, X, kind="EVID")
    if mid.endswith("MLPENS5_HETERO") or mid.endswith("MLP_HETERO"):
        return _predict_mlp_family(mid, fold, X, kind="HETERO")
    if mid.endswith("MLPENS5_MCDROPOUT") or mid.endswith("MLP_MCDROPOUT"):
        return _predict_mlp_family(mid, fold, X, kind="MCDROPOUT")
    if mid.endswith("MLPENS5_BAYES_FULL") or mid.endswith("MLP_BAYES_FULL"):
        return _predict_bnn_family(mid, fold, X)
    if mid.endswith("SNGPENS5") or mid.endswith("_SNGP"):
        return _predict_sngp_family(mid, fold, X)
    if mid.endswith("GPR_SPARSE"):
        return _predict_sparse_gpr(mid, fold, X)
    if mid.endswith("MLP_POINT"):
        return _predict_point_mlp(mid, fold, X)
    if mid.endswith("XGB_POINT"):
        return _predict_xgb(mid, fold, X)
    raise KeyError(f"no inference path for {model_id}")


def _n_members(model_id: str) -> int:
    return 5 if "ENS5" in str(model_id) else 1


def _predict_mlp_family(model_id: str, fold: str, X, kind: str) -> dict:
    import torch

    from uq_remake.models.uq_tabular import (
        EvidMLP,
        HeteroMLP,
        MCDropoutMLP,
        aggregate_members,
        load_torch,
        predict_evid,
        predict_hetero_mc,
        predict_mcdropout_mc,
    )

    grid = load_active_grid_cfg()
    n_mc = int(grid.get("mc_dropout_passes", 30))
    device = _device()
    members = []
    for m in range(_n_members(model_id)):
        ckpt = checkpoint_dir(model_id, fold, m) / "model.pt"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        meta = blob.get("meta", {})
        hidden = _hidden_dims(meta)
        in_dim = int(np.asarray(X).shape[1])
        if kind == "EVID":
            model = EvidMLP(in_dim, hidden, dropout=0.1)
            model, _ = load_torch(model, ckpt, device=device)
            members.append(predict_evid(model, X, device=device))
        elif kind == "HETERO":
            model = HeteroMLP(in_dim, hidden, dropout=0.2)
            model, _ = load_torch(model, ckpt, device=device)
            seed = int(meta.get("seed", 0))
            members.append(predict_hetero_mc(model, X, n_mc=n_mc, device=device, seed=seed + 11))
        else:
            model = MCDropoutMLP(in_dim, hidden, dropout=0.2)
            model, _ = load_torch(model, ckpt, device=device)
            seed = int(meta.get("seed", 0))
            members.append(predict_mcdropout_mc(model, X, n_mc=n_mc, device=device, seed=seed + 11))
    if len(members) == 1:
        return members[0]
    return aggregate_members(members)


def _predict_bnn_family(model_id: str, fold: str, X) -> dict:
    from uq_remake.models.uq_tabular import aggregate_members
    from uq_remake.models.variational_bnn import load_bnn, predict_variational_bnn

    grid = load_active_grid_cfg()
    n_s = int(grid.get("bayes_posterior_samples", 30))
    device = _device()
    members = []
    for m in range(_n_members(model_id)):
        ckpt = checkpoint_dir(model_id, fold, m) / "model.pt"
        model, meta = load_bnn(ckpt, device=device)
        seed = int(meta.get("seed", 0)) if isinstance(meta, dict) else 0
        members.append(
            predict_variational_bnn(
                model, X, n_samples=n_s, device=device, seed=seed + 11, single_model=len(members) == 0
            )
        )
    if len(members) == 1:
        return members[0]
    return aggregate_members(members)


def _predict_sngp_family(model_id: str, fold: str, X) -> dict:
    from uq_remake.models.sngp import load_sngp, predict_sngp
    from uq_remake.models.uq_tabular import aggregate_members

    device = _device()
    members = []
    for m in range(_n_members(model_id)):
        ckpt = checkpoint_dir(model_id, fold, m) / "model.pt"
        model, _meta = load_sngp(ckpt, device=device)
        members.append(predict_sngp(model, X, device=device, single_model=True))
    if len(members) == 1:
        return members[0]
    return aggregate_members(members)


def _predict_sparse_gpr(model_id: str, fold: str, X) -> dict:
    from uq_remake.models.gpr import SparseGPR, TargetScaler, load_gpr, predict_sparse_gpr

    ckpt = checkpoint_dir(model_id, fold, 0) / "model.pt"
    blob = load_gpr(ckpt)
    if isinstance(blob, dict) and "model" in blob:
        return predict_sparse_gpr(blob["model"], blob["scaler"], X)
    if not isinstance(blob, dict) or "state_dict" not in blob:
        return predict_sparse_gpr(blob, getattr(blob, "scaler", None), X)
    z = blob.get("Z")
    if z is None:
        raise RuntimeError(f"{model_id}: sparse GPR checkpoint is missing inducing points")
    if not torch.is_tensor(z):
        z = torch.as_tensor(z, dtype=torch.float32)
    meta = blob.get("meta") or {}
    jitter = float(meta.get("gpr_jitter", 1e-6))
    model = SparseGPR(z, jitter=jitter)
    model.load_state_dict(blob["state_dict"])
    sc = blob.get("scaler")
    if isinstance(sc, TargetScaler):
        scaler = sc
    elif isinstance(sc, dict):
        scaler = TargetScaler(sc["target_scaler_mean"], sc["target_scaler_scale"])
    else:
        raise RuntimeError(f"{model_id}: sparse GPR checkpoint is missing a target scaler")
    return predict_sparse_gpr(model, scaler, X)


def _predict_point_mlp(model_id: str, fold: str, X) -> dict:
    from uq_remake.models.point_tabular import PointMLP, predict_mlp
    from uq_remake.models.uq_tabular import load_torch

    ckpt = checkpoint_dir(model_id, fold, 0) / "model.pt"
    import torch

    blob = torch.load(ckpt, map_location=_device(), weights_only=False)
    hidden = _hidden_dims(blob.get("meta", {}))
    model = PointMLP(int(np.asarray(X).shape[1]), hidden)
    model, _ = load_torch(model, ckpt, device=_device())
    mu = predict_mlp(model, X, device=_device())
    return {"mu": np.asarray(mu, dtype=float), "sigma_total": np.full(len(mu), np.nan)}


def _predict_xgb(model_id: str, fold: str, X) -> dict:
    import xgboost as xgb

    ckpt = checkpoint_dir(model_id, fold, 0) / "model.json"
    if not ckpt.exists():
        ckpt = checkpoint_dir(model_id, fold, 0) / "model.pt"
    model = xgb.XGBRegressor()
    model.load_model(str(ckpt))
    mu = np.asarray(model.predict(np.asarray(X)), dtype=float)
    return {"mu": mu, "sigma_total": np.full(len(mu), np.nan)}


def cache_path(model_id: str) -> Path:
    return REMAKE_ROOT / "results" / "downstream_cv" / "ad_hoc_selectivity" / "ds1_heldout_preds" / f"{model_id}.csv"
