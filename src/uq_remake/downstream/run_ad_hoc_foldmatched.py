"""Retrain ad hoc DS1 models on fold-matched subsets.

For each DS2 outer fold, DS1 train/val exclude every ligand in that fold's DS2
test partition. Artifacts are stored separately from the frozen active grid.

Table 2 models first; remaining SI S13 ad hoc models via --models si.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402  — before rdkit on Windows

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from uq_remake import REMAKE_ROOT  # noqa: E402
from uq_remake.data.build_m4 import write_chemprop_fold_csvs  # noqa: E402
from uq_remake.data.rdkit_features import preprocess_fold  # noqa: E402
from uq_remake.downstream.ad_hoc_selectivity import (  # noqa: E402
    SCOPES,
    assemble_ad_hoc_pairs,
    assert_no_ligand_overlap,
    evaluate_ad_hoc_fold,
    summarize_fold_metrics,
)
from uq_remake.downstream.protocol import FOLDS  # noqa: E402
from uq_remake.models.chemprop_dmpnn import (  # noqa: E402
    predict_dmpnn_evidential_uq,
    predict_dmpnn_point,
    train_dmpnn_point,
    write_chemprop_identity,
)
from uq_remake.models.point_tabular import (  # noqa: E402
    predict_mlp,
    save_mlp_checkpoint,
    train_point_mlp,
    train_point_xgb,
)
from uq_remake.models.uq_tabular import (  # noqa: E402
    aggregate_members,
    predict_evid,
    predict_hetero_mc,
    predict_mcdropout_mc,
    save_torch,
    select_evidential_lambda,
    train_evid_mlp,
    train_hetero_mlp,
    train_mcdropout_mlp,
)
from uq_remake.models.variational_bnn import (  # noqa: E402
    predict_variational_bnn,
    save_bnn,
    train_variational_bnn,
)
from uq_remake.models.sngp import predict_sngp, save_sngp, train_sngp  # noqa: E402
from uq_remake.models.gpr import predict_sparse_gpr, save_gpr, train_sparse_gpr  # noqa: E402
from uq_remake.training.protocol import (  # noqa: E402
    checkpoint_meta,
    ligand_split_hash,
    load_active_grid_cfg,
    load_folds,
    load_shared_cfg,
    load_table,
    model_seed,
    seed_everything,
    write_json,
)

CKPT_ROOT = REMAKE_ROOT / "checkpoints" / "ad_hoc_comparison"
FEAT_ROOT = REMAKE_ROOT / "data" / "processed" / "features" / "ad_hoc_foldmatched"
OUT_ROOT = REMAKE_ROOT / "results" / "downstream_cv" / "ad_hoc_selectivity_foldmatched"
TABLES = REMAKE_ROOT / "manuscripts" / "final" / "tables"

TABLE2_MODELS = (
    "DS1_RDKIT_XGB_POINT",
    "DS1_RDKIT_MLP_POINT",
    "DS1_GRAPH_DMPNN_POINT",
    "DS1_GRAPH_DMPNN_EVID",
)
# Remaining SI S13 ad hoc models (same fold-matched protocol; run after Table 2).
SI_REMAINING_MODELS = (
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
)
SI_SPECS = {
    "DS1_RDKIT_MLP_EVID": {"kind": "EVID", "ens": False, "offset": "EVID", "family": "EVID_MLP", "single_id": None},
    "DS1_RDKIT_MLPENS5_EVID": {"kind": "EVID", "ens": True, "offset": "EVID", "family": "EVID_MLPENS5", "single_id": "DS1_RDKIT_MLP_EVID"},
    "DS1_RDKIT_MLP_HETERO": {"kind": "HETERO", "ens": False, "offset": "HETERO", "family": "HETERO_MLP", "single_id": None},
    "DS1_RDKIT_MLPENS5_HETERO": {"kind": "HETERO", "ens": True, "offset": "HETERO", "family": "HETERO_MLPENS5", "single_id": "DS1_RDKIT_MLP_HETERO"},
    "DS1_RDKIT_MLP_MCDROPOUT": {"kind": "MCDROPOUT", "ens": False, "offset": "MCDROPOUT", "family": "MCDROPOUT_MLP", "single_id": None},
    "DS1_RDKIT_MLPENS5_MCDROPOUT": {"kind": "MCDROPOUT", "ens": True, "offset": "MCDROPOUT", "family": "MCDROPOUT_MLPENS5", "single_id": "DS1_RDKIT_MLP_MCDROPOUT"},
    "DS1_RDKIT_SNGP": {"kind": "SNGP", "ens": False, "offset": "SNGP", "family": "SNGP", "single_id": None},
    "DS1_RDKIT_SNGPENS5": {"kind": "SNGP", "ens": True, "offset": "SNGP", "family": "SNGPENS5", "single_id": "DS1_RDKIT_SNGP"},
    "DS1_RDKIT_GPR_SPARSE": {"kind": "GPR_SPARSE", "ens": False, "offset": "GPR_SPARSE", "family": "GPR_SPARSE", "single_id": None},
    "DS1_RDKIT_MLP_BAYES_FULL": {"kind": "BAYES", "ens": False, "offset": "BAYES_FULL", "family": "BAYES_FULL", "single_id": None},
    "DS1_RDKIT_MLPENS5_BAYES_FULL": {"kind": "BAYES", "ens": True, "offset": "BAYES_FULL", "family": "BAYES_FULLENS5", "single_id": "DS1_RDKIT_MLP_BAYES_FULL"},
}
_EVID_LAMBDA: dict[str, float] = {}
FAMILY = {
    "DS1_RDKIT_XGB_POINT": "POINT_XGB",
    "DS1_RDKIT_MLP_POINT": "POINT_MLP",
    "DS1_GRAPH_DMPNN_POINT": "CHEMPROP_POINT_NATIVE",
    "DS1_GRAPH_DMPNN_EVID": "CHEMPROP_EVIDENTIAL_NATIVE",
}
OFFSET = {
    "DS1_RDKIT_XGB_POINT": "POINT_XGB",
    "DS1_RDKIT_MLP_POINT": "POINT_MLP",
    "DS1_GRAPH_DMPNN_POINT": "CHEMPROP_POINT",
    "DS1_GRAPH_DMPNN_EVID": "CHEMPROP_EVID",
}


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def split_path(fold: str) -> Path:
    return OUT_ROOT / "splits" / f"{fold}.json"


def rdkit_feat_dir(fold: str) -> Path:
    return FEAT_ROOT / "rdkit_ds1" / fold


def chem_dir(fold: str) -> Path:
    return FEAT_ROOT / "chemprop_ds1" / fold


def ckpt_dir(model_id: str, fold: str, member: int = 0) -> Path:
    return CKPT_ROOT / model_id / fold / f"member_{member}"


def pred_path(model_id: str, fold: str) -> Path:
    return OUT_ROOT / "ds1_preds" / model_id / f"{fold}.csv"


def reconstruct_unscaled_ds1_features(ds1: pd.DataFrame) -> np.ndarray:
    """Invert the fold-0 StandardScaler to recover the full unscaled DS1 matrix."""
    fold = "fold0"
    feat_dir = REMAKE_ROOT / "data" / "processed" / "features" / "rdkit_ds1" / fold
    man = json.loads((feat_dir / "manifest.json").read_text(encoding="utf-8"))
    z = np.load(feat_dir / "arrays.npz")
    mean = np.asarray(man["scaler_mean"], dtype=float)
    scale = np.asarray(man["scaler_scale"], dtype=float)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    folds = load_folds("DS1")[fold]
    n = len(ds1)
    d = int(z["X_train"].shape[1])
    raw = np.full((n, d), np.nan, dtype=np.float64)
    filled = np.zeros(n, dtype=bool)
    for key, idx_key in (("X_train", "train_idx"), ("X_val", "val_idx"), ("X_test", "test_idx")):
        idx = np.asarray(folds[idx_key], dtype=int)
        raw[idx] = np.asarray(z[key], dtype=float) * scale + mean
        filled[idx] = True
    if not bool(filled.all()):
        raise RuntimeError(f"DS1 fold0 arrays do not cover the table: missing {int((~filled).sum())} rows")
    if not np.isfinite(raw).all():
        raise RuntimeError("Unscaled DS1 feature matrix contains non-finite values")
    return raw


def build_fold_split(ds1: pd.DataFrame, ds2: pd.DataFrame, fold: str) -> dict:
    ds1_folds = load_folds("DS1")
    ds2_folds = load_folds("DS2")
    lig = ds1["ligand_smiles"].astype(str)
    ds2_te = np.asarray(ds2_folds[fold]["test_idx"], dtype=int)
    blocked = set(ds2.iloc[ds2_te]["ligand_smiles"].astype(str))
    orig = ds1_folds[fold]
    orig_train = set(lig.iloc[np.asarray(orig["train_idx"], dtype=int)])
    orig_val = set(lig.iloc[np.asarray(orig["val_idx"], dtype=int)])
    orig_test = set(lig.iloc[np.asarray(orig["test_idx"], dtype=int)])
    available = set(lig) - blocked
    val_lig = orig_val & available
    train_lig = available - val_lig
    if len(val_lig) < 20:
        raise RuntimeError(f"{fold}: only {len(val_lig)} remaining val ligands after blocking DS2 test ligands")
    train_idx = np.flatnonzero(lig.isin(train_lig)).astype(int)
    val_idx = np.flatnonzero(lig.isin(val_lig)).astype(int)
    test_idx = np.flatnonzero(lig.isin(blocked)).astype(int)
    leaked_train = orig_train & blocked
    leaked_val = orig_val & blocked
    assert_no_ligand_overlap(train_lig | val_lig, blocked)
    return {
        "fold": fold,
        "protocol": "ds2_fold_matched_ds1_subset",
        "blocked_ligand_count": len(blocked),
        "train_ligand_count": len(train_lig),
        "val_ligand_count": len(val_lig),
        "test_ligand_count": len(blocked),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "orig_ds1_train_ligands_in_ds2_test": len(leaked_train),
        "orig_ds1_val_ligands_in_ds2_test": len(leaked_val),
        "orig_ds1_test_ligands_reused_in_train": len(orig_test & train_lig),
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "ligand_hash_train": ligand_split_hash(sorted(train_lig)),
        "ligand_hash_val": ligand_split_hash(sorted(val_lig)),
        "ligand_hash_test": ligand_split_hash(sorted(blocked)),
        "ds2_test_idx": ds2_te.tolist(),
    }


def prepare() -> list[dict]:
    ds1 = load_table("DS1")
    ds2 = load_table("DS2")
    raw = reconstruct_unscaled_ds1_features(ds1)
    y = ds1["logK"].to_numpy(dtype=float)
    rows = []
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for fold in FOLDS:
        split = build_fold_split(ds1, ds2, fold)
        split_path(fold).parent.mkdir(parents=True, exist_ok=True)
        split_path(fold).write_text(json.dumps(split, indent=2), encoding="utf-8")
        tr = np.asarray(split["train_idx"], dtype=int)
        va = np.asarray(split["val_idx"], dtype=int)
        te = np.asarray(split["test_idx"], dtype=int)
        xtr, xva, xte, prep = preprocess_fold(raw[tr], raw[va], raw[te])
        rd = rdkit_feat_dir(fold)
        rd.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            rd / "arrays.npz",
            X_train=np.asarray(xtr, dtype=np.float32),
            X_val=np.asarray(xva, dtype=np.float32),
            X_test=np.asarray(xte, dtype=np.float32),
            y_train=np.asarray(y[tr], dtype=np.float32),
            y_val=np.asarray(y[va], dtype=np.float32),
            y_test=np.asarray(y[te], dtype=np.float32),
        )
        write_json(rd / "prep.json", prep)
        write_chemprop_fold_csvs(ds1, tr, va, te, chem_dir(fold))
        rec = {k: v for k, v in split.items() if not str(k).endswith("_idx")}
        rows.append(rec)
        print(
            f"{fold}: train {split['n_train']} val {split['n_val']} test {split['n_test']} "
            f"blocked_ligands {split['blocked_ligand_count']} "
            f"leaked_orig_train_ligands {split['orig_ds1_train_ligands_in_ds2_test']}",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(OUT_ROOT / "split_summary.csv", index=False)
    return rows


def load_split(fold: str) -> dict:
    return json.loads(split_path(fold).read_text(encoding="utf-8"))


def load_rdkit_arrays(fold: str) -> dict[str, np.ndarray]:
    z = np.load(rdkit_feat_dir(fold) / "arrays.npz")
    return {k: np.asarray(z[k]) for k in z.files}


def _meta_blob(model_id: str, fold: str, family: str, seed: int, hps: dict, extra: dict, split: dict, member_id: int = 0) -> dict:
    return checkpoint_meta(
        dataset="DS1",
        fold=fold,
        model_id=model_id,
        family=family,
        member_id=int(member_id),
        seed=seed,
        feature_hash="ad_hoc_foldmatched",
        rdkit_version="frozen_unscaled_from_ds1_fold0",
        n_train=int(split["n_train"]),
        n_val=int(split["n_val"]),
        n_test=int(split["n_test"]),
        ligand_hash_train=split["ligand_hash_train"],
        ligand_hash_val=split["ligand_hash_val"],
        ligand_hash_test=split["ligand_hash_test"],
        selected_hps=hps,
        extra={"protocol": "ds2_fold_matched_ds1_subset", **extra},
    )


def _predict_si_uq(kind: str, model, X, grid: dict, seed: int) -> dict:
    device = _device()
    n_mc = int(grid["mc_dropout_passes"])
    n_s = int(grid.get("bayes_posterior_samples", 30))
    if kind == "EVID":
        return predict_evid(model, X, device=device)
    if kind == "HETERO":
        return predict_hetero_mc(model, X, n_mc=n_mc, device=device, seed=seed + 11)
    if kind == "MCDROPOUT":
        return predict_mcdropout_mc(model, X, n_mc=n_mc, device=device, seed=seed + 11)
    if kind == "BAYES":
        return predict_variational_bnn(model, X, n_samples=n_s, device=device, seed=seed + 11)
    if kind == "SNGP":
        return predict_sngp(model, X, device=device)
    raise KeyError(kind)


def _train_si_member(kind: str, arr: dict, cfg: dict, grid: dict, seed: int, lambda_reg=None):
    device = _device()
    n_mc = int(grid["mc_validation_passes"])
    if kind == "EVID":
        return train_evid_mlp(arr["X_train"], arr["y_train"], arr["X_val"], arr["y_val"], cfg, device=device, seed=seed, lambda_reg=float(lambda_reg))
    if kind == "HETERO":
        return train_hetero_mlp(arr["X_train"], arr["y_train"], arr["X_val"], arr["y_val"], cfg, device=device, seed=seed, n_mc_val=n_mc)
    if kind == "MCDROPOUT":
        return train_mcdropout_mlp(arr["X_train"], arr["y_train"], arr["X_val"], arr["y_val"], cfg, device=device, seed=seed, n_mc_val=n_mc)
    if kind == "BAYES":
        return train_variational_bnn(
            arr["X_train"], arr["y_train"], arr["X_val"], arr["y_val"], cfg,
            device=device, seed=seed, n_samples_val=n_mc, beta_kl=float(grid.get("beta_KL", 1.0)),
        )
    if kind == "SNGP":
        local = dict(cfg)
        local["sngp_num_features"] = grid.get("sngp_num_features", 512)
        local["sngp_lengthscale"] = grid.get("sngp_lengthscale", 2.0)
        local["sngp_ridge"] = grid.get("sngp_ridge", 1.0)
        return train_sngp(arr["X_train"], arr["y_train"], arr["X_val"], arr["y_val"], local, device=device, seed=seed)
    raise KeyError(kind)


def _save_si_member(kind: str, model, path: Path, blob: dict, extra=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind in {"EVID", "HETERO", "MCDROPOUT"}:
        save_torch(model, path, blob)
    elif kind == "BAYES":
        save_bnn(model, path, blob)
    elif kind == "SNGP":
        save_sngp(model, path, blob)
    elif kind == "GPR_SPARSE":
        save_gpr(extra if extra is not None else {"meta": blob}, path)
    else:
        raise KeyError(kind)
    write_json(path.with_name("meta.json"), blob)


def _train_si_model(model_id: str, fold: str, cfg: dict, grid: dict, rec: pd.DataFrame, split: dict) -> dict:
    spec = SI_SPECS[model_id]
    arr = load_rdkit_arrays(fold)
    fid = int(str(fold).replace("fold", ""))
    kind = spec["kind"]
    n_members = 5 if spec["ens"] else 1
    member_outs = []
    seconds = []
    lambda_reg = None
    if kind == "EVID":
        if fold not in _EVID_LAMBDA:
            seed0 = model_seed(fid, 0, grid["model_offsets"]["EVID"], grid["base_seed"], dataset="DS1")
            seed_everything(seed0)
            _model, _meta, lam = select_evidential_lambda(
                arr["X_train"], arr["y_train"], arr["X_val"], arr["y_val"], cfg,
                grid=list(grid["evidential_lambda_grid"]), device=_device(), seed=seed0,
            )
            _EVID_LAMBDA[fold] = float(lam)
            print(f"  evidential lambda {fold}={lam}", flush=True)
        lambda_reg = _EVID_LAMBDA[fold]
    for m in range(n_members):
        seed = model_seed(fid, m, grid["model_offsets"][spec["offset"]], grid["base_seed"], dataset="DS1")
        ckpt = ckpt_dir(model_id, fold, m)
        if kind == "GPR_SPARSE":
            seed_everything(seed)
            local = dict(cfg)
            for k in ("gpr_n_inducing", "gpr_sparse_epochs", "gpr_sparse_patience", "gpr_sparse_batch_size", "gpr_jitter"):
                if k in grid:
                    local[k] = grid[k]
            model, meta, scaler = train_sparse_gpr(arr["X_train"], arr["y_train"], arr["X_val"], arr["y_val"], local, device=_device(), seed=seed)
            blob = _meta_blob(model_id, fold, spec["family"], seed, {"gpr_type": "sparse_variational"}, meta, split, member_id=0)
            payload = {"state_dict": model.state_dict(), "meta": meta, "scaler": scaler.hash_payload(), "class": "SparseGPR", "Z": model.Z.detach().cpu()}
            _save_si_member(kind, model, ckpt / "model.pt", blob, extra=payload)
            uq = predict_sparse_gpr(model, scaler, arr["X_test"], device=_device())
            seconds.append(float(meta.get("train_seconds", 0)))
            member_outs.append(uq)
            continue
        seed_everything(seed)
        model, meta = _train_si_member(kind, arr, cfg, grid, seed, lambda_reg=lambda_reg)
        blob = _meta_blob(model_id, fold, spec["family"], seed, {"kind": kind, "lambda_reg": lambda_reg}, meta, split, member_id=m)
        _save_si_member(kind, model, ckpt / "model.pt", blob)
        seconds.append(float(meta.get("train_seconds", 0)))
        uq = _predict_si_uq(kind, model, arr["X_test"], grid, seed)
        member_outs.append(uq)
        print(f"  trained {model_id} {fold} member_{m} s={seconds[-1]}", flush=True)
    if spec["ens"]:
        uq = aggregate_members(member_outs)
    else:
        uq = member_outs[0]
    rec["y_pred"] = np.asarray(uq["mu"], dtype=float)
    rec["sigma_total"] = np.asarray(uq.get("sigma_total", np.full(len(rec), np.nan)), dtype=float)
    return {
        "train_seconds": float(np.nansum(seconds)),
        "n_members": n_members,
        "lambda_reg": lambda_reg,
        "kind": kind,
    }


def train_one(model_id: str, fold: str, cfg: dict, grid: dict) -> dict:
    split = load_split(fold)
    ckpt = ckpt_dir(model_id, fold)
    preds = pred_path(model_id, fold)
    if preds.exists() and (ckpt / "meta.json").exists():
        print(f"skip existing {model_id} {fold}", flush=True)
        return json.loads((ckpt / "meta.json").read_text(encoding="utf-8"))
    ds1 = load_table("DS1")
    te = np.asarray(split["test_idx"], dtype=int)
    rec = pd.DataFrame(
        {
            "record_id": ds1.iloc[te]["record_id"].astype(str).to_numpy(),
            "ligand_smiles": ds1.iloc[te]["ligand_smiles"].astype(str).to_numpy(),
            "y_true": ds1.iloc[te]["logK"].to_numpy(dtype=float),
            "fold_id": fold,
            "source_fold": fold,
            "model_id": model_id,
            "evaluation_partition": "ds2_fold_matched_heldout_ligands",
        }
    )
    if model_id in SI_SPECS:
        meta = _train_si_model(model_id, fold, cfg, grid, rec, split)
        blob = _meta_blob(model_id, fold, SI_SPECS[model_id]["family"], 0, {"si_foldmatched": True}, meta, split)
        write_json(ckpt / "meta.json", blob)
        preds.parent.mkdir(parents=True, exist_ok=True)
        rec.to_csv(preds, index=False)
        print(f"trained {model_id} {fold} s={blob.get('train_seconds')}", flush=True)
        return blob
    fid = int(str(fold).replace("fold", ""))
    seed = model_seed(fid, 0, grid["model_offsets"][OFFSET[model_id]], grid["base_seed"], dataset="DS1")
    seed_everything(seed)
    family = FAMILY[model_id]
    if model_id == "DS1_RDKIT_MLP_POINT":
        arr = load_rdkit_arrays(fold)
        model, meta = train_point_mlp(arr["X_train"], arr["y_train"], arr["X_val"], arr["y_val"], cfg, device=_device(), seed=seed)
        hps = {
            "mlp_hidden_dims": cfg["mlp_hidden_dims"],
            "mlp_lr": cfg["mlp_lr"],
            "mlp_weight_decay": cfg["mlp_weight_decay"],
            "dropout": cfg["mlp_dropout_point"],
        }
        blob = _meta_blob(model_id, fold, family, seed, hps, meta, split)
        save_mlp_checkpoint(model, ckpt / "model.pt", blob)
        rec["y_pred"] = predict_mlp(model, arr["X_test"], device=_device())
        rec["sigma_total"] = np.nan
    elif model_id == "DS1_RDKIT_XGB_POINT":
        arr = load_rdkit_arrays(fold)
        model, meta = train_point_xgb(arr["X_train"], arr["y_train"], arr["X_val"], arr["y_val"], cfg, seed=seed)
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_model(ckpt / "model.json")
        blob = _meta_blob(model_id, fold, family, seed, {"xgb": {k: cfg[k] for k in cfg if str(k).startswith("xgb_")}}, meta, split)
        rec["y_pred"] = np.asarray(model.predict(arr["X_test"]), dtype=float)
        rec["sigma_total"] = np.nan
    elif model_id == "DS1_GRAPH_DMPNN_POINT":
        work = chem_dir(fold)
        model_pt = ckpt / "fold_0" / "model_0" / "model.pt"
        if not model_pt.exists():
            meta = train_dmpnn_point(work / "train.csv", work / "train_features.csv", work / "val.csv", work / "val_features.csv", ckpt, seed=seed)
        else:
            meta = {"train_seconds": 0, "reused_local": True}
        write_chemprop_identity(ckpt, dataset="DS1", fold=fold, member_id=0, model_id=model_id)
        tmp = ckpt / "chemprop_raw_preds.csv"
        rec["y_pred"] = np.asarray(predict_dmpnn_point(work / "test.csv", work / "test_features.csv", ckpt, tmp), dtype=float)
        rec["sigma_total"] = np.nan
        blob = _meta_blob(model_id, fold, family, seed, {"chemprop": "point", "loss": "mse"}, meta, split)
    elif model_id == "DS1_GRAPH_DMPNN_EVID":
        work = chem_dir(fold)
        lam = float(grid["chemprop_evidential_regularization"])
        extra = ["--loss_function", "evidential", "--evidential_regularization", str(lam)]
        model_pt = ckpt / "fold_0" / "model_0" / "model.pt"
        if not model_pt.exists():
            meta = train_dmpnn_point(
                work / "train.csv",
                work / "train_features.csv",
                work / "val.csv",
                work / "val_features.csv",
                ckpt,
                seed=seed,
                extra_args=extra,
            )
        else:
            meta = {"train_seconds": 0, "reused_local": True}
        write_chemprop_identity(ckpt, dataset="DS1", fold=fold, member_id=0, model_id=model_id)
        uq = predict_dmpnn_evidential_uq(
            work / "test.csv",
            work / "test_features.csv",
            ckpt,
            n_expected=int(split["n_test"]),
            dataset="DS1",
            fold=fold,
            member_id=0,
            model_id=model_id,
        )
        rec["y_pred"] = np.asarray(uq["mu"], dtype=float)
        rec["sigma_total"] = np.asarray(uq["sigma_total"], dtype=float)
        blob = _meta_blob(
            model_id,
            fold,
            family,
            seed,
            {"loss_function": "evidential", "evidential_regularization": lam},
            meta,
            split,
        )
    else:
        raise KeyError(model_id)
    write_json(ckpt / "meta.json", blob)
    preds.parent.mkdir(parents=True, exist_ok=True)
    rec.to_csv(preds, index=False)
    print(f"trained {model_id} {fold} s={blob.get('train_seconds')}", flush=True)
    return blob


def train_models(model_ids: tuple[str, ...]) -> None:
    cfg = load_shared_cfg()
    grid = load_active_grid_cfg()
    for model_id in model_ids:
        for fold in FOLDS:
            train_one(model_id, fold, cfg, grid)


def evaluate(model_ids: tuple[str, ...]) -> pd.DataFrame:
    ds2 = load_table("DS2")
    ds2_folds = load_folds("DS2")
    rows = []
    for model_id in model_ids:
        frames = [pd.read_csv(pred_path(model_id, fold)) for fold in FOLDS]
        pred = pd.concat(frames, ignore_index=True)
        dup = pred["record_id"].astype(str).duplicated()
        if bool(dup.any()):
            raise RuntimeError(f"{model_id}: duplicate DS1 record_id in fold-matched predictions")
        for fold in FOLDS:
            te = np.asarray(ds2_folds[fold]["test_idx"], dtype=int)
            pairs = ds2.iloc[te].copy()
            assembled = assemble_ad_hoc_pairs(pairs, pred, model_id, fold)
            for scope in SCOPES:
                rows.append(evaluate_ad_hoc_fold(assembled, model_id, scope))
    fold_df = pd.DataFrame(rows)
    dest_fold = OUT_ROOT / "ad_hoc_selectivity_fold_metrics.csv"
    if dest_fold.exists():
        old = pd.read_csv(dest_fold)
        old = old[~old.source_model_id.astype(str).isin(set(model_ids))].copy()
        fold_df = pd.concat([old, fold_df], ignore_index=True)
    summary = summarize_fold_metrics(fold_df)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(dest_fold, index=False)
    summary.to_csv(OUT_ROOT / "ad_hoc_selectivity_summary.csv", index=False)
    return summary


def _pm(mean: float, sd: float, nd: int) -> str:
    if mean is None or (isinstance(mean, float) and not np.isfinite(mean)):
        return "NA (no UQ)"
    return f"{float(mean):.{nd}f} ± {float(sd):.{nd}f}"


def training_time_rows(model_ids: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for model_id in model_ids:
        secs = []
        ntr = []
        for fold in FOLDS:
            meta = json.loads((ckpt_dir(model_id, fold) / "meta.json").read_text(encoding="utf-8"))
            split = load_split(fold)
            secs.append(float(meta.get("train_seconds", np.nan)))
            ntr.append(float(split["n_train"]))
        per = np.asarray(secs, dtype=float) / (np.asarray(ntr, dtype=float) / 1000.0)
        rows.append(
            {
                "model_id": model_id,
                "construction": "ad hoc DS1 difference",
                "seconds_per_1000_mean": float(np.nanmean(per)),
                "seconds_per_1000_sd": float(np.nanstd(per, ddof=1)),
                "n_members": 1,
                "n_train_mean": float(np.mean(ntr)),
            }
        )
    return pd.DataFrame(rows)


def write_iz_table2(summary: pd.DataFrame) -> Path:
    times = training_time_rows(TABLE2_MODELS)
    times.to_csv(OUT_ROOT / "iz_ad_hoc_training_time.csv", index=False)
    combined = pd.read_csv(TABLES / "iz_combined_model_comparison.csv")
    adhoc = summary[summary.scope.astype(str) == "all_pairs"].copy()
    by_id = adhoc.set_index("source_model_id")
    tmap = times.set_index("model_id")
    out_rows = []
    for _, row in combined.iterrows():
        rec = row.to_dict()
        if str(rec.get("selectivity construction")) != "ad hoc DS1 difference":
            out_rows.append(rec)
            continue
        name = str(rec["model"])
        mid = {
            "RDKit XGBoost": "DS1_RDKIT_XGB_POINT",
            "RDKit MLP": "DS1_RDKIT_MLP_POINT",
            "Chemprop point": "DS1_GRAPH_DMPNN_POINT",
            "Chemprop evidential": "DS1_GRAPH_DMPNN_EVID",
        }[name]
        s = by_id.loc[mid]
        t = tmap.loc[mid]
        rec["training time / 1,000 rows (s)"] = _pm(t.seconds_per_1000_mean, t.seconds_per_1000_sd, 2)
        rec["MoP RMSE"] = _pm(s.mop_rmse, s.mop_rmse_sd, 3)
        rec["MoP $R^2$"] = _pm(s.mop_r2, s.mop_r2_sd, 3)
        rec["row-level $R^2$"] = _pm(s.row_r2, s.row_r2_sd, 3)
        if mid == "DS1_GRAPH_DMPNN_EVID":
            rec["MoP PICP@90"] = _pm(s.mop_picp_90, s.mop_picp_90_sd, 3)
            rec["MoP ACE"] = _pm(s.mop_ace, s.mop_ace_sd, 3)
        else:
            rec["MoP PICP@90"] = "NA (no UQ)"
            rec["MoP ACE"] = "NA (no UQ)"
        out_rows.append(rec)
    out = pd.DataFrame(out_rows)
    dest = TABLES / "iz_combined_model_comparison.csv"
    out.to_csv(dest, index=False)
    return dest


SI_CMP = REMAKE_ROOT / "results" / "downstream_cv" / "ad_hoc_selectivity" / "ad_hoc_selectivity_comparison.csv"
SI_SUM = REMAKE_ROOT / "results" / "downstream_cv" / "ad_hoc_selectivity" / "ad_hoc_selectivity_summary.csv"
SI_FOLD = REMAKE_ROOT / "results" / "downstream_cv" / "ad_hoc_selectivity" / "ad_hoc_selectivity_fold_metrics.csv"
MAN_CMP = TABLES / "ad_hoc_selectivity_comparison.csv"


def patch_si_table2_ad_hoc_rows(summary: pd.DataFrame, fold_df: pd.DataFrame) -> None:
    """Overwrite fold-matched ad hoc rows in SI CSVs. Direct DS2 and other models stay."""
    ids = set(summary.source_model_id.astype(str))

    def _patch_summary(path: Path) -> None:
        if not path.exists():
            return
        old = pd.read_csv(path)
        keep = old[~(old.source_model_id.astype(str).isin(ids) & (old.selectivity_method.astype(str) == "ad_hoc_ds1_difference"))].copy()
        new = summary[summary.source_model_id.astype(str).isin(ids)].copy()
        extra = [c for c in old.columns if c not in new.columns]
        for c in extra:
            new[c] = np.nan
        if "method_class" in old.columns:
            point_ids = {"DS1_RDKIT_XGB_POINT", "DS1_RDKIT_MLP_POINT", "DS1_GRAPH_DMPNN_POINT"}
            new["method_class"] = np.where(
                new.source_model_id.astype(str).isin(point_ids),
                "ad_hoc_ds1_point",
                "ad_hoc_ds1_uncertainty",
            )
        aligned = new.reindex(columns=old.columns)
        pd.concat([keep, aligned], ignore_index=True).to_csv(path, index=False)

    _patch_summary(SI_CMP)
    _patch_summary(SI_SUM)
    _patch_summary(MAN_CMP)
    if SI_FOLD.exists() and not fold_df.empty:
        old = pd.read_csv(SI_FOLD)
        keep = old[~(old.source_model_id.astype(str).isin(ids) & (old.selectivity_method.astype(str) == "ad_hoc_ds1_difference"))].copy()
        add = fold_df[fold_df.source_model_id.astype(str).isin(ids)].copy()
        extra = [c for c in old.columns if c not in add.columns]
        for c in extra:
            add[c] = np.nan
        aligned = add.reindex(columns=old.columns)
        pd.concat([keep, aligned], ignore_index=True).to_csv(SI_FOLD, index=False)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fold-matched Table 2 ad hoc DS1 retraining")
    p.add_argument("--stage", choices=("prepare", "train", "eval", "all"), default="all")
    p.add_argument(
        "--models",
        default="xgb,mlp,chemprop_point,chemprop_evid",
        help="Comma-separated model ids or aliases: xgb,mlp,chemprop_point,chemprop_evid,table2,si",
    )
    return p.parse_args(argv)


def _models_from_flag(raw: str) -> tuple[str, ...]:
    alias = {
        "xgb": "DS1_RDKIT_XGB_POINT",
        "mlp": "DS1_RDKIT_MLP_POINT",
        "chemprop_point": "DS1_GRAPH_DMPNN_POINT",
        "chemprop_evid": "DS1_GRAPH_DMPNN_EVID",
        "table2": ",".join(TABLE2_MODELS),
        "si": ",".join(SI_REMAINING_MODELS),
        "si_remaining": ",".join(SI_REMAINING_MODELS),
    }
    out = []
    for tok in str(raw).split(","):
        tok = tok.strip()
        if not tok:
            continue
        mapped = alias.get(tok, tok)
        if "," in mapped and tok in {"table2", "si", "si_remaining"}:
            out.extend([p for p in mapped.split(",") if p])
        else:
            out.append(mapped)
    return tuple(out)


def main(argv=None) -> int:
    args = parse_args(argv)
    models = _models_from_flag(args.models)
    if args.stage == "eval":
        # Called as a pipeline step to re-apply the fold-matched rows that
        # run_ad_hoc_selectivity.py overwrites. The stored predictions live in
        # ad_hoc_foldmatched_results.7z / checkpoints_ad_hoc.7z (Git LFS). If they were
        # never unpacked there is nothing to re-apply; say so and leave the frozen-grid
        # rows in place rather than failing the whole rebuild.
        needed = [m for m in TABLE2_MODELS if not all(pred_path(m, f).exists() for f in FOLDS)]
        if needed:
            print(
                "ad_hoc_foldmatched: stored fold-matched predictions missing for "
                + ", ".join(needed)
                + "; skipping. Table 2 and SI S13 will show the independently folded DS1 "
                "alignment, not the fold-matched one. Unpack artifacts/ad_hoc_foldmatched_results.7z "
                "(see artifacts/README.md) or run --stage all to retrain.",
                flush=True,
            )
            return 0
    if args.stage in {"prepare", "all"}:
        prepare()
    if args.stage in {"train", "all"}:
        train_models(models)
    if args.stage in {"eval", "all"}:
        summary = evaluate(TABLE2_MODELS if args.stage == "all" else models)
        if set(TABLE2_MODELS).issubset(set(models)) or args.stage == "eval":
            if all(pred_path(m, "fold0").exists() for m in TABLE2_MODELS):
                summary = evaluate(TABLE2_MODELS)
                write_iz_table2(summary)
                fold_df = pd.read_csv(OUT_ROOT / "ad_hoc_selectivity_fold_metrics.csv")
                patch_si_table2_ad_hoc_rows(summary, fold_df)
        si_ready = [m for m in SI_REMAINING_MODELS if all(pred_path(m, f).exists() for f in FOLDS)]
        if si_ready:
            si_summary = evaluate(tuple(si_ready))
            si_fold = pd.read_csv(OUT_ROOT / "ad_hoc_selectivity_fold_metrics.csv")
            patch_si_table2_ad_hoc_rows(si_summary, si_fold)
            print(si_summary[si_summary.scope == "ree_pairs"][["source_model", "mop_rmse", "mop_rmse_sd"]].to_string(index=False), flush=True)
        print(summary[summary.scope == "all_pairs"][["source_model", "mop_rmse", "mop_rmse_sd", "row_r2"]].to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
