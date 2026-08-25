"""Shared DS1/DS2 training / checkpoint / prediction protocol."""
from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from uq_remake import REMAKE_ROOT

NAN = np.nan

SCHEMA_COLUMNS = [
    "record_id",
    "row_id",
    "table_idx",
    "fold_id",
    "evaluation_partition",
    "member_id",
    "model_id",
    "dataset",
    "target_name",
    "y_true",
    "y_pred",
    "prediction_type",
    "uncertainty_status",
    "uncertainty_method",
    "interval_method",
    "sigma_aleatoric",
    "var_aleatoric",
    "sigma_epistemic",
    "var_epistemic",
    "sigma_epistemic_within",
    "var_epistemic_within",
    "sigma_epistemic_between",
    "var_epistemic_between",
    "sigma_epistemic_total",
    "var_epistemic_total",
    "sigma_total",
    "var_total",
    "gamma",
    "nu",
    "alpha",
    "beta",
    "std_total",
    "n_posterior_samples",
    "prediction_seed",
    "checkpoint_id",
    "feature_schema_hash",
    "preprocessing_schema_hash",
    "lower_90",
    "upper_90",
    "chemprop_version",
    "ligand_smiles",
    "family",
    "metal_species_pair",
    "is_ree_pair",
    "pair_id",
]

UQ_SIGMA_COLS = [
    "sigma_aleatoric",
    "var_aleatoric",
    "sigma_epistemic",
    "var_epistemic",
    "sigma_epistemic_within",
    "var_epistemic_within",
    "sigma_epistemic_between",
    "var_epistemic_between",
    "sigma_epistemic_total",
    "var_epistemic_total",
    "sigma_total",
    "var_total",
]


def load_active_grid_cfg() -> dict:
    return json.loads((REMAKE_ROOT / "configs" / "active_grid.json").read_text(encoding="utf-8"))


def load_shared_cfg() -> dict:
    return json.loads((REMAKE_ROOT / "configs" / "shared_dmpnn.json").read_text(encoding="utf-8"))


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


PREPROCESSING_SCHEMA_HASH = "imputation=false;pca=false;fold_train_standardscaler"


def model_seed(fold_id: int, member_id: int, model_offset: int, base_seed: int = 42, dataset: str = "DS1") -> int:
    ds_off = 0 if str(dataset).upper() == "DS1" else 100_000
    return int(base_seed) + ds_off + int(fold_id) * 1000 + int(member_id) * 10 + int(model_offset)


def fold_int(fold_name: str) -> int:
    return int(str(fold_name).replace("fold", ""))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ligand_split_hash(ligands) -> str:
    payload = "\n".join(sorted(str(x) for x in ligands))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def canonical_model_id(dataset: str, family: str) -> str:
    """Map active-grid family names to glossary identifiers."""
    ds = dataset.upper()
    mapping = {
        "POINT_MLP": f"{ds}_RDKIT_MLP_POINT",
        "POINT_XGB": f"{ds}_RDKIT_XGB_POINT",
        "CHEMPROP_POINT_NATIVE": f"{ds}_GRAPH_DMPNN_POINT",
        "EVID_MLP": f"{ds}_RDKIT_MLP_EVID",
        "EVID_MLPENS5": f"{ds}_RDKIT_MLPENS5_EVID",
        "HETERO_MLP": f"{ds}_RDKIT_MLP_HETERO",
        "HETERO_MLPENS5": f"{ds}_RDKIT_MLPENS5_HETERO",
        "MCDROPOUT_MLP": f"{ds}_RDKIT_MLP_MCDROPOUT",
        "MCDROPOUT_MLPENS5": f"{ds}_RDKIT_MLPENS5_MCDROPOUT",
        "CHEMPROP_EVIDENTIAL_NATIVE": f"{ds}_GRAPH_DMPNN_EVID",
        "BAYES_FULL": f"{ds}_RDKIT_MLP_BAYES_FULL",
        "BAYES_FULLENS5": f"{ds}_RDKIT_MLPENS5_BAYES_FULL",
        "SNGP": f"{ds}_RDKIT_SNGP",
        "SNGPENS5": f"{ds}_RDKIT_SNGPENS5",
        "GPR": f"{ds}_RDKIT_GPR",
        "GPR_SPARSE": f"{ds}_RDKIT_GPR_SPARSE",
    }
    if family not in mapping:
        raise KeyError(f"Unknown active family {family!r}")
    return mapping[family]


def checkpoint_dir(model_id: str, fold: str, member_id: int | None = None) -> Path:
    root = REMAKE_ROOT / "checkpoints" / "active" / model_id / fold
    if member_id is None:
        return root
    return root / f"member_{int(member_id)}"


def preds_dir(model_id: str, fold: str) -> Path:
    return REMAKE_ROOT / "results" / "active_grid" / model_id / fold


def load_folds(dataset: str) -> dict:
    name = "project_folds.json" if dataset.upper() == "DS1" else "ds2_project_folds.json"
    return json.loads((REMAKE_ROOT / "data" / "splits" / name).read_text(encoding="utf-8"))


def load_table(dataset: str) -> pd.DataFrame:
    path = REMAKE_ROOT / "data" / "processed" / ("ds1.csv" if dataset.upper() == "DS1" else "ds2.csv")
    return pd.read_csv(path)


def target_name(dataset: str) -> str:
    return "logK" if dataset.upper() == "DS1" else "selectivity_logK"


def record_ids_for_split(df: pd.DataFrame, dataset: str, idx: np.ndarray) -> np.ndarray:
    sub = df.iloc[np.asarray(idx, dtype=int)]
    if dataset.upper() == "DS1":
        return sub["record_id"].astype(str).to_numpy()
    return sub["pair_id"].astype(str).to_numpy()


def extra_meta_frame(df: pd.DataFrame, dataset: str, idx: np.ndarray) -> pd.DataFrame:
    sub = df.iloc[np.asarray(idx, dtype=int)].reset_index(drop=True)
    out = pd.DataFrame(index=sub.index)
    out["ligand_smiles"] = sub["ligand_smiles"].astype(str).to_numpy()
    if dataset.upper() == "DS1":
        out["family"] = sub["family"].astype(str).to_numpy()
        out["metal_species_pair"] = ""
        out["is_ree_pair"] = ""
        out["pair_id"] = ""
    else:
        out["family"] = "metal_metal_pair"
        out["metal_species_pair"] = sub["metal_species_pair"].astype(str).to_numpy()
        out["is_ree_pair"] = sub["is_ree_pair"].astype(bool).to_numpy()
        out["pair_id"] = sub["pair_id"].astype(str).to_numpy()
    return out


def assert_row_alignment(
    *,
    n_pred: int,
    n_table: int,
    n_features: int | None = None,
    n_chemprop: int | None = None,
    y_pred_source: np.ndarray | None = None,
    y_table: np.ndarray | None = None,
    context: str = "",
) -> dict:
    """Fail rather than truncate or SMILES-join."""
    if n_pred != n_table:
        raise ValueError(f"{context}: n_pred={n_pred} != n_table={n_table}")
    if n_features is not None and n_features != n_table:
        raise ValueError(f"{context}: n_features={n_features} != n_table={n_table}")
    if n_chemprop is not None and n_chemprop != n_table:
        raise ValueError(f"{context}: n_chemprop={n_chemprop} != n_table={n_table}")
    y_match = None
    if y_pred_source is not None and y_table is not None:
        a = np.asarray(y_pred_source, dtype=float)
        b = np.asarray(y_table, dtype=float)
        if len(a) != len(b):
            raise ValueError(f"{context}: y length {len(a)} != {len(b)}")
        y_match = bool(np.allclose(a, b, equal_nan=True, rtol=0, atol=1e-8))
        if not y_match:
            raise ValueError(f"{context}: Chemprop/table target values are not aligned")
    return {
        "n_pred": int(n_pred),
        "n_table": int(n_table),
        "n_features": None if n_features is None else int(n_features),
        "n_chemprop": None if n_chemprop is None else int(n_chemprop),
        "y_values_match": y_match,
        "truncated": False,
        "joined_on_smiles_only": False,
        "context": context,
        "ok": True,
    }


def empty_uq_fields(n: int) -> dict[str, np.ndarray]:
    nan = np.full(n, NAN, dtype=float)
    return {c: nan.copy() for c in UQ_SIGMA_COLS}


def prediction_frame(
    *,
    dataset: str,
    fold: str,
    model_id: str,
    member_id: int,
    table_idx: np.ndarray,
    record_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    extra: pd.DataFrame,
    prediction_type: str,
    uncertainty_status: str,
    uncertainty_method: str,
    uq: dict | None = None,
    interval_method: str = "",
    chemprop_version: str = "",
    artifact_meta: dict | None = None,
) -> pd.DataFrame:
    n = len(y_pred)
    if len(y_true) != n or len(record_ids) != n or len(table_idx) != n or len(extra) != n:
        raise ValueError("prediction_frame length mismatch")
    am = artifact_meta or {}
    if uncertainty_status == "unavailable":
        fields = empty_uq_fields(n)
        gamma = nu = alpha = beta = np.full(n, NAN)
        interval_method = interval_method or ""
        uq_use: dict = {}
    else:
        if uq is None:
            raise ValueError("valid UQ requires a uq dict; never fill with zeros or residual SD")
        uq_use = uq
        fields = {}
        for c in UQ_SIGMA_COLS:
            if c not in uq:
                raise KeyError(f"UQ dict missing {c}")
            arr = np.asarray(uq[c], dtype=float).reshape(-1)
            if len(arr) != n:
                raise ValueError(f"{c} length {len(arr)} != {n}")
            fields[c] = arr
        gamma = np.asarray(uq.get("gamma", np.full(n, NAN)), dtype=float).reshape(-1)
        nu = np.asarray(uq.get("nu", np.full(n, NAN)), dtype=float).reshape(-1)
        alpha = np.asarray(uq.get("alpha", np.full(n, NAN)), dtype=float).reshape(-1)
        beta = np.asarray(uq.get("beta", np.full(n, NAN)), dtype=float).reshape(-1)
        if len(gamma) != n:
            gamma = np.full(n, NAN)
        interval_method = interval_method or str(uq.get("interval_method", ""))
        chemprop_version = chemprop_version or str(uq.get("chemprop_version", ""))
    df = pd.DataFrame(
        {
            "record_id": np.asarray(record_ids).astype(str),
            "row_id": np.arange(n, dtype=int),
            "table_idx": np.asarray(table_idx, dtype=int),
            "fold_id": fold,
            "evaluation_partition": "heldout_fold",
            "member_id": int(member_id),
            "model_id": model_id,
            "dataset": dataset.upper(),
            "target_name": target_name(dataset),
            "y_true": np.asarray(y_true, dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
            "prediction_type": prediction_type,
            "uncertainty_status": uncertainty_status,
            "uncertainty_method": uncertainty_method,
            "interval_method": interval_method,
            "chemprop_version": chemprop_version,
            "gamma": gamma,
            "nu": nu,
            "alpha": alpha,
            "beta": beta,
        }
    )
    for c, arr in fields.items():
        df[c] = arr
    if "sigma_total" in df.columns:
        df["std_total"] = df["sigma_total"]
    else:
        df["std_total"] = np.full(n, NAN)
    if uq_use.get("std_total") is not None:
        df["std_total"] = np.asarray(uq_use["std_total"], dtype=float).reshape(-1)
    for c in ("lower_90", "upper_90"):
        if c in uq_use:
            df[c] = np.asarray(uq_use[c], dtype=float).reshape(-1)
        else:
            df[c] = np.full(n, NAN)
    n_post = int(uq_use.get("n_posterior_samples", am.get("n_posterior_samples", 0) or 0) or 0)
    if n_post <= 0:
        n_post = int(uq_use.get("n_mc", am.get("n_mc", 0) or 0) or 0)
    df["n_posterior_samples"] = n_post
    df["prediction_seed"] = str(uq_use.get("prediction_seed", am.get("prediction_seed", "")))
    df["checkpoint_id"] = str(am.get("checkpoint_id", f"{model_id}/{fold}/member_{member_id}"))
    df["feature_schema_hash"] = str(am.get("feature_schema_hash", ""))
    df["preprocessing_schema_hash"] = str(am.get("preprocessing_schema_hash", PREPROCESSING_SCHEMA_HASH))
    for col in extra.columns:
        df[col] = extra[col].to_numpy()
    return df[SCHEMA_COLUMNS]


def write_predictions(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"prediction schema missing {missing}")
    df[SCHEMA_COLUMNS].to_csv(path, index=False)
    return path


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def checkpoint_meta(
    *,
    dataset: str,
    fold: str,
    model_id: str,
    family: str,
    member_id: int,
    seed: int,
    feature_hash: str,
    rdkit_version: str,
    n_train: int,
    n_val: int,
    n_test: int,
    ligand_hash_train: str,
    ligand_hash_val: str,
    ligand_hash_test: str,
    selected_hps: dict,
    extra: dict | None = None,
) -> dict:
    meta = {
        "dataset": dataset,
        "fold": fold,
        "model_id": model_id,
        "family": family,
        "member_id": int(member_id),
        "seed": int(seed),
        "feature_hash": feature_hash,
        "rdkit_version": rdkit_version,
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),
        "ligand_hash_train": ligand_hash_train,
        "ligand_hash_val": ligand_hash_val,
        "ligand_hash_test": ligand_hash_test,
        "selected_hps": selected_hps,
        "pca": False,
        "imputation": False,
        "created_utc": utc_now(),
        "grid": str(load_active_grid_cfg().get("grid_name", "shared_ds1_ds2_uq_v3")),
        "chemprop_mc_dropout": False,
    }
    if extra:
        meta.update(extra)
    return meta
