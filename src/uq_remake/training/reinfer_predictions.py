"""Re-infer held-out fold predictions from existing checkpoints. No retraining.

Writes to results/cv_predictions/fold_test_predictions/<MODEL_ID>/fold*/fold_test_predictions.csv.
Used after full retrain or to refresh OOF dumps for pair-event support diagnostics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from uq_remake import REMAKE_ROOT
from uq_remake.downstream.ad_hoc_infer import _predict_model
from uq_remake.metrics.cv_protocol import EVALUATION_PARTITION
from uq_remake.training.protocol import (
    extra_meta_frame,
    load_folds,
    load_table,
    record_ids_for_split,
    target_name,
)

PRED_ROOT = REMAKE_ROOT / "results" / "cv_predictions" / "fold_test_predictions"
CKPT_ROOT = REMAKE_ROOT / "checkpoints" / "active"
FOLDS = ("fold0", "fold1", "fold2", "fold3", "fold4")
SKIP = {"DS1_RDKIT_GPR", "DS2_RDKIT_GPR"}


def _models() -> list[str]:
    return sorted(p.name for p in CKPT_ROOT.iterdir() if p.is_dir() and p.name not in SKIP)


def _load_heldout(dataset: str, fold: str) -> dict:
    table = load_table(dataset)
    split = load_folds(dataset)[fold]
    te = np.asarray(split["test_idx"], dtype=int)
    schema = "rdkit_ds1" if dataset == "DS1" else "rdkit_ds2"
    arrays = np.load(REMAKE_ROOT / "data" / "processed" / "features" / schema / fold / "arrays.npz")
    x_test = np.asarray(arrays["X_test"])
    if len(x_test) != len(te):
        raise RuntimeError(f"{dataset} {fold}: X_test {len(x_test)} != split {len(te)}")
    return {
        "X_test": x_test,
        "te": te,
        "y_true": table.iloc[te][target_name(dataset)].to_numpy(dtype=float),
        "record_ids": record_ids_for_split(table, dataset, te),
        "extra": extra_meta_frame(table, dataset, te),
    }


def _out_path(model_id: str, fold: str) -> Path:
    return PRED_ROOT / model_id / fold / "fold_test_predictions.csv"


def _status(model_id: str, uq: dict, n: int) -> np.ndarray:
    if "POINT" in model_id or "XGB" in model_id:
        return np.full(n, "unavailable")
    if "nig_valid" in uq:
        valid = np.asarray(uq["nig_valid"], dtype=bool).reshape(-1)
        return np.where(valid, "valid", "invalid_nig")
    return np.full(n, "valid")


def _arr(uq: dict, key: str, n: int) -> np.ndarray:
    if key not in uq or uq[key] is None:
        return np.full(n, np.nan)
    val = uq[key]
    if np.isscalar(val):
        return np.full(n, np.nan)
    out = np.asarray(val, dtype=float).reshape(-1)
    if len(out) != n:
        return np.full(n, np.nan)
    return out


def write_tabular(model_id: str, fold: str) -> Path:
    dest = _out_path(model_id, fold)
    if dest.exists():
        return dest
    dataset = "DS1" if model_id.startswith("DS1_") else "DS2"
    bundle = _load_heldout(dataset, fold)
    uq = _predict_model(model_id, fold, bundle["X_test"], bundle)
    n = len(bundle["y_true"])
    extra = bundle["extra"]
    member_id = -1 if "ENS5" in model_id else 0
    pred = pd.DataFrame(
        {
            "dataset": dataset,
            "model_id": model_id,
            "fold_id": fold,
            "record_id": np.asarray(bundle["record_ids"]).astype(str),
            "row_id": np.arange(n, dtype=int),
            "table_idx": np.asarray(bundle["te"], dtype=int),
            "evaluation_partition": EVALUATION_PARTITION,
            "member_id": member_id,
            "y_true": np.asarray(bundle["y_true"], dtype=float),
            "y_pred": np.asarray(uq["mu"], dtype=float),
            "uncertainty_status": _status(model_id, uq, n),
            "sigma_total": _arr(uq, "sigma_total", n),
            "gamma": _arr(uq, "gamma", n),
            "nu": _arr(uq, "nu", n),
            "alpha": _arr(uq, "alpha", n),
            "beta": _arr(uq, "beta", n),
            "lower_90": _arr(uq, "lower_90", n),
            "upper_90": _arr(uq, "upper_90", n),
            "ligand_smiles": extra["ligand_smiles"].to_numpy(),
            "family": extra["family"].to_numpy(),
            "metal_species_pair": extra["metal_species_pair"].to_numpy(),
            "is_ree_pair": extra["is_ree_pair"].to_numpy(),
            "pair_id": extra["pair_id"].to_numpy(),
        }
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(dest, index=False)
    return dest


def write_chemprop(model_id: str, fold: str) -> Path:
    dest = _out_path(model_id, fold)
    if dest.exists() or model_id == "DS2_GRAPH_DMPNN_EVID":
        return dest
    from uq_remake.models.chemprop_dmpnn import predict_dmpnn_evidential_uq, predict_dmpnn_point
    from uq_remake.models.chemprop_evidential_validate import validate_native_evidential_output
    from uq_remake.training.protocol import checkpoint_dir

    dataset = "DS1" if model_id.startswith("DS1_") else "DS2"
    table = load_table(dataset)
    split = load_folds(dataset)[fold]
    te = np.asarray(split["test_idx"], dtype=int)
    y_true = table.iloc[te][target_name(dataset)].to_numpy(dtype=float)
    chem_dir = REMAKE_ROOT / "data" / "processed" / "features" / ("chemprop_ds1" if dataset == "DS1" else "chemprop_ds2") / fold
    extra = extra_meta_frame(table, dataset, te)
    ckpt = checkpoint_dir(model_id, fold, 0)
    n = len(te)
    if "EVID" in model_id:
        native = predict_dmpnn_evidential_uq(
            chem_dir / "test.csv",
            chem_dir / "test_features.csv",
            ckpt,
            n_expected=n,
            dataset=dataset,
            fold=fold,
            member_id=0,
            model_id=model_id,
        )
        validated = validate_native_evidential_output(
            {"mu": native["mu"], "nu": native["nu"], "alpha": native["alpha"], "beta": native["beta"]}
        )
        y_pred = validated["mu"]
        status = validated["uncertainty_status"]
        gamma = validated["mu"]
        nu = validated["nu"]
        alpha = validated["alpha"]
        beta = validated["beta"]
        sigma = np.sqrt(np.maximum(np.asarray(validated["var_total"], dtype=float), 0.0))
    else:
        tmp = dest.parent / "chemprop_raw_preds.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        y_pred = np.asarray(
            predict_dmpnn_point(chem_dir / "test.csv", chem_dir / "test_features.csv", ckpt, tmp),
            dtype=float,
        )
        status = np.full(n, "unavailable")
        gamma = nu = alpha = beta = sigma = np.full(n, np.nan)
    pred = pd.DataFrame(
        {
            "dataset": dataset,
            "model_id": model_id,
            "fold_id": fold,
            "record_id": record_ids_for_split(table, dataset, te).astype(str),
            "row_id": np.arange(n, dtype=int),
            "table_idx": te,
            "evaluation_partition": EVALUATION_PARTITION,
            "member_id": 0,
            "y_true": y_true,
            "y_pred": y_pred,
            "uncertainty_status": status,
            "sigma_total": sigma,
            "gamma": gamma,
            "nu": nu,
            "alpha": alpha,
            "beta": beta,
            "ligand_smiles": extra["ligand_smiles"].to_numpy(),
            "family": extra["family"].to_numpy(),
            "metal_species_pair": extra["metal_species_pair"].to_numpy(),
            "is_ree_pair": extra["is_ree_pair"].to_numpy(),
            "pair_id": extra["pair_id"].to_numpy(),
        }
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(dest, index=False)
    return dest


def main() -> int:
    models = _models()
    print(f"models={len(models)}", flush=True)
    for model_id in models:
        for fold in FOLDS:
            dest = _out_path(model_id, fold)
            if dest.exists() and model_id != "DS2_GRAPH_DMPNN_EVID":
                print(f"skip {model_id} {fold}", flush=True)
                continue
            print(f"infer {model_id} {fold}", flush=True)
            if "GRAPH_DMPNN" in model_id:
                if model_id == "DS2_GRAPH_DMPNN_EVID":
                    from uq_remake.metrics.reeval_chemprop_evidential_v2 import run_fold

                    run_fold(fold)
                else:
                    write_chemprop(model_id, fold)
            else:
                write_tabular(model_id, fold)
            print(f"  wrote {dest if model_id != 'DS2_GRAPH_DMPNN_EVID' else 'v2 ' + fold}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
