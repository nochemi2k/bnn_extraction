"""Train the shared DS1/DS2 active grid. Chemprop MC-dropout is not included."""
from __future__ import annotations

import argparse
import io
import json
import shutil
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
from uq_remake.data.rdkit_features import load_fold_features  # noqa: E402
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
from uq_remake.models.gpr import (  # noqa: E402
    EXACT_GPR_MAX_N,
    predict_exact_gpr,
    predict_sparse_gpr,
    save_gpr,
    train_exact_gpr,
    train_sparse_gpr,
)
from uq_remake.training.protocol import (  # noqa: E402
    PREPROCESSING_SCHEMA_HASH,
    assert_row_alignment,
    canonical_model_id,
    checkpoint_dir,
    checkpoint_meta,
    extra_meta_frame,
    fold_int,
    ligand_split_hash,
    load_active_grid_cfg,
    load_folds,
    load_shared_cfg,
    load_table,
    model_seed,
    preds_dir,
    prediction_frame,
    record_ids_for_split,
    seed_everything,
    target_name,
    write_json,
    write_predictions,
)
from uq_remake.training.write_model_audit import write_static_audit  # noqa: E402


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _already_done(path: Path, n_expected: int) -> bool:
    if not path.exists():
        return False
    try:
        n = len(pd.read_csv(path, usecols=["y_pred"]))
    except Exception:
        return False
    return n == int(n_expected)


def _log_run(row: dict) -> None:
    print(
        f"  [{row.get('status')}] {row.get('model_id')} {row.get('fold')} "
        f"member={row.get('member_id')} s={row.get('train_seconds')}",
        flush=True,
    )


def _log_ckpt(row: dict) -> None:
    return


def _log_align(row: dict) -> None:
    return


def _bundle(dataset: str, fold: str) -> dict:
    table = load_table(dataset)
    folds = load_folds(dataset)
    split = folds[fold]
    tr = np.asarray(split["train_idx"], dtype=int)
    va = np.asarray(split["val_idx"], dtype=int)
    te = np.asarray(split["test_idx"], dtype=int)
    feat_dir = REMAKE_ROOT / "data" / "processed" / "features" / f"rdkit_{dataset.lower()}" / fold
    arrays, manifest = load_fold_features(feat_dir)
    if len(arrays["X_train"]) != len(tr) or len(arrays["X_val"]) != len(va) or len(arrays["X_test"]) != len(te):
        raise RuntimeError(
            f"{dataset} {fold}: array/split length mismatch "
            f"{arrays['X_train'].shape[0]}/{len(tr)} "
            f"{arrays['X_val'].shape[0]}/{len(va)} "
            f"{arrays['X_test'].shape[0]}/{len(te)}"
        )
    y_table = table.iloc[te][target_name(dataset)].to_numpy(dtype=float)
    y_arr = np.asarray(arrays["y_test"], dtype=np.float32)
    if not np.allclose(y_arr, np.asarray(y_table, dtype=np.float32), equal_nan=True, rtol=0, atol=0):
        max_diff = float(np.nanmax(np.abs(np.asarray(y_table, dtype=float) - np.asarray(arrays["y_test"], dtype=float))))
        raise RuntimeError(f"{dataset} {fold}: y_test does not match table target (max_diff={max_diff})")
    chem_dir = REMAKE_ROOT / "data" / "processed" / "features" / f"chemprop_{dataset.lower()}" / fold
    chem_test = pd.read_csv(chem_dir / "test.csv")
    if len(chem_test) != len(te):
        raise RuntimeError(f"{dataset} {fold}: chemprop test.csv {len(chem_test)} != split {len(te)}")
    if not np.allclose(chem_test["logK"].to_numpy(dtype=float), y_table, equal_nan=True, rtol=0, atol=1e-6):
        raise RuntimeError(f"{dataset} {fold}: chemprop test targets not aligned to table")
    ligands = "ligand_smiles"
    return {
        "dataset": dataset,
        "fold": fold,
        "table": table,
        "tr": tr,
        "va": va,
        "te": te,
        "arrays": arrays,
        "manifest": manifest,
        "chem_dir": chem_dir,
        "y_true": y_table,
        "record_ids": record_ids_for_split(table, dataset, te),
        "extra": extra_meta_frame(table, dataset, te),
        "ligand_hash_train": ligand_split_hash(table.iloc[tr][ligands]),
        "ligand_hash_val": ligand_split_hash(table.iloc[va][ligands]),
        "ligand_hash_test": ligand_split_hash(table.iloc[te][ligands]),
        "feature_hash": manifest.get("feature_hash", ""),
        "rdkit_version": manifest.get("rdkit_version", ""),
    }


def _base_meta(bundle: dict, model_id: str, family: str, member_id: int, seed: int, hps: dict, extra=None):
    return checkpoint_meta(
        dataset=bundle["dataset"],
        fold=bundle["fold"],
        model_id=model_id,
        family=family,
        member_id=member_id,
        seed=seed,
        feature_hash=bundle["feature_hash"],
        rdkit_version=bundle["rdkit_version"],
        n_train=len(bundle["tr"]),
        n_val=len(bundle["va"]),
        n_test=len(bundle["te"]),
        ligand_hash_train=bundle["ligand_hash_train"],
        ligand_hash_val=bundle["ligand_hash_val"],
        ligand_hash_test=bundle["ligand_hash_test"],
        selected_hps=hps,
        extra=extra,
    )


def _artifact_meta(bundle, model_id: str, member_id: int, extra: dict | None = None) -> dict:
    meta = {
        "feature_schema_hash": bundle.get("feature_hash", ""),
        "preprocessing_schema_hash": PREPROCESSING_SCHEMA_HASH,
        "checkpoint_id": f"{model_id}/{bundle['fold']}/member_{int(member_id)}",
        "training_row_hash": bundle.get("ligand_hash_train", ""),
        "validation_row_hash": bundle.get("ligand_hash_val", ""),
        "heldout_row_hash": bundle.get("ligand_hash_test", ""),
    }
    if extra:
        meta.update(extra)
    return meta


def _write_point_preds(bundle, model_id, y_pred, method: str):
    n = len(y_pred)
    align = assert_row_alignment(
        n_pred=n,
        n_table=len(bundle["te"]),
        n_features=len(bundle["arrays"]["X_test"]),
        context=f"{model_id} {bundle['fold']}",
    )
    _log_align({"model_id": model_id, "fold": bundle["fold"], "member_id": 0, **align})
    frame = prediction_frame(
        dataset=bundle["dataset"],
        fold=bundle["fold"],
        model_id=model_id,
        member_id=0,
        table_idx=bundle["te"],
        record_ids=bundle["record_ids"],
        y_true=bundle["y_true"],
        y_pred=y_pred,
        extra=bundle["extra"],
        prediction_type="point",
        uncertainty_status="unavailable",
        uncertainty_method=method,
        artifact_meta=_artifact_meta(bundle, model_id, 0),
    )
    path = preds_dir(model_id, bundle["fold"]) / "test_preds.csv"
    write_predictions(frame, path)
    return path


def _write_uq_preds(bundle, model_id, member_id, y_pred, uq, method: str, fname="test_preds.csv"):
    n = len(y_pred)
    align = assert_row_alignment(
        n_pred=n,
        n_table=len(bundle["te"]),
        n_features=len(bundle["arrays"]["X_test"]),
        context=f"{model_id} {bundle['fold']} member={member_id}",
    )
    _log_align({"model_id": model_id, "fold": bundle["fold"], "member_id": member_id, **align})
    frame = prediction_frame(
        dataset=bundle["dataset"],
        fold=bundle["fold"],
        model_id=model_id,
        member_id=member_id,
        table_idx=bundle["te"],
        record_ids=bundle["record_ids"],
        y_true=bundle["y_true"],
        y_pred=y_pred,
        extra=bundle["extra"],
        prediction_type="uq",
        uncertainty_status="valid",
        uncertainty_method=method,
        uq=uq,
        interval_method=str(uq.get("interval_method", "normal_moment_approximation")),
        chemprop_version=str(uq.get("chemprop_version", "")),
        artifact_meta=_artifact_meta(bundle, model_id, member_id, {"prediction_seed": uq.get("prediction_seed", "")}),
    )
    if "nig_valid" in uq:
        valid = np.asarray(uq["nig_valid"], dtype=bool).reshape(-1)
        if len(valid) != n:
            raise ValueError(f"nig_valid length {len(valid)} != {n}")
        frame["uncertainty_status"] = np.where(valid, "valid", "invalid_nig")
    path = preds_dir(model_id, bundle["fold"]) / fname
    write_predictions(frame, path)
    return path


def run_point_mlp(bundle, cfg, grid, family="POINT_MLP"):
    dataset, fold = bundle["dataset"], bundle["fold"]
    model_id = canonical_model_id(dataset, family)
    out = preds_dir(model_id, fold) / "test_preds.csv"
    if _already_done(out, len(bundle["te"])):
        _log_run({"model_id": model_id, "fold": fold, "member_id": 0, "status": "skipped_existing", "train_seconds": 0})
        return
    fid = fold_int(fold)
    seed = model_seed(fid, 0, grid["model_offsets"]["POINT_MLP"], grid["base_seed"], dataset=dataset)
    seed_everything(seed)
    ckpt = checkpoint_dir(model_id, fold, 0)
    arr = bundle["arrays"]
    model, meta = train_point_mlp(arr["X_train"], arr["y_train"], arr["X_val"], arr["y_val"], cfg, device=_device(), seed=seed)
    hps = {"mlp_hidden_dims": cfg["mlp_hidden_dims"], "mlp_lr": cfg["mlp_lr"], "mlp_weight_decay": cfg["mlp_weight_decay"], "dropout": cfg["mlp_dropout_point"]}
    blob = _base_meta(bundle, model_id, family, 0, seed, hps, meta)
    save_mlp_checkpoint(model, ckpt / "model.pt", blob)
    write_json(ckpt / "meta.json", blob)
    yhat = predict_mlp(model, arr["X_test"], device=_device())
    _write_point_preds(bundle, model_id, yhat, "none")
    _log_ckpt({"model_id": model_id, "fold": fold, "member_id": 0, "path": str(ckpt / "model.pt"), "seed": seed})
    _log_run({"model_id": model_id, "fold": fold, "member_id": 0, "status": "trained", "train_seconds": meta.get("train_seconds"), "val_loss": meta.get("best_val_mse")})


def run_point_xgb(bundle, cfg, grid, family="POINT_XGB"):
    dataset, fold = bundle["dataset"], bundle["fold"]
    model_id = canonical_model_id(dataset, family)
    out = preds_dir(model_id, fold) / "test_preds.csv"
    if _already_done(out, len(bundle["te"])):
        _log_run({"model_id": model_id, "fold": fold, "member_id": 0, "status": "skipped_existing", "train_seconds": 0})
        return
    fid = fold_int(fold)
    seed = model_seed(fid, 0, grid["model_offsets"]["POINT_XGB"], grid["base_seed"], dataset=dataset)
    seed_everything(seed)
    ckpt = checkpoint_dir(model_id, fold, 0)
    arr = bundle["arrays"]
    model, meta = train_point_xgb(arr["X_train"], arr["y_train"], arr["X_val"], arr["y_val"], cfg, seed=seed)
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_model(ckpt / "model.json")
    blob = _base_meta(bundle, model_id, family, 0, seed, {"xgb": {k: cfg[k] for k in cfg if str(k).startswith("xgb_")}}, meta)
    write_json(ckpt / "meta.json", blob)
    yhat = np.asarray(model.predict(arr["X_test"]), dtype=float)
    _write_point_preds(bundle, model_id, yhat, "none")
    _log_ckpt({"model_id": model_id, "fold": fold, "member_id": 0, "path": str(ckpt / "model.json"), "seed": seed})
    _log_run({"model_id": model_id, "fold": fold, "member_id": 0, "status": "trained", "train_seconds": meta.get("train_seconds")})


def _train_one_uq(kind, bundle, cfg, grid, member_id, lambda_reg=None):
    arr = bundle["arrays"]
    fid = fold_int(bundle["fold"])
    offset = grid["model_offsets"][kind]
    seed = model_seed(fid, member_id, offset, grid["base_seed"], dataset=bundle["dataset"])
    seed_everything(seed)
    n_mc = int(grid["mc_validation_passes"])
    device = _device()
    if kind == "EVID":
        model, meta = train_evid_mlp(
            arr["X_train"], arr["y_train"], arr["X_val"], arr["y_val"], cfg,
            device=device, seed=seed, lambda_reg=float(lambda_reg),
        )
    elif kind == "HETERO":
        model, meta = train_hetero_mlp(
            arr["X_train"], arr["y_train"], arr["X_val"], arr["y_val"], cfg,
            device=device, seed=seed, n_mc_val=n_mc,
        )
    elif kind == "MCDROPOUT":
        model, meta = train_mcdropout_mlp(
            arr["X_train"], arr["y_train"], arr["X_val"], arr["y_val"], cfg,
            device=device, seed=seed, n_mc_val=n_mc,
        )
    else:
        raise KeyError(kind)
    meta["seed"] = seed
    return model, meta, seed


def _predict_one_uq(kind, model, X, grid, seed):
    n_mc = int(grid["mc_dropout_passes"])
    device = _device()
    if kind == "EVID":
        return predict_evid(model, X, device=device)
    if kind == "HETERO":
        return predict_hetero_mc(model, X, n_mc=n_mc, device=device, seed=seed + 11)
    if kind == "MCDROPOUT":
        return predict_mcdropout_mc(model, X, n_mc=n_mc, device=device, seed=seed + 11)
    raise KeyError(kind)


def run_tabular_uq(kind: str, bundle, cfg, grid, families, evid_cache: dict):
    want_single = f"{kind}_MLP" in families
    want_ens = f"{kind}_MLPENS5" in families
    if not (want_single or want_ens):
        return
    dataset, fold = bundle["dataset"], bundle["fold"]
    arr = bundle["arrays"]
    n_test = len(bundle["te"])
    single_id = canonical_model_id(dataset, f"{kind}_MLP")
    ens_id = canonical_model_id(dataset, f"{kind}_MLPENS5")
    single_done = (not want_single) or _already_done(preds_dir(single_id, fold) / "test_preds.csv", n_test)
    ens_done = (not want_ens) or _already_done(preds_dir(ens_id, fold) / "test_preds.csv", n_test)
    if single_done and ens_done:
        _log_run({"model_id": ens_id if want_ens else single_id, "fold": fold, "member_id": -1, "status": "skipped_existing", "train_seconds": 0})
        return
    lambda_reg = None
    selected_model = None
    selected_meta = None
    if kind == "EVID":
        key = (dataset, fold)
        if key not in evid_cache:
            fid = fold_int(fold)
            seed = model_seed(fid, 0, grid["model_offsets"]["EVID"], grid["base_seed"], dataset=dataset)
            seed_everything(seed)
            model, meta, lam = select_evidential_lambda(
                arr["X_train"], arr["y_train"], arr["X_val"], arr["y_val"], cfg,
                grid=list(grid["evidential_lambda_grid"]),
                device=_device(),
                seed=seed,
            )
            evid_cache[key] = (model, meta, lam, seed)
            _log_run({
                "model_id": single_id,
                "fold": fold,
                "member_id": 0,
                "status": "lambda_selected",
                "train_seconds": meta.get("train_seconds"),
                "lambda_reg": lam,
                "val_nll": meta.get("best_val"),
            })
        selected_model, selected_meta, lambda_reg, _seed0 = evid_cache[key]
    n_members = 5 if want_ens else 1
    member_outs = []
    for m in range(n_members):
        mid = ens_id if want_ens else single_id
        member_path = preds_dir(ens_id if want_ens else single_id, fold) / f"member_{m}.csv"
        # skip member train if ensemble aggregate already done and this is only ens
        if m == 0 and kind == "EVID" and selected_model is not None:
            model, meta, seed = selected_model, selected_meta, evid_cache[(dataset, fold)][3]
        else:
            model, meta, seed = _train_one_uq(kind, bundle, cfg, grid, m, lambda_reg=lambda_reg)
        ckpt = checkpoint_dir(ens_id if want_ens else single_id, fold, m)
        hps = {
            "mlp_hidden_dims": cfg["mlp_hidden_dims"],
            "mlp_lr": cfg["mlp_lr"],
            "log_var_clamp": grid["log_var_clamp"],
            "mc_dropout_passes": grid["mc_dropout_passes"],
            "lambda_reg": lambda_reg,
            "checkpoint_rule": "validation_predictive_nll",
        }
        blob = _base_meta(bundle, mid, f"{kind}_MLPENS5" if want_ens else f"{kind}_MLP", m, seed, hps, meta)
        save_torch(model, ckpt / "model.pt", blob)
        write_json(ckpt / "meta.json", blob)
        if kind == "HETERO":
            val_uq = _predict_one_uq(kind, model, arr["X_val"], grid, seed)
            write_json(
                ckpt / "hetero_diagnostics.json",
                {
                    "split": "val",
                    "log_var_quantiles": val_uq.get("log_var_quantiles"),
                    "frac_at_clamp_lo": val_uq.get("frac_at_clamp_lo"),
                    "frac_at_clamp_hi": val_uq.get("frac_at_clamp_hi"),
                },
            )
        uq = _predict_one_uq(kind, model, arr["X_test"], grid, seed)
        member_outs.append(uq)
        method = uq.get("uncertainty_method", kind.lower())
        if want_ens:
            _write_uq_preds(bundle, ens_id, m, uq["mu"], uq, method, fname=f"member_{m}.csv")
        if m == 0 and want_single:
            s_ckpt = checkpoint_dir(single_id, fold, 0)
            s_ckpt.mkdir(parents=True, exist_ok=True)
            if want_ens:
                shutil.copy2(ckpt / "model.pt", s_ckpt / "model.pt")
                write_json(s_ckpt / "meta.json", {**blob, "model_id": single_id, "family": f"{kind}_MLP"})
            if not _already_done(preds_dir(single_id, fold) / "test_preds.csv", n_test):
                _write_uq_preds(bundle, single_id, 0, uq["mu"], uq, method)
            _log_ckpt({"model_id": single_id, "fold": fold, "member_id": 0, "path": str(s_ckpt / "model.pt"), "seed": seed})
            _log_run({"model_id": single_id, "fold": fold, "member_id": 0, "status": "trained", "train_seconds": meta.get("train_seconds"), "val_loss": meta.get("best_val"), "lambda_reg": lambda_reg})
        _log_ckpt({"model_id": mid, "fold": fold, "member_id": m, "path": str(ckpt / "model.pt"), "seed": seed})
        _log_run({"model_id": mid, "fold": fold, "member_id": m, "status": "trained", "train_seconds": meta.get("train_seconds"), "val_loss": meta.get("best_val"), "lambda_reg": lambda_reg})
    if want_ens:
        agg = aggregate_members(member_outs)
        _write_uq_preds(bundle, ens_id, -1, agg["mu"], agg, agg.get("uncertainty_method", "ensemble_law_of_total_variance"))
        _log_run({"model_id": ens_id, "fold": fold, "member_id": -1, "status": "aggregated", "train_seconds": 0, "n_members": 5, "variance_ddof": 0})


def run_chemprop_point(bundle, cfg, grid):
    dataset, fold = bundle["dataset"], bundle["fold"]
    family = "CHEMPROP_POINT_NATIVE"
    model_id = canonical_model_id(dataset, family)
    out = preds_dir(model_id, fold) / "test_preds.csv"
    n = len(bundle["te"])
    if _already_done(out, n):
        _log_run({"model_id": model_id, "fold": fold, "member_id": 0, "status": "skipped_existing", "train_seconds": 0})
        return
    fid = fold_int(fold)
    seed = model_seed(fid, 0, grid["model_offsets"]["CHEMPROP_POINT"], grid["base_seed"], dataset=dataset)
    seed_everything(seed)
    ckpt = checkpoint_dir(model_id, fold, 0)
    work = bundle["chem_dir"]
    model_pt = ckpt / "fold_0" / "model_0" / "model.pt"
    if not model_pt.exists():
        meta = train_dmpnn_point(work / "train.csv", work / "train_features.csv", work / "val.csv", work / "val_features.csv", ckpt, seed=seed)
    else:
        meta = {"train_seconds": 0, "reused_local": True}
    write_chemprop_identity(ckpt, dataset=dataset, fold=fold, member_id=0, model_id=model_id)
    tmp = preds_dir(model_id, fold) / "chemprop_raw_preds.csv"
    yhat = predict_dmpnn_point(work / "test.csv", work / "test_features.csv", ckpt, tmp)
    if len(yhat) != n:
        raise ValueError(f"{model_id} {fold}: chemprop preds {len(yhat)} != {n}")
    blob = _base_meta(bundle, model_id, family, 0, seed, {"chemprop": "point", "loss": "mse"}, meta)
    write_json(ckpt / "meta.json", blob)
    align = assert_row_alignment(n_pred=len(yhat), n_table=n, n_chemprop=n, y_pred_source=pd.read_csv(work / "test.csv")["logK"].to_numpy(), y_table=bundle["y_true"], context=f"{model_id} {fold}")
    _log_align({"model_id": model_id, "fold": fold, "member_id": 0, **align})
    _write_point_preds(bundle, model_id, np.asarray(yhat, dtype=float), "none")
    _log_ckpt({"model_id": model_id, "fold": fold, "member_id": 0, "path": str(model_pt), "seed": seed})
    _log_run({"model_id": model_id, "fold": fold, "member_id": 0, "status": "trained", "train_seconds": meta.get("train_seconds")})


def run_chemprop_evidential(bundle, cfg, grid):
    dataset, fold = bundle["dataset"], bundle["fold"]
    family = "CHEMPROP_EVIDENTIAL_NATIVE"
    model_id = canonical_model_id(dataset, family)
    out = preds_dir(model_id, fold) / "test_preds.csv"
    n = len(bundle["te"])
    if _already_done(out, n):
        _log_run({"model_id": model_id, "fold": fold, "member_id": 0, "status": "skipped_existing", "train_seconds": 0})
        return
    fid = fold_int(fold)
    seed = model_seed(fid, 0, grid["model_offsets"]["CHEMPROP_EVID"], grid["base_seed"], dataset=dataset)
    seed_everything(seed)
    ckpt = checkpoint_dir(model_id, fold, 0)
    work = bundle["chem_dir"]
    lam = float(grid["chemprop_evidential_regularization"])
    model_pt = ckpt / "fold_0" / "model_0" / "model.pt"
    extra = ["--loss_function", "evidential", "--evidential_regularization", str(lam)]
    if not model_pt.exists():
        meta = train_dmpnn_point(
            work / "train.csv", work / "train_features.csv", work / "val.csv", work / "val_features.csv",
            ckpt, seed=seed, extra_args=extra,
        )
    else:
        meta = {"train_seconds": 0, "reused_local": True}
    write_chemprop_identity(ckpt, dataset=dataset, fold=fold, member_id=0, model_id=model_id)
    uq = predict_dmpnn_evidential_uq(
        work / "test.csv",
        work / "test_features.csv",
        ckpt,
        n_expected=n,
        dataset=dataset,
        fold=fold,
        member_id=0,
        model_id=model_id,
    )
    blob = _base_meta(
        bundle, model_id, family, 0, seed,
        {"loss_function": "evidential", "evidential_regularization": lam, "selection": "a_priori_not_test"},
        meta,
    )
    write_json(ckpt / "meta.json", blob)
    align = assert_row_alignment(
        n_pred=len(uq["mu"]), n_table=n, n_chemprop=n,
        y_pred_source=pd.read_csv(work / "test.csv")["logK"].to_numpy(),
        y_table=bundle["y_true"],
        context=f"{model_id} {fold}",
    )
    _log_align({"model_id": model_id, "fold": fold, "member_id": 0, **align})
    _write_uq_preds(bundle, model_id, 0, uq["mu"], uq, uq["uncertainty_method"])
    _log_ckpt({"model_id": model_id, "fold": fold, "member_id": 0, "path": str(model_pt), "seed": seed})
    _log_run({
        "model_id": model_id,
        "fold": fold,
        "member_id": 0,
        "status": "trained",
        "train_seconds": meta.get("train_seconds"),
        "lambda_reg": lam,
        "n_invalid_nig": int(uq.get("n_invalid_nig", 0)),
        "n_valid_nig": int(uq.get("n_valid_nig", n)),
    })


def _run_ensemble_family(
    *,
    bundle,
    cfg,
    grid,
    families,
    single_fam: str,
    ens_fam: str,
    offset_key: str,
    train_fn,
    predict_fn,
    save_fn,
    single_within_zero: bool,
):
    want_single = single_fam in families
    want_ens = ens_fam in families
    if not (want_single or want_ens):
        return
    dataset, fold = bundle["dataset"], bundle["fold"]
    arr = bundle["arrays"]
    n_test = len(bundle["te"])
    single_id = canonical_model_id(dataset, single_fam)
    ens_id = canonical_model_id(dataset, ens_fam)
    single_done = (not want_single) or _already_done(preds_dir(single_id, fold) / "test_preds.csv", n_test)
    ens_done = (not want_ens) or _already_done(preds_dir(ens_id, fold) / "test_preds.csv", n_test)
    if single_done and ens_done:
        _log_run({"model_id": ens_id if want_ens else single_id, "fold": fold, "member_id": -1, "status": "skipped_existing", "train_seconds": 0})
        return
    n_members = 5 if want_ens else 1
    member_outs = []
    fid = fold_int(fold)
    offset = grid["model_offsets"][offset_key]
    n_s = int(grid.get("bayes_posterior_samples", grid["mc_dropout_passes"]))
    for m in range(n_members):
        seed = model_seed(fid, m, offset, grid["base_seed"], dataset=dataset)
        seed_everything(seed)
        model, meta = train_fn(arr, cfg, grid, seed)
        mid = ens_id if want_ens else single_id
        ckpt = checkpoint_dir(mid, fold, m)
        blob = _base_meta(bundle, mid, ens_fam if want_ens else single_fam, m, seed, {"seed": seed}, meta)
        blob.update(_artifact_meta(bundle, mid, m, {"prediction_seed": seed + 11, "base_seed": grid["base_seed"], "fold_seed": seed, "member_seed": seed}))
        save_fn(model, ckpt / "model.pt", blob)
        write_json(ckpt / "meta.json", blob)
        uq = predict_fn(model, arr["X_test"], seed, single_model=False)
        member_outs.append(uq)
        if want_ens:
            _write_uq_preds(bundle, ens_id, m, uq["mu"], uq, uq.get("uncertainty_method", single_fam.lower()), fname=f"member_{m}.csv")
        if m == 0 and want_single:
            s_ckpt = checkpoint_dir(single_id, fold, 0)
            s_ckpt.mkdir(parents=True, exist_ok=True)
            if want_ens:
                shutil.copy2(ckpt / "model.pt", s_ckpt / "model.pt")
                write_json(s_ckpt / "meta.json", {**blob, "model_id": single_id, "family": single_fam})
            uq_single = dict(uq)
            if single_within_zero:
                # Single models have no between-member term. Epistemic from
                # posterior/MC samples is within-model epistemic, not zero.
                z = np.zeros_like(uq["var_epistemic"])
                ve = np.asarray(uq["var_epistemic"], dtype=float)
                uq_single["var_epistemic_within"] = ve
                uq_single["var_epistemic_between"] = z
                uq_single["var_epistemic_total"] = ve
                uq_single["sigma_epistemic_within"] = np.sqrt(np.maximum(ve, 0.0))
                uq_single["sigma_epistemic_between"] = z
            if not _already_done(preds_dir(single_id, fold) / "test_preds.csv", n_test):
                _write_uq_preds(bundle, single_id, 0, uq_single["mu"], uq_single, uq_single.get("uncertainty_method", single_fam.lower()))
            _log_ckpt({"model_id": single_id, "fold": fold, "member_id": 0, "path": str(s_ckpt / "model.pt"), "seed": seed})
            _log_run({"model_id": single_id, "fold": fold, "member_id": 0, "status": "trained", "train_seconds": meta.get("train_seconds")})
        _log_ckpt({"model_id": mid, "fold": fold, "member_id": m, "path": str(ckpt / "model.pt"), "seed": seed})
        _log_run({"model_id": mid, "fold": fold, "member_id": m, "status": "trained", "train_seconds": meta.get("train_seconds")})
        _ = n_s
    if want_ens:
        agg = aggregate_members(member_outs)
        agg["interval_method"] = member_outs[0].get("interval_method", "gaussian_predictive_moment")
        if "lower_90" in member_outs[0]:
            z = 1.6448536269514722
            agg["lower_90"] = agg["mu"] - z * agg["sigma_total"]
            agg["upper_90"] = agg["mu"] + z * agg["sigma_total"]
            if member_outs[0].get("interval_method") == "empirical_posterior_predictive":
                agg["interval_method"] = "empirical_predictive_mixture"
        _write_uq_preds(bundle, ens_id, -1, agg["mu"], agg, agg.get("uncertainty_method", "ensemble_law_of_total_variance"))
        _log_run({"model_id": ens_id, "fold": fold, "member_id": -1, "status": "aggregated", "train_seconds": 0, "n_members": 5, "variance_ddof": 0})


def run_bayes_uq(bundle, cfg, grid, families):
    n_s = int(grid.get("bayes_posterior_samples", 30))
    n_val = int(grid.get("mc_validation_passes", 30))
    beta = float(grid.get("beta_KL", 1.0))

    def train_fn(arr, cfg_, grid_, seed):
        return train_variational_bnn(
            arr["X_train"], arr["y_train"], arr["X_val"], arr["y_val"], cfg_,
            device=_device(), seed=seed, n_samples_val=n_val, beta_kl=beta,
        )

    def predict_fn(model, X, seed, single_model=False):
        return predict_variational_bnn(model, X, n_samples=n_s, device=_device(), seed=seed + 11, single_model=single_model)

    _run_ensemble_family(
        bundle=bundle, cfg=cfg, grid=grid, families=families,
        single_fam="BAYES_FULL", ens_fam="BAYES_FULLENS5", offset_key="BAYES_FULL",
        train_fn=train_fn, predict_fn=predict_fn, save_fn=save_bnn, single_within_zero=True,
    )


def run_sngp_uq(bundle, cfg, grid, families):
    def train_fn(arr, cfg_, grid_, seed):
        local = dict(cfg_)
        local["sngp_num_features"] = grid_.get("sngp_num_features", 512)
        local["sngp_lengthscale"] = grid_.get("sngp_lengthscale", 2.0)
        local["sngp_ridge"] = grid_.get("sngp_ridge", 1.0)
        return train_sngp(arr["X_train"], arr["y_train"], arr["X_val"], arr["y_val"], local, device=_device(), seed=seed)

    def predict_fn(model, X, seed, single_model=True):
        return predict_sngp(model, X, device=_device(), single_model=single_model)

    _run_ensemble_family(
        bundle=bundle, cfg=cfg, grid=grid, families=families,
        single_fam="SNGP", ens_fam="SNGPENS5", offset_key="SNGP",
        train_fn=train_fn, predict_fn=predict_fn, save_fn=save_sngp, single_within_zero=False,
    )


def run_gpr_family(bundle, cfg, grid, families):
    dataset, fold = bundle["dataset"], bundle["fold"]
    arr = bundle["arrays"]
    n_train = len(bundle["tr"])
    n_test = len(bundle["te"])
    fid = fold_int(fold)
    max_n = int(grid.get("exact_gpr_max_n", EXACT_GPR_MAX_N))
    if "GPR" in families:
        model_id = canonical_model_id(dataset, "GPR")
        out = preds_dir(model_id, fold) / "test_preds.csv"
        if _already_done(out, n_test):
            _log_run({"model_id": model_id, "fold": fold, "member_id": 0, "status": "skipped_existing", "train_seconds": 0})
        elif n_train > max_n:
            ckpt = checkpoint_dir(model_id, fold, 0)
            write_json(
                ckpt / "infeasible.json",
                {
                    "status": "computationally_infeasible",
                    "n_train": n_train,
                    "exact_gpr_max_n": max_n,
                    "reason": grid.get("exact_gpr_infeasible_reason", "n_train exceeds exact Cholesky limit"),
                    "use_instead": canonical_model_id(dataset, "GPR_SPARSE"),
                },
            )
            _log_run({"model_id": model_id, "fold": fold, "member_id": 0, "status": "computationally_infeasible", "train_seconds": 0, "n_train": n_train})
        else:
            seed = model_seed(fid, 0, grid["model_offsets"]["GPR"], grid["base_seed"], dataset=dataset)
            seed_everything(seed)
            local = dict(cfg)
            local.update({k: grid[k] for k in ("gpr_jitter", "gpr_exact_steps") if k in grid})
            model, meta, scaler = train_exact_gpr(arr["X_train"], arr["y_train"], arr["X_val"], arr["y_val"], local, device=_device(), seed=seed, max_n=max_n)
            ckpt = checkpoint_dir(model_id, fold, 0)
            blob = _base_meta(bundle, model_id, "GPR", 0, seed, {"gpr_type": "exact"}, meta)
            save_gpr({"state_dict": model.state_dict(), "meta": meta, "scaler": scaler.hash_payload(), "class": "ExactGPR", "n_features": arr["X_train"].shape[1]}, ckpt / "model.pt")
            write_json(ckpt / "meta.json", {**blob, **scaler.hash_payload()})
            uq = predict_exact_gpr(model, scaler, arr["X_test"], device=_device())
            _write_uq_preds(bundle, model_id, 0, uq["mu"], uq, uq["uncertainty_method"])
            _log_run({"model_id": model_id, "fold": fold, "member_id": 0, "status": "trained", "train_seconds": meta.get("train_seconds")})
    if "GPR_SPARSE" not in families:
        return
    model_id = canonical_model_id(dataset, "GPR_SPARSE")
    out = preds_dir(model_id, fold) / "test_preds.csv"
    if _already_done(out, n_test):
        _log_run({"model_id": model_id, "fold": fold, "member_id": 0, "status": "skipped_existing", "train_seconds": 0})
        return
    seed = model_seed(fid, 0, grid["model_offsets"]["GPR_SPARSE"], grid["base_seed"], dataset=dataset)
    seed_everything(seed)
    local = dict(cfg)
    for k in ("gpr_n_inducing", "gpr_sparse_epochs", "gpr_sparse_patience", "gpr_sparse_batch_size", "gpr_jitter"):
        if k in grid:
            local[k] = grid[k]
    model, meta, scaler = train_sparse_gpr(arr["X_train"], arr["y_train"], arr["X_val"], arr["y_val"], local, device=_device(), seed=seed)
    ckpt = checkpoint_dir(model_id, fold, 0)
    blob = _base_meta(bundle, model_id, "GPR_SPARSE", 0, seed, {"gpr_type": "sparse_variational"}, meta)
    save_gpr({"state_dict": model.state_dict(), "meta": meta, "scaler": scaler.hash_payload(), "class": "SparseGPR", "Z": model.Z.detach().cpu()}, ckpt / "model.pt")
    write_json(ckpt / "meta.json", {**blob, **scaler.hash_payload()})
    uq = predict_sparse_gpr(model, scaler, arr["X_test"], device=_device())
    _write_uq_preds(bundle, model_id, 0, uq["mu"], uq, uq["uncertainty_method"])
    _log_ckpt({"model_id": model_id, "fold": fold, "member_id": 0, "path": str(ckpt / "model.pt"), "seed": seed})
    _log_run({"model_id": model_id, "fold": fold, "member_id": 0, "status": "trained", "train_seconds": meta.get("train_seconds")})


def run_dataset(dataset: str, families: list[str], cfg: dict, grid: dict, evid_cache: dict):
    for fold in grid["fold_names"]:
        print(f"\n======== {dataset} {fold} families={families} ========", flush=True)
        bundle = _bundle(dataset, fold)
        if "POINT_MLP" in families:
            run_point_mlp(bundle, cfg, grid)
        if "POINT_XGB" in families:
            run_point_xgb(bundle, cfg, grid)
        for kind in ("EVID", "HETERO", "MCDROPOUT"):
            run_tabular_uq(kind, bundle, cfg, grid, families, evid_cache)
        run_bayes_uq(bundle, cfg, grid, families)
        run_sngp_uq(bundle, cfg, grid, families)
        run_gpr_family(bundle, cfg, grid, families)
        if "CHEMPROP_POINT_NATIVE" in families:
            run_chemprop_point(bundle, cfg, grid)
        if "CHEMPROP_EVIDENTIAL_NATIVE" in families:
            run_chemprop_evidential(bundle, cfg, grid)
        if any("CHEMPROP" in f and "EVIDENTIAL" not in f and "POINT" not in f for f in families):
            raise RuntimeError("Chemprop MC-dropout is not part of the active grid")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Shared DS1/DS2 active grid (no Chemprop MC-dropout)")
    p.add_argument("--dataset", default="DS1,DS2", help="Comma-separated DS1 and/or DS2")
    p.add_argument("--families", default="all", help="Comma-separated family names, or all")
    p.add_argument("--audit-only", action="store_true")
    return p.parse_args(argv)


def _tee_stdout() -> None:
    log_dir = REMAKE_ROOT / "tmp"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "active_grid_train.log"
    fh = open(path, "a", encoding="utf-8", buffering=1)
    for stream in (sys.__stdout__, sys.__stderr__):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    class Tee(io.TextIOBase):
        encoding = "utf-8"

        def write(self, s):
            if not isinstance(s, str):
                s = str(s)
            try:
                sys.__stdout__.write(s)
            except UnicodeEncodeError:
                enc = getattr(sys.__stdout__, "encoding", None) or "utf-8"
                sys.__stdout__.write(s.encode(enc, errors="replace").decode(enc, errors="replace"))
            fh.write(s)
            return len(s)

        def flush(self):
            sys.__stdout__.flush()
            fh.flush()

        def isatty(self) -> bool:
            return False

    sys.stdout = Tee()
    sys.stderr = Tee()


def main(argv=None):
    args = parse_args(argv)
    if not args.audit_only:
        _tee_stdout()
    grid = load_active_grid_cfg()
    cfg = load_shared_cfg()
    import uq_remake.models.uq_tabular as uq_mod

    uq_mod.LOG_VAR_CLAMP = tuple(grid["log_var_clamp"])
    for k in (
        "sngp_num_features",
        "sngp_lengthscale",
        "sngp_ridge",
        "gpr_n_inducing",
        "gpr_sparse_epochs",
        "gpr_sparse_patience",
        "gpr_sparse_batch_size",
        "gpr_jitter",
        "beta_KL",
    ):
        if k in grid:
            cfg[k] = grid[k]
    paths = write_static_audit()
    print("Wrote model audit:", json.dumps(paths, indent=2), flush=True)
    if args.audit_only:
        return
    if grid.get("native_chemprop_mc_dropout_in_active_grid"):
        raise RuntimeError("active_grid.json must keep Chemprop MC-dropout disabled")
    families = list(grid["active_families"]) if args.families == "all" else [x.strip() for x in args.families.split(",") if x.strip()]
    banned = [f for f in families if "CHEMPROP" in f and "EVIDENTIAL" not in f and "POINT" not in f]
    banned += [f for f in families if "MCDROPOUT" in f and "CHEMPROP" in f]
    if banned:
        raise SystemExit(f"Chemprop MC-dropout families are forbidden: {banned}")
    unknown = [f for f in families if f not in grid["active_families"]]
    if unknown:
        raise SystemExit(f"Unknown families {unknown}; allowed {grid['active_families']}")
    evid_cache: dict = {}
    datasets = [x.strip().upper() for x in args.dataset.split(",") if x.strip()]
    print(f"Active grid device={_device()} datasets={datasets} families={families}", flush=True)
    for ds in datasets:
        run_dataset(ds, families, cfg, grid, evid_cache)
    print("Active grid complete.", flush=True)


if __name__ == "__main__":
    main()
