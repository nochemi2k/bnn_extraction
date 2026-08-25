"""Reproduce CMI M4 Chemprop: released weights and architecture-matched retrain.

On Windows this script imports torch before RDKit (WinError 1114).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402  — must precede RDKit on Windows

from uq_remake import REMAKE_ROOT  # noqa: E402
from uq_remake.data.cmi_reproduction import (  # noqa: E402
    CMI_CKPT_ROOT,
    CMI_DATA_WORK,
    N_FOLDS,
    download_best_model_folds,
    dump_cmi_architecture,
    fold_model_pt,
    write_official_chemprop_csvs,
)
from uq_remake.metrics.point_metrics import compute_point_metrics  # noqa: E402
from uq_remake.models.chemprop_dmpnn import (  # noqa: E402
    chemprop_version,
    predict_dmpnn_point,
    train_dmpnn_point,
)

OUT_JSON = REMAKE_ROOT / "results" / "stage1" / "cmi_reproduction_results.json"
ARCH_CKPT = REMAKE_ROOT / "checkpoints" / "M4_GRAPH_DMPNN_POINT_cmiarch"


def _y_from_csv(path: Path) -> np.ndarray:
    return pd.read_csv(path)["logK"].to_numpy(dtype=float)


def _metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    return compute_point_metrics(y, yhat)


def _predict_logged(data_csv: Path, feat_csv: Path, ckpt: Path, preds_path: Path) -> np.ndarray:
    return predict_dmpnn_point(data_csv, feat_csv, ckpt, preds_path)


def eval_released_weights() -> dict:
    download_best_model_folds()
    arch = dump_cmi_architecture()
    written = write_official_chemprop_csvs()
    pred_dir = REMAKE_ROOT / "results" / "stage1" / "cmi_released_preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    evaluations = {}
    for split in ("test", "ext"):
        data_csv = CMI_DATA_WORK / f"{split}.csv"
        feat_csv = CMI_DATA_WORK / f"{split}_features.csv"
        y = _y_from_csv(data_csv)
        fold_rows = []
        fold_preds = []
        for fold in range(N_FOLDS):
            yhat = _predict_logged(
                data_csv,
                feat_csv,
                CMI_CKPT_ROOT / f"fold_{fold}",
                pred_dir / f"{split}_fold{fold}.csv",
            )
            m = _metrics(y, yhat)
            fold_rows.append({"fold": fold, **m})
            fold_preds.append(np.asarray(yhat, dtype=float))
        stacked = np.vstack(fold_preds)
        ens = stacked.mean(axis=0)
        ens_path = pred_dir / f"{split}_ensemble.csv"
        pd.DataFrame({"logK": y, "y_pred": ens}).to_csv(ens_path, index=False)
        evaluations[f"cmi_{split}"] = {
            "n": int(len(y)),
            "ensemble": _metrics(y, ens),
            "fold_mean_mae": float(np.mean([r["mae"] for r in fold_rows])),
            "fold_mean_r2": float(np.mean([r["r2"] for r in fold_rows])),
            "fold_sd_mae": float(np.std([r["mae"] for r in fold_rows], ddof=1)),
            "fold_sd_r2": float(np.std([r["r2"] for r in fold_rows], ddof=1)),
            "per_fold": fold_rows,
            "pred_path": str(ens_path.relative_to(REMAKE_ROOT)).replace("\\", "/"),
        }
    return {
        "status": "complete",
        "chemprop_version_used": chemprop_version(),
        "n_checkpoints": N_FOLDS,
        "architecture": arch.get("architecture", {}),
        "architecture_mismatches": arch.get("fold_mismatch", []),
        "official_csvs": written,
        "evaluations": evaluations,
        "note": (
            "Released CMI best_model fold_0–fold_4 weights evaluated with Chemprop "
            "1.6.1 on the official CMI train/test/validation CSVs. Ensemble = mean "
            "of five fold predictions. fold_mean_* averages the five per-fold metrics "
            "on the same open file (closer to a CV-average estimand than the ensemble)."
        ),
    }


def train_cmi_architecture(*, skip_if_exists: bool = True) -> dict:
    """From-scratch Chemprop using architecture recovered from released checkpoints.

    Uses official CMI train CSV with Chemprop ``--num_folds 5`` and seed 0, matching
    the released checkpoint layout. Official test/ext stay out of training.
    """
    download_best_model_folds()
    arch_dump = dump_cmi_architecture()
    arch = dict(arch_dump.get("architecture") or {})
    write_official_chemprop_csvs()
    train_csv = CMI_DATA_WORK / "train.csv"
    train_feat = CMI_DATA_WORK / "train_features.csv"
    model_pt = ARCH_CKPT / "fold_0" / "model_0" / "model.pt"
    if skip_if_exists and model_pt.exists() and (ARCH_CKPT / "fold_4" / "model_0" / "model.pt").exists():
        trained = False
        print(f"[cmiarch] skipping train; found {ARCH_CKPT}", flush=True)
    else:
        print("[cmiarch] training 5-fold Chemprop with CMI hidden/depth/batch/epochs ...", flush=True)
        train_dmpnn_point(
            train_data=train_csv,
            train_features=train_feat,
            save_dir=ARCH_CKPT,
            architecture=arch,
            num_folds=5,
            seed=int(arch.get("seed", 0)),
        )
        trained = True
    pred_dir = REMAKE_ROOT / "results" / "stage1" / "cmi_arch_preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    evaluations = {}
    for split in ("test", "ext"):
        data_csv = CMI_DATA_WORK / f"{split}.csv"
        feat_csv = CMI_DATA_WORK / f"{split}_features.csv"
        y = _y_from_csv(data_csv)
        fold_rows = []
        fold_preds = []
        for fold in range(N_FOLDS):
            fold_dir = ARCH_CKPT / f"fold_{fold}"
            if not (fold_dir / "model_0" / "model.pt").exists():
                raise FileNotFoundError(fold_dir)
            yhat = _predict_logged(
                data_csv,
                feat_csv,
                fold_dir,
                pred_dir / f"{split}_fold{fold}.csv",
            )
            m = _metrics(y, yhat)
            fold_rows.append({"fold": fold, **m})
            fold_preds.append(np.asarray(yhat, dtype=float))
        ens = np.vstack(fold_preds).mean(axis=0)
        ens_path = pred_dir / f"{split}_ensemble.csv"
        pd.DataFrame({"logK": y, "y_pred": ens}).to_csv(ens_path, index=False)
        evaluations[f"cmi_{split}"] = {
            "n": int(len(y)),
            "ensemble": _metrics(y, ens),
            "fold_mean_mae": float(np.mean([r["mae"] for r in fold_rows])),
            "fold_mean_r2": float(np.mean([r["r2"] for r in fold_rows])),
            "fold_sd_mae": float(np.std([r["mae"] for r in fold_rows], ddof=1)),
            "fold_sd_r2": float(np.std([r["r2"] for r in fold_rows], ddof=1)),
            "per_fold": fold_rows,
            "pred_path": str(ens_path.relative_to(REMAKE_ROOT)).replace("\\", "/"),
        }
    return {
        "status": "complete",
        "trained_this_run": trained,
        "chemprop_version_used": chemprop_version(),
        "architecture": arch,
        "checkpoint_dir": str(ARCH_CKPT.relative_to(REMAKE_ROOT)).replace("\\", "/"),
        "evaluations": evaluations,
        "note": (
            "From-scratch Chemprop 1.6.1 training with architecture recovered from "
            "the released CMI best_model checkpoints (hidden 2400, depth 6, three "
            "FFN layers, batch 50, 30 epochs, seed 0, five random folds on the "
            "official train file). Published CMI weights were not loaded."
        ),
    }


def _load_existing() -> dict:
    if OUT_JSON.exists():
        return json.loads(OUT_JSON.read_text(encoding="utf-8"))
    return {}


def _save(payload: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("Wrote", OUT_JSON, flush=True)


def _section_complete(section) -> bool:
    return isinstance(section, dict) and str(section.get("status", "")).lower() == "complete"


def main() -> None:
    p = argparse.ArgumentParser(description="CMI M4 checkpoint and architecture reproduction")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--eval-weights", action="store_true")
    p.add_argument("--train-architecture", action="store_true")
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument(
        "--skip-if-complete",
        action="store_true",
        help="Skip released-weight eval and architecture retrain when cmi_reproduction_results.json is already complete.",
    )
    args = p.parse_args()
    payload = _load_existing()
    payload.setdefault("cmi_published", {
        "cmi_published_m4_test_mae": 0.599,
        "cmi_published_m4_test_r2": 0.934,
        "cmi_published_m4_ext_mae": 0.834,
        "cmi_published_m4_ext_r2": 0.942,
    })
    eval_weights = bool(args.eval_weights)
    train_architecture = bool(args.train_architecture)
    if not any((args.download_only, eval_weights, train_architecture)):
        eval_weights = True
        train_architecture = True
    if args.skip_if_complete:
        if _section_complete(payload.get("released_weights")):
            eval_weights = False
        if _section_complete(payload.get("architecture_retrain")):
            train_architecture = False
        if not args.download_only and not eval_weights and not train_architecture:
            print(f"[cmi] skipping; complete results in {OUT_JSON}", flush=True)
            return
    if args.download_only:
        download_best_model_folds()
        dump_cmi_architecture()
        write_official_chemprop_csvs()
        payload["download"] = {"status": "complete", "checkpoint_dir": str(CMI_CKPT_ROOT)}
        _save(payload)
        return
    if eval_weights:
        payload["released_weights"] = eval_released_weights()
        _save(payload)
    if train_architecture:
        payload["architecture_retrain"] = train_cmi_architecture(skip_if_exists=not args.force_retrain)
        _save(payload)


if __name__ == "__main__":
    main()
