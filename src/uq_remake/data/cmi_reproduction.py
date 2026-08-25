"""Download and inspect published CMI M4 Chemprop checkpoints."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.request import urlopen, Request

from uq_remake import REMAKE_ROOT
from uq_remake.data.feature_schema import CHEMPROP_DS1_FEATURE_COLUMNS

CMI_GITHUB = "Critical-Materials-Institute/stabilityconstant-ml-models"
CMI_REF = "main"
CMI_BEST_REL = "models/stability_constan_protonation_constant_model_M4/best_model"
CMI_CKPT_ROOT = REMAKE_ROOT / "checkpoints" / "cmi_published" / "M4_best_model"
CMI_DATA_WORK = REMAKE_ROOT / "data" / "processed" / "features" / "chemprop_cmi_official"
RAW_M4 = REMAKE_ROOT / "data" / "raw_m4"
N_FOLDS = 5

ARCH_KEYS = (
    "hidden_size",
    "depth",
    "dropout",
    "ffn_num_layers",
    "ffn_hidden_size",
    "epochs",
    "batch_size",
    "init_lr",
    "max_lr",
    "final_lr",
    "warmup_epochs",
    "aggregation",
    "activation",
    "bias",
    "features_scaling",
    "no_features_scaling",
    "seed",
    "pytorch_seed",
    "metric",
    "dataset_type",
    "ensemble_size",
    "number_of_molecules",
)


def cmi_raw_url(rel: str) -> str:
    return f"https://raw.githubusercontent.com/{CMI_GITHUB}/{CMI_REF}/{rel.replace(' ', '%20')}"


def fold_model_pt(fold: int) -> Path:
    return CMI_CKPT_ROOT / f"fold_{fold}" / "model_0" / "model.pt"


def download_file(url: str, dest: Path, *, timeout: int = 600) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = Request(url, headers={"User-Agent": "uq-remake-cmi-reproduction"})
    with urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)


def download_best_model_folds(*, force: bool = False) -> list[Path]:
    paths = []
    for fold in range(N_FOLDS):
        dest = fold_model_pt(fold)
        rel = f"{CMI_BEST_REL}/fold_{fold}/model_0/model.pt"
        if dest.exists() and dest.stat().st_size > 1_000_000 and not force:
            paths.append(dest)
            continue
        print(f"downloading CMI M4 best_model fold_{fold} ...", flush=True)
        download_file(cmi_raw_url(rel), dest)
        print(f"  wrote {dest} ({dest.stat().st_size} bytes)", flush=True)
        paths.append(dest)
    (CMI_CKPT_ROOT / "provenance.json").write_text(
        json.dumps(
            {
                "github": f"https://github.com/{CMI_GITHUB}",
                "ref": CMI_REF,
                "relative": CMI_BEST_REL,
                "n_folds": N_FOLDS,
                "files": [str(p.relative_to(REMAKE_ROOT)).replace("\\", "/") for p in paths],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths


def _args_to_dict(args: Any) -> dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, dict):
        src = args
    else:
        src = vars(args) if hasattr(args, "__dict__") else {}
    out: dict[str, Any] = {}
    for key in ARCH_KEYS:
        if key in src:
            val = src[key]
            if isinstance(val, (str, int, float, bool)) or val is None:
                out[key] = val
            else:
                out[key] = str(val)
    n_f = src.get("features_size") or src.get("ffn_input_size")
    if n_f is not None:
        out["features_size"] = int(n_f) if not isinstance(n_f, str) else n_f
    return out


def dump_checkpoint_args(model_pt: Path) -> dict[str, Any]:
    import torch

    raw = torch.load(str(model_pt), map_location="cpu", weights_only=False)
    args = None
    if isinstance(raw, dict):
        args = raw.get("args")
    dumped = _args_to_dict(args)
    dumped["checkpoint_path"] = str(model_pt.relative_to(REMAKE_ROOT)).replace("\\", "/")
    dumped["checkpoint_bytes"] = int(model_pt.stat().st_size)
    if isinstance(raw, dict):
        dumped["checkpoint_keys"] = sorted(str(k) for k in raw.keys())
        scaler = raw.get("scaler")
        features_scaler = raw.get("features_scaler")
        dumped["has_target_scaler"] = scaler is not None
        dumped["has_features_scaler"] = features_scaler is not None
    return dumped


def dump_cmi_architecture() -> dict[str, Any]:
    per_fold = []
    for fold in range(N_FOLDS):
        pt = fold_model_pt(fold)
        if not pt.exists():
            raise FileNotFoundError(pt)
        rec = dump_checkpoint_args(pt)
        rec["fold"] = fold
        per_fold.append(rec)
    ref = {k: per_fold[0].get(k) for k in ARCH_KEYS if k in per_fold[0]}
    mismatched = []
    for rec in per_fold[1:]:
        for k, v in ref.items():
            if rec.get(k) != v:
                mismatched.append({"fold": rec["fold"], "key": k, "ref": v, "got": rec.get(k)})
    payload = {
        "source": "CMI released best_model checkpoints",
        "n_folds": N_FOLDS,
        "architecture": ref,
        "fold_mismatch": mismatched,
        "folds": per_fold,
        "chemprop_predict_checkpoint_dir": str(CMI_CKPT_ROOT.relative_to(REMAKE_ROOT)).replace("\\", "/"),
    }
    out = REMAKE_ROOT / "results" / "stage1" / "cmi_released_architecture.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def write_official_chemprop_csvs(*, raw_dir: Path | None = None, out_dir: Path | None = None) -> dict[str, str]:
    """Write Chemprop CSVs from the official CMI train/test/validation files."""
    import pandas as pd
    from uq_remake.data.build_m4 import _harmonize_feature_columns

    raw_dir = raw_dir or RAW_M4
    out_dir = out_dir or CMI_DATA_WORK
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = list(CHEMPROP_DS1_FEATURE_COLUMNS)
    mapping = {
        "train": ("train_input.csv", "train_features.csv", "train"),
        "test": ("test_input.csv", "test_features.csv", "test"),
        "ext": ("validation_input.csv", "validation_features.csv", "ext"),
    }
    written = {}
    for split, (inp_name, feat_name, stem) in mapping.items():
        inp = pd.read_csv(raw_dir / inp_name)
        feat = pd.read_csv(raw_dir / feat_name, encoding="utf-8-sig")
        feat = _harmonize_feature_columns(feat)
        feat.columns = [str(c).strip() for c in feat.columns]
        if "cmplx_smiles_neutral_std" in inp.columns:
            smiles = inp["cmplx_smiles_neutral_std"]
            y = inp["Exp LgK1"]
        else:
            smiles_col = "SMILES" if "SMILES" in inp.columns else inp.columns[0]
            y_col = "Experimental Lg K1" if "Experimental Lg K1" in inp.columns else inp.columns[1]
            smiles = inp[smiles_col]
            y = inp[y_col]
        missing = [c for c in cols if c not in feat.columns]
        if missing:
            raise KeyError(f"{split}: missing feature columns {missing}; have {list(feat.columns)}")
        feats = feat[cols].apply(pd.to_numeric, errors="coerce")
        n_na = int(feats.isna().any(axis=1).sum())
        data_path = out_dir / f"{stem}.csv"
        feat_path = out_dir / f"{stem}_features.csv"
        pd.DataFrame({"smiles": smiles.values, "logK": pd.to_numeric(y, errors="coerce").values}).to_csv(
            data_path, index=False
        )
        feats.to_csv(feat_path, index=False)
        written[split] = {
            "data": str(data_path.relative_to(REMAKE_ROOT)).replace("\\", "/"),
            "features": str(feat_path.relative_to(REMAKE_ROOT)).replace("\\", "/"),
            "n": int(len(feats)),
            "n_na_feature_rows": n_na,
        }
        if n_na:
            raise ValueError(f"{split}: {n_na} rows have NaN additional features after harmonization")
    (out_dir / "schema.json").write_text(
        json.dumps({"columns": cols, "n_features": len(cols), "splits": written}, indent=2),
        encoding="utf-8",
    )
    return written
