"""Fold-local Chemprop MPN graph embeddings for AD distances.

Embeddings come from the outer-fold Chemprop evidential checkpoint. Additional
metal/medium features are *not* concatenated: Chemprop's encoder would otherwise
mix pair metadata into the supposed molecular representation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from uq_remake.downstream.protocol import rel, sha256_file
from uq_remake.models.chemprop_dmpnn import (
    _ensure_rdbase,
    _load_chemprop_test_loader,
    _stub_descriptastorus,
    resolve_chemprop_model_pt,
)


def chemprop_feature_dir(dataset: str, fold: str) -> Path:
    from uq_remake import REMAKE_ROOT

    tag = "chemprop_ds1" if str(dataset).upper() == "DS1" else "chemprop_ds2"
    return REMAKE_ROOT / "data" / "processed" / "features" / tag / fold


def chemprop_checkpoint_dir(model_id: str, fold: str) -> Path:
    from uq_remake import REMAKE_ROOT

    return REMAKE_ROOT / "checkpoints" / "active" / model_id / fold / "member_0"


def unique_smiles_tables(data_csv: Path, feat_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(data_csv)
    feat = pd.read_csv(feat_csv)
    if len(data) != len(feat):
        raise RuntimeError(f"Chemprop data/features length mismatch: {data_csv} {len(data)} vs {feat_csv} {len(feat)}")
    smiles = data["smiles"].astype(str)
    keep = ~smiles.duplicated(keep="first")
    return data.loc[keep].reset_index(drop=True), feat.loc[keep].reset_index(drop=True)


def _unpack_batch(batch):
    if hasattr(batch, "batch_graph"):
        return (
            batch.batch_graph(),
            batch.features() if hasattr(batch, "features") else None,
            None,
            None,
            None,
            None,
        )
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        mol_batch = batch[0]
        features_batch = batch[1]
        atom_descriptors_batch = batch[3] if len(batch) > 3 else None
        atom_features_batch = batch[4] if len(batch) > 4 else None
        bond_descriptors_batch = batch[5] if len(batch) > 5 else None
        bond_features_batch = batch[6] if len(batch) > 6 else None
        return (
            mol_batch,
            features_batch,
            atom_descriptors_batch,
            atom_features_batch,
            bond_descriptors_batch,
            bond_features_batch,
        )
    raise TypeError(f"Unrecognized Chemprop batch type {type(batch)}")


def extract_graph_embeddings(
    data_csv: Path,
    feat_csv: Path,
    checkpoint_dir: Path,
    out_npz: Path,
    *,
    dataset: str,
    fold: str,
    model_id: str,
    split: str,
    unique_smiles: bool = True,
) -> dict:
    """Extract MPN graph hidden states. Cached at out_npz when checkpoint hash matches."""
    _ensure_rdbase()
    _stub_descriptastorus()
    import torch
    from chemprop.utils import load_checkpoint

    model_path = resolve_chemprop_model_pt(
        Path(checkpoint_dir),
        dataset=dataset,
        fold=fold,
        member_id=0,
        model_id=model_id,
    )
    ck_hash = sha256_file(model_path)
    meta_path = Path(out_npz).with_suffix(".json")
    if Path(out_npz).exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("checkpoint_hash") == ck_hash and meta.get("graph_only") is True:
            z = np.load(out_npz, allow_pickle=True)
            return {
                "smiles": np.asarray(z["smiles"], dtype=object),
                "embedding": np.asarray(z["embedding"], dtype=float),
                "meta": meta,
                "cached": True,
            }

    work_data, work_feat = Path(data_csv), Path(feat_csv)
    tmp_dir = None
    if unique_smiles:
        udata, ufeat = unique_smiles_tables(data_csv, feat_csv)
        tmp_dir = Path(out_npz).parent / f"_tmp_{split}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        work_data = tmp_dir / "unique.csv"
        work_feat = tmp_dir / "unique_features.csv"
        udata.to_csv(work_data, index=False)
        ufeat.to_csv(work_feat, index=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(model_path, device=device)
    model.eval()
    encoder = model.encoder
    old_flag = bool(getattr(encoder, "use_input_features", False))
    encoder.use_input_features = False
    data, loader = _load_chemprop_test_loader(work_data, work_feat, model_path, batch_size=256)
    fps = []
    with torch.no_grad():
        for batch in loader:
            mol_batch, features_batch, atom_d, atom_f, bond_d, bond_f = _unpack_batch(batch)
            fp = model.fingerprint(
                mol_batch,
                None,
                atom_d,
                atom_f,
                bond_d,
                bond_f,
                fingerprint_type="MPN",
            )
            fps.append(fp.detach().cpu().numpy())
    encoder.use_input_features = old_flag
    emb = np.concatenate(fps, axis=0) if fps else np.zeros((0, 0), dtype=float)
    smiles = pd.read_csv(work_data)["smiles"].astype(str).to_numpy()
    if len(smiles) != len(emb):
        raise RuntimeError(f"Embedding length {len(emb)} != smiles {len(smiles)} for {work_data}")
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, smiles=smiles.astype(object), embedding=np.asarray(emb, dtype=np.float32))
    meta = {
        "dataset": dataset,
        "fold": fold,
        "model_id": model_id,
        "split": split,
        "checkpoint_path": rel(model_path),
        "checkpoint_hash": ck_hash,
        "graph_only": True,
        "use_input_features_during_ad": False,
        "encoder_use_input_features_trained": old_flag,
        "n_embeddings": int(len(emb)),
        "embedding_dim": int(emb.shape[1]) if emb.ndim == 2 and emb.size else 0,
        "unique_smiles": bool(unique_smiles),
        "source_data": rel(data_csv),
        "note": (
            "MPN graph hidden state without concatenating additional features. "
            "Chemprop-specific AD score; not available for MLP/SNGP/GPR."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if tmp_dir is not None:
        for p in tmp_dir.glob("*"):
            p.unlink()
        try:
            tmp_dir.rmdir()
        except OSError:
            pass
    return {"smiles": smiles, "embedding": np.asarray(emb, dtype=float), "meta": meta, "cached": False}


def map_embeddings(row_smiles: np.ndarray, unique_smiles: np.ndarray, unique_emb: np.ndarray) -> np.ndarray:
    lookup = {str(s): unique_emb[i] for i, s in enumerate(unique_smiles)}
    dim = int(unique_emb.shape[1]) if unique_emb.ndim == 2 and unique_emb.size else 0
    out = np.full((len(row_smiles), dim), np.nan, dtype=float)
    missing = 0
    for i, s in enumerate(row_smiles):
        vec = lookup.get(str(s))
        if vec is None:
            missing += 1
            continue
        out[i] = vec
    return out
