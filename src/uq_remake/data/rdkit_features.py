"""RDKit 2D descriptors + row-level CMI metal_props + fixed continuous columns.

No electrolyte one-hot. No metal one-hot. Dimension comes from the fold
manifest / feature_schema, never from a hard-coded width.
RDKit columns with missing values, near-zero variance, or duplicate information
are dropped; remaining values are never imputed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import rdBase
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import StandardScaler

from uq_remake.data.feature_schema import (
    DS1_CONTINUOUS_COLS,
    FEATURE_ABS_MAX,
    assert_manifest_matches_runtime,
    assert_runtime_schema,
    ds1_rdkit_column_names,
    rdkit_selected_descriptor_names,
    schema_dict,
    verify_feature_name_layout,
)
from uq_remake.data.metal_props import METAL_PROP_COLS, metal_props_matrix

_CALC = None
_CALC_NAMES: list[str] | None = None


def _calculator() -> MoleculeDescriptors.MolecularDescriptorCalculator:
    global _CALC, _CALC_NAMES
    names = rdkit_selected_descriptor_names()
    if _CALC is None or _CALC_NAMES != names:
        _CALC = MoleculeDescriptors.MolecularDescriptorCalculator(names)
        _CALC_NAMES = list(names)
    return _CALC


def rdkit_version() -> str:
    return str(rdBase.rdkitVersion)


def rdkit_descriptor_vector(smiles: str) -> np.ndarray:
    """Selected RDKit 2D vector. Non-finite or |x|>1e6 is an error (no impute)."""
    names = rdkit_selected_descriptor_names()
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        raise ValueError(f"RDKit failed to parse ligand SMILES {smiles!r}; cannot impute descriptors.")
    vals = np.asarray(_calculator().CalcDescriptors(mol), dtype=float)
    if vals.shape[0] != len(names):
        raise RuntimeError(f"RDKit returned {vals.shape[0]} descriptors, expected {len(names)}")
    bad = ~np.isfinite(vals) | (np.abs(vals) > FEATURE_ABS_MAX)
    if bad.any():
        which = [names[i] for i, flag in enumerate(bad) if flag]
        raise ValueError(
            f"Non-finite or extreme RDKit descriptors {which} for SMILES {smiles!r}. "
            "These columns should have been dropped; imputation is disabled."
        )
    return vals


def _require_columns(df: pd.DataFrame, cols: list[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{context}: required columns absent {missing}. "
            "Do not silently shorten the feature vector."
        )


def build_rdkit_matrix(
    df: pd.DataFrame,
    ligand_col: str = "ligand_smiles",
    metal_el_col: str = "metal_el",
) -> tuple[np.ndarray, list[str]]:
    """DS1-style X = RDKit 2D || metal_props(metal) || DS1_CONTINUOUS_COLS.

    ``include_electrolyte`` is intentionally not a parameter: electrolyte identity
    never enters the tabular matrix. metal_props IE/valence are row-level CMI values.
    """
    assert_runtime_schema("DS1")
    _require_columns(df, [ligand_col, metal_el_col], "build_rdkit_matrix")
    _require_columns(df, DS1_CONTINUOUS_COLS, "build_rdkit_matrix")

    lig_cache: dict[str, np.ndarray] = {}
    rows = []
    for smi in df[ligand_col].astype(str):
        if smi not in lig_cache:
            lig_cache[smi] = rdkit_descriptor_vector(smi)
        rows.append(lig_cache[smi])
    X_desc = np.vstack(rows)
    X_metal = metal_props_matrix(df, metal_el_col=metal_el_col)
    X_cont = df[DS1_CONTINUOUS_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    X = np.hstack([X_desc, X_metal, X_cont])
    col_names = ds1_rdkit_column_names()
    if X.shape[1] != len(col_names):
        raise RuntimeError(
            f"DS1 RDKit width {X.shape[1]} != schema {len(col_names)}"
        )
    if not np.isfinite(X).all():
        n = int((~np.isfinite(X)).sum())
        raise ValueError(
            f"build_rdkit_matrix: {n} non-finite values after selection. Imputation is disabled."
        )
    verify_feature_name_layout("DS1", col_names)
    return X, col_names


def build_ds1_side_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """metal_props || DS1_CONTINUOUS_COLS (CPEMB / graph-side pack, no one-hot)."""
    _require_columns(df, ["metal_el"], "build_ds1_side_matrix")
    _require_columns(df, DS1_CONTINUOUS_COLS, "build_ds1_side_matrix")
    X_metal = metal_props_matrix(df, metal_el_col="metal_el")
    X_cont = df[DS1_CONTINUOUS_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    names = list(METAL_PROP_COLS) + list(DS1_CONTINUOUS_COLS)
    X = np.hstack([X_metal, X_cont])
    if X.shape[1] != len(names):
        raise RuntimeError(f"DS1 side width {X.shape[1]} != {len(names)}")
    return X, names


def assert_finite_features(X: np.ndarray, context: str) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if not np.isfinite(X).all():
        n = int((~np.isfinite(X)).sum())
        raise ValueError(
            f"{context}: {n} non-finite feature values. Imputation is disabled; "
            "drop incomplete descriptors instead."
        )
    return X


def fit_fold_preprocessor(X_train: np.ndarray) -> dict:
    """Fit StandardScaler on training rows only. No imputation."""
    X_tr = assert_finite_features(X_train, "fit_fold_preprocessor")
    scaler = StandardScaler().fit(X_tr)
    scale = np.asarray(scaler.scale_, dtype=float)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    scaler.scale_ = scale
    return {
        "imputation_medians": None,
        "imputation": False,
        "scaler_mean": np.asarray(scaler.mean_, dtype=float).tolist(),
        "scaler_scale": scale.tolist(),
    }


def transform_with_preprocessor(X: np.ndarray, prep: dict) -> np.ndarray:
    mean = np.asarray(prep["scaler_mean"], dtype=float)
    scale = np.asarray(prep["scaler_scale"], dtype=float)
    X = assert_finite_features(X, "transform_with_preprocessor")
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    return (X - mean) / scale


def preprocess_fold(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    prep = fit_fold_preprocessor(X_train)
    return (
        transform_with_preprocessor(X_train, prep),
        transform_with_preprocessor(X_val, prep),
        transform_with_preprocessor(X_test, prep),
        prep,
    )


def fold_manifest(
    dataset: str,
    fold_name: str,
    prep: dict,
    n_train: int,
    n_val: int,
    n_test: int,
) -> dict:
    schema = schema_dict(dataset)
    return {
        "dataset": dataset,
        "fold": fold_name,
        "rdkit_version": rdkit_version(),
        "pinned_rdkit_version": schema["pinned_rdkit_version"],
        "descriptor_names": schema["descriptor_names"],
        "rdkit_n_descriptors": schema["rdkit_n_descriptors"],
        "continuous_feature_names": schema["continuous_feature_names"],
        "metal_prop_cols": schema["metal_prop_cols"],
        "metal_props_convention": schema["metal_props_convention"],
        "feature_names": schema["feature_names"],
        "feature_dimension": schema["feature_dimension"],
        "feature_hash": schema["feature_hash"],
        "preprocessing": schema["preprocessing"],
        "imputation_medians": prep["imputation_medians"],
        "scaler_mean": prep["scaler_mean"],
        "scaler_scale": prep["scaler_scale"],
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),
        "electrolyte_in_X": False,
        "metal_onehot_in_X": False,
        "include_solvent_properties": schema["include_solvent_properties"],
        "note": schema["note"],
    }


def load_fold_features(fold_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load arrays.npz + manifest; fail if the descriptor schema does not match runtime."""
    fold_dir = Path(fold_dir)
    manifest = json.loads((fold_dir / "manifest.json").read_text(encoding="utf-8"))
    assert_manifest_matches_runtime(manifest)
    z = np.load(fold_dir / "arrays.npz")
    arrays = {k: np.asarray(z[k]) for k in z.files}
    dim = int(manifest["feature_dimension"])
    for key in ("X_train", "X_val", "X_test"):
        if key in arrays and int(arrays[key].shape[1]) != dim:
            raise RuntimeError(
                f"{fold_dir}: {key} width {arrays[key].shape[1]} != manifest {dim}"
            )
    return arrays, manifest


def save_fold_features(
    out_dir: Path,
    Xtr: np.ndarray,
    Xva: np.ndarray,
    Xte: np.ndarray,
    ytr: np.ndarray,
    yva: np.ndarray,
    yte: np.ndarray,
    manifest: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "arrays.npz",
        X_train=np.asarray(Xtr, dtype=np.float32),
        X_val=np.asarray(Xva, dtype=np.float32),
        X_test=np.asarray(Xte, dtype=np.float32),
        y_train=np.asarray(ytr, dtype=np.float32),
        y_val=np.asarray(yva, dtype=np.float32),
        y_test=np.asarray(yte, dtype=np.float32),
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
