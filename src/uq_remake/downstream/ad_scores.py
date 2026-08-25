"""Outer-training-only AD / risk-ordering scores. Not OOD detectors."""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.covariance import LedoitWolf

from uq_remake.downstream.protocol import (
    COVARIANCE_ESTIMATOR,
    FEATURE_KNN_K,
    FINGERPRINT_N_BITS,
    FINGERPRINT_RADIUS,
    FINGERPRINT_TYPE,
    PAIR_COLUMN,
    PAIR_DENSITY_BINS,
)
from uq_remake.metrics.cv_protocol import SPECIES_PAIR_RE


def morgan_fp(smiles: str, radius: int = FINGERPRINT_RADIUS, n_bits: int = FINGERPRINT_N_BITS):
    mol = Chem.MolFromSmiles(str(smiles)) if smiles not in (None, "", "nan") else None
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, int(radius), nBits=int(n_bits))


def fingerprint_library(smiles_list: list[str], radius: int = FINGERPRINT_RADIUS, n_bits: int = FINGERPRINT_N_BITS):
    uniq = list(dict.fromkeys(str(s) for s in smiles_list))
    fps = {}
    n_fail = 0
    for s in uniq:
        fp = morgan_fp(s, radius=radius, n_bits=n_bits)
        if fp is None:
            n_fail += 1
        fps[s] = fp
    return fps, n_fail


def tanimoto_nn_and_median(query_smiles: np.ndarray, ref_fps: dict, radius: int, n_bits: int) -> tuple[np.ndarray, np.ndarray, int]:
    """d_NN = 1 - max_z S(x,z); d_med = median_z (1 - S(x,z)). Higher = more novel."""
    bundle = tanimoto_distance_bundle(query_smiles, ref_fps, radius=radius, n_bits=n_bits, ks=(1,))
    return bundle["tanimoto_nn_distance"], bundle["tanimoto_median_distance"], bundle["n_fail"]


def tanimoto_distance_bundle(
    query_smiles: np.ndarray,
    ref_fps: dict,
    radius: int,
    n_bits: int,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """Fold-local Tanimoto distances. Unique query SMILES are scored once, then mapped."""
    ref_fp_only = [fp for fp in ref_fps.values() if fp is not None]
    n_ref = len(ref_fp_only)
    query = np.asarray(query_smiles, dtype=object)
    uniq = list(dict.fromkeys(str(s) for s in query))
    uniq_nn = {}
    uniq_med = {}
    uniq_knn = {int(k): {} for k in ks}
    n_fail = 0
    for s in uniq:
        fp = morgan_fp(s, radius=radius, n_bits=n_bits)
        if fp is None or n_ref == 0:
            n_fail += 1
            uniq_nn[s] = 1.0
            uniq_med[s] = 1.0
            for k in ks:
                uniq_knn[int(k)][s] = 1.0
            continue
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, ref_fp_only), dtype=float)
        dists = 1.0 - sims
        uniq_nn[s] = float(np.min(dists))
        uniq_med[s] = float(np.median(dists))
        for k in ks:
            k_eff = min(int(k), n_ref)
            if k_eff <= 0:
                uniq_knn[int(k)][s] = 1.0
            else:
                uniq_knn[int(k)][s] = float(np.mean(np.partition(dists, k_eff - 1)[:k_eff]))
    n = len(query)
    out = {
        "tanimoto_nn_distance": np.array([uniq_nn[str(s)] for s in query], dtype=float),
        "tanimoto_median_distance": np.array([uniq_med[str(s)] for s in query], dtype=float),
        "n_fail": int(n_fail),
        "n_ref": int(n_ref),
        "n_unique_query": int(len(uniq)),
        "n_query": int(n),
    }
    for k in ks:
        out[f"tanimoto_knn_mean_{int(k)}"] = np.array(
            [uniq_knn[int(k)][str(s)] for s in query], dtype=float
        )
    return out


def fit_mahalanobis_ledoit(X_train: np.ndarray) -> dict:
    X_train = np.asarray(X_train, dtype=float)
    est = LedoitWolf().fit(X_train)
    cov = np.asarray(est.covariance_, dtype=float)
    rank = int(np.linalg.matrix_rank(cov, tol=1e-10))
    try:
        cond = float(np.linalg.cond(cov))
    except Exception:
        cond = float("inf")
    flag = "stable" if np.isfinite(cond) and cond < 1e12 else "ill_conditioned"
    return {
        "estimator": est,
        "mean": np.asarray(est.location_, dtype=float),
        "covariance": cov,
        "effective_rank": rank,
        "condition_number": cond,
        "stability_flag": flag,
        "shrinkage": float(getattr(est, "shrinkage_", np.nan)),
        "reference_row_count": int(X_train.shape[0]),
        "n_features": int(X_train.shape[1]),
        "covariance_estimator": COVARIANCE_ESTIMATOR,
    }


def mahalanobis_distance(fit: dict, X: np.ndarray) -> np.ndarray:
    """sqrt of sklearn LedoitWolf.mahalanobis (which returns squared distance)."""
    X = np.asarray(X, dtype=float)
    d2 = fit["estimator"].mahalanobis(X)
    d2 = np.maximum(np.asarray(d2, dtype=float), 0.0)
    return np.sqrt(d2)


def _whiten_ledoit(fit: dict, X: np.ndarray) -> np.ndarray:
    """Center and whiten with the same Ledoit–Wolf covariance used for centroid Mahalanobis."""
    X = np.asarray(X, dtype=float)
    mu = np.asarray(fit["mean"], dtype=float)
    cov = np.asarray(fit["covariance"], dtype=float)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 1e-12)
    w = evecs * (1.0 / np.sqrt(evals))
    return (X - mu) @ w


def _median_euclidean_to_all(X_ref: np.ndarray, X_query: np.ndarray, batch: int = 8) -> np.ndarray:
    """Median Euclidean distance to every training row. Small batches keep peak RAM low."""
    X_ref = np.asarray(X_ref, dtype=np.float32)
    X_query = np.asarray(X_query, dtype=np.float32)
    ref_norm = np.sum(X_ref * X_ref, axis=1)
    out = np.empty(len(X_query), dtype=np.float64)
    step = max(int(batch), 1)
    for i in range(0, len(X_query), step):
        q = X_query[i : i + step]
        q_norm = np.sum(q * q, axis=1, keepdims=True)
        d2 = q_norm + ref_norm - 2.0 * (q @ X_ref.T)
        np.maximum(d2, 0.0, out=d2)
        out[i : i + step] = np.sqrt(np.median(d2, axis=1))
    return out


def feature_distance_bundle(
    X_train: np.ndarray,
    X_query: np.ndarray,
    ks: tuple[int, ...] = FEATURE_KNN_K,
) -> dict:
    """RDKit+metal+medium feature-space distances. Matrices are already StandardScaled.

    Centroid Mahalanobis is the published retention score. NN / kNN Mahalanobis,
    Euclidean NN, and median Euclidean-to-all are SI Spearman diagnostics.
    """
    from sklearn.neighbors import NearestNeighbors

    X_train = np.asarray(X_train, dtype=float)
    X_query = np.asarray(X_query, dtype=float)
    fit = fit_mahalanobis_ledoit(X_train)
    d_cent = mahalanobis_distance(fit, X_query)
    zw_train = _whiten_ledoit(fit, X_train)
    zw_query = _whiten_ledoit(fit, X_query)
    k_vals = tuple(int(k) for k in ks if int(k) > 0)
    k_nn = min(max(k_vals, default=1), len(X_train)) if len(X_train) else 1
    if len(X_train) == 0 or len(X_query) == 0:
        nan = np.full(len(X_query), np.nan, dtype=float)
        out = {
            "mahalanobis_feature_centroid": nan.copy(),
            "mahalanobis_joint": nan.copy(),
            "mahalanobis_feature_nn": nan.copy(),
            "euclidean_feature_nn": nan.copy(),
            "euclidean_feature_median": nan.copy(),
            "mahalanobis_fit": fit,
        }
        for k in k_vals:
            out[f"mahalanobis_feature_knn_mean_{k}"] = nan.copy()
        return out

    d_med = _median_euclidean_to_all(X_train, X_query)
    nn_mah = NearestNeighbors(n_neighbors=k_nn, metric="euclidean", algorithm="brute")
    nn_mah.fit(zw_train)
    d_mah, _ = nn_mah.kneighbors(zw_query)
    del nn_mah
    nn_euc = NearestNeighbors(n_neighbors=1, metric="euclidean", algorithm="brute")
    nn_euc.fit(X_train)
    d_euc, _ = nn_euc.kneighbors(X_query)
    del nn_euc
    out = {
        "mahalanobis_feature_centroid": np.asarray(d_cent, dtype=float),
        "mahalanobis_joint": np.asarray(d_cent, dtype=float),
        "mahalanobis_feature_nn": d_mah[:, 0].astype(float),
        "euclidean_feature_nn": d_euc[:, 0].astype(float),
        "euclidean_feature_median": np.asarray(d_med, dtype=float),
        "mahalanobis_fit": fit,
    }
    for k in k_vals:
        k_eff = min(int(k), d_mah.shape[1])
        out[f"mahalanobis_feature_knn_mean_{int(k)}"] = np.mean(d_mah[:, :k_eff], axis=1).astype(float)
    return out


def pair_training_ligand_counts(train_df: pd.DataFrame, pair_col: str = PAIR_COLUMN) -> dict[str, int]:
    counts: dict[str, int] = {}
    if pair_col not in train_df.columns or "ligand_smiles" not in train_df.columns:
        return counts
    g = train_df.groupby(pair_col, sort=False)["ligand_smiles"].nunique()
    return {str(k): int(v) for k, v in g.items()}


def density_score_from_count(n: np.ndarray) -> np.ndarray:
    n = np.asarray(n, dtype=float)
    return 1.0 / (1.0 + np.maximum(n, 0.0))


def assert_species_pairs(values) -> int:
    vals = pd.Series(values).astype(str)
    bad = vals[~vals.str.match(SPECIES_PAIR_RE)]
    return int(len(bad))


def medium_indicators(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=test_df.index)
    if "electrolyte" in train_df.columns and "electrolyte" in test_df.columns:
        seen_e = set(train_df["electrolyte"].astype(str))
        out["electrolyte_seen"] = test_df["electrolyte"].astype(str).isin(seen_e).to_numpy()
        out["electrolyte"] = test_df["electrolyte"].astype(str).to_numpy()
    else:
        out["electrolyte_seen"] = np.nan
        out["electrolyte"] = ""
    if "ligand_smiles" in train_df.columns and "electrolyte" in train_df.columns:
        train_lm = set(zip(train_df["ligand_smiles"].astype(str), train_df["electrolyte"].astype(str)))
        out["ligand_medium_seen"] = [
            (str(l), str(e)) in train_lm
            for l, e in zip(test_df["ligand_smiles"].astype(str), test_df["electrolyte"].astype(str))
        ]
    else:
        out["ligand_medium_seen"] = np.nan
    if PAIR_COLUMN in train_df.columns and "electrolyte" in train_df.columns:
        train_mm = set(zip(train_df[PAIR_COLUMN].astype(str), train_df["electrolyte"].astype(str)))
        out["metal_medium_seen"] = [
            (str(p), str(e)) in train_mm
            for p, e in zip(test_df[PAIR_COLUMN].astype(str), test_df["electrolyte"].astype(str))
        ]
    else:
        out["metal_medium_seen"] = np.nan
    return out


def standardize_train_only(X_train: np.ndarray, X_query: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-dimension z-score using training-fold statistics only."""
    X_train = np.asarray(X_train, dtype=float)
    X_query = np.asarray(X_query, dtype=float)
    mu = np.mean(X_train, axis=0)
    sd = np.std(X_train, axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X_train - mu) / sd, (X_query - mu) / sd, {"mean": mu, "std": sd}


def _knn_means(dists: np.ndarray, ks: tuple[int, ...]) -> dict[str, np.ndarray]:
    out = {}
    n_avail = int(dists.shape[1]) if dists.ndim == 2 else 0
    for k in ks:
        k_eff = min(int(k), n_avail)
        if k_eff <= 0:
            out[int(k)] = np.full(len(dists), np.nan, dtype=float)
        else:
            out[int(k)] = np.mean(dists[:, :k_eff], axis=1)
    return out


def embedding_distance_suite(
    H_train: np.ndarray,
    H_query: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """Fold-local Chemprop embedding distances. Train stats/covariance only.

    Default Mahalanobis is distance to the training centroid (Ledoit–Wolf).
    Nearest-embedding Mahalanobis is a sensitivity score.
    """
    from sklearn.neighbors import NearestNeighbors

    H_train = np.asarray(H_train, dtype=float)
    H_query = np.asarray(H_query, dtype=float)
    n_train = int(H_train.shape[0])
    n_query = int(H_query.shape[0])
    nan = np.full(n_query, np.nan, dtype=float)
    empty = {
        "chemprop_euclidean_nn": nan.copy(),
        "chemprop_std_euclidean_nn": nan.copy(),
        "chemprop_cosine_nn": nan.copy(),
        "chemprop_mahalanobis_centroid": nan.copy(),
        "chemprop_mahalanobis_nn": nan.copy(),
        "mahalanobis_fit": None,
        "n_train_embeddings": n_train,
        "n_query_embeddings": n_query,
    }
    for k in ks:
        empty[f"chemprop_std_euclidean_knn_{int(k)}"] = nan.copy()
    if n_train == 0 or n_query == 0 or H_train.ndim != 2:
        return empty

    k_nn = min(max(int(k) for k in ks), n_train)
    nn_raw = NearestNeighbors(n_neighbors=k_nn, metric="euclidean", algorithm="auto")
    nn_raw.fit(H_train)
    d_raw, _ = nn_raw.kneighbors(H_query)
    Z_train, Z_query, _stats = standardize_train_only(H_train, H_query)
    nn_std = NearestNeighbors(n_neighbors=k_nn, metric="euclidean", algorithm="auto")
    nn_std.fit(Z_train)
    d_std, _ = nn_std.kneighbors(Z_query)
    nn_cos = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="auto")
    nn_cos.fit(H_train)
    d_cos, _ = nn_cos.kneighbors(H_query)

    mahal_fit = fit_mahalanobis_ledoit(Z_train)
    d_cent = mahalanobis_distance(mahal_fit, Z_query)

    cov = np.asarray(mahal_fit["covariance"], dtype=float)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 1e-12)
    W = evecs * (1.0 / np.sqrt(evals))
    Zw_train = Z_train @ W
    Zw_query = Z_query @ W
    nn_mah = NearestNeighbors(n_neighbors=1, metric="euclidean", algorithm="auto")
    nn_mah.fit(Zw_train)
    d_mah_nn, _ = nn_mah.kneighbors(Zw_query)

    knn = _knn_means(d_std, ks)
    out = {
        "chemprop_euclidean_nn": d_raw[:, 0].astype(float),
        "chemprop_std_euclidean_nn": d_std[:, 0].astype(float),
        "chemprop_cosine_nn": d_cos[:, 0].astype(float),
        "chemprop_mahalanobis_centroid": np.asarray(d_cent, dtype=float),
        "chemprop_mahalanobis_nn": d_mah_nn[:, 0].astype(float),
        "mahalanobis_fit": mahal_fit,
        "n_train_embeddings": n_train,
        "n_query_embeddings": n_query,
        "embedding_dim": int(H_train.shape[1]),
        "standardized": True,
        "mahalanobis_default": "centroid",
    }
    for k in ks:
        out[f"chemprop_std_euclidean_knn_{int(k)}"] = knn[int(k)]
    return out


def bin_pair_count(n: int) -> str:
    for name, lo, hi in PAIR_DENSITY_BINS:
        if lo <= int(n) <= hi:
            return name
    return "0"
