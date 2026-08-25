"""Fold-local AD retention study. No grid retraining."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402  — before rdkit on Windows

from uq_remake.downstream.ad_scores import (  # noqa: E402
    assert_species_pairs,
    density_score_from_count,
    embedding_distance_suite,
    feature_distance_bundle,
    fingerprint_library,
    medium_indicators,
    pair_training_ligand_counts,
    tanimoto_distance_bundle,
)
from uq_remake.downstream.analyses import (  # noqa: E402
    random_retention_fold,
    risk_coverage_fold,
    summarize_fold_metrics,
    utility_overlap_fold,
)
from uq_remake.downstream.chemprop_embeddings import (  # noqa: E402
    chemprop_checkpoint_dir,
    chemprop_feature_dir,
    extract_graph_embeddings,
    map_embeddings,
)
from uq_remake.downstream.protocol import (  # noqa: E402
    AD_RETENTION_DIR,
    CHEMPROP_ONLY_SCORES,
    CHEMPROP_PRIMARY_SCORES,
    CHEMPROP_SENSITIVITY_SCORES,
    COVARIANCE_ESTIMATOR,
    DS1_CHEMPROP,
    DS1_WORKHORSE,
    DS2_CHEMPROP,
    DS2_WORKHORSE,
    EMBEDDING_K,
    FEATURE_KNN_K,
    FEATURE_SPACE_SI_SCORES,
    EVALUATION_PARTITION,
    FINGERPRINT_N_BITS,
    FINGERPRINT_RADIUS,
    FINGERPRINT_TYPE,
    FOLDS,
    family_display_name,
    HIGH_CONFIDENCE_RETENTION,
    PAIR_COLUMN,
    PRIMARY_RETENTION,
    PRIMARY_SCORE_SET,
    RANDOM_N_REPEATS,
    RANDOM_SEED,
    RETENTION_FRACTIONS,
    TANIMOTO_K,
    WORKHORSE_PRIMARY_SCORES,
    WORKHORSE_SENSITIVITY_SCORES,
    rel,
)
from uq_remake.metrics.cv_protocol import fold_mean_sd  # noqa: E402
from uq_remake.training.protocol import load_folds, load_table  # noqa: E402


def _outer_frames(table: pd.DataFrame, folds: dict, fold: str):
    split = folds[fold]
    tr = np.asarray(split["train_idx"], dtype=int)
    va = np.asarray(split["val_idx"], dtype=int)
    te = np.asarray(split["test_idx"], dtype=int)
    outer = np.concatenate([tr, va])
    train_df = table.iloc[outer].copy()
    test_df = table.iloc[te].copy()
    train_df["_table_idx"] = outer
    test_df["_table_idx"] = te
    return tr, va, te, outer, train_df, test_df

JOBS = (
    {"dataset": "DS1", "model_id": DS1_WORKHORSE, "ad_family": "workhorse"},
    {"dataset": "DS2", "model_id": DS2_WORKHORSE, "ad_family": "workhorse"},
    {"dataset": "DS1", "model_id": DS1_CHEMPROP, "ad_family": "chemprop"},
    {"dataset": "DS2", "model_id": DS2_CHEMPROP, "ad_family": "chemprop"},
)


def load_frozen_fold_arrays(fold_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load precomputed fold arrays without re-checking the live RDKit 2D list.

    AD scoring consumes frozen StandardScaled matrices. Descriptor regeneration is
    not performed.
    """
    fold_dir = Path(fold_dir)
    manifest = json.loads((fold_dir / "manifest.json").read_text(encoding="utf-8"))
    z = np.load(fold_dir / "arrays.npz")
    arrays = {k: np.asarray(z[k]) for k in z.files}
    dim = int(manifest["feature_dimension"])
    for key in ("X_train", "X_val", "X_test"):
        if key in arrays and int(arrays[key].shape[1]) != dim:
            raise RuntimeError(f"{fold_dir}: {key} width {arrays[key].shape[1]} != manifest {dim}")
    return arrays, manifest


def _write_csv(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
    else:
        pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path


def pred_path(model_id: str, fold: str) -> Path:
    return REPO_ROOT / "results" / "active_grid" / model_id / fold / "test_preds.csv"


def feature_dir(dataset: str, fold: str) -> Path:
    tag = "rdkit_ds1" if dataset.upper() == "DS1" else "rdkit_ds2"
    return REPO_ROOT / "data" / "processed" / "features" / tag / fold


def score_role(ad_family: str, score: str) -> str:
    if ad_family == "chemprop":
        return "primary" if score in CHEMPROP_PRIMARY_SCORES else "sensitivity"
    return "primary" if score in WORKHORSE_PRIMARY_SCORES else "sensitivity"


def scores_for_family(ad_family: str, dataset: str) -> tuple[str, ...]:
    if ad_family == "chemprop":
        scores = CHEMPROP_PRIMARY_SCORES + CHEMPROP_SENSITIVITY_SCORES
    else:
        scores = WORKHORSE_PRIMARY_SCORES + WORKHORSE_SENSITIVITY_SCORES
        if dataset.upper() == "DS1":
            scores = tuple(s for s in scores if s != "pair_density_risk")
    overlap = set(FEATURE_SPACE_SI_SCORES) & set(scores)
    if overlap:
        raise RuntimeError(f"feature-space SI scores leaked into retention inventory: {sorted(overlap)}")
    return scores


def attach_tanimoto(scores: pd.DataFrame, train_lig, test_lig) -> dict:
    ref_fps, n_fail_ref = fingerprint_library(list(train_lig))
    bundle = tanimoto_distance_bundle(
        np.asarray(test_lig, dtype=object),
        ref_fps,
        radius=FINGERPRINT_RADIUS,
        n_bits=FINGERPRINT_N_BITS,
        ks=TANIMOTO_K,
    )
    for key in ("tanimoto_nn_distance", "tanimoto_median_distance", "tanimoto_knn_mean_5", "tanimoto_knn_mean_10"):
        scores[key] = bundle[key]
    return {
        "tanimoto_reference_ligand_count": int(pd.Series(list(train_lig)).nunique()),
        "tanimoto_query_count": int(len(test_lig)),
        "tanimoto_ref_parse_failures": int(n_fail_ref),
        "tanimoto_query_parse_failures": int(bundle["n_fail"]),
        "tanimoto_unique_query": int(bundle["n_unique_query"]),
        "heldout_in_tanimoto_reference": bool(set(map(str, test_lig)) & set(map(str, train_lig))),
        "fingerprint_type": FINGERPRINT_TYPE,
        "fingerprint_radius": FINGERPRINT_RADIUS,
        "fingerprint_n_bits": FINGERPRINT_N_BITS,
    }


def attach_feature_mahalanobis(scores: pd.DataFrame, dataset: str, fold: str) -> dict:
    arrays, _man = load_frozen_fold_arrays(feature_dir(dataset, fold))
    X_train = np.concatenate([arrays["X_train"], arrays["X_val"]], axis=0)
    X_test = arrays["X_test"]
    if len(X_test) != len(scores):
        raise RuntimeError(f"{dataset} {fold}: feature rows {len(X_test)} != preds {len(scores)}")
    bundle = feature_distance_bundle(X_train, X_test, ks=FEATURE_KNN_K)
    fit = bundle["mahalanobis_fit"]
    for key, val in bundle.items():
        if key == "mahalanobis_fit" or not isinstance(val, np.ndarray):
            continue
        if val.ndim == 1 and len(val) == len(scores):
            scores[key] = val
    return {
        "feature_space": "joint_ligand_metal_outer_train_standardscaled",
        "covariance_estimator": fit["covariance_estimator"],
        "mahalanobis_center": "training_centroid",
        "feature_si_scores": list(FEATURE_SPACE_SI_SCORES),
        "effective_rank": fit["effective_rank"],
        "condition_number": fit["condition_number"],
        "stability_flag": fit["stability_flag"],
        "shrinkage": fit["shrinkage"],
        "reference_row_count": fit["reference_row_count"],
        "n_features": fit["n_features"],
        "fit_partition": "outer_train",
    }


def attach_chemprop_embeddings(scores: pd.DataFrame, dataset: str, fold: str, model_id: str) -> dict:
    feat_dir = chemprop_feature_dir(dataset, fold)
    ckpt = chemprop_checkpoint_dir(model_id, fold)
    cache = AD_RETENTION_DIR / "embeddings" / model_id / fold
    train_emb = extract_graph_embeddings(
        feat_dir / "train.csv",
        feat_dir / "train_features.csv",
        ckpt,
        cache / "train.npz",
        dataset=dataset,
        fold=fold,
        model_id=model_id,
        split="train",
        unique_smiles=True,
    )
    val_emb = extract_graph_embeddings(
        feat_dir / "val.csv",
        feat_dir / "val_features.csv",
        ckpt,
        cache / "val.npz",
        dataset=dataset,
        fold=fold,
        model_id=model_id,
        split="val",
        unique_smiles=True,
    )
    test_csv = feat_dir / "test.csv"
    test_smiles_all = pd.read_csv(test_csv)["smiles"].astype(str).to_numpy()
    test_emb = extract_graph_embeddings(
        test_csv,
        feat_dir / "test_features.csv",
        ckpt,
        cache / "test.npz",
        dataset=dataset,
        fold=fold,
        model_id=model_id,
        split="test",
        unique_smiles=True,
    )
    train_s = np.concatenate([train_emb["smiles"], val_emb["smiles"]])
    train_h = np.concatenate([train_emb["embedding"], val_emb["embedding"]], axis=0)
    # Unique training ligands: first occurrence after concat(train, val).
    keep = pd.Series(train_s.astype(str)).duplicated(keep="first").to_numpy() == False  # noqa: E712
    train_s = train_s[keep]
    train_h = train_h[keep]
    query_smiles = scores["chemprop_smiles"].astype(str).to_numpy() if "chemprop_smiles" in scores.columns else scores["ligand_smiles"].astype(str).to_numpy()
    if len(query_smiles) != len(test_smiles_all):
        # Predictions are table-aligned; Chemprop CSV is the same held-out rows.
        if len(test_smiles_all) == len(scores):
            query_smiles = test_smiles_all
            scores["chemprop_smiles"] = test_smiles_all
        else:
            raise RuntimeError(
                f"{model_id} {fold}: chemprop test smiles {len(test_smiles_all)} != preds {len(scores)}"
            )
    else:
        scores["chemprop_smiles"] = query_smiles
    H_query = map_embeddings(query_smiles, test_emb["smiles"], test_emb["embedding"])
    suite = embedding_distance_suite(train_h, H_query, ks=EMBEDDING_K)
    for key, val in suite.items():
        if key in {"mahalanobis_fit"} or not isinstance(val, np.ndarray):
            continue
        if val.ndim == 1 and len(val) == len(scores):
            scores[key] = val
    overlap = bool(set(map(str, query_smiles)) & set(map(str, train_s)))
    fit = suite.get("mahalanobis_fit") or {}
    return {
        "n_train_embeddings": int(suite.get("n_train_embeddings", 0)),
        "n_query_embeddings": int(suite.get("n_query_embeddings", 0)),
        "embedding_dim": int(suite.get("embedding_dim", 0)),
        "graph_only": True,
        "heldout_in_embedding_reference": overlap,
        "mahalanobis_default": "centroid",
        "covariance_estimator": COVARIANCE_ESTIMATOR,
        "shrinkage": fit.get("shrinkage"),
        "effective_rank": fit.get("effective_rank"),
        "train_cached": bool(train_emb.get("cached")),
        "test_cached": bool(test_emb.get("cached")),
        "checkpoint_path": rel(ckpt),
    }


def build_fold_scores(dataset: str, model_id: str, ad_family: str, fold: str, table: pd.DataFrame, folds: dict) -> dict:
    pred = pd.read_csv(pred_path(model_id, fold))
    _tr, _va, _te, _outer, train_df, test_table = _outer_frames(table, folds, fold)
    if len(pred) != len(test_table):
        raise RuntimeError(f"{model_id} {fold}: pred {len(pred)} != test {len(test_table)}")
    train_lig = train_df["ligand_smiles"].astype(str).tolist()
    test_lig = test_table["ligand_smiles"].astype(str).to_numpy()
    scores = pd.DataFrame(
        {
            "dataset": dataset,
            "model_id": model_id,
            "ad_family": ad_family,
            "outer_fold_id": fold,
            "fold_id": fold,
            "record_id": pred["record_id"].astype(str) if "record_id" in pred.columns else pred.index.astype(str),
            "evaluation_partition": pred["evaluation_partition"] if "evaluation_partition" in pred.columns else EVALUATION_PARTITION,
            "y_true": pred["y_true"],
            "y_pred": pred["y_pred"],
            "ligand_smiles": test_lig,
            "score_orientation": "higher_is_greater_risk",
        }
    )
    for col in ("sigma_total", "sigma_epistemic_within", "sigma_epistemic_between", "sigma_aleatoric"):
        if col in pred.columns:
            scores[col] = pred[col]
    if dataset.upper() == "DS1":
        fam = None
        if "family" in pred.columns:
            fam = pred["family"].astype(str).to_numpy()
        elif "family" in test_table.columns:
            fam = test_table["family"].astype(str).to_numpy()
        if fam is not None:
            scores["family"] = fam
    else:
        scores[PAIR_COLUMN] = test_table[PAIR_COLUMN].astype(str).to_numpy()
        scores["is_ree_pair"] = test_table["is_ree_pair"].to_numpy()
        n_bad = assert_species_pairs(scores[PAIR_COLUMN])
        if n_bad:
            raise RuntimeError(f"{fold}: {n_bad} non-species pair IDs")
        counts = pair_training_ligand_counts(train_df)
        n_pair = np.asarray([counts.get(str(p), 0) for p in scores[PAIR_COLUMN].astype(str)], dtype=int)
        scores["n_training_ligands_for_pair"] = n_pair
        scores["pair_density_risk"] = density_score_from_count(n_pair)
        med = medium_indicators(train_df, test_table)
        for c in med.columns:
            scores[c] = med[c].to_numpy()

    tan = attach_tanimoto(scores, train_lig, test_lig)
    cov = attach_feature_mahalanobis(scores, dataset, fold)
    chem = {}
    if ad_family != "workhorse":
        feat_dir = chemprop_feature_dir(dataset, fold)
        scores["chemprop_smiles"] = pd.read_csv(feat_dir / "test.csv")["smiles"].astype(str).to_numpy()
        chem = attach_chemprop_embeddings(scores, dataset, fold, model_id)
    leak = {
        "dataset": dataset,
        "model_id": model_id,
        "ad_family": ad_family,
        "outer_fold_id": fold,
        "tanimoto_heldout_in_reference": tan["heldout_in_tanimoto_reference"],
        "embedding_heldout_in_reference": chem.get("heldout_in_embedding_reference", False),
        "fit_partition": "outer_train",
        "evaluation_partition": EVALUATION_PARTITION,
        "chemprop_scores_on_non_chemprop": ad_family != "chemprop" and any(c in scores.columns for c in CHEMPROP_ONLY_SCORES),
        "status": "pass",
        "failure_reason": "",
    }
    if leak["tanimoto_heldout_in_reference"] or leak["embedding_heldout_in_reference"] or leak["chemprop_scores_on_non_chemprop"]:
        leak["status"] = "fix_required"
        reasons = []
        if leak["tanimoto_heldout_in_reference"]:
            reasons.append("heldout_ligand_in_tanimoto_reference")
        if leak["embedding_heldout_in_reference"]:
            reasons.append("heldout_smiles_in_embedding_reference")
        if leak["chemprop_scores_on_non_chemprop"]:
            reasons.append("chemprop_score_on_non_chemprop_model")
        leak["failure_reason"] = ";".join(reasons)
    audit = {"tanimoto": tan, "covariance": cov, "chemprop": chem, "leakage": leak}
    return {"scores": scores, "audit": audit}


def enrichment_from_frames(score_df: pd.DataFrame, random_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["dataset", "model_id", "family", "outer_fold_id", "scope", "score_name", "metric_name"]
    rand = random_df[
        (random_df.score_name == "random_retention")
        & (pd.to_numeric(random_df.retention_fraction, errors="coerce") == HIGH_CONFIDENCE_RETENTION)
    ]
    scored = score_df[
        (pd.to_numeric(score_df.retention_fraction, errors="coerce") == HIGH_CONFIDENCE_RETENTION)
        & (score_df.metric_name.isin(["rmse", "mop_rmse"]))
    ]
    for rec in scored.to_dict(orient="records"):
        rhit = rand[
            (rand.dataset == rec["dataset"])
            & (rand.model_id == rec["model_id"])
            & (rand.outer_fold_id == rec["outer_fold_id"])
            & (rand.scope == rec["scope"])
            & (rand.metric_name == rec["metric_name"])
        ]
        rmse_r = float(rhit.iloc[0]["value"]) if len(rhit) else np.nan
        rmse_s = float(rec["value"]) if rec["value"] == rec["value"] else np.nan
        if rec["metric_name"] in {"rmse", "mop_rmse"}:
            value = rmse_r / rmse_s if np.isfinite(rmse_r) and np.isfinite(rmse_s) and rmse_s != 0 else np.nan
            kind = "rmse_enrichment_ratio"
            out_metric = f"enrichment_10_{rec['metric_name']}"
            enrichment_10 = value
        else:
            continue
        rows.append(
            {
                **{k: rec.get(k) for k in keys},
                "metric_name": out_metric,
                "source_metric": rec["metric_name"],
                "retention_fraction": HIGH_CONFIDENCE_RETENTION,
                "score_value": rmse_s,
                "random_value": rmse_r,
                "value": value,
                "enrichment_10": enrichment_10,
                "comparison_kind": kind,
                "n_rows_used": rec.get("n_rows_used"),
                "n_pairs_used": rec.get("n_pairs_used"),
                "metric_status": rec.get("metric_status"),
                "evaluation_partition": EVALUATION_PARTITION,
            }
        )
    return pd.DataFrame(rows)


def table_50pct(summary: pd.DataFrame, fold_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = summary[
        (pd.to_numeric(summary.retention_fraction, errors="coerce") == PRIMARY_RETENTION)
        & (summary.baseline.fillna("score") == "score")
    ]
    full = summary[
        (pd.to_numeric(summary.retention_fraction, errors="coerce") == 1.0)
        & (summary.baseline.fillna("score") == "score")
    ]
    for rec in sub.itertuples(index=False):
        if rec.metric_name not in {"rmse", "mop_rmse"}:
            continue
        fhit = full[
            (full.dataset == rec.dataset)
            & (full.model_id == rec.model_id)
            & (full.scope == rec.scope)
            & (full.score_name == rec.score_name)
            & (full.metric_name == rec.metric_name)
        ]
        full_mean = float(fhit.iloc[0].fold_mean) if len(fhit) else np.nan
        rel_change = np.nan
        if np.isfinite(full_mean) and full_mean != 0 and np.isfinite(rec.fold_mean):
            if rec.metric_name in {"rmse", "mop_rmse"}:
                rel_change = (rec.fold_mean - full_mean) / abs(full_mean)
            else:
                rel_change = (rec.fold_mean - full_mean) / abs(full_mean)
        def _support(metric_name: str) -> dict:
            hit = fold_df[
                (fold_df.dataset == rec.dataset)
                & (fold_df.model_id == rec.model_id)
                & (fold_df.scope == rec.scope)
                & (fold_df.score_name == rec.score_name)
                & (pd.to_numeric(fold_df.retention_fraction, errors="coerce") == PRIMARY_RETENTION)
                & (fold_df.metric_name == metric_name)
            ]
            if not len(hit):
                return {"fold_mean": np.nan, "fold_sd": np.nan, "n_valid_folds": 0}
            return fold_mean_sd(pd.to_numeric(hit["value"], errors="coerce").to_numpy())

        n_pairs_mop = _support("n_pairs_used")
        n_pairs_any = _support("n_pairs_total")
        n_rows = _support("n_rows_retained")
        is_ds1 = str(rec.dataset).upper() == "DS1"
        rows.append(
            {
                "dataset": rec.dataset,
                "model_id": rec.model_id,
                "family": rec.family,
                "family_display": family_display_name(rec.family, rec.model_id),
                "score_role": rec.score_role,
                "scope": rec.scope,
                "score_name": rec.score_name,
                "retained_fraction": PRIMARY_RETENTION,
                "metric_name": rec.metric_name,
                "retained_mean": rec.fold_mean,
                "retained_sd": rec.fold_sd,
                "full_data_mean": full_mean,
                "relative_change": rel_change,
                "n_valid_folds": rec.n_valid_folds,
                "n_rows_retained_mean": n_rows["fold_mean"],
                "n_rows_retained_sd": n_rows["fold_sd"],
                "pairs_contributing_to_MoP_mean": np.nan if is_ds1 else n_pairs_mop["fold_mean"],
                "pairs_contributing_to_MoP_sd": np.nan if is_ds1 else n_pairs_mop["fold_sd"],
                "pairs_with_any_retained_row_mean": np.nan if is_ds1 else n_pairs_any["fold_mean"],
                "pairs_with_any_retained_row_sd": np.nan if is_ds1 else n_pairs_any["fold_sd"],
                "pairs_retained_mean": np.nan if is_ds1 else n_pairs_mop["fold_mean"],
                "pairs_retained_sd": np.nan if is_ds1 else n_pairs_mop["fold_sd"],
                "primary_comparison": rec.score_name in PRIMARY_SCORE_SET,
            }
        )
    return pd.DataFrame(rows)


def aurc_table(summary: pd.DataFrame) -> pd.DataFrame:
    sub = summary[summary.metric_name.astype(str).str.startswith("retention_curve_auc_")].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub[sub.baseline.fillna("score") == "score"].copy()
    return sub


SCORE_DEFINITIONS = [
    {"score_name": "sigma_total", "method_name": "sigma_total", "display_name": "sigma_total", "definition": "predictive total uncertainty from the fold-local model", "fit_partition": "none_model_output", "orientation": "higher_is_greater_risk", "availability": "all_uq_models"},
    {"score_name": "tanimoto_nn_distance", "method_name": "tanimoto_nn_distance", "display_name": "Tanimoto nearest-neighbor distance", "definition": "1 - max Tanimoto similarity to outer-training ligands (Morgan r=2, 2048 bits)", "fit_partition": "outer_train", "orientation": "higher_is_greater_risk", "availability": "all_models"},
    {"score_name": "mahalanobis_feature_centroid", "method_name": "rdkit_feature_mahalanobis_centroid", "display_name": "RDKit feature-centroid Mahalanobis", "definition": "Ledoit-Wolf Mahalanobis distance to the training centroid in fold-local StandardScaled RDKit+metal features. The obsolete label mahalanobis_joint refers to this same centroid score.", "fit_partition": "outer_train", "orientation": "higher_is_greater_risk", "availability": "rdkit_feature_space"},
    {"score_name": "mahalanobis_feature_nn", "method_name": "rdkit_feature_mahalanobis_nn", "display_name": "RDKit feature Mahalanobis-NN", "definition": "nearest-row Ledoit-Wolf Mahalanobis distance in the same StandardScaled RDKit+metal+medium space. SI Spearman only; not a retention-figure score.", "fit_partition": "outer_train", "orientation": "higher_is_greater_risk", "availability": "rdkit_feature_space_si"},
    {"score_name": "mahalanobis_feature_knn_mean_5", "method_name": "rdkit_feature_mahalanobis_knn_mean_k5", "display_name": "RDKit feature Mahalanobis kNN mean, k=5", "definition": "mean Ledoit-Wolf Mahalanobis distance to the 5 nearest training rows in the same feature space. SI Spearman only.", "fit_partition": "outer_train", "orientation": "higher_is_greater_risk", "availability": "rdkit_feature_space_si"},
    {"score_name": "mahalanobis_feature_knn_mean_10", "method_name": "rdkit_feature_mahalanobis_knn_mean_k10", "display_name": "RDKit feature Mahalanobis kNN mean, k=10", "definition": "mean Ledoit-Wolf Mahalanobis distance to the 10 nearest training rows in the same feature space. SI Spearman only.", "fit_partition": "outer_train", "orientation": "higher_is_greater_risk", "availability": "rdkit_feature_space_si"},
    {"score_name": "euclidean_feature_nn", "method_name": "rdkit_feature_euclidean_nn", "display_name": "RDKit feature Euclidean-NN", "definition": "nearest-row Euclidean distance in the same StandardScaled RDKit+metal+medium space. SI Spearman only.", "fit_partition": "outer_train", "orientation": "higher_is_greater_risk", "availability": "rdkit_feature_space_si"},
    {"score_name": "euclidean_feature_median", "method_name": "rdkit_feature_euclidean_median", "display_name": "RDKit feature median Euclidean distance", "definition": "median Euclidean distance to all outer-training rows in the same feature space. SI Spearman only.", "fit_partition": "outer_train", "orientation": "higher_is_greater_risk", "availability": "rdkit_feature_space_si"},
    {"score_name": "chemprop_std_euclidean_nn", "method_name": "chemprop_embedding_euclidean_nn_standardized", "display_name": "Chemprop Euclidean-NN", "definition": "nearest-neighbor Euclidean distance in train-standardized Chemprop MPN graph embeddings", "fit_partition": "outer_train", "orientation": "higher_is_greater_risk", "availability": "chemprop_only"},
    {"score_name": "chemprop_std_euclidean_knn_5", "method_name": "chemprop_embedding_euclidean_knn_mean_standardized_k5", "display_name": "Chemprop Euclidean-kNN, k=5", "definition": "mean Euclidean distance to 5 nearest training-fold Chemprop embeddings after training-fold standardization", "fit_partition": "outer_train", "orientation": "higher_is_greater_risk", "availability": "chemprop_only"},
    {"score_name": "chemprop_std_euclidean_knn_10", "method_name": "chemprop_embedding_euclidean_knn_mean_standardized_k10", "display_name": "Chemprop Euclidean-kNN, k=10", "definition": "mean Euclidean distance to 10 nearest training-fold Chemprop embeddings after training-fold standardization", "fit_partition": "outer_train", "orientation": "higher_is_greater_risk", "availability": "chemprop_only"},
    {"score_name": "chemprop_mahalanobis_centroid", "method_name": "chemprop_embedding_mahalanobis_centroid", "display_name": "Chemprop embedding-centroid Mahalanobis", "definition": "Ledoit-Wolf Mahalanobis distance to the training-embedding centroid after train-fold standardization", "fit_partition": "outer_train", "orientation": "higher_is_greater_risk", "availability": "chemprop_only"},
    {"score_name": "chemprop_mahalanobis_nn", "method_name": "chemprop_embedding_mahalanobis_nn", "display_name": "Chemprop embedding Mahalanobis-NN", "definition": "nearest-embedding Ledoit-Wolf Mahalanobis distance (sensitivity; not the default)", "fit_partition": "outer_train", "orientation": "higher_is_greater_risk", "availability": "chemprop_only"},
    {"score_name": "chemprop_cosine_nn", "method_name": "chemprop_embedding_cosine_nn", "display_name": "Chemprop cosine-NN", "definition": "1 - max cosine similarity to training-fold Chemprop embeddings", "fit_partition": "outer_train", "orientation": "higher_is_greater_risk", "availability": "chemprop_only"},
    {"score_name": "pair_density_risk", "method_name": "pair_density_risk", "display_name": "pair-density risk", "definition": "inverse training-ligand support for the species-level metal pair (higher = greater risk)", "fit_partition": "outer_train", "orientation": "higher_is_greater_risk", "availability": "ds2_workhorse"},
    {"score_name": "random_retention", "method_name": "random_retention", "display_name": "random retention", "definition": "fold-local repeated random subsets at the same retention fractions", "fit_partition": "none", "orientation": "not_a_score", "availability": "baseline"},
]


def write_derived_ad_tables(risk_df: pd.DataFrame, random_df: pd.DataFrame, risk_sum: pd.DataFrame) -> None:
    enrich = enrichment_from_frames(risk_df, random_df) if len(random_df) else pd.DataFrame()
    _write_csv(AD_RETENTION_DIR / "enrichment_fold.csv", enrich)
    if len(enrich):
        enrich_sum = summarize_fold_metrics(
            enrich,
            ["dataset", "model_id", "family", "scope", "score_name", "metric_name", "comparison_kind", "source_metric"],
        )
        _write_csv(AD_RETENTION_DIR / "enrichment_summary.csv", enrich_sum)
    _write_csv(AD_RETENTION_DIR / "retention_50pct.csv", table_50pct(risk_sum, risk_df))
    _write_csv(AD_RETENTION_DIR / "retention_aurc.csv", aurc_table(risk_sum))
    _write_csv(AD_RETENTION_DIR / "score_definitions.csv", SCORE_DEFINITIONS)


def _ad_snapshot(spec_src: Path) -> dict:
    return {
        "experiment_id": "ad_retention_study",
        "spec_path": rel(spec_src),
        "retention_grid": list(RETENTION_FRACTIONS),
        "primary_retention": PRIMARY_RETENTION,
        "random_n_repeats": RANDOM_N_REPEATS,
        "random_seed": RANDOM_SEED,
        "chemprop_mahalanobis_default": "centroid",
        "pooled_oof_forbidden": True,
        "formal_risk_control": False,
        "ds2_primary_error_ordering": "all_pair_mop_rmse",
        "ds1_primary_error_ordering": "row_rmse",
        "enrichment_10": "RMSE_random_10 / RMSE_score_10",
        "ranking_k": "floor(r n)",
        "tie_break": "score_asc_then_record_id",
        "mahalanobis_feature_centroid_alias": "mahalanobis_joint",
        "interpretation": "fixed_retention_rmse_is_error_ordering_diagnostic_not_screening_utility",
        "jobs": JOBS,
    }


def backfill_feature_space_scores(
    dataset: str = "DS2",
    model_ids: tuple[str, ...] | None = None,
) -> dict:
    """Add RDKit feature-space distances onto existing fold_scores. No retraining."""
    model_ids = tuple(model_ids or (DS2_WORKHORSE, DS2_CHEMPROP))
    reports = []
    for fold in FOLDS:
        arrays, _man = load_frozen_fold_arrays(feature_dir(dataset, fold))
        x_train = np.concatenate([arrays["X_train"], arrays["X_val"]], axis=0)
        x_test = np.asarray(arrays["X_test"], dtype=float)
        print(f"feature-space {dataset} {fold} train={len(x_train)} test={len(x_test)}", flush=True)
        bundle = feature_distance_bundle(x_train, x_test, ks=FEATURE_KNN_K)
        cols = [
            key
            for key, val in bundle.items()
            if key != "mahalanobis_fit" and isinstance(val, np.ndarray) and val.ndim == 1
        ]
        for model_id in model_ids:
            path = AD_RETENTION_DIR / "fold_scores" / model_id / f"{fold}.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            scores = pd.read_csv(path)
            if len(scores) != len(x_test):
                raise RuntimeError(f"{model_id} {fold}: scores {len(scores)} != features {len(x_test)}")
            if "mahalanobis_feature_centroid" in scores.columns:
                old = pd.to_numeric(scores["mahalanobis_feature_centroid"], errors="coerce").to_numpy(dtype=float)
                new = np.asarray(bundle["mahalanobis_feature_centroid"], dtype=float)
                ok = np.isfinite(old) & np.isfinite(new)
                if ok.any():
                    delta = float(np.max(np.abs(old[ok] - new[ok])))
                    if delta > 1e-6:
                        raise RuntimeError(f"{model_id} {fold}: centroid mismatch max|d|={delta}")
            for key in cols:
                scores[key] = bundle[key]
            _write_csv(path, scores)
            reports.append({"model_id": model_id, "fold": fold, "n": int(len(scores)), "columns": cols})
            print(f"  wrote {model_id} {fold} +{len(cols)} feature-space columns", flush=True)
    return {"dataset": dataset, "model_ids": list(model_ids), "folds": reports}


def analyze_existing_fold_scores(jobs: list[dict], skip_random: bool = False) -> dict:
    risk_rows = []
    random_rows = []
    util_rows = []
    for job in jobs:
        dataset, model_id, ad_family = job["dataset"], job["model_id"], job["ad_family"]
        print(f"=== scores {dataset} {model_id} ===", flush=True)
        wanted = scores_for_family(ad_family, dataset)
        for fold in FOLDS:
            path = AD_RETENTION_DIR / "fold_scores" / model_id / f"{fold}.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            scores = pd.read_csv(path)
            print(f"  {fold} n={len(scores)}", flush=True)
            available = [s for s in wanted if s in scores.columns]
            primary_scope = "metal_complex" if dataset == "DS1" else "ree_pairs"
            for score in available:
                risk_rows.extend(
                    risk_coverage_fold(
                        scores,
                        score,
                        fold,
                        experiment_id="ad_retention",
                        dataset=dataset,
                        model_id=model_id,
                        family=ad_family,
                        score_role=score_role(ad_family, score),
                    )
                )
                if score in PRIMARY_SCORE_SET:
                    util_rows.extend(
                        utility_overlap_fold(
                            scores,
                            score,
                            fold,
                            dataset,
                            primary_scope,
                            model_id=model_id,
                            family=ad_family,
                            score_role=score_role(ad_family, score),
                        )
                    )
            if not skip_random:
                random_rows.extend(
                    random_retention_fold(
                        scores,
                        fold,
                        dataset=dataset,
                        model_id=model_id,
                        family=ad_family,
                    )
                )
    return {"risk": risk_rows, "random": random_rows, "utility": util_rows}


def upsert_retention_outputs(risk_rows, random_rows, model_ids, util_rows=None) -> None:
    """Replace retention rows for selected models; keep every other model intact."""
    wanted = {str(m) for m in model_ids}
    risk_new = pd.DataFrame(risk_rows)
    risk_path = AD_RETENTION_DIR / "fold_retention.csv"
    if risk_path.exists() and wanted:
        old = pd.read_csv(risk_path)
        old = old[~old.model_id.astype(str).isin(wanted)].copy()
        risk_df = pd.concat([old, risk_new], ignore_index=True)
    else:
        risk_df = risk_new
    rand_path = AD_RETENTION_DIR / "random_fold_retention.csv"
    if random_rows:
        rand_new = pd.DataFrame(random_rows)
        if rand_path.exists() and wanted:
            old_r = pd.read_csv(rand_path)
            old_r = old_r[~old_r.model_id.astype(str).isin(wanted)].copy()
            random_df = pd.concat([old_r, rand_new], ignore_index=True)
        else:
            random_df = rand_new
    elif rand_path.exists():
        random_df = pd.read_csv(rand_path)
    else:
        random_df = pd.DataFrame()
    _write_csv(risk_path, risk_df)
    if len(random_df):
        _write_csv(rand_path, random_df)
    risk_sum = summarize_fold_metrics(
        risk_df,
        ["dataset", "model_id", "family", "score_role", "experiment_id", "scope", "score_name", "retention_fraction", "metric_name", "baseline"],
    )
    rand_sum = summarize_fold_metrics(
        random_df,
        ["dataset", "model_id", "family", "score_role", "experiment_id", "scope", "score_name", "retention_fraction", "metric_name", "baseline"],
    ) if len(random_df) else pd.DataFrame()
    _write_csv(AD_RETENTION_DIR / "retention_summary.csv", risk_sum)
    _write_csv(AD_RETENTION_DIR / "random_retention_summary.csv", rand_sum)
    write_derived_ad_tables(risk_df, random_df, risk_sum)
    if util_rows is not None:
        _write_csv(AD_RETENTION_DIR / "utility_fold.csv", util_rows)


def write_retention_outputs(risk_rows, random_rows, util_rows=None) -> None:
    risk_df = pd.DataFrame(risk_rows)
    random_df = pd.DataFrame(random_rows)
    _write_csv(AD_RETENTION_DIR / "fold_retention.csv", risk_df)
    _write_csv(AD_RETENTION_DIR / "random_fold_retention.csv", random_df)
    risk_sum = summarize_fold_metrics(
        risk_df,
        ["dataset", "model_id", "family", "score_role", "experiment_id", "scope", "score_name", "retention_fraction", "metric_name", "baseline"],
    )
    rand_sum = summarize_fold_metrics(
        random_df,
        ["dataset", "model_id", "family", "score_role", "experiment_id", "scope", "score_name", "retention_fraction", "metric_name", "baseline"],
    ) if len(random_df) else pd.DataFrame()
    _write_csv(AD_RETENTION_DIR / "retention_summary.csv", risk_sum)
    _write_csv(AD_RETENTION_DIR / "random_retention_summary.csv", rand_sum)
    write_derived_ad_tables(risk_df, random_df, risk_sum)
    if util_rows is not None:
        _write_csv(AD_RETENTION_DIR / "utility_fold.csv", util_rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Fold-local AD retention study")
    p.add_argument("--skip-chemprop", action="store_true")
    p.add_argument("--skip-random", action="store_true")
    p.add_argument(
        "--rebuild-derived",
        action="store_true",
        help="Rebuild enrichment, 50% table, and AURC from existing fold CSVs without rescoring.",
    )
    p.add_argument(
        "--from-existing-scores",
        action="store_true",
        help="Recompute RMSE retention from fold_scores without Tanimoto/Mahalanobis/embedding rescoring.",
    )
    p.add_argument(
        "--backfill-feature-space",
        action="store_true",
        help="Attach RDKit feature-space SI distances onto existing DS2 fold_scores.",
    )
    p.add_argument("--model-id", action="append", default=None, help="Restrict jobs to these model_id values.")
    args = p.parse_args(argv)
    spec_src = REPO_ROOT / "configs" / "experiments" / "ad_retention_spec.yaml"
    AD_RETENTION_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = _ad_snapshot(spec_src)
    if args.backfill_feature_space:
        report = backfill_feature_space_scores(model_ids=tuple(args.model_id) if args.model_id else None)
        _write_json(AD_RETENTION_DIR / "ad_retention_spec_snapshot.json", snapshot)
        print(json.dumps({"backfill_feature_space": report["model_ids"]}, indent=2), flush=True)
        return 0
    if args.rebuild_derived:
        risk_df = pd.read_csv(AD_RETENTION_DIR / "fold_retention.csv")
        random_df = pd.read_csv(AD_RETENTION_DIR / "random_fold_retention.csv")
        risk_sum = pd.read_csv(AD_RETENTION_DIR / "retention_summary.csv")
        write_derived_ad_tables(risk_df, random_df, risk_sum)
        _write_json(AD_RETENTION_DIR / "ad_retention_spec_snapshot.json", snapshot)
        print(f"Rebuilt derived AD tables in {AD_RETENTION_DIR}", flush=True)
        return 0

    _write_json(AD_RETENTION_DIR / "ad_retention_spec_snapshot.json", snapshot)
    jobs = [j for j in JOBS if not (args.skip_chemprop and j["ad_family"] == "chemprop")]
    if args.model_id:
        wanted = set(args.model_id)
        jobs = [j for j in jobs if j["model_id"] in wanted]

    if args.from_existing_scores:
        analyzed = analyze_existing_fold_scores(jobs, skip_random=args.skip_random)
        selected = [j["model_id"] for j in jobs]
        if selected and set(selected) != {j["model_id"] for j in JOBS}:
            upsert_retention_outputs(analyzed["risk"], analyzed["random"], selected, analyzed["utility"])
        else:
            write_retention_outputs(analyzed["risk"], analyzed["random"], analyzed["utility"])
        print(f"Recomputed RMSE retention from existing fold_scores in {AD_RETENTION_DIR}", flush=True)
        return 0

    tables = {ds: load_table(ds) for ds in ("DS1", "DS2")}
    folds_map = {ds: load_folds(ds) for ds in ("DS1", "DS2")}

    score_frames = []
    risk_rows = []
    random_rows = []
    util_rows = []
    leak_rows = []
    cov_rows = []
    chem_rows = []

    for job in jobs:
        dataset, model_id, ad_family = job["dataset"], job["model_id"], job["ad_family"]
        print(f"=== {dataset} {model_id} ===", flush=True)
        wanted = scores_for_family(ad_family, dataset)
        for fold in FOLDS:
            print(f"  {fold}", flush=True)
            built = build_fold_scores(dataset, model_id, ad_family, fold, tables[dataset], folds_map[dataset])
            scores = built["scores"]
            _write_csv(AD_RETENTION_DIR / "fold_scores" / model_id / f"{fold}.csv", scores)
            score_frames.append(scores)
            leak_rows.append(built["audit"]["leakage"])
            if built["audit"]["covariance"]:
                cov_rows.append({"dataset": dataset, "model_id": model_id, "outer_fold_id": fold, **built["audit"]["covariance"]})
            if built["audit"]["chemprop"]:
                chem_rows.append({"dataset": dataset, "model_id": model_id, "outer_fold_id": fold, **built["audit"]["chemprop"]})
            available = [s for s in wanted if s in scores.columns]
            primary_scope = "metal_complex" if dataset == "DS1" else "ree_pairs"
            for score in available:
                risk_rows.extend(
                    risk_coverage_fold(
                        scores,
                        score,
                        fold,
                        experiment_id="ad_retention",
                        dataset=dataset,
                        model_id=model_id,
                        family=ad_family,
                        score_role=score_role(ad_family, score),
                    )
                )
                if score in PRIMARY_SCORE_SET:
                    util_rows.extend(
                        utility_overlap_fold(
                            scores,
                            score,
                            fold,
                            dataset,
                            primary_scope,
                            model_id=model_id,
                            family=ad_family,
                            score_role=score_role(ad_family, score),
                        )
                    )
            if not args.skip_random:
                random_rows.extend(
                    random_retention_fold(
                        scores,
                        fold,
                        dataset=dataset,
                        model_id=model_id,
                        family=ad_family,
                    )
                )

    write_retention_outputs(risk_rows, random_rows, util_rows)
    _write_csv(AD_RETENTION_DIR / "leakage_audit.csv", leak_rows)
    _write_csv(AD_RETENTION_DIR / "covariance_diagnostics.csv", cov_rows)
    _write_csv(AD_RETENTION_DIR / "chemprop_embedding_audit.csv", chem_rows)

    n_leak = sum(1 for r in leak_rows if r.get("status") == "fix_required")
    print(f"Wrote AD retention artifacts to {AD_RETENTION_DIR} (leakage_fail={n_leak})", flush=True)
    return 0 if n_leak == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
