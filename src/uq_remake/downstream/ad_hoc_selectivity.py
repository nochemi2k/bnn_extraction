"""Aligned ad hoc DS1-difference versus direct DS2 selectivity.

DS2 pairs are evaluated on the existing ligand-disjoint DS2 outer folds.
Ad hoc predictions are formed by subtracting two held-out DS1 absolute-binding
predictions for the same ligand. The DS1 checkpoint is the fold that held that
ligand out. No held-out targets are used to form the difference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from uq_remake.downstream.protocol import DS2_WORKHORSE
from uq_remake.metrics.cv_protocol import (
    EVALUATION_PARTITION,
    MIN_PAIR_N,
    PAIR_COLUMN,
    fold_mean_sd,
)
from uq_remake.metrics.fold_metrics import (
    apply_scope,
    filter_finite_rows,
    interval_metrics_fold,
    mop_block,
    mop_interval_block,
    row_point_metrics,
    sign_metrics,
)

PROPAGATION_METHOD = "independent_quadrature_sigma_total"
SELECTIVITY_DIRECT = "direct_ds2_selectivity"
SELECTIVITY_AD_HOC = "ad_hoc_ds1_difference"
SCOPES = ("ree_pairs", "all_pairs")


class AdHocAlignmentError(ValueError):
    """Raised when DS1/DS2 rows cannot be aligned for ad hoc differencing."""


def ad_hoc_difference(pred_x, pred_y) -> np.ndarray:
    """ΔlogK_ad_hoc = logK_hat(X) − logK_hat(Y). Never uses abs()."""
    return np.asarray(pred_x, dtype=float) - np.asarray(pred_y, dtype=float)


def reverse_orientation(delta) -> np.ndarray:
    return -np.asarray(delta, dtype=float)


def propagate_difference_sigma(sigma_x, sigma_y) -> np.ndarray:
    """σ_Δ = sqrt(σ_X^2 + σ_Y^2). Independent-error diagnostic, not a covariance model."""
    sx = np.asarray(sigma_x, dtype=float)
    sy = np.asarray(sigma_y, dtype=float)
    out = np.sqrt(np.square(sx) + np.square(sy))
    if np.isfinite(out).any() and float(np.nanmin(out[np.isfinite(out)])) < 0:
        raise AdHocAlignmentError("propagated difference uncertainty is negative")
    return out


def assert_no_ligand_overlap(train_ligands, heldout_ligands) -> None:
    overlap = set(map(str, train_ligands)) & set(map(str, heldout_ligands))
    if overlap:
        raise AdHocAlignmentError(f"ligand overlap between outer train and held-out rows: {sorted(overlap)[:5]}")


def index_ds1_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    need = {"record_id", "y_pred"}
    missing = need - set(pred_df.columns)
    if missing:
        raise AdHocAlignmentError(f"DS1 predictions missing columns {sorted(missing)}")
    work = pred_df.copy()
    work["record_id"] = work["record_id"].astype(str)
    work["y_pred"] = pd.to_numeric(work["y_pred"], errors="coerce")
    if "sigma_total" in work.columns:
        work["sigma_total"] = pd.to_numeric(work["sigma_total"], errors="coerce")
    if "ligand_smiles" in work.columns:
        work["ligand_smiles"] = work["ligand_smiles"].astype(str)
    if work["record_id"].duplicated().any():
        raise AdHocAlignmentError("DS1 predictions contain duplicate record_id values")
    return work.set_index("record_id", drop=False)


def assemble_ad_hoc_pairs(
    ds2_pairs: pd.DataFrame,
    ds1_preds: pd.DataFrame,
    source_model_id: str,
    outer_fold: str,
) -> pd.DataFrame:
    """Join DS2 held-out pairs to two DS1 endpoint predictions and subtract."""
    if "record_id_x" not in ds2_pairs.columns or "record_id_y" not in ds2_pairs.columns:
        raise AdHocAlignmentError("DS2 pairs must include record_id_x and record_id_y")
    indexed = index_ds1_predictions(ds1_preds)
    left = ds2_pairs.copy()
    left["record_id_x"] = left["record_id_x"].astype(str)
    left["record_id_y"] = left["record_id_y"].astype(str)
    miss_x = ~left["record_id_x"].isin(indexed.index)
    miss_y = ~left["record_id_y"].isin(indexed.index)
    if bool(miss_x.any() or miss_y.any()):
        n_miss = int((miss_x | miss_y).sum())
        raise AdHocAlignmentError(f"{n_miss} DS2 pairs are missing a DS1 endpoint prediction")
    px = indexed.loc[left["record_id_x"]]
    py = indexed.loc[left["record_id_y"]]
    y_pred = ad_hoc_difference(px["y_pred"].to_numpy(), py["y_pred"].to_numpy())
    if "selectivity_logK" in left.columns:
        y_true = pd.to_numeric(left["selectivity_logK"], errors="coerce").to_numpy(dtype=float)
    elif "y_true" in left.columns:
        y_true = pd.to_numeric(left["y_true"], errors="coerce").to_numpy(dtype=float)
    else:
        raise AdHocAlignmentError("DS2 pairs must include selectivity_logK or y_true")
    if "sigma_total" in px.columns and "sigma_total" in py.columns:
        sigma = propagate_difference_sigma(px["sigma_total"].to_numpy(), py["sigma_total"].to_numpy())
        uq_status = "propagated_independent_quadrature"
    else:
        sigma = np.full(len(left), np.nan)
        uq_status = "unavailable"
    if np.isfinite(sigma).any() and float(np.nanmin(sigma[np.isfinite(sigma)])) < 0:
        raise AdHocAlignmentError("propagated difference uncertainty is negative")
    out = pd.DataFrame(
        {
            "pair_id": left["pair_id"].astype(str) if "pair_id" in left.columns else left.index.astype(str),
            "record_id": left["pair_id"].astype(str) if "pair_id" in left.columns else left.index.astype(str),
            "record_id_x": left["record_id_x"].to_numpy(),
            "record_id_y": left["record_id_y"].to_numpy(),
            "ligand_smiles": left["ligand_smiles"].astype(str).to_numpy() if "ligand_smiles" in left.columns else "",
            PAIR_COLUMN: left[PAIR_COLUMN].astype(str).to_numpy() if PAIR_COLUMN in left.columns else "",
            "is_ree_pair": left["is_ree_pair"].to_numpy() if "is_ree_pair" in left.columns else False,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_pred_x": px["y_pred"].to_numpy(dtype=float),
            "y_pred_y": py["y_pred"].to_numpy(dtype=float),
            "sigma_total": sigma,
            "sigma_x": px["sigma_total"].to_numpy(dtype=float) if "sigma_total" in px.columns else np.nan,
            "sigma_y": py["sigma_total"].to_numpy(dtype=float) if "sigma_total" in py.columns else np.nan,
            "fold_id": outer_fold,
            "outer_fold_id": outer_fold,
            "evaluation_partition": EVALUATION_PARTITION,
            "source_model_id": source_model_id,
            "selectivity_method": SELECTIVITY_AD_HOC,
            "propagation_method": PROPAGATION_METHOD if uq_status != "unavailable" else "none",
            "uncertainty_status": uq_status,
        }
    )
    if (out[PAIR_COLUMN].astype(str).str.len() == 0).any():
        raise AdHocAlignmentError("assembled ad hoc pairs are missing metal_species_pair labels")
    return out


def evaluate_ad_hoc_fold(pair_df: pd.DataFrame, source_model_id: str, scope: str) -> dict:
    scoped = apply_scope(pair_df, "DS2", scope)
    used, diag = filter_finite_rows(scoped)
    row_m = row_point_metrics(used["y_true"], used["y_pred"]) if len(used) else {"rmse": np.nan, "r2": np.nan}
    signs = sign_metrics(used["y_true"], used["y_pred"]) if len(used) else sign_metrics([], [])
    mop = mop_block(used) if len(used) else {
        "mop_rmse": np.nan,
        "mop_r2": np.nan,
        "n_pairs_total": 0,
        "n_pairs_retained": 0,
        "n_rows_in_retained_pairs": 0,
        "min_pair_size": MIN_PAIR_N,
    }
    if mop.get("n_pairs_retained", 0) and mop["n_pairs_retained"] > 0:
        # pair-support rule is already applied inside mop_block
        pass
    has_sigma = "sigma_total" in used.columns and bool(np.isfinite(used["sigma_total"]).any())
    mop_picp = mop_ace = np.nan
    if has_sigma:
        # Force normal-moment intervals from propagated sigma. Do not reuse NIG/Student-t.
        used_iv = used.drop(columns=["alpha", "beta", "nu", "gamma"], errors="ignore")
        im = interval_metrics_fold(used_iv, "ADHOC_PROPAGATED")
        picp = im["marginal_picp_90"]
        ace = im["marginal_ace"]
        interval_method = "normal_moment_propagated_difference"
        n_valid_intervals = im["n_valid_intervals"]
        if len(used_iv):
            mop_iv = mop_interval_block(used_iv, "ADHOC_PROPAGATED")
            mop_picp = mop_iv.get("mop_picp_90", np.nan)
            mop_ace = mop_iv.get("mop_ace", np.nan)
    else:
        picp = ace = np.nan
        interval_method = "unavailable"
        n_valid_intervals = 0
    uq_status = str(used["uncertainty_status"].iloc[0]) if len(used) and "uncertainty_status" in used.columns else "unavailable"
    return {
        "dataset": "DS2",
        "outer_fold": str(used["fold_id"].iloc[0]) if len(used) else str(pair_df["fold_id"].iloc[0]),
        "source_model": _display_model(source_model_id),
        "source_model_id": source_model_id,
        "selectivity_method": SELECTIVITY_AD_HOC,
        "scope": scope,
        "n_rows": int(diag["n_rows_used"]),
        "n_pairs": int(mop.get("n_pairs_retained") or 0),
        "n_pairs_total": int(mop.get("n_pairs_total") or 0),
        "mop_rmse": mop.get("mop_rmse"),
        "mop_r2": mop.get("mop_r2"),
        "mop_picp_90": mop_picp,
        "mop_ace": mop_ace,
        "sign_accuracy": signs.get("sign_accuracy_all"),
        "row_rmse": row_m.get("rmse"),
        "row_r2": row_m.get("r2"),
        "picp_90": picp,
        "ace": ace,
        "uncertainty_status": uq_status,
        "propagation_method": PROPAGATION_METHOD if has_sigma else "none",
        "interval_method": interval_method,
        "n_valid": 1 if np.isfinite(mop.get("mop_rmse", np.nan)) else 0,
        "n_valid_intervals": int(n_valid_intervals),
        "min_pair_size": MIN_PAIR_N,
        "evaluation_partition": EVALUATION_PARTITION,
    }


def summarize_fold_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    if fold_df is None or fold_df.empty:
        return pd.DataFrame()
    keys = ["dataset", "source_model", "source_model_id", "selectivity_method", "scope"]
    metric_cols = [
        "mop_rmse",
        "mop_r2",
        "mop_picp_90",
        "mop_ace",
        "sign_accuracy",
        "row_rmse",
        "row_r2",
        "picp_90",
        "ace",
        "n_rows",
        "n_pairs",
    ]
    rows = []
    present = [k for k in keys if k in fold_df.columns]
    for key, g in fold_df.groupby(present, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        rec = {k: v for k, v in zip(present, key)}
        rec["n_valid"] = int(g["outer_fold"].nunique()) if "outer_fold" in g.columns else int(len(g))
        rec["uncertainty_status"] = str(g["uncertainty_status"].iloc[0]) if "uncertainty_status" in g.columns else ""
        rec["propagation_method"] = str(g["propagation_method"].iloc[0]) if "propagation_method" in g.columns else ""
        rec["aggregation"] = "fold_mean_plus_sd"
        rec["sd_definition"] = "ddof_1"
        for col in metric_cols:
            if col not in g.columns:
                continue
            stats = fold_mean_sd(pd.to_numeric(g[col], errors="coerce").to_numpy())
            rec[col] = stats["fold_mean"]
            rec[f"{col}_sd"] = stats["fold_sd"]
        rows.append(rec)
    return pd.DataFrame(rows)


def _display_model(model_id: str) -> str:
    mid = str(model_id)
    mapping = {
        "RDKIT_MLPENS5_EVID": "Evidential MLP ensemble",
        "RDKIT_MLP_EVID": "Evidential MLP",
        "RDKIT_MLPENS5_HETERO": "Heteroscedastic MLP ensemble",
        "RDKIT_MLP_HETERO": "Heteroscedastic MLP",
        "RDKIT_MLPENS5_MCDROPOUT": "MC-dropout MLP ensemble",
        "RDKIT_MLP_MCDROPOUT": "MC-dropout MLP",
        "RDKIT_SNGPENS5": "SNGP ensemble",
        "RDKIT_SNGP": "SNGP",
        "RDKIT_GPR_SPARSE": "Sparse variational GPR",
        "RDKIT_MLPENS5_BAYES_FULL": "Full variational BNN ensemble",
        "RDKIT_MLP_BAYES_FULL": "Full variational BNN",
        "GRAPH_DMPNN_EVID": "Chemprop evidential",
        "RDKIT_MLP_POINT": "RDKit MLP",
        "RDKIT_XGB_POINT": "RDKit XGBoost",
        "GRAPH_DMPNN_POINT": "Chemprop point",
    }
    for key, name in mapping.items():
        if mid.endswith(key):
            return name
    return mid


def method_class(selectivity_method: str, model_id: str) -> str:
    if selectivity_method == SELECTIVITY_DIRECT:
        return "direct_ds2"
    mid = str(model_id).upper()
    if "POINT" in mid or "XGB" in mid:
        return "ad_hoc_ds1_point"
    return "ad_hoc_ds1_uncertainty"


def ad_hoc_comparison_phrase(summary: pd.DataFrame) -> str:
    """Choose only the restrained wording supported by the aligned REE–REE MoP RMSE result."""
    if summary is None or summary.empty:
        return "was compared with"
    work = summary[
        (summary.scope.astype(str) == "ree_pairs")
        & (summary.source_model_id.astype(str) == DS2_WORKHORSE)
        & (summary.selectivity_method.astype(str) == SELECTIVITY_DIRECT)
    ]
    adhoc = summary[
        (summary.scope.astype(str) == "ree_pairs")
        & (summary.selectivity_method.astype(str) == SELECTIVITY_AD_HOC)
        & (~summary.source_model_id.astype(str).str.contains("POINT|XGB", regex=True))
    ]
    if work.empty or adhoc.empty:
        return "was compared with"
    metric = "mop_rmse" if "mop_rmse" in summary.columns else "mop_mae"
    direct_rmse = float(work.iloc[0][metric])
    adhoc_rmses = pd.to_numeric(adhoc[metric], errors="coerce").dropna()
    if adhoc_rmses.empty or not np.isfinite(direct_rmse):
        return "was compared with"
    if bool((adhoc_rmses > direct_rmse).all()):
        return "performed worse than"
    if bool((adhoc_rmses < direct_rmse).all()):
        return "was comparable to"
    span = float(adhoc_rmses.max() - adhoc_rmses.min())
    if span <= 0.05 and abs(float(adhoc_rmses.mean()) - direct_rmse) <= 0.05:
        return "was comparable to"
    return "did not consistently outperform"
