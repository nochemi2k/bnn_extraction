"""SI diagnostics from stored fold scores and CV metrics. No retraining.

Called from the downstream pipeline (fold scores) and, after workhorse
selection, from manuscript assembly (paired differences).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from uq_remake import REMAKE_ROOT
from uq_remake.downstream.analyses import (
    coverage_at_normal,
    curve_aurc,
    retained_metrics,
    risk_coverage_fold,
)
from uq_remake.downstream.protocol import (
    AD_DIR,
    AD_RETENTION_DIR,
    DS2_WORKHORSE,
    FOLDS,
    PAIR_ACQ_EVENT_MIN_EXCLUSIVE,
    PAIR_EVENT_DIR,
)
from uq_remake.metrics.cv_protocol import EVALUATION_PARTITION, fold_mean_sd
from uq_remake.metrics.fold_metrics import apply_scope as scope_filter

MSRC = REMAKE_ROOT / "results" / "manuscript_final"
MAN = REMAKE_ROOT / "manuscripts" / "final"
TABLES = MAN / "tables"
ACTIVE = REMAKE_ROOT / "results" / "active_grid"
PRED_ROOT = REMAKE_ROOT / "results" / "cv_predictions" / "fold_test_predictions"
WORKHORSE = DS2_WORKHORSE
POINT_TOKENS = ("_POINT",)
PRED_COLS = [
    "record_id",
    "y_true",
    "y_pred",
    "sigma_total",
    "std_total",
    "var_total",
    "uq_valid",
    "metal_species_pair",
    "is_ree_pair",
    "fold_id",
    "evaluation_partition",
    "member_id",
    "uncertainty_status",
]


def _write(path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        return s.astype(str).str.lower().isin({"true", "1", "yes"})
    return s.astype(bool)


def pair_level_ad_from_existing_curves() -> dict[str, Path]:
    fold = pd.read_csv(AD_RETENTION_DIR / "fold_retention.csv")
    if "dataset" in fold.columns:
        fold = fold[fold.dataset.astype(str) == "DS2"].copy()
    mop = fold[fold["metric_name"] == "mop_rmse"].copy()
    mop["retention_fraction"] = pd.to_numeric(mop["retention_fraction"], errors="coerce")
    mop["value"] = pd.to_numeric(mop["value"], errors="coerce")
    rows = []
    for keys, g in mop.groupby(["scope", "score_name", "outer_fold_id"], dropna=False):
        scope, score, fold_id = keys
        aurc = curve_aurc(g["retention_fraction"], g["value"])
        at50 = g.loc[np.isclose(g["retention_fraction"].to_numpy(dtype=float), 0.5), "value"]
        rows.append(
            {
                "dataset": "DS2",
                "experiment_id": "ad_pair_level_risk_coverage",
                "outer_fold_id": fold_id,
                "scope": scope,
                "score_name": score,
                "metric_name": "mop_aurc",
                "value": aurc,
                "metric_status": "ok" if np.isfinite(aurc) else "undefined_metric",
                "evaluation_partition": EVALUATION_PARTITION,
                "aggregation_unit": "mean_of_pairs_on_row_ranked_retention",
            }
        )
        rows.append(
            {
                "dataset": "DS2",
                "experiment_id": "ad_pair_level_risk_coverage",
                "outer_fold_id": fold_id,
                "scope": scope,
                "score_name": score,
                "metric_name": "mop_rmse_at_50pct_retention",
                "value": float(at50.iloc[0]) if len(at50) else float("nan"),
                "metric_status": "ok" if len(at50) and np.isfinite(float(at50.iloc[0])) else "undefined_metric",
                "evaluation_partition": EVALUATION_PARTITION,
                "aggregation_unit": "mean_of_pairs_on_row_ranked_retention",
            }
        )
    fold_out = pd.DataFrame(rows)
    summary_rows = []
    for (scope, score, metric), g in fold_out.groupby(["scope", "score_name", "metric_name"], dropna=False):
        stats = fold_mean_sd(g["value"].to_numpy(dtype=float))
        summary_rows.append(
            {
                "dataset": "DS2",
                "experiment_id": "ad_pair_level_risk_coverage",
                "scope": scope,
                "score_name": score,
                "metric_name": metric,
                "fold_mean": stats["fold_mean"],
                "fold_sd": stats["fold_sd"],
                "n_folds_attempted": int(g["outer_fold_id"].nunique()),
                "n_valid_folds": stats["n_valid_folds"],
                "metric_aggregation": "fold_mean_plus_sd",
                "sd_definition": "ddof_1",
                "note": "Rows ranked fold-locally; MoP RMSE computed on retained rows; retention-curve AUC is trapz(mop_rmse, retention) on the canonical 10-100% grid.",
            }
        )
    summary = pd.DataFrame(summary_rows)
    return {
        "fold": _write(AD_DIR / "ad_pair_level_fold_risk_coverage.csv", fold_out),
        "summary": _write(AD_DIR / "ad_pair_level_risk_coverage_summary.csv", summary),
        "msrc": _write(MSRC / "ad_pair_level_risk_coverage_summary.csv", summary),
    }


def metal_medium_strata() -> dict[str, Path]:
    frames = []
    score_root = AD_RETENTION_DIR / "fold_scores" / WORKHORSE
    for fold_id in FOLDS:
        path = score_root / f"{fold_id}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"missing workhorse fold scores under {score_root}")
    scores = pd.concat(frames, ignore_index=True)
    need = [
        "outer_fold_id",
        "fold_id",
        "record_id",
        "evaluation_partition",
        "y_true",
        "y_pred",
        "sigma_total",
        "metal_species_pair",
        "is_ree_pair",
        "metal_medium_seen",
    ]
    scores = scores[[c for c in need if c in scores.columns]].copy()
    scores["metal_medium_seen"] = _bool_series(scores["metal_medium_seen"])
    fold_rows = []
    for fold, g in scores.groupby("outer_fold_id", sort=False):
        g = g.copy()
        if "fold_id" not in g.columns:
            g["fold_id"] = fold
        g["fold_id"] = str(fold)
        g["evaluation_partition"] = EVALUATION_PARTITION
        for scope in ("all_pairs", "ree_pairs"):
            scoped = scope_filter(g, "DS2", scope)
            for seen_flag, label in ((True, "seen"), (False, "unseen")):
                part = scoped.loc[scoped["metal_medium_seen"] == seen_flag].copy()
                n_rows = int(len(part))
                n_pairs = int(part["metal_species_pair"].nunique()) if n_rows else 0
                if n_rows == 0:
                    met = {
                        "rmse": float("nan"),
                        "sign_accuracy": float("nan"),
                        "mop_rmse": float("nan"),
                        "mop_r2": float("nan"),
                    }
                    picp = float("nan")
                    status = "empty_stratum"
                else:
                    met = retained_metrics(part)
                    picp = coverage_at_normal(part["y_true"], part["y_pred"], part["sigma_total"])
                    status = "ok"
                for metric_name, value in (
                    ("n_rows", float(n_rows)),
                    ("n_pairs", float(n_pairs)),
                    ("rmse", met["rmse"]),
                    ("sign_accuracy", met["sign_accuracy"]),
                    ("mop_rmse", met["mop_rmse"]),
                    ("mop_r2", met.get("mop_r2", float("nan"))),
                    ("picp_90", picp),
                ):
                    fold_rows.append(
                        {
                            "dataset": "DS2",
                            "model_id": WORKHORSE,
                            "experiment_id": "metal_medium_strata",
                            "outer_fold_id": fold,
                            "scope": scope,
                            "stratum": "metal_medium_seen" if seen_flag else "metal_medium_unseen",
                            "stratum_label": label,
                            "metric_name": metric_name,
                            "value": value,
                            "n_rows_used": n_rows,
                            "n_pairs_used": n_pairs,
                            "metric_status": status if metric_name in {"n_rows", "n_pairs"} or np.isfinite(value) else met.get("metric_status", status),
                            "evaluation_partition": EVALUATION_PARTITION,
                            "identifiability": "identifiable",
                            "note": "metal–medium = (species pair, electrolyte) observed in outer-train; not a medium-novelty claim",
                        }
                    )
    fold_df = pd.DataFrame(fold_rows)
    summary_rows = []
    keys = ["scope", "stratum", "stratum_label", "metric_name"]
    for key, g in fold_df.groupby(keys, dropna=False):
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        stats = fold_mean_sd(pd.to_numeric(g["value"], errors="coerce").to_numpy())
        rec.update(
            {
                "dataset": "DS2",
                "model_id": WORKHORSE,
                "experiment_id": "metal_medium_strata",
                "fold_mean": stats["fold_mean"],
                "fold_sd": stats["fold_sd"],
                "n_folds_attempted": int(g["outer_fold_id"].nunique()),
                "n_valid_folds": stats["n_valid_folds"],
                "metric_aggregation": "fold_mean_plus_sd",
                "sd_definition": "ddof_1",
                "identifiability": "identifiable",
            }
        )
        summary_rows.append(rec)
    summary = pd.DataFrame(summary_rows)
    return {
        "fold": _write(AD_DIR / "ad_metal_medium_strata_fold.csv", fold_df),
        "summary": _write(AD_DIR / "ad_metal_medium_strata_summary.csv", summary),
        "msrc": _write(MSRC / "ad_metal_medium_strata_summary.csv", summary),
    }


def fold_paired_differences() -> dict[str, Path]:
    fold = pd.read_csv(REMAKE_ROOT / "results" / "cv_metrics" / "fold_metrics.csv")
    intervals = pd.read_csv(REMAKE_ROOT / "results" / "cv_metrics" / "fold_interval_metrics.csv")
    cand_path = MSRC / "workhorse_selection_candidates.csv"
    if not cand_path.exists():
        cand_path = REMAKE_ROOT / "manuscripts" / "final" / "data" / "workhorse_selection_candidates.csv"
    cand = pd.read_csv(cand_path)
    usable = cand[cand["calibration_usable"].astype(str).str.lower().isin({"true", "1"})].copy()
    usable_ids = [WORKHORSE] + [m for m in usable["model_id"].astype(str) if m != WORKHORSE]

    def _vals(src: pd.DataFrame, mid: str, scope: str, metric: str) -> pd.Series:
        hit = src[
            (src["dataset"].astype(str).str.upper() == "DS2")
            & (src["model_id"] == mid)
            & (src["scope"] == scope)
            & (src["metric_name"] == metric)
        ].copy()
        hit["fold_id"] = hit["fold_id"].astype(str)
        return hit.set_index("fold_id")["value"].astype(float)

    wh_mae = _vals(fold, WORKHORSE, "all_pairs", "mop_rmse")
    wh_ree = _vals(fold, WORKHORSE, "ree_pairs", "mop_rmse")
    wh_picp = _vals(intervals, WORKHORSE, "all_pairs", "marginal_picp_90")
    wh_ace = _vals(intervals, WORKHORSE, "all_pairs", "marginal_ace")
    fold_rows = []
    for mid in usable_ids:
        if mid == WORKHORSE:
            continue
        mae = _vals(fold, mid, "all_pairs", "mop_rmse")
        ree = _vals(fold, mid, "ree_pairs", "mop_rmse")
        picp = _vals(intervals, mid, "all_pairs", "marginal_picp_90")
        ace = _vals(intervals, mid, "all_pairs", "marginal_ace")
        folds_ok = set(wh_mae.index) & set(mae.index) & set(picp.index) & set(ace.index)
        for fold_id in sorted(folds_ok):
            fold_rows.append(
                {
                    "dataset": "DS2",
                    "comparator_model_id": mid,
                    "reference_model_id": WORKHORSE,
                    "outer_fold_id": fold_id,
                    "delta_all_pair_mop_rmse": float(mae.loc[fold_id] - wh_mae.loc[fold_id]),
                    "delta_ree_mop_rmse": float(ree.loc[fold_id] - wh_ree.loc[fold_id]) if fold_id in ree.index and fold_id in wh_ree.index else float("nan"),
                    "delta_all_pair_picp_90": float(picp.loc[fold_id] - wh_picp.loc[fold_id]),
                    "delta_all_pair_ace": float(ace.loc[fold_id] - wh_ace.loc[fold_id]),
                    "comparator_all_pair_mop_rmse": float(mae.loc[fold_id]),
                    "reference_all_pair_mop_rmse": float(wh_mae.loc[fold_id]),
                    "evaluation_partition": EVALUATION_PARTITION,
                    "note": "descriptive fold-wise difference; not a nested significance test",
                }
            )
    fold_df = pd.DataFrame(fold_rows)
    summary_rows = []
    if len(fold_df):
        for mid, g in fold_df.groupby("comparator_model_id"):
            rec = {
                "dataset": "DS2",
                "comparator_model_id": mid,
                "reference_model_id": WORKHORSE,
                "n_folds": int(g["outer_fold_id"].nunique()),
            }
            for col in ("delta_all_pair_mop_rmse", "delta_ree_mop_rmse", "delta_all_pair_picp_90", "delta_all_pair_ace"):
                stats = fold_mean_sd(g[col].to_numpy(dtype=float))
                rec[f"{col}_mean"] = stats["fold_mean"]
                rec[f"{col}_sd"] = stats["fold_sd"]
                rec[f"{col}_n_valid"] = stats["n_valid_folds"]
            rec["n_folds_comparator_worse_rmse"] = int((g["delta_all_pair_mop_rmse"] > 0).sum())
            rec["note"] = "Positive RMSE delta means the comparator has higher (worse) all-pair MoP RMSE than the workhorse. Not a hypothesis test."
            summary_rows.append(rec)
    summary = pd.DataFrame(summary_rows)
    return {
        "fold": _write(MSRC / "workhorse_fold_paired_differences.csv", fold_df),
        "summary": _write(MSRC / "workhorse_fold_paired_difference_summary.csv", summary),
    }


def mop_r2_pair_validity() -> dict[str, Path]:
    fold = pd.read_csv(REMAKE_ROOT / "results" / "cv_metrics" / "fold_metrics.csv")
    hit = fold[
        (fold["model_id"] == WORKHORSE)
        & (fold["scope"].isin(["ree_pairs", "all_pairs"]))
        & (fold["metric_name"].isin(["mop_r2", "mop_rmse"]))
    ].copy()
    keep = hit[
        [
            "dataset",
            "model_id",
            "fold_id",
            "scope",
            "metric_name",
            "n_pairs_total",
            "n_pairs_retained",
            "n_pairs_valid",
            "n_pairs_invalid",
            "value",
        ]
    ]
    return {"fold": _write(MSRC / "workhorse_mop_pair_validity.csv", keep)}


def _pred_path(model_id: str, fold: str) -> Path:
    if model_id == "DS2_GRAPH_DMPNN_EVID":
        p = PRED_ROOT / "DS2_GRAPH_DMPNN_EVID_v2" / fold / "fold_test_predictions.csv"
        if p.exists():
            return p
    p = ACTIVE / model_id / fold / "test_preds.csv"
    if p.exists():
        return p
    return PRED_ROOT / model_id / fold / "fold_test_predictions.csv"


def _load_pred(model_id: str, fold: str) -> pd.DataFrame:
    path = _pred_path(model_id, fold)
    header = pd.read_csv(path, nrows=0)
    cols = [c for c in PRED_COLS if c in header.columns]
    df = pd.read_csv(path, usecols=cols)
    df["fold_id"] = fold
    df["evaluation_partition"] = EVALUATION_PARTITION
    if "member_id" in df.columns:
        members = set(df["member_id"].astype(str).unique())
        if "ALL" in members:
            df = df.loc[df["member_id"].astype(str) == "ALL"].copy()
    if "record_id" in df.columns and df["record_id"].duplicated().any():
        df = df.drop_duplicates("record_id", keep="first")
    if "uncertainty_status" in df.columns:
        status = df["uncertainty_status"].astype(str).str.lower()
        bad = status.isin({"invalid", "failed", "partial_invalid"})
        if bad.any() and "sigma_total" in df.columns:
            df.loc[bad, "sigma_total"] = np.nan
    if "uq_valid" in df.columns:
        flag = df["uq_valid"]
        if flag.dtype == object:
            flag = flag.astype(str).str.lower().isin({"true", "1", "yes"})
        if "sigma_total" not in df.columns:
            df["sigma_total"] = np.nan
        df.loc[~flag.astype(bool), "sigma_total"] = np.nan
    if "sigma_total" not in df.columns:
        if "std_total" in df.columns:
            df["sigma_total"] = pd.to_numeric(df["std_total"], errors="coerce")
        elif "var_total" in df.columns:
            vt = pd.to_numeric(df["var_total"], errors="coerce")
            df["sigma_total"] = np.sqrt(np.maximum(vt, 0.0))
        else:
            df["sigma_total"] = np.nan
    df["sigma_total"] = pd.to_numeric(df["sigma_total"], errors="coerce")
    if "sigma_total" in df.columns and df["sigma_total"].isna().all() and "var_total" in df.columns:
        vt = pd.to_numeric(df["var_total"], errors="coerce")
        df["sigma_total"] = np.sqrt(np.maximum(vt, 0.0))
        if "uq_valid" in df.columns:
            flag = df["uq_valid"]
            if flag.dtype == object:
                flag = flag.astype(str).str.lower().isin({"true", "1", "yes"})
            df.loc[~flag.astype(bool), "sigma_total"] = np.nan
    return df


def grid_sigma_total_risk_ordering() -> dict[str, Path]:
    eligibility = pd.read_csv(REMAKE_ROOT / "results" / "model_audit" / "workhorse_eligibility.csv")
    ds2 = eligibility[eligibility["dataset"].astype(str).str.upper() == "DS2"]
    models = []
    for mid in ds2["model_id"].astype(str):
        if any(tok in mid for tok in POINT_TOKENS):
            continue
        if mid.endswith("_GPR") and "SPARSE" not in mid:
            continue
        models.append(mid)
    if "DS2_GRAPH_DMPNN_EVID" not in models:
        models.append("DS2_GRAPH_DMPNN_EVID")
    fold_rows = []
    for mid in models:
        for fold in FOLDS:
            path = _pred_path(mid, fold)
            if not path.exists():
                continue
            df = _load_pred(mid, fold)
            if "sigma_total" not in df.columns or not np.isfinite(pd.to_numeric(df["sigma_total"], errors="coerce")).any():
                print(f"grid_sigma_total skip {mid} {fold} (no sigma_total)", flush=True)
                continue
            rows = risk_coverage_fold(df, "sigma_total", fold, experiment_id="grid_sigma_total_risk_ordering")
            mop = [r for r in rows if r["metric_name"] == "mop_rmse" and r.get("retention_fraction") not in ("", None)]
            mae = [r for r in rows if r["metric_name"] == "rmse" and r.get("retention_fraction") not in ("", None)]
            for scope in ("all_pairs", "ree_pairs"):
                mop_s = [r for r in mop if r["scope"] == scope]
                mae_s = [r for r in mae if r["scope"] == scope]
                mop_aurc = curve_aurc(
                    [float(r["retention_fraction"]) for r in mop_s],
                    [r["value"] for r in mop_s],
                )
                row_aurc = curve_aurc(
                    [float(r["retention_fraction"]) for r in mae_s],
                    [r["value"] for r in mae_s],
                )
                mop50 = [r["value"] for r in mop_s if np.isclose(float(r["retention_fraction"]), 0.5)]
                mae50 = [r["value"] for r in mae_s if np.isclose(float(r["retention_fraction"]), 0.5)]
                fold_rows.append(
                    {
                        "dataset": "DS2",
                        "model_id": mid,
                        "outer_fold_id": fold,
                        "scope": scope,
                        "score_name": "sigma_total",
                        "row_aurc": row_aurc,
                        "mop_aurc": mop_aurc,
                        "row_rmse_at_50pct_retention": float(mae50[0]) if mae50 else float("nan"),
                        "mop_rmse_at_50pct_retention": float(mop50[0]) if mop50 else float("nan"),
                        "evaluation_partition": EVALUATION_PARTITION,
                        "used_for_workhorse_selection": False,
                        "note": "Reconstructed fold-local sigma_total risk-ordering; not a selection key",
                    }
                )
            print(f"grid_sigma_total {mid} {fold}", flush=True)
    fold_df = pd.DataFrame(fold_rows)
    summary_rows = []
    if len(fold_df):
        for (mid, scope), g in fold_df.groupby(["model_id", "scope"]):
            rec = {
                "dataset": "DS2",
                "model_id": mid,
                "scope": scope,
                "score_name": "sigma_total",
                "used_for_workhorse_selection": False,
            }
            for col in ("row_aurc", "mop_aurc", "row_rmse_at_50pct_retention", "mop_rmse_at_50pct_retention"):
                stats = fold_mean_sd(g[col].to_numpy(dtype=float))
                rec[f"{col}_mean"] = stats["fold_mean"]
                rec[f"{col}_sd"] = stats["fold_sd"]
                rec[f"{col}_n_valid"] = stats["n_valid_folds"]
            rec["n_folds_attempted"] = int(g["outer_fold_id"].nunique())
            summary_rows.append(rec)
    summary = pd.DataFrame(summary_rows)
    return {
        "fold": _write(MSRC / "grid_sigma_total_risk_ordering_fold.csv", fold_df),
        "summary": _write(MSRC / "grid_sigma_total_risk_ordering.csv", summary),
    }


def _pair_event_rows(pred: pd.DataFrame, *, aggregation: str) -> list[dict]:
    rows = []
    for pair, g in pred.groupby("metal_species_pair", dropna=False):
        y = pd.to_numeric(g["y_true"], errors="coerce").to_numpy(dtype=float)
        yhat = pd.to_numeric(g["y_pred"], errors="coerce").to_numpy(dtype=float)
        sig = pd.to_numeric(g["sigma_total"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(y) & np.isfinite(yhat)
        y, yhat, sig = y[ok], yhat[ok], sig[ok]
        n = int(y.size)
        if n == 0:
            continue
        n_hi = int(np.sum(y > 1.0))
        n_lo = int(np.sum(y < -1.0))
        n_nz = int(np.sum(np.abs(y) < 1.0))
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        mean_sigma = float(np.nanmean(sig)) if np.isfinite(sig).any() else np.nan
        n_folds = int(g["fold_id"].nunique()) if "fold_id" in g.columns else 1
        if "is_ree_pair" in g.columns:
            is_ree = bool(g["is_ree_pair"].astype(str).str.lower().isin({"true", "1", "yes"}).any())
        else:
            is_ree = False
        rows.append(
            {
                "dataset": "DS2",
                "model_id": WORKHORSE,
                "metal_species_pair": pair,
                "is_ree_pair": is_ree,
                "n_heldout_rows": n,
                "n_high_selectivity_canonical": n_hi,
                "n_high_selectivity_reversed": n_lo,
                "n_high_selectivity_events": n_hi + n_lo,
                "n_near_zero": n_nz,
                "high_selectivity_fraction": (n_hi + n_lo) / n if n else np.nan,
                "out_of_fold_pooled_per_pair_rmse": rmse if aggregation == "out_of_fold_pooled" else np.nan,
                "fold_local_per_pair_rmse": rmse if aggregation == "fold_local" else np.nan,
                "mean_sigma_total": mean_sigma,
                "n_folds_represented": n_folds,
                "aggregation": aggregation,
                "label_source": "heldout_fold_observed_y",
                "used_for_workhorse_selection": False,
            }
        )
    return rows


def pair_event_support() -> dict[str, Path]:
    """Held-out pair-event support from workhorse AD fold scores (no cv_predictions)."""
    score_root = AD_RETENTION_DIR / "fold_scores" / WORKHORSE
    frames = []
    fold_rows = []
    for fold in FOLDS:
        path = score_root / f"{fold}.csv"
        if not path.exists():
            continue
        pred = pd.read_csv(path)
        if "evaluation_partition" in pred.columns:
            pred = pred[pred.evaluation_partition.astype(str) == "heldout_fold"].copy()
        pred["fold_id"] = fold if "fold_id" not in pred.columns else pred["fold_id"]
        frames.append(pred)
        fold_rows.extend(_pair_event_rows(pred, aggregation="fold_local"))
    if not frames:
        raise FileNotFoundError(f"missing workhorse fold scores under {score_root}")
    all_pred = pd.concat(frames, ignore_index=True)
    pooled = pd.DataFrame(_pair_event_rows(all_pred, aggregation="out_of_fold_pooled"))
    pooled = pooled.sort_values(["n_heldout_rows", "n_high_selectivity_events"], ascending=False)
    frequent = pooled.nlargest(15, "n_heldout_rows").copy()
    frequent["selection_rule"] = "top_15_heldout_rows_across_five_folds"
    n_pairs = int(len(pooled))
    n_ge10_rows = int((pooled.n_heldout_rows >= 10).sum())
    n_gt10_events = int((pooled.n_high_selectivity_events > 10).sum())
    n_gt10_canonical = int((pooled.n_high_selectivity_canonical > 10).sum())
    n_gt_min_events = int((pooled.n_high_selectivity_events > PAIR_ACQ_EVENT_MIN_EXCLUSIVE).sum())
    n_gt_min_canonical = int((pooled.n_high_selectivity_canonical > PAIR_ACQ_EVENT_MIN_EXCLUSIVE).sum())
    min_ex = int(PAIR_ACQ_EVENT_MIN_EXCLUSIVE)
    summary = pd.DataFrame(
        [
            {
                "dataset": "DS2",
                "model_id": WORKHORSE,
                "n_pairs_with_heldout_rows": n_pairs,
                "n_pairs_ge_10_heldout_rows": n_ge10_rows,
                "n_pairs_gt_10_high_selectivity_events": n_gt10_events,
                "n_pairs_gt_10_canonical_high_selectivity": n_gt10_canonical,
                f"n_pairs_gt_{min_ex}_high_selectivity_events": n_gt_min_events,
                f"n_pairs_gt_{min_ex}_canonical_high_selectivity": n_gt_min_canonical,
                "event_min_exclusive": min_ex,
                "aggregation": "out_of_fold_pooled_counts",
                "label_source": "heldout_fold_observed_y",
            }
        ]
    )
    fold_df = pd.DataFrame(fold_rows)
    PAIR_EVENT_DIR.mkdir(parents=True, exist_ok=True)
    written = {
        "fold": _write(PAIR_EVENT_DIR / "pair_event_support_fold.csv", fold_df),
        "pooled": _write(PAIR_EVENT_DIR / "pair_event_support_pooled.csv", pooled),
        "frequent": _write(PAIR_EVENT_DIR / "pair_event_support_frequent.csv", frequent),
        "summary": _write(PAIR_EVENT_DIR / "pair_event_support_summary.csv", summary),
    }
    for dest_root in (MSRC, TABLES):
        dest_root.mkdir(parents=True, exist_ok=True)
        for src in written.values():
            _write(dest_root / src.name, pd.read_csv(src))
    return written


def run_from_fold_scores() -> dict[str, dict[str, Path]]:
    """Downstream SI tables that only need AD fold scores and CV metrics."""
    MSRC.mkdir(parents=True, exist_ok=True)
    from uq_remake.downstream.magnitude_adjusted_risk import magnitude_adjusted_risk

    return {
        "pair_level_ad": pair_level_ad_from_existing_curves(),
        "metal_medium": metal_medium_strata(),
        "mop_r2_validity": mop_r2_pair_validity(),
        "pair_event_support": pair_event_support(),
        "magnitude_adjusted_risk": magnitude_adjusted_risk(),
    }


def run_after_workhorse_selection() -> dict[str, dict[str, Path]]:
    """Paired differences need the manuscript workhorse-candidate table."""
    MSRC.mkdir(parents=True, exist_ok=True)
    return {"paired_differences": fold_paired_differences()}


def run_all(*, skip_grid: bool = False) -> dict[str, dict[str, Path]]:
    out = run_from_fold_scores()
    out.update(run_after_workhorse_selection())
    if not skip_grid:
        out["grid_sigma_total"] = grid_sigma_total_risk_ordering()
    return out


if __name__ == "__main__":
    import sys

    skip = "--skip-grid" in sys.argv
    result = run_all(skip_grid=skip)
    for name, paths in result.items():
        print(name, {k: str(v) for k, v in paths.items()})
