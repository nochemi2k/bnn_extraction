"""Fold-wise AD-distance versus row-level absolute-residual association.

This is an SI diagnostic. It is not a workhorse-selection metric and is not a
substitute for fixed-retention error-ordering.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from uq_remake.downstream.analyses import summarize_fold_metrics
from uq_remake.downstream.protocol import (
    AD_RETENTION_DIR,
    CHEMPROP_DISTANCE_ERROR_CORR_SCORES,
    CORR_DIR,
    DATASET,
    DISTANCE_ERROR_CORR_MIN_N,
    DISTANCE_ERROR_CORR_SCOPES,
    DISTANCE_ERROR_CORR_SCORES,
    DS2_CHEMPROP,
    DS2_WORKHORSE,
    EVALUATION_PARTITION,
    FOLDS,
    rel,
    score_display_name,
)
from uq_remake.metrics.fold_metrics import apply_scope

ERROR_DEFINITION = "row_level_absolute_residual"
EXPERIMENT_ID = "ad_distance_error_corr"


def _safe_corr(x, y, kind: str, min_n: int = DISTANCE_ERROR_CORR_MIN_N) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    ok = np.isfinite(xa) & np.isfinite(ya)
    n = int(ok.sum())
    if n < int(min_n):
        return float("nan")
    xv, yv = xa[ok], ya[ok]
    if float(np.std(xv)) == 0.0 or float(np.std(yv)) == 0.0:
        return float("nan")
    if kind == "pearson":
        r, _ = pearsonr(xv, yv)
    elif kind == "spearman":
        r, _ = spearmanr(xv, yv)
    else:
        raise ValueError(f"unknown correlation kind: {kind}")
    return float(r)


def fold_distance_error_corr(
    scores: pd.DataFrame,
    fold: str,
    model_id: str = DS2_WORKHORSE,
    scopes: tuple[str, ...] = DISTANCE_ERROR_CORR_SCOPES,
    score_names: tuple[str, ...] = DISTANCE_ERROR_CORR_SCORES,
    min_n: int = DISTANCE_ERROR_CORR_MIN_N,
) -> list[dict]:
    """One fold: Pearson and Spearman between each AD distance and |y − ŷ|."""
    rows: list[dict] = []
    work = scores.copy()
    yt = pd.to_numeric(work["y_true"], errors="coerce")
    yp = pd.to_numeric(work["y_pred"], errors="coerce")
    work["_abs_residual"] = (yt - yp).abs()
    for scope in scopes:
        scoped = apply_scope(work, DATASET, scope)
        n_total = int(len(scoped))
        for score_name in score_names:
            rec = {
                "dataset": DATASET,
                "experiment_id": EXPERIMENT_ID,
                "model_id": model_id,
                "outer_fold_id": fold,
                "scope": scope,
                "score_name": score_name,
                "evaluation_partition": EVALUATION_PARTITION,
                "error_definition": ERROR_DEFINITION,
                "n_rows_total": n_total,
            }
            if score_name not in scoped.columns:
                for metric_name in ("pearson_abs_residual", "spearman_abs_residual"):
                    rows.append(
                        {
                            **rec,
                            "metric_name": metric_name,
                            "value": float("nan"),
                            "n_rows_used": 0,
                            "metric_status": "missing_score",
                        }
                    )
                continue
            x = pd.to_numeric(scoped[score_name], errors="coerce")
            y = pd.to_numeric(scoped["_abs_residual"], errors="coerce")
            ok = x.notna() & y.notna() & np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
            n_used = int(ok.sum())
            status = "ok" if n_used >= int(min_n) else "insufficient_n"
            pearson = _safe_corr(x, y, "pearson", min_n=min_n)
            spearman = _safe_corr(x, y, "spearman", min_n=min_n)
            if status == "ok" and (not np.isfinite(pearson) or not np.isfinite(spearman)):
                status = "undefined_zero_variance"
            for metric_name, value in (
                ("pearson_abs_residual", pearson),
                ("spearman_abs_residual", spearman),
            ):
                rows.append(
                    {
                        **rec,
                        "metric_name": metric_name,
                        "value": value,
                        "n_rows_used": n_used,
                        "metric_status": status,
                    }
                )
    return rows


def fold_score_path(model_id: str, fold: str) -> Path:
    return AD_RETENTION_DIR / "fold_scores" / model_id / f"{fold}.csv"


def corr_scores_for_model(model_id: str) -> tuple[str, ...]:
    if str(model_id) == DS2_CHEMPROP or "GRAPH_DMPNN" in str(model_id):
        return CHEMPROP_DISTANCE_ERROR_CORR_SCORES
    return DISTANCE_ERROR_CORR_SCORES


def rebuild_ad_distance_error_corr(model_id: str | tuple[str, ...] | None = None) -> dict:
    if model_id is None:
        targets = (DS2_WORKHORSE, DS2_CHEMPROP)
    elif isinstance(model_id, str):
        targets = (model_id,)
    else:
        targets = tuple(model_id)
    fold_rows: list[dict] = []
    for mid in targets:
        names = corr_scores_for_model(mid)
        for fold in FOLDS:
            path = fold_score_path(mid, fold)
            if not path.exists():
                raise FileNotFoundError(path)
            scores = pd.read_csv(path)
            print(f"ad_distance_error_corr {mid} {fold} n={len(scores)} scores={len(names)}", flush=True)
            fold_rows.extend(fold_distance_error_corr(scores, fold, model_id=mid, score_names=names))
    fold_new = pd.DataFrame(fold_rows)
    CORR_DIR.mkdir(parents=True, exist_ok=True)
    fold_path = CORR_DIR / "ad_distance_error_corr_fold.csv"
    sum_path = CORR_DIR / "ad_distance_error_corr_summary.csv"
    if fold_path.exists() and targets:
        old = pd.read_csv(fold_path)
        old = old[~old.model_id.astype(str).isin(set(targets))].copy()
        fold_df = pd.concat([old, fold_new], ignore_index=True)
    else:
        fold_df = fold_new
    fold_df.to_csv(fold_path, index=False)
    summary = summarize_fold_metrics(
        fold_df,
        ["dataset", "experiment_id", "model_id", "scope", "score_name", "metric_name", "error_definition"],
    )
    if len(summary):
        n_used = (
            fold_df.groupby(["model_id", "scope", "score_name", "metric_name"], dropna=False)["n_rows_used"]
            .mean()
            .rename("n_rows_used_fold_mean")
            .reset_index()
        )
        summary = summary.merge(n_used, on=["model_id", "scope", "score_name", "metric_name"], how="left")
        summary["score_display"] = summary["score_name"].map(score_display_name)
        summary["role"] = "si_diagnostic_only"
        summary["used_for_selection"] = False
    summary.to_csv(sum_path, index=False)
    snapshot = {
        "experiment_id": EXPERIMENT_ID,
        "model_id": list(targets),
        "computation_status": "complete",
        "source": [rel(AD_RETENTION_DIR / "fold_scores" / mid) for mid in targets],
        "n_fold_rows": int(len(fold_df)),
        "error_definition": ERROR_DEFINITION,
        "score_names": {
            mid: list(corr_scores_for_model(mid)) for mid in targets
        },
        "scopes": list(DISTANCE_ERROR_CORR_SCOPES),
        "min_n": DISTANCE_ERROR_CORR_MIN_N,
        "used_for_selection": False,
        "rank_correlation_role": "si_diagnostic_only",
        "pooled_oof_metrics": False,
    }
    (CORR_DIR / "ad_distance_error_corr_snapshot.json").write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8"
    )
    print(f"Wrote AD-distance vs |error| correlations for {list(targets)}", flush=True)
    return snapshot


def main() -> dict:
    return rebuild_ad_distance_error_corr()


if __name__ == "__main__":
    main()
