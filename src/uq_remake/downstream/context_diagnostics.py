"""Rebuild pair-density and sign-stability summaries from stored fold scores.

Does not retrain models or recompute Tanimoto/Mahalanobis/embeddings.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from uq_remake.downstream.analyses import coverage_at_normal, retained_metrics, sign_stability_fold, summarize_fold_metrics
from uq_remake.downstream.ad_distance_error_corr import rebuild_ad_distance_error_corr
from uq_remake.downstream.protocol import (
    AD_RETENTION_DIR,
    DATASET,
    DS2_WORKHORSE,
    EVALUATION_PARTITION,
    FOLDS,
    PAIR_COLUMN,
    PAIR_DENSITY_BINS,
    PAIR_DIR,
    SIGN_DIR,
    rel,
)
from uq_remake.metrics.fold_metrics import apply_scope


def bin_pair_count(n: int) -> str:
    for name, lo, hi in PAIR_DENSITY_BINS:
        if lo <= int(n) <= hi:
            return name
    return "0"


def pair_density_fold(scores: pd.DataFrame, fold: str) -> list[dict]:
    rows = []
    work = scores.copy()
    work["_bin"] = [bin_pair_count(int(n)) for n in work["n_training_ligands_for_pair"].to_numpy()]
    for scope in ("all_pairs", "ree_pairs"):
        scoped = apply_scope(work, DATASET, scope)
        for bname, _lo, _hi in PAIR_DENSITY_BINS:
            part = scoped.loc[scoped["_bin"] == bname]
            if len(part) == 0:
                rows.append(
                    {
                        "dataset": DATASET,
                        "experiment_id": "pair_density",
                        "outer_fold_id": fold,
                        "scope": scope,
                        "score_name": "pair_density_risk",
                        "retention_fraction": "",
                        "metric_name": "n_rows",
                        "value": float("nan"),
                        "n_rows_total": int(len(scoped)),
                        "n_rows_used": 0,
                        "n_pairs_total": 0,
                        "n_pairs_used": 0,
                        "metric_status": "missing_in_fold",
                        "bin": bname,
                        "evaluation_partition": EVALUATION_PARTITION,
                    }
                )
                continue
            met = retained_metrics(part)
            picp = coverage_at_normal(part["y_true"], part["y_pred"], part["sigma_total"])
            mean_u = float(pd.to_numeric(part["sigma_total"], errors="coerce").mean())
            n_pairs = int(part[PAIR_COLUMN].nunique())
            for metric_name, value in (
                ("n_rows", float(len(part))),
                ("n_pairs", float(n_pairs)),
                ("mean_sigma_total", mean_u),
                ("rmse", met["rmse"]),
                ("mop_rmse", met["mop_rmse"]),
                ("picp_90", picp),
            ):
                rows.append(
                    {
                        "dataset": DATASET,
                        "experiment_id": "pair_density",
                        "outer_fold_id": fold,
                        "scope": scope,
                        "score_name": "pair_density_risk",
                        "retention_fraction": "",
                        "metric_name": metric_name,
                        "value": value,
                        "n_rows_total": int(len(scoped)),
                        "n_rows_used": int(len(part)),
                        "n_pairs_total": n_pairs,
                        "n_pairs_used": met.get("n_pairs_used", n_pairs),
                        "metric_status": "ok" if np.isfinite(value) else met["metric_status"],
                        "bin": bname,
                        "evaluation_partition": EVALUATION_PARTITION,
                    }
                )
    return rows


def fold_score_path(model_id: str, fold: str) -> Path:
    return AD_RETENTION_DIR / "fold_scores" / model_id / f"{fold}.csv"


def rebuild_context_diagnostics(model_id: str = DS2_WORKHORSE) -> dict:
    pair_rows: list[dict] = []
    sign_rows: list[dict] = []
    for fold in FOLDS:
        path = fold_score_path(model_id, fold)
        if not path.exists():
            raise FileNotFoundError(path)
        scores = pd.read_csv(path)
        print(f"context {model_id} {fold} n={len(scores)}", flush=True)
        pair_rows.extend(pair_density_fold(scores, fold))
        sign_rows.extend(sign_stability_fold(scores, fold, dataset="DS2"))
    pair_df = pd.DataFrame(pair_rows)
    sign_df = pd.DataFrame(sign_rows)
    PAIR_DIR.mkdir(parents=True, exist_ok=True)
    SIGN_DIR.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(PAIR_DIR / "pair_density_fold.csv", index=False)
    sign_df.to_csv(SIGN_DIR / "sign_stability_fold.csv", index=False)
    pair_sum = summarize_fold_metrics(pair_df, ["dataset", "experiment_id", "scope", "score_name", "bin", "metric_name"])
    sign_sum = summarize_fold_metrics(sign_df, ["dataset", "experiment_id", "scope", "score_name", "category", "metric_name"])
    pair_sum.to_csv(PAIR_DIR / "pair_density_summary.csv", index=False)
    sign_sum.to_csv(SIGN_DIR / "sign_stability_summary.csv", index=False)
    snapshot = {
        "experiment_id": "context_diagnostics",
        "model_id": model_id,
        "computation_status": "complete",
        "source": rel(AD_RETENTION_DIR / "fold_scores" / model_id),
        "n_pair_fold_rows": int(len(pair_df)),
        "n_sign_fold_rows": int(len(sign_df)),
        "rank_correlation_used": False,
        "pooled_oof_metrics": False,
    }
    (PAIR_DIR / "pair_density_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    (SIGN_DIR / "sign_stability_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"Wrote pair-density and sign-stability summaries for {model_id}", flush=True)
    rebuild_ad_distance_error_corr(model_id)
    return snapshot
