"""Fold-local magnitude-adjusted risk diagnostics for the selected DS2 workhorse.

Computes partial Spearman correlation between sigma_total and |y-yhat| controlling
for |yhat|, plus an optional |yhat| fixed-retention comparator. Does not overwrite
canonical AD retention or pair-acquisition ranking artifacts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from uq_remake.downstream.analyses import curve_aurc, risk_coverage_fold
from uq_remake.downstream.protocol import (
    AD_RETENTION_DIR,
    DS2_WORKHORSE,
    FOLDS,
    PAIR_ACQ_DIR,
    PRIMARY_RETENTION,
)
from uq_remake.metrics.cv_protocol import EVALUATION_PARTITION, FOLD_SD_DDOF, fold_mean_sd

SCOPE = "all_pairs"
SCORE_ABS_YHAT = "abs_yhat"


def _nan() -> float:
    return float("nan")


def partial_spearman(x, y, z) -> float:
    """Spearman partial correlation of x and y controlling for z via rank-Pearson."""
    xs = pd.Series(np.asarray(x, dtype=float))
    ys = pd.Series(np.asarray(y, dtype=float))
    zs = pd.Series(np.asarray(z, dtype=float))
    mask = xs.notna() & ys.notna() & zs.notna()
    if int(mask.sum()) < 3:
        return _nan()
    rx = xs[mask].rank(method="average")
    ry = ys[mask].rank(method="average")
    rz = zs[mask].rank(method="average")
    if rx.nunique() < 2 or ry.nunique() < 2 or rz.nunique() < 2:
        return _nan()
    rxy = float(rx.corr(ry, method="pearson"))
    rxz = float(rx.corr(rz, method="pearson"))
    ryz = float(ry.corr(rz, method="pearson"))
    denom = float(np.sqrt((1.0 - rxz * rxz) * (1.0 - ryz * ryz)))
    if not np.isfinite(denom) or denom < 1e-12:
        return _nan()
    val = (rxy - rxz * ryz) / denom
    return float(val) if np.isfinite(val) else _nan()


def spearman(x, y) -> float:
    xs = pd.Series(np.asarray(x, dtype=float))
    ys = pd.Series(np.asarray(y, dtype=float))
    rho = xs.corr(ys, method="spearman")
    return float(rho) if pd.notna(rho) else _nan()


def _load_fold(fold: str, model_id: str) -> pd.DataFrame:
    path = AD_RETENTION_DIR / "fold_scores" / model_id / f"{fold}.csv"
    df = pd.read_csv(path)
    if "evaluation_partition" in df.columns:
        part = df["evaluation_partition"].astype(str)
        if not part.eq(EVALUATION_PARTITION).all():
            raise RuntimeError(f"{fold}: expected {EVALUATION_PARTITION}")
    return df


def fold_correlations(df: pd.DataFrame) -> dict:
    y = pd.to_numeric(df["y_true"], errors="coerce")
    yhat = pd.to_numeric(df["y_pred"], errors="coerce")
    sigma = pd.to_numeric(df["sigma_total"], errors="coerce")
    mask = y.notna() & yhat.notna() & sigma.notna()
    err = (y[mask] - yhat[mask]).abs()
    mag = yhat[mask].abs()
    sig = sigma[mask]
    return {
        "n_rows": int(mask.sum()),
        "rho_sigma_abs_error": spearman(sig, err),
        "rho_sigma_abs_yhat": spearman(sig, mag),
        "rho_abs_yhat_abs_error": spearman(mag, err),
        "rho_partial_sigma_error_given_abs_yhat": partial_spearman(sig, err, mag),
    }


def _summarize(fold_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in sorted(fold_rows.metric_name.unique()):
        part = fold_rows[fold_rows.metric_name.astype(str) == metric]
        stats = fold_mean_sd(part.value)
        rows.append(
            {
                "model_id": str(part.model_id.iloc[0]),
                "scope": str(part.scope.iloc[0]),
                "metric_name": metric,
                "fold_mean": stats["fold_mean"],
                "fold_sd": stats["fold_sd"],
                "n_valid_folds": stats["n_valid_folds"],
                "aggregation": "fold_mean_plus_sd",
                "evaluation_partition": EVALUATION_PARTITION,
                "fold_sd_ddof": FOLD_SD_DDOF,
            }
        )
    return pd.DataFrame(rows)


def magnitude_adjusted_risk(model_id: str = DS2_WORKHORSE) -> dict:
    corr_rows = []
    ret_rows = []
    for fold in FOLDS:
        df = _load_fold(fold, model_id)
        cor = fold_correlations(df)
        for name, value in cor.items():
            if name == "n_rows":
                continue
            corr_rows.append(
                {
                    "model_id": model_id,
                    "outer_fold": fold,
                    "scope": "all_heldout_rows",
                    "metric_name": name,
                    "value": value,
                    "n_rows": cor["n_rows"],
                    "evaluation_partition": EVALUATION_PARTITION,
                }
            )
        work = df.copy()
        work[SCORE_ABS_YHAT] = pd.to_numeric(work["y_pred"], errors="coerce").abs()
        ret_rows.extend(
            risk_coverage_fold(
                work,
                SCORE_ABS_YHAT,
                fold,
                experiment_id="magnitude_adjusted_risk_sensitivity",
                dataset="DS2",
                model_id=model_id,
                family="workhorse_sensitivity",
                score_role="magnitude_comparator",
            )
        )
    corr_fold = pd.DataFrame(corr_rows)
    corr_sum = _summarize(corr_fold.rename(columns={"outer_fold": "outer_fold_id"}).assign(scope="all_heldout_rows"))
    ret_fold = pd.DataFrame(ret_rows)
    mop50 = ret_fold[
        (ret_fold.scope.astype(str) == SCOPE)
        & (ret_fold.metric_name.astype(str) == "mop_rmse")
        & np.isclose(pd.to_numeric(ret_fold.retention_fraction, errors="coerce"), PRIMARY_RETENTION)
    ]
    aurc = []
    for fold, part in ret_fold[
        (ret_fold.scope.astype(str) == SCOPE) & (ret_fold.metric_name.astype(str) == "mop_rmse")
    ].groupby("outer_fold_id"):
        aurc.append(
            {
                "model_id": model_id,
                "outer_fold": fold,
                "scope": SCOPE,
                "metric_name": "retention_curve_auc_mop_rmse_abs_yhat",
                "value": curve_aurc(part.retention_fraction, part.value),
                "n_rows": int(part.n_rows_total.iloc[0]) if "n_rows_total" in part.columns else np.nan,
                "evaluation_partition": EVALUATION_PARTITION,
            }
        )
    ret50_sum = _summarize(
        mop50.rename(columns={"outer_fold_id": "outer_fold"}).assign(
            metric_name="mop_rmse_at_50pct_abs_yhat",
            scope=SCOPE,
        )
    )
    aurc_sum = _summarize(pd.DataFrame(aurc).assign(scope=SCOPE))
    summary = pd.concat([corr_sum, ret50_sum, aurc_sum], ignore_index=True)
    PAIR_ACQ_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "fold": PAIR_ACQ_DIR / "magnitude_adjusted_risk_fold.csv",
        "summary": PAIR_ACQ_DIR / "magnitude_adjusted_risk_summary.csv",
        "abs_yhat_retention_fold": PAIR_ACQ_DIR / "magnitude_abs_yhat_retention_fold.csv",
    }
    corr_fold.to_csv(paths["fold"], index=False)
    summary.to_csv(paths["summary"], index=False)
    ret_fold.to_csv(paths["abs_yhat_retention_fold"], index=False)
    return paths
