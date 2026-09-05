"""JACS Task 3: scale-free error companions for every retention/error claim.

`e_norm = |y_true - y_pred| / SD_p(y_true)`, where `SD_p` is the standard
deviation (ddof=1) of the observed target for one species pair, pooled across
that pair's held-out rows from all five folds -- the same pooling as
`within_pair_ordering`'s Task 1 headline (see that module's docstring; this
is the reading that reproduces the prototype ratios below). Pairs with
pooled n < 10 or SD_p == 0 are dropped from every scale-free statistic; the
count dropped is reported.

Reproduces, from `manuscripts/jacs/analysis/chemical_sanity_checks.json`:
  raw |error| ratio (non-REE / REE)          ~ 3.734
  scale-free e_norm ratio (non-REE / REE)    ~ 0.877   (the reversal is the point)
  pooled rho(sigma_total, e_norm)            ~ +0.271
  pooled rho(sigma_total / SD_p, e_norm)     ~ +0.410

Writes `manuscripts/jacs/tables/scale_free_error.csv` (REE comparison +
pooled diagnostics + retention-on-e_norm for the primary scores) and
`manuscripts/jacs/tables/per_pair_difficulty.csv` (one row per pair with
n >= 10: SD_p, mean |error|, mean e_norm, mean sigma_total).

    python -m uq_remake.manuscript.scale_free_error
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from uq_remake.downstream.analyses import deterministic_rank, k_at_retention
from uq_remake.downstream.protocol import FOLDS, PAIR_COLUMN, RETENTION_FRACTIONS
from uq_remake.manuscript.jacs_common import (
    FOLD_SCORES_DIR,
    JACS_TABLES,
    MIN_CELL_N,
    MIN_PAIR_N_SCALE_FREE,
    PRIMARY_SCORES,
    WORKHORSE,
    concat_folds,
    fold_mean_sd_row,
    load_all_folds,
    with_abs_residual,
    write_provenance,
)

OUT_SCALE_FREE = JACS_TABLES / "scale_free_error.csv"
OUT_PER_PAIR = JACS_TABLES / "per_pair_difficulty.csv"


def _nan() -> float:
    return float("nan")


def _spearman(x, y) -> float:
    rho, _ = spearmanr(x, y)
    return float(rho) if np.isfinite(rho) else _nan()


def add_pair_pooled_sdp(all_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Attach SD_p, n_pair, e_norm; drop pairs with n<10 or SD_p==0."""
    work = with_abs_residual(all_df)
    n_pair = work.groupby(PAIR_COLUMN)["y_true"].transform("size")
    sdp = work.groupby(PAIR_COLUMN)["y_true"].transform(lambda s: pd.to_numeric(s, errors="coerce").std(ddof=1))
    work["_n_pair"] = n_pair
    work["_sd_p"] = sdp
    eligible = (work["_n_pair"] >= MIN_PAIR_N_SCALE_FREE) & work["_sd_p"].notna() & (work["_sd_p"] > 0)
    n_total_rows = int(len(work))
    n_dropped = int((~eligible).sum())
    n_pairs_total = int(work[PAIR_COLUMN].nunique())
    used = work.loc[eligible].copy()
    used["e_norm"] = used["_abs_residual"] / used["_sd_p"]
    diag = {
        "n_rows_total": n_total_rows,
        "n_rows_dropped": n_dropped,
        "n_rows_used": int(len(used)),
        "n_pairs_total": n_pairs_total,
        "n_pairs_used": int(used[PAIR_COLUMN].nunique()),
        "min_pair_n": MIN_PAIR_N_SCALE_FREE,
    }
    return used, diag


def ree_comparison(used: pd.DataFrame) -> pd.DataFrame:
    ree = used[used["is_ree_pair"] == True]  # noqa: E712
    nonree = used[used["is_ree_pair"] == False]  # noqa: E712
    raw_ree = float(ree["_abs_residual"].mean())
    raw_nonree = float(nonree["_abs_residual"].mean())
    sf_ree = float(ree["e_norm"].mean())
    sf_nonree = float(nonree["e_norm"].mean())
    rows = [
        {"metric_name": "mean_abs_error_raw", "group": "ree", "value": raw_ree, "n_rows": int(len(ree))},
        {"metric_name": "mean_abs_error_raw", "group": "non_ree", "value": raw_nonree, "n_rows": int(len(nonree))},
        {"metric_name": "mean_abs_error_raw", "group": "ratio_nonree_over_ree", "value": raw_nonree / raw_ree if raw_ree else _nan(), "n_rows": ""},
        {"metric_name": "mean_e_norm_scale_free", "group": "ree", "value": sf_ree, "n_rows": int(len(ree))},
        {"metric_name": "mean_e_norm_scale_free", "group": "non_ree", "value": sf_nonree, "n_rows": int(len(nonree))},
        {"metric_name": "mean_e_norm_scale_free", "group": "ratio_nonree_over_ree", "value": sf_nonree / sf_ree if sf_ree else _nan(), "n_rows": ""},
    ]
    return pd.DataFrame(rows)


def pooled_sigma_enorm_correlations(used: pd.DataFrame) -> pd.DataFrame:
    sigma = pd.to_numeric(used["sigma_total"], errors="coerce")
    sigma_over_sd = sigma / used["_sd_p"]
    e_norm = used["e_norm"]
    rows = [
        {
            "metric_name": "pooled_rho_sigma_total_vs_e_norm",
            "value": _spearman(sigma, e_norm),
            "n_rows": int(len(used)),
            "role": "si_diagnostic_only",
        },
        {
            "metric_name": "pooled_rho_sigma_over_sdp_vs_e_norm",
            "value": _spearman(sigma_over_sd, e_norm),
            "n_rows": int(len(used)),
            "role": "si_diagnostic_only",
        },
    ]
    return pd.DataFrame(rows)


def per_pair_difficulty(used: pd.DataFrame) -> pd.DataFrame:
    grouped = used.groupby(PAIR_COLUMN).agg(
        n_rows=("y_true", "size"),
        sd_p=("_sd_p", "first"),
        mean_abs_error=("_abs_residual", "mean"),
        mean_e_norm=("e_norm", "mean"),
        mean_sigma_total=("sigma_total", "mean"),
        is_ree_pair=("is_ree_pair", "first"),
    ).reset_index()
    return grouped.sort_values(PAIR_COLUMN).reset_index(drop=True)


def _retention_on_enorm(fold_frames_enorm: dict[str, pd.DataFrame], score_col: str) -> list[dict]:
    """Same mechanics as within_pair_ordering's Task 2, but the retained-subset
    metric is mean(e_norm) rather than RMSE(y_true, y_pred) -- e_norm is already
    a normalized absolute error, not a signed residual to square."""
    out = []
    per_fold: dict[str, dict[float, tuple[float, int]]] = {}
    for fold, df in fold_frames_enorm.items():
        per_ret: dict[float, list[float]] = {ret: [] for ret in RETENTION_FRACTIONS}
        for _pair, g in df.groupby(PAIR_COLUMN, sort=False):
            if len(g) < MIN_CELL_N:
                continue
            ranked = deterministic_rank(g, score_col, higher_is_risk=True)
            n = len(ranked)
            for ret in RETENTION_FRACTIONS:
                k = k_at_retention(n, ret)
                if k == 0:
                    continue
                part = ranked.iloc[:k]
                val = pd.to_numeric(part["e_norm"], errors="coerce").mean()
                if np.isfinite(val):
                    per_ret[ret].append(float(val))
        per_fold[fold] = {ret: (float(np.mean(v)) if v else _nan(), len(v)) for ret, v in per_ret.items()}
    for ret in RETENTION_FRACTIONS:
        fold_vals = [per_fold[fold][ret][0] for fold in FOLDS]
        n_cells = [per_fold[fold][ret][1] for fold in FOLDS]
        fm = fold_mean_sd_row(fold_vals)
        out.append(
            {
                "score_name": score_col,
                "retention_fraction": ret,
                "metric_name": "mean_e_norm_within_pair",
                "fold_mean": fm["fold_mean"],
                "fold_sd": fm["fold_sd"],
                "n_valid_folds": fm["n_valid_folds"],
                "n_cells_total": int(sum(n_cells)),
            }
        )
    return out


def main() -> dict:
    fold_frames = load_all_folds()
    all_df = concat_folds(fold_frames)
    used, diag = add_pair_pooled_sdp(all_df)
    print(f"scale_free_error: {diag}", flush=True)

    ree_rows = ree_comparison(used)
    corr_rows = pooled_sigma_enorm_correlations(used)

    # e_norm-restricted per-fold frames for the retention-curve reuse of the
    # per-(pair, fold)-cell ranking machinery.
    used_keys = set(zip(used[PAIR_COLUMN], used["outer_fold_id"], used["record_id"]))
    fold_frames_enorm = {}
    for fold, df in fold_frames.items():
        sub = df.copy()
        key = list(zip(sub[PAIR_COLUMN], sub["outer_fold_id"], sub["record_id"]))
        mask = [k in used_keys for k in key]
        sub = sub.loc[mask].copy()
        merged = used[[PAIR_COLUMN, "outer_fold_id", "record_id", "e_norm"]]
        sub = sub.merge(merged, on=[PAIR_COLUMN, "outer_fold_id", "record_id"], how="inner")
        fold_frames_enorm[fold] = sub

    retention_rows: list[dict] = []
    for score in PRIMARY_SCORES:
        retention_rows.extend(_retention_on_enorm(fold_frames_enorm, score))
    retention_df = pd.DataFrame(retention_rows)
    retention_df["metric_name"] = retention_df["metric_name"].where(retention_df["metric_name"].notna(), "mean_e_norm_within_pair")

    ree_rows["table_section"] = "ree_comparison"
    corr_rows["table_section"] = "pooled_correlations"
    retention_df["table_section"] = "retention_on_e_norm"
    combined = pd.concat([ree_rows, corr_rows, retention_df], ignore_index=True)
    combined["source"] = f"results/downstream_cv/ad_retention/fold_scores/{WORKHORSE}/fold{{0..4}}.csv"
    combined["min_pair_n"] = MIN_PAIR_N_SCALE_FREE
    combined["n_rows_dropped_insufficient_n_or_zero_sd"] = diag["n_rows_dropped"]

    JACS_TABLES.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_SCALE_FREE, index=False)
    write_provenance(
        OUT_SCALE_FREE,
        module="scale_free_error.main",
        inputs=[FOLD_SCORES_DIR / f"{f}.csv" for f in FOLDS],
        notes={"e_norm_eligibility": diag},
    )
    print(f"wrote {OUT_SCALE_FREE} ({len(combined)} rows)", flush=True)

    per_pair = per_pair_difficulty(used)
    per_pair["source"] = f"results/downstream_cv/ad_retention/fold_scores/{WORKHORSE}/fold{{0..4}}.csv"
    per_pair.to_csv(OUT_PER_PAIR, index=False)
    write_provenance(
        OUT_PER_PAIR,
        module="scale_free_error.per_pair_difficulty",
        inputs=[FOLD_SCORES_DIR / f"{f}.csv" for f in FOLDS],
    )
    print(f"wrote {OUT_PER_PAIR} ({len(per_pair)} rows)", flush=True)
    return {"combined": combined, "per_pair": per_pair, "diag": diag}


if __name__ == "__main__":
    main()
