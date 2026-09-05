"""JACS Task 1 & 2: within-pair risk ordering and within-pair retention curves.

Task 1. For every candidate risk score, Spearman(score, |y_true - y_pred|)
computed on one pair's held-out rows pooled across all five folds (see
`jacs_common` module docstring for why pooling-across-folds, not per-(pair,
fold) cells, is the unit that reproduces the prototype numbers). Aggregated
two ways, both kept:

  (a) "distribution_over_pairs" -- one Spearman rho per eligible pair, then
      mean/median/IQR/p10/p90/frac_positive over that distribution. This is
      the reading that reproduces the prototype figures in
      `manuscripts/jacs/analysis/chemical_sanity_checks.json`.
  (b) "fold_mean_plus_sd" -- true (pair, fold) cells (this fold's held-out
      rows for this pair only), pair-equal mean within each fold, then
      fold mean +/- SD (ddof=1) across the 5 folds. Genuinely fold-local.

`pair_density_risk` is a training-set support count: exactly constant within
every (pair, fold) cell (asserted below over all 7,574 cells). Pooled across
folds it can display up to 5 distinct values purely because different folds
have different training support for the same pair -- that is fold-membership
confounding, not ligand-level ranking signal, so it is reported as undefined
in aggregation (a) too rather than left to the generic distinct-value filter.

Task 2. Within each (pair, fold) cell: rank ligands by score, retain the
lowest-risk fraction r on the canonical grid, RMSE on the retained subset,
mean over cells (pair-equal) within a fold, then fold mean +/- SD across
folds. Includes a random-retention baseline (20 permutations, seed 0) and a
y_pred-magnitude comparator (score = -|y_pred|).

Reads only frozen fold scores under
`results/downstream_cv/ad_retention/fold_scores/DS2_RDKIT_MLPENS5_HETERO/`.
Writes `manuscripts/jacs/tables/within_pair_score_residual.csv` and
`manuscripts/jacs/tables/within_pair_retention.csv` plus provenance sidecars.

    python -m uq_remake.manuscript.within_pair_ordering
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from uq_remake.downstream.analyses import deterministic_rank, k_at_retention
from uq_remake.downstream.protocol import FOLDS, PAIR_COLUMN, RANDOM_N_REPEATS, RANDOM_SEED, RETENTION_FRACTIONS
from uq_remake.manuscript.jacs_common import (
    CANDIDATE_SCORES,
    CONSTANT_WITHIN_CELL_SCORES,
    FOLD_SCORES_DIR,
    JACS_TABLES,
    MIN_CELL_DISTINCT,
    MIN_CELL_N,
    WORKHORSE,
    assert_pair_density_constant_within_cells,
    concat_folds,
    fold_mean_sd_row,
    load_all_folds,
    with_abs_residual,
    write_provenance,
)

OUT_ORDERING = JACS_TABLES / "within_pair_score_residual.csv"
OUT_RETENTION = JACS_TABLES / "within_pair_retention.csv"
MAG_SCORE_NAME = "abs_yhat_magnitude_comparator"
RANDOM_SCORE_NAME = "random_retention"


def _nan() -> float:
    return float("nan")


def _spearman(x: pd.Series, y: pd.Series) -> float:
    rho, _ = spearmanr(x, y)
    return float(rho) if np.isfinite(rho) else _nan()


# ------------------------------------------------------------------ Task 1 ---
def _pair_pooled_rho(all_df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """One row per distinct species pair, pooling that pair's rows across all folds."""
    rows = []
    forced_undefined = score_col in CONSTANT_WITHIN_CELL_SCORES
    for pair, g in all_df.groupby(PAIR_COLUMN, sort=False):
        n_rows = int(len(g))
        x = pd.to_numeric(g[score_col], errors="coerce")
        y = g["_abs_residual"]
        ok = x.notna() & y.notna()
        n_used = int(ok.sum())
        n_distinct = int(x[ok].nunique()) if n_used else 0
        if n_used < MIN_CELL_N:
            status = "insufficient_n"
            rho = _nan()
        elif forced_undefined:
            status = "constant_within_cell_zero_resolving_power"
            rho = _nan()
        elif n_distinct < MIN_CELL_DISTINCT:
            status = "constant_or_low_distinct"
            rho = _nan()
        else:
            rho = _spearman(x[ok], y[ok])
            status = "ok" if np.isfinite(rho) else "undefined_zero_variance"
        rows.append(
            {
                PAIR_COLUMN: pair,
                "score_name": score_col,
                "n_rows_total": n_rows,
                "n_rows_used": n_used,
                "n_distinct": n_distinct,
                "rho": rho,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def _cell_rho(fold_frames: dict[str, pd.DataFrame], score_col: str) -> pd.DataFrame:
    """One row per (pair, fold) cell -- this fold's held-out rows for this pair only."""
    rows = []
    forced_undefined = score_col in CONSTANT_WITHIN_CELL_SCORES
    for fold, df in fold_frames.items():
        work = with_abs_residual(df)
        for pair, g in work.groupby(PAIR_COLUMN, sort=False):
            n_rows = int(len(g))
            x = pd.to_numeric(g[score_col], errors="coerce")
            y = g["_abs_residual"]
            ok = x.notna() & y.notna()
            n_used = int(ok.sum())
            n_distinct = int(x[ok].nunique()) if n_used else 0
            if n_used < MIN_CELL_N:
                status = "insufficient_n"
                rho = _nan()
            elif forced_undefined or n_distinct < MIN_CELL_DISTINCT:
                status = "constant_within_cell_zero_resolving_power" if forced_undefined else "constant_or_low_distinct"
                rho = _nan()
            else:
                rho = _spearman(x[ok], y[ok])
                status = "ok" if np.isfinite(rho) else "undefined_zero_variance"
            rows.append(
                {
                    "outer_fold_id": fold,
                    PAIR_COLUMN: pair,
                    "score_name": score_col,
                    "n_rows_total": n_rows,
                    "n_rows_used": n_used,
                    "n_distinct": n_distinct,
                    "rho": rho,
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def _distribution_stats(ok: pd.Series) -> dict:
    vals = ok.to_numpy(dtype=float)
    return {
        "mean": float(np.mean(vals)) if len(vals) else _nan(),
        "median": float(np.median(vals)) if len(vals) else _nan(),
        "iqr_low_p25": float(np.percentile(vals, 25)) if len(vals) else _nan(),
        "iqr_high_p75": float(np.percentile(vals, 75)) if len(vals) else _nan(),
        "p10": float(np.percentile(vals, 10)) if len(vals) else _nan(),
        "p90": float(np.percentile(vals, 90)) if len(vals) else _nan(),
        "frac_positive": float(np.mean(vals > 0)) if len(vals) else _nan(),
        "n": int(len(vals)),
    }


def within_pair_score_residual(fold_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    all_df = with_abs_residual(concat_folds(fold_frames))
    out_rows: list[dict] = []
    for score in CANDIDATE_SCORES:
        pooled = _pair_pooled_rho(all_df, score)
        n_pairs_total = int(len(pooled))
        n_excl_n = int((pooled.status == "insufficient_n").sum())
        n_excl_distinct = int(pooled.status.isin(["constant_or_low_distinct", "constant_within_cell_zero_resolving_power"]).sum())
        ok_pooled = pooled.loc[pooled.status == "ok", "rho"]
        dist = _distribution_stats(ok_pooled)
        out_rows.append(
            {
                "score_name": score,
                "aggregation": "distribution_over_pairs",
                "unit": "pair_pooled_across_folds",
                "n_pairs_total": n_pairs_total,
                "n_pairs_excluded_insufficient_n": n_excl_n,
                "n_pairs_excluded_constant_or_low_distinct": n_excl_distinct,
                "n_pairs_eligible": dist["n"],
                "mean_rho": dist["mean"],
                "median_rho": dist["median"],
                "iqr_low_p25": dist["iqr_low_p25"],
                "iqr_high_p75": dist["iqr_high_p75"],
                "p10_rho": dist["p10"],
                "p90_rho": dist["p90"],
                "frac_positive": dist["frac_positive"],
            }
        )

        cells = _cell_rho(fold_frames, score)
        n_cells_total = int(len(cells))
        fold_means = []
        for fold in FOLDS:
            fc = cells[(cells.outer_fold_id == fold) & (cells.status == "ok")]
            fold_means.append(float(fc.rho.mean()) if len(fc) else _nan())
        fm = fold_mean_sd_row(fold_means)
        out_rows.append(
            {
                "score_name": score,
                "aggregation": "fold_mean_plus_sd",
                "unit": "pair_fold_cell",
                "n_pairs_total": "",
                "n_pairs_excluded_insufficient_n": "",
                "n_pairs_excluded_constant_or_low_distinct": "",
                "n_pairs_eligible": int((cells.status == "ok").sum()),
                "mean_rho": fm["fold_mean"],
                "median_rho": "",
                "iqr_low_p25": "",
                "iqr_high_p75": "",
                "p10_rho": "",
                "p90_rho": "",
                "frac_positive": "",
                "fold_sd": fm["fold_sd"],
                "n_valid_folds": fm["n_valid_folds"],
                "n_cells_total": n_cells_total,
            }
        )
    return pd.DataFrame(out_rows)


# ------------------------------------------------------------------ Task 2 ---
def _cell_retained_rmse(g: pd.DataFrame, score_col: str, higher_is_risk: bool) -> dict[float, float]:
    ranked = deterministic_rank(g, score_col, higher_is_risk=higher_is_risk)
    n = len(ranked)
    out = {}
    for ret in RETENTION_FRACTIONS:
        k = k_at_retention(n, ret)
        part = ranked.iloc[:k]
        if len(part) == 0:
            out[ret] = _nan()
            continue
        yt = pd.to_numeric(part["y_true"], errors="coerce").to_numpy(dtype=float)
        yp = pd.to_numeric(part["y_pred"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(yt) & np.isfinite(yp)
        out[ret] = float(np.sqrt(np.mean((yt[ok] - yp[ok]) ** 2))) if ok.any() else _nan()
    return out


def _fold_retention_for_score(df: pd.DataFrame, score_col: str, higher_is_risk: bool) -> dict[float, tuple[float, int]]:
    """Pair-equal mean retained-RMSE across (pair, fold) cells for one fold, one score."""
    per_ret: dict[float, list[float]] = {ret: [] for ret in RETENTION_FRACTIONS}
    for _pair, g in df.groupby(PAIR_COLUMN, sort=False):
        if len(g) < MIN_CELL_N:
            continue
        curve = _cell_retained_rmse(g, score_col, higher_is_risk)
        for ret, val in curve.items():
            if np.isfinite(val):
                per_ret[ret].append(val)
    return {ret: (float(np.mean(v)) if v else _nan(), len(v)) for ret, v in per_ret.items()}


def _fold_retention_random(df: pd.DataFrame, fold_index: int, n_repeats: int = RANDOM_N_REPEATS, seed: int = RANDOM_SEED) -> dict[float, tuple[float, int]]:
    fold_seed = int(seed) + fold_index * 1009
    rng = np.random.default_rng(fold_seed)
    per_ret: dict[float, list[float]] = {ret: [] for ret in RETENTION_FRACTIONS}
    for _pair, g in df.groupby(PAIR_COLUMN, sort=False):
        n = len(g)
        if n < MIN_CELL_N:
            continue
        yt_all = pd.to_numeric(g["y_true"], errors="coerce").to_numpy(dtype=float)
        yp_all = pd.to_numeric(g["y_pred"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(yt_all) & np.isfinite(yp_all)
        idx = np.flatnonzero(finite)
        n_finite = len(idx)
        if n_finite == 0:
            continue
        cell_ret_vals: dict[float, list[float]] = {ret: [] for ret in RETENTION_FRACTIONS}
        for _rep in range(int(n_repeats)):
            perm = rng.permutation(n_finite)
            order = idx[perm]
            for ret in RETENTION_FRACTIONS:
                k = k_at_retention(n_finite, ret)
                if k == 0:
                    continue
                sel = order[:k]
                rmse = float(np.sqrt(np.mean((yt_all[sel] - yp_all[sel]) ** 2)))
                cell_ret_vals[ret].append(rmse)
        for ret, vals in cell_ret_vals.items():
            if vals:
                per_ret[ret].append(float(np.mean(vals)))
    return {ret: (float(np.mean(v)) if v else _nan(), len(v)) for ret, v in per_ret.items()}


def within_pair_retention(fold_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    scored_frames = {fold: with_abs_residual(df).assign(**{MAG_SCORE_NAME: -pd.to_numeric(df["y_pred"], errors="coerce").abs()}) for fold, df in fold_frames.items()}
    scores_to_run = list(CANDIDATE_SCORES) + [MAG_SCORE_NAME]

    out_rows: list[dict] = []
    for score in scores_to_run:
        per_fold_curves: dict[str, dict[float, tuple[float, int]]] = {}
        for fold, df in scored_frames.items():
            per_fold_curves[fold] = _fold_retention_for_score(df, score, higher_is_risk=True)
        for ret in RETENTION_FRACTIONS:
            fold_vals = [per_fold_curves[fold][ret][0] for fold in FOLDS]
            n_cells = [per_fold_curves[fold][ret][1] for fold in FOLDS]
            fm = fold_mean_sd_row(fold_vals)
            out_rows.append(
                {
                    "score_name": score,
                    "baseline": "score",
                    "retention_fraction": ret,
                    "metric_name": "mop_rmse_within_pair",
                    "fold_mean": fm["fold_mean"],
                    "fold_sd": fm["fold_sd"],
                    "n_valid_folds": fm["n_valid_folds"],
                    "n_cells_total": int(sum(n_cells)),
                }
            )

    per_fold_random: dict[str, dict[float, tuple[float, int]]] = {}
    for i, fold in enumerate(FOLDS):
        per_fold_random[fold] = _fold_retention_random(scored_frames[fold], i)
    for ret in RETENTION_FRACTIONS:
        fold_vals = [per_fold_random[fold][ret][0] for fold in FOLDS]
        n_cells = [per_fold_random[fold][ret][1] for fold in FOLDS]
        fm = fold_mean_sd_row(fold_vals)
        out_rows.append(
            {
                "score_name": RANDOM_SCORE_NAME,
                "baseline": "random",
                "retention_fraction": ret,
                "metric_name": "mop_rmse_within_pair",
                "fold_mean": fm["fold_mean"],
                "fold_sd": fm["fold_sd"],
                "n_valid_folds": fm["n_valid_folds"],
                "n_cells_total": int(sum(n_cells)),
                "n_repeats": RANDOM_N_REPEATS,
                "random_seed": RANDOM_SEED,
            }
        )
    return pd.DataFrame(out_rows)


def main() -> dict:
    fold_frames = load_all_folds()
    cell_check = assert_pair_density_constant_within_cells(fold_frames)
    print(f"pair_density_risk constant in all {cell_check['n_cells_total']} (pair, fold) cells", flush=True)

    ordering = within_pair_score_residual(fold_frames)
    ordering["source"] = f"results/downstream_cv/ad_retention/fold_scores/{WORKHORSE}/fold{{0..4}}.csv"
    JACS_TABLES.mkdir(parents=True, exist_ok=True)
    ordering.to_csv(OUT_ORDERING, index=False)
    write_provenance(
        OUT_ORDERING,
        module="within_pair_ordering.within_pair_score_residual",
        inputs=[FOLD_SCORES_DIR / f"{f}.csv" for f in FOLDS],
        notes={"pair_density_risk_cell_check": cell_check},
    )
    print(f"wrote {OUT_ORDERING} ({len(ordering)} rows)", flush=True)

    retention = within_pair_retention(fold_frames)
    retention["source"] = f"results/downstream_cv/ad_retention/fold_scores/{WORKHORSE}/fold{{0..4}}.csv"
    retention.to_csv(OUT_RETENTION, index=False)
    write_provenance(
        OUT_RETENTION,
        module="within_pair_ordering.within_pair_retention",
        inputs=[FOLD_SCORES_DIR / f"{f}.csv" for f in FOLDS],
    )
    print(f"wrote {OUT_RETENTION} ({len(retention)} rows)", flush=True)
    return {"ordering": ordering, "retention": retention, "cell_check": cell_check}


if __name__ == "__main__":
    main()
