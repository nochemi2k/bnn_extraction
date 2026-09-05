"""Analysis task 6: does sigma track interaction strength after controlling for coverage?

Joins the frozen DS2_RDKIT_MLPENS5_HETERO fold scores to `data/processed/ds2.csv`.

Join-key note: the task spec says to join on `(ligand_smiles, metal_species_pair)`.
That pair is **not unique** in ds2.csv -- 11,127 of 70,390 (ligand, pair) groups have
more than one row (up to 10), because the same ligand-pair combination can appear
under different replicate conditions (temperature, ionic strength, electrolyte).
Joining on it as specified would fan out the sample: some held-out rows would be
duplicated and attached to an arbitrary alternate replicate's covariates, silently
corrupting every correlation/regression below. `ds2.csv`'s `pair_id` column is
`record_id_x + "__" + record_id_y`, which is exactly the fold-score `record_id`
(verified: 100% match, 1:1, no duplicates on either side). This module joins on
`record_id == pair_id` instead -- the evidently-intended precise key -- and reports
the (ligand_smiles, metal_species_pair) duplication as a diagnostic rather than
using it as the join.

Aggregation units, per the standing instruction: the fold-local unit
(`pair_fold_cell` for Task 6.2; row-level-within-one-fold for 6.1/6.3/6.4) is
primary throughout. Where Task 6.2's distribution is also computable pooling a
pair's rows across folds (as `within_pair_ordering.py` did), that reading is
reported too, secondary, for comparability.

Four analyses, run for response = sigma_total and again for response = |error|
(Task 6.4 asks for the same four analyses with |error| as the response):

  6.1  Partial Spearman rho(response, favoured_logK | support) and
       rho(response, support | favoured_logK), fold-local (all held-out rows in
       one fold), fold mean +/- SD across folds. `support` = n_training_ligands_for_pair.
  6.2  Within-pair (pair, fold cell) simple Spearman(response, favoured_logK):
       metal identity, coverage and stratum are removed by fixing the cell.
       Distribution over cells: mean/median/IQR/p10/p90/frac_positive.
  6.3  Multiple rank regression of response on [favoured_logK, d_electronegativity,
       d_covalent_radius, d_oxidation_state, log1p(support)]; standardized
       (rank-then-z-score) coefficients, fold mean +/- SD, plus VIFs (computed once,
       response-independent -- collinearity is a property of the predictor set).

Writes `manuscripts/derived/tables/sigma_chemical_axes.csv`.

    python -m uq_remake.manuscript.sigma_chemical_axes
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from uq_remake.downstream.magnitude_adjusted_risk import partial_spearman, spearman
from uq_remake.downstream.protocol import FOLDS, PAIR_COLUMN
from uq_remake.manuscript.manuscript_common import (
    DS2_PATH,
    FOLD_SCORES_DIR,
    DERIVED_TABLES,
    MIN_CELL_DISTINCT,
    MIN_CELL_N,
    concat_folds,
    fold_mean_sd_row,
    join_ds2_chemical_axes,
    load_all_folds,
    load_ds2_chemical_axes,
    with_abs_residual,
    write_provenance,
)

OUT = DERIVED_TABLES / "sigma_chemical_axes.csv"

PREDICTORS = ("favoured_logK", "d_electronegativity", "d_covalent_radius", "d_oxidation_state", "log1p_support")
RESPONSES = ("sigma_total", "abs_error")


def _nan() -> float:
    return float("nan")


# ---------------------------------------------------------------- 6.1 ---
def _fold_partial_corr(df: pd.DataFrame, response_col: str) -> tuple[float, float]:
    x = pd.to_numeric(df[response_col], errors="coerce")
    y = pd.to_numeric(df["favoured_logK"], errors="coerce")
    z = pd.to_numeric(df["n_training_ligands_for_pair"], errors="coerce")
    rho_resp_fav_given_support = partial_spearman(x, y, z)
    rho_resp_support_given_fav = partial_spearman(x, z, y)
    return rho_resp_fav_given_support, rho_resp_support_given_fav


def analysis_6_1(fold_frames: dict[str, pd.DataFrame], response_col: str) -> list[dict]:
    a_vals, b_vals = [], []
    for fold in FOLDS:
        a, b = _fold_partial_corr(fold_frames[fold], response_col)
        a_vals.append(a)
        b_vals.append(b)
    fm_a = fold_mean_sd_row(a_vals)
    fm_b = fold_mean_sd_row(b_vals)
    return [
        {"analysis": "6.1_partial_corr_support_controlled", "response": response_col, "metric_name": "rho_response_favouredlogK_given_support", **fm_a},
        {"analysis": "6.1_partial_corr_support_controlled", "response": response_col, "metric_name": "rho_response_support_given_favouredlogK", **fm_b},
    ]


# ---------------------------------------------------------------- 6.2 ---
def _cell_two_var_rho(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    rows = []
    for pair, g in df.groupby(PAIR_COLUMN, sort=False):
        x = pd.to_numeric(g[x_col], errors="coerce")
        y = pd.to_numeric(g[y_col], errors="coerce")
        ok = x.notna() & y.notna()
        n_used = int(ok.sum())
        if n_used < MIN_CELL_N:
            status, rho = "insufficient_n", _nan()
        else:
            n_distinct = int(min(x[ok].nunique(), y[ok].nunique()))
            if n_distinct < MIN_CELL_DISTINCT:
                status, rho = "constant_or_low_distinct", _nan()
            else:
                rho = spearman(x[ok], y[ok])
                status = "ok" if np.isfinite(rho) else "undefined_zero_variance"
        rows.append({PAIR_COLUMN: pair, "n_rows_used": n_used, "rho": rho, "status": status})
    return pd.DataFrame(rows)


def _distribution_stats(vals: pd.Series) -> dict:
    v = vals.to_numpy(dtype=float)
    return {
        "mean_rho": float(np.mean(v)) if len(v) else _nan(),
        "median_rho": float(np.median(v)) if len(v) else _nan(),
        "iqr_low_p25": float(np.percentile(v, 25)) if len(v) else _nan(),
        "iqr_high_p75": float(np.percentile(v, 75)) if len(v) else _nan(),
        "p10_rho": float(np.percentile(v, 10)) if len(v) else _nan(),
        "p90_rho": float(np.percentile(v, 90)) if len(v) else _nan(),
        "frac_positive": float(np.mean(v > 0)) if len(v) else _nan(),
        "n_cells_eligible": int(len(v)),
    }


def analysis_6_2(fold_frames: dict[str, pd.DataFrame], response_col: str) -> list[dict]:
    rows: list[dict] = []

    # Primary: pair_fold_cell -- distribution over every (pair, fold) cell, pooled
    # across the 5 folds' cell-level rho values (each rho itself computed from a
    # single fold's own held-out rows for that pair -- genuinely fold-local).
    cell_frames = []
    for fold, df in fold_frames.items():
        c = _cell_two_var_rho(df, response_col, "favoured_logK")
        c["outer_fold_id"] = fold
        cell_frames.append(c)
    cells = pd.concat(cell_frames, ignore_index=True)
    ok_cells = cells.loc[cells.status == "ok", "rho"]
    dist = _distribution_stats(ok_cells)
    rows.append(
        {
            "analysis": "6.2_within_pair_corr",
            "response": response_col,
            "metric_name": "rho_response_favouredlogK",
            "unit": "pair_fold_cell",
            **dist,
        }
    )

    # Secondary: pair pooled across folds (comparable to within_pair_ordering.py).
    all_df = concat_folds(fold_frames)
    pooled = _cell_two_var_rho(all_df, response_col, "favoured_logK")
    ok_pooled = pooled.loc[pooled.status == "ok", "rho"]
    dist_pooled = _distribution_stats(ok_pooled)
    rows.append(
        {
            "analysis": "6.2_within_pair_corr",
            "response": response_col,
            "metric_name": "rho_response_favouredlogK",
            "unit": "pair_pooled_across_folds",
            **dist_pooled,
        }
    )
    return rows


# ---------------------------------------------------------------- 6.3 ---
def _rank_standardize(s: pd.Series) -> np.ndarray:
    r = rankdata(s.to_numpy(dtype=float), method="average")
    sd = r.std(ddof=1)
    return (r - r.mean()) / sd if sd > 0 else np.zeros_like(r)


def _ols_beta(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    Xd = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return coef[1:]


def _vif(X: np.ndarray) -> list[float]:
    out = []
    for j in range(X.shape[1]):
        y = X[:, j]
        others = np.delete(X, j, axis=1)
        Xd = np.column_stack([np.ones(len(others)), others])
        coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        pred = Xd @ coef
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        out.append(float(1.0 / (1.0 - r2)) if np.isfinite(r2) and r2 < 1.0 else float("inf"))
    return out


def _fold_design(df: pd.DataFrame, response_col: str | None) -> tuple[np.ndarray, np.ndarray | None]:
    cols = list(PREDICTORS) + ([response_col] if response_col else [])
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    X = np.column_stack([_rank_standardize(sub[p]) for p in PREDICTORS])
    y = _rank_standardize(sub[response_col]) if response_col else None
    return X, y


def analysis_6_3(fold_frames: dict[str, pd.DataFrame], response_col: str, *, with_vif: bool) -> list[dict]:
    coef_by_pred: dict[str, list[float]] = {p: [] for p in PREDICTORS}
    vif_by_pred: dict[str, list[float]] = {p: [] for p in PREDICTORS}
    for fold in FOLDS:
        X, y = _fold_design(fold_frames[fold], response_col)
        beta = _ols_beta(X, y)
        for p, b in zip(PREDICTORS, beta):
            coef_by_pred[p].append(float(b))
        if with_vif:
            vifs = _vif(X)
            for p, v in zip(PREDICTORS, vifs):
                vif_by_pred[p].append(float(v))
    rows = []
    for p in PREDICTORS:
        fm = fold_mean_sd_row(coef_by_pred[p])
        rows.append({"analysis": "6.3_multiple_rank_regression", "response": response_col, "metric_name": f"standardized_coef_{p}", **fm})
    if with_vif:
        for p in PREDICTORS:
            fm = fold_mean_sd_row(vif_by_pred[p])
            rows.append({"analysis": "6.3_multiple_rank_regression", "response": "n/a_predictor_only", "metric_name": f"vif_{p}", **fm})
    return rows


def main() -> dict:
    fold_frames_raw = load_all_folds()
    ds2_derived, join_diag = load_ds2_chemical_axes()
    print(f"sigma_chemical_axes: join diagnostics {join_diag}", flush=True)

    fold_frames = {fold: join_ds2_chemical_axes(with_abs_residual(df), ds2_derived) for fold, df in fold_frames_raw.items()}

    rows: list[dict] = []
    for i, response in enumerate(RESPONSES):
        rows.extend(analysis_6_1(fold_frames, response))
        rows.extend(analysis_6_2(fold_frames, response))
        rows.extend(analysis_6_3(fold_frames, response, with_vif=(i == 0)))

    out = pd.DataFrame(rows)
    out["source"] = "results/downstream_cv/ad_retention/fold_scores/DS2_RDKIT_MLPENS5_HETERO/fold{0..4}.csv + data/processed/ds2.csv"
    DERIVED_TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    write_provenance(
        OUT,
        module="sigma_chemical_axes.main",
        inputs=[FOLD_SCORES_DIR / f"{f}.csv" for f in FOLDS] + [DS2_PATH],
        notes={"join_diagnostics": join_diag, "predictors": list(PREDICTORS), "responses": list(RESPONSES)},
    )
    print(f"wrote {OUT} ({len(out)} rows)", flush=True)
    return {"table": out, "join_diag": join_diag}


if __name__ == "__main__":
    main()
