"""JACS Task 7 & 8: the coverage-override grid, productionised, and the chemical
composition of its flagged cells.

Support-quartile x sigma-quartile 4x4 grid of mean |error|. Quartiles are
assigned **within each fold** (qcut, 4 bins, on that fold's own row-level
distribution of `n_training_ligands_for_pair` ["support"] and `sigma_total`),
so no fold's boundary depends on another fold's held-out rows. Two readings
of every cell/marginal are reported, fold-local primary per the standing
instruction:

  * `fold_mean_plus_sd` -- per-fold cell mean, then fold mean +/- SD (ddof=1)
    across the 5 folds. This is the headline.
  * `pooled_across_folds` -- all rows pooled after per-fold quartile
    assignment. This is the reading the task's prototype figures were
    computed on (labelled "fold-pooled" in the spec); kept as the secondary,
    cross-check unit. **Do not treat it as the headline** -- it is reported
    only because the spec's reproduction targets are stated in this unit.

Task 7 also emits a pairs-clustered bootstrap 95% interval (1000 resamples,
seed 42, resampling species pairs with replacement -- mirrors
`metrics.bootstrap_fold.ligand_cluster_bootstrap` but clustered by
`metal_species_pair`, and deliberately run on the pooled-across-folds cell
population, since that is the population the interval is meant to describe)
for the two cells of interest: sparse-support/low-sigma (support Q1, sigma
Q1) and dense-support/high-sigma (support Q4, sigma Q4).

Task 8 profiles the chemical composition of the two DENSE-support cells at
opposite sigma extremes -- dense-support/high-sigma (support Q4, sigma Q4)
and dense-support/low-sigma (support Q4, sigma Q1) -- to test whether sigma
still separates high- from low-error rows once coverage is held fixed, and
reports the species-pair overlap between the two cells (needed for the
"sigma partitions within pairs" claim).

Reads: frozen `DS2_RDKIT_MLPENS5_HETERO` fold scores, joined to
`data/processed/ds2.csv` via `jacs_common.load_ds2_chemical_axes` (same
join-key fix as `sigma_chemical_axes.py` -- see that module's docstring).

Writes `manuscripts/jacs/tables/coverage_override_grid.csv` and
`manuscripts/jacs/tables/flagged_region_composition.csv`.

    python -m uq_remake.manuscript.coverage_override_grid
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from uq_remake.downstream.protocol import FOLDS, PAIR_COLUMN
from uq_remake.manuscript.jacs_common import (
    DS2_PATH,
    FOLD_SCORES_DIR,
    JACS_TABLES,
    concat_folds,
    fold_mean_sd_row,
    join_ds2_chemical_axes,
    load_all_folds,
    load_ds2_chemical_axes,
    with_abs_residual,
    write_provenance,
)

OUT_GRID = JACS_TABLES / "coverage_override_grid.csv"
OUT_COMPOSITION = JACS_TABLES / "flagged_region_composition.csv"

N_QUANTILES = 4
QUARTILE_LABELS = [1, 2, 3, 4]
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42
TOP_N_PAIRS = 15

CELLS_OF_INTEREST_TASK7 = {
    "sparse_support_low_sigma": (1, 1),
    "dense_support_high_sigma": (4, 4),
}
CELLS_OF_INTEREST_TASK8 = {
    "dense_support_high_sigma": (4, 4),
    "dense_support_low_sigma": (4, 1),
}


def _nan() -> float:
    return float("nan")


def _assign_quartiles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["support_quartile"] = pd.qcut(
        pd.to_numeric(out["n_training_ligands_for_pair"], errors="coerce"), N_QUANTILES, labels=QUARTILE_LABELS, duplicates="drop"
    ).astype("Int64")
    out["sigma_quartile"] = pd.qcut(
        pd.to_numeric(out["sigma_total"], errors="coerce"), N_QUANTILES, labels=QUARTILE_LABELS, duplicates="drop"
    ).astype("Int64")
    return out


def _cell_stats(df: pd.DataFrame, support_q: int, sigma_q: int) -> tuple[float, int]:
    sub = df[(df.support_quartile == support_q) & (df.sigma_quartile == sigma_q)]
    if len(sub) == 0:
        return _nan(), 0
    return float(sub["_abs_residual"].mean()), int(len(sub))


def _cluster_bootstrap_mean(df: pd.DataFrame, value_col: str, cluster_col: str = PAIR_COLUMN, n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> dict:
    """Percentile bootstrap 95% CI for the mean of `value_col`, resampling
    distinct `cluster_col` groups with replacement (not individual rows)."""
    rng = np.random.default_rng(seed)
    groups = df.groupby(cluster_col, sort=False)
    group_keys = list(groups.groups.keys())
    group_vals = [groups.get_group(k)[value_col].to_numpy(dtype=float) for k in group_keys]
    n_g = len(group_keys)
    point = float(df[value_col].mean())
    if n_g == 0:
        return {"point": _nan(), "ci_low": _nan(), "ci_high": _nan(), "n_boot": n_boot, "n_clusters": 0}
    boots = []
    for _ in range(int(n_boot)):
        chosen = rng.integers(0, n_g, size=n_g)
        sample = np.concatenate([group_vals[i] for i in chosen])
        if len(sample):
            boots.append(float(np.mean(sample)))
    if not boots:
        return {"point": point, "ci_low": _nan(), "ci_high": _nan(), "n_boot": n_boot, "n_clusters": n_g}
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    return {"point": point, "ci_low": lo, "ci_high": hi, "n_boot": n_boot, "n_clusters": n_g}


# ------------------------------------------------------------------ Task 7 ---
def build_grid(fold_frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    fold_level: list[dict] = []
    per_cell_fold_vals: dict[tuple[int, int], list[float]] = {(s, g): [] for s in QUARTILE_LABELS for g in QUARTILE_LABELS}
    supp_marg_fold_vals: dict[int, list[float]] = {s: [] for s in QUARTILE_LABELS}
    sig_marg_fold_vals: dict[int, list[float]] = {g: [] for g in QUARTILE_LABELS}

    for fold, df in fold_frames.items():
        fold_n = int(len(df))
        cell_n_sum = 0
        for s in QUARTILE_LABELS:
            for g in QUARTILE_LABELS:
                mean_val, n = _cell_stats(df, s, g)
                per_cell_fold_vals[(s, g)].append(mean_val)
                cell_n_sum += n
                fold_level.append({"outer_fold_id": fold, "support_quartile": s, "sigma_quartile": g, "mean_abs_error": mean_val, "n_rows": n})
        if cell_n_sum != fold_n:
            raise AssertionError(f"{fold}: cell row counts sum to {cell_n_sum}, expected {fold_n} (unassigned quartile rows)")
        for s in QUARTILE_LABELS:
            sub = df[df.support_quartile == s]
            supp_marg_fold_vals[s].append(float(sub["_abs_residual"].mean()) if len(sub) else _nan())
        for g in QUARTILE_LABELS:
            sub = df[df.sigma_quartile == g]
            sig_marg_fold_vals[g].append(float(sub["_abs_residual"].mean()) if len(sub) else _nan())

    # fold_frames already carry per-fold quartile assignments (done before this
    # function is called) and `_abs_residual` -- concatenating does NOT re-derive
    # quartiles globally, which would silently break the within-fold requirement.
    all_df = concat_folds(fold_frames)

    for s in QUARTILE_LABELS:
        for g in QUARTILE_LABELS:
            fm = fold_mean_sd_row(per_cell_fold_vals[(s, g)])
            pooled_mean, pooled_n = _cell_stats(all_df, s, g)
            rows.append(
                {
                    "table_section": "grid",
                    "support_quartile": s,
                    "sigma_quartile": g,
                    "aggregation": "fold_mean_plus_sd",
                    "mean_abs_error": fm["fold_mean"],
                    "sd": fm["fold_sd"],
                    "n_valid_folds": fm["n_valid_folds"],
                    "n_rows": "",
                }
            )
            rows.append(
                {
                    "table_section": "grid",
                    "support_quartile": s,
                    "sigma_quartile": g,
                    "aggregation": "pooled_across_folds",
                    "mean_abs_error": pooled_mean,
                    "sd": "",
                    "n_valid_folds": "",
                    "n_rows": pooled_n,
                }
            )

    for s in QUARTILE_LABELS:
        fm = fold_mean_sd_row(supp_marg_fold_vals[s])
        sub = all_df[all_df.support_quartile == s]
        rows.append({"table_section": "support_marginal", "support_quartile": s, "sigma_quartile": "", "aggregation": "fold_mean_plus_sd", "mean_abs_error": fm["fold_mean"], "sd": fm["fold_sd"], "n_valid_folds": fm["n_valid_folds"], "n_rows": ""})
        rows.append({"table_section": "support_marginal", "support_quartile": s, "sigma_quartile": "", "aggregation": "pooled_across_folds", "mean_abs_error": float(sub["_abs_residual"].mean()) if len(sub) else _nan(), "sd": "", "n_valid_folds": "", "n_rows": int(len(sub))})
    for g in QUARTILE_LABELS:
        fm = fold_mean_sd_row(sig_marg_fold_vals[g])
        sub = all_df[all_df.sigma_quartile == g]
        rows.append({"table_section": "sigma_marginal", "support_quartile": "", "sigma_quartile": g, "aggregation": "fold_mean_plus_sd", "mean_abs_error": fm["fold_mean"], "sd": fm["fold_sd"], "n_valid_folds": fm["n_valid_folds"], "n_rows": ""})
        rows.append({"table_section": "sigma_marginal", "support_quartile": "", "sigma_quartile": g, "aggregation": "pooled_across_folds", "mean_abs_error": float(sub["_abs_residual"].mean()) if len(sub) else _nan(), "sd": "", "n_valid_folds": "", "n_rows": int(len(sub))})

    overall_pooled_mean = float(all_df["_abs_residual"].mean())
    overall_fold_vals = [float(with_abs_residual(fold_frames[f])["_abs_residual"].mean()) for f in FOLDS]
    fm_overall = fold_mean_sd_row(overall_fold_vals)
    rows.append({"table_section": "overall", "support_quartile": "", "sigma_quartile": "", "aggregation": "fold_mean_plus_sd", "mean_abs_error": fm_overall["fold_mean"], "sd": fm_overall["fold_sd"], "n_valid_folds": fm_overall["n_valid_folds"], "n_rows": ""})
    rows.append({"table_section": "overall", "support_quartile": "", "sigma_quartile": "", "aggregation": "pooled_across_folds", "mean_abs_error": overall_pooled_mean, "sd": "", "n_valid_folds": "", "n_rows": int(len(all_df))})

    for name, (s, g) in CELLS_OF_INTEREST_TASK7.items():
        cell_rows = all_df[(all_df.support_quartile == s) & (all_df.sigma_quartile == g)]
        boot = _cluster_bootstrap_mean(cell_rows, "_abs_residual")
        excludes_overall = bool(np.isfinite(boot["ci_low"]) and np.isfinite(boot["ci_high"]) and not (boot["ci_low"] <= overall_pooled_mean <= boot["ci_high"]))
        rows.append(
            {
                "table_section": "bootstrap_ci",
                "support_quartile": s,
                "sigma_quartile": g,
                "aggregation": "pooled_across_folds_cluster_bootstrap",
                "mean_abs_error": boot["point"],
                "sd": "",
                "n_valid_folds": "",
                "n_rows": int(len(cell_rows)),
                "cell_name": name,
                "ci_low_95": boot["ci_low"],
                "ci_high_95": boot["ci_high"],
                "n_boot": boot["n_boot"],
                "n_clusters_pairs": boot["n_clusters"],
                "bootstrap_seed": BOOTSTRAP_SEED,
                "excludes_overall_mean": excludes_overall,
                "overall_mean_abs_error_pooled": overall_pooled_mean,
            }
        )

    grid_df = pd.DataFrame(rows)
    fold_level_df = pd.DataFrame(fold_level)
    return grid_df, fold_level_df


# ------------------------------------------------------------------ Task 8 ---
def _pair_composition(df: pd.DataFrame) -> list[dict]:
    n_total = len(df)
    counts = df[PAIR_COLUMN].value_counts()
    rows = []
    for rank, (pair, n) in enumerate(counts.head(TOP_N_PAIRS).items(), start=1):
        rows.append({"row_type": "top_pair", "rank": rank, "key": pair, "n_rows": int(n), "share": float(n / n_total) if n_total else _nan()})
    return rows


def _electrolyte_distribution(df: pd.DataFrame) -> list[dict]:
    n_total = len(df)
    counts = df["electrolyte"].value_counts()
    rows = []
    for electrolyte, n in counts.items():
        rows.append({"row_type": "electrolyte", "key": str(electrolyte), "n_rows": int(n), "share": float(n / n_total) if n_total else _nan()})
    return rows


def _summary_stats(df: pd.DataFrame) -> list[dict]:
    metrics = ("favoured_logK", "abs_dlogK", "d_electronegativity", "d_covalent_radius", "d_oxidation_state", "_abs_residual")
    rows = []
    for m in metrics:
        v = pd.to_numeric(df[m], errors="coerce")
        rows.append(
            {
                "row_type": "summary_stat",
                "key": "abs_error" if m == "_abs_residual" else m,
                "mean": float(v.mean()) if v.notna().any() else _nan(),
                "sd": float(v.std(ddof=1)) if v.notna().sum() > 1 else _nan(),
                "n_rows": int(v.notna().sum()),
            }
        )
    return rows


def build_composition(all_df: pd.DataFrame) -> pd.DataFrame:
    cell_members: dict[str, pd.DataFrame] = {}
    all_rows: list[dict] = []
    for name, (s, g) in CELLS_OF_INTEREST_TASK8.items():
        sub = all_df[(all_df.support_quartile == s) & (all_df.sigma_quartile == g)]
        cell_members[name] = sub
        for row in _pair_composition(sub):
            row["cell"] = name
            all_rows.append(row)
        for row in _electrolyte_distribution(sub):
            row["cell"] = name
            all_rows.append(row)
        for row in _summary_stats(sub):
            row["cell"] = name
            all_rows.append(row)
        all_rows.append({"row_type": "cell_n_rows", "cell": name, "n_rows": int(len(sub))})

    names = list(cell_members.keys())
    a_name, b_name = names[0], names[1]
    a_pairs = set(cell_members[a_name][PAIR_COLUMN].unique())
    b_pairs = set(cell_members[b_name][PAIR_COLUMN].unique())
    overlap_pairs = a_pairs & b_pairs
    a_overlap_rows = cell_members[a_name][cell_members[a_name][PAIR_COLUMN].isin(overlap_pairs)]
    b_overlap_rows = cell_members[b_name][cell_members[b_name][PAIR_COLUMN].isin(overlap_pairs)]
    all_rows.append(
        {
            "row_type": "overlap",
            "cell": f"{a_name}_vs_{b_name}",
            "key": "n_pairs_overlap",
            "n_rows": len(overlap_pairs),
        }
    )
    all_rows.append(
        {
            "row_type": "overlap",
            "cell": a_name,
            "key": "n_pairs_in_cell",
            "n_rows": len(a_pairs),
        }
    )
    all_rows.append(
        {
            "row_type": "overlap",
            "cell": b_name,
            "key": "n_pairs_in_cell",
            "n_rows": len(b_pairs),
        }
    )
    all_rows.append(
        {
            "row_type": "overlap",
            "cell": a_name,
            "key": "fraction_of_cell_rows_from_overlap_pairs",
            "share": float(len(a_overlap_rows) / len(cell_members[a_name])) if len(cell_members[a_name]) else _nan(),
            "n_rows": int(len(a_overlap_rows)),
        }
    )
    all_rows.append(
        {
            "row_type": "overlap",
            "cell": b_name,
            "key": "fraction_of_cell_rows_from_overlap_pairs",
            "share": float(len(b_overlap_rows) / len(cell_members[b_name])) if len(cell_members[b_name]) else _nan(),
            "n_rows": int(len(b_overlap_rows)),
        }
    )
    return pd.DataFrame(all_rows)


def main() -> dict:
    fold_frames_raw = load_all_folds()
    ds2_derived, join_diag = load_ds2_chemical_axes()
    print(f"coverage_override_grid: join diagnostics {join_diag}", flush=True)

    fold_frames = {}
    for fold, df in fold_frames_raw.items():
        merged = join_ds2_chemical_axes(with_abs_residual(df), ds2_derived)
        fold_frames[fold] = _assign_quartiles(merged)

    grid_df, fold_level_df = build_grid(fold_frames)
    grid_df["source"] = "results/downstream_cv/ad_retention/fold_scores/DS2_RDKIT_MLPENS5_HETERO/fold{0..4}.csv"
    JACS_TABLES.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(OUT_GRID, index=False)
    fold_level_path = JACS_TABLES / "coverage_override_grid_fold_level.csv"
    fold_level_df.to_csv(fold_level_path, index=False)
    write_provenance(
        OUT_GRID,
        module="coverage_override_grid.build_grid",
        inputs=[FOLD_SCORES_DIR / f"{f}.csv" for f in FOLDS] + [DS2_PATH],
        notes={
            "join_diagnostics": join_diag,
            "fold_level_detail_table": str(fold_level_path.name),
            "bootstrap_n": BOOTSTRAP_N,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
    )
    print(f"wrote {OUT_GRID} ({len(grid_df)} rows) and {fold_level_path.name} ({len(fold_level_df)} rows)", flush=True)

    all_df = concat_folds(fold_frames)
    composition_df = build_composition(all_df)
    composition_df["source"] = "results/downstream_cv/ad_retention/fold_scores/DS2_RDKIT_MLPENS5_HETERO/fold{0..4}.csv (pooled_across_folds cell membership)"
    composition_df.to_csv(OUT_COMPOSITION, index=False)
    write_provenance(
        OUT_COMPOSITION,
        module="coverage_override_grid.build_composition",
        inputs=[FOLD_SCORES_DIR / f"{f}.csv" for f in FOLDS] + [DS2_PATH],
        notes={"cells": CELLS_OF_INTEREST_TASK8, "top_n_pairs": TOP_N_PAIRS},
    )
    print(f"wrote {OUT_COMPOSITION} ({len(composition_df)} rows)", flush=True)
    return {"grid": grid_df, "fold_level": fold_level_df, "composition": composition_df}


if __name__ == "__main__":
    main()
