"""Decision-level retention, in both units, with paired uncertainty.

Recomputes the retention protocol inside every eligible (metal pair, fold) cell
from the frozen fold-level score files. No model is loaded, trained or rescored.

Three tables are emitted:

  decision_level_retention.csv   fold mean and s.d. of the retained error for
                                 every score and retention fraction, in absolute
                                 dlog K and in units of each cell's own spread.

  decision_level_contrasts.csv   paired contrasts between scores. The fold s.d.
                                 of a single curve is dominated by fold-to-fold
                                 variation in the cell population, which is
                                 common to every score, so it is the wrong
                                 uncertainty for comparing two scores. Each
                                 contrast is therefore reported as a per-fold
                                 paired difference, a cluster bootstrap over
                                 metal pairs, and the fraction of individual
                                 cells in which one score retains lower error.

  decision_level_rho.csv         within-cell Spearman association of every score
                                 with the absolute residual, with the same
                                 cluster bootstrap and the fraction of cells in
                                 which the association is negative.

Rank statistics and retained subsets are identical in the two units, because the
divisor is constant inside a cell; only magnitudes and cross-cell weighting change.
"""
import json
import os
import subprocess

import numpy as np
import pandas as pd

FOLDS = [os.path.join("results", "downstream_cv", "ad_retention", "fold_scores",
                      "DS2_RDKIT_MLPENS5_HETERO", f"fold{i}.csv") for i in range(5)]
T = os.environ.get("BNN_TABLES", os.path.join("manuscripts", "jacs", "tables")) + os.sep

GRID = [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
MIN_N, MIN_DISTINCT, NPERM, SEED = 20, 5, 20, 0
NBOOT, BOOT_SEED = 2000, 1

SCORES = ["sigma_total", "sigma_aleatoric",
          "euclidean_feature_median", "euclidean_feature_nn",
          "mahalanobis_feature_centroid", "mahalanobis_feature_nn",
          "mahalanobis_feature_knn_mean_5", "mahalanobis_feature_knn_mean_10",
          "tanimoto_median_distance", "tanimoto_nn_distance",
          "tanimoto_knn_mean_5", "tanimoto_knn_mean_10",
          "pair_density_risk"]

CONTRASTS = [("sigma_total", "unfiltered"), ("sigma_total", "random"),
             ("sigma_total", "euclidean_feature_median"),
             ("sigma_total", "tanimoto_nn_distance"),
             ("euclidean_feature_median", "random"),
             ("mahalanobis_feature_centroid", "random"),
             ("tanimoto_median_distance", "random"),
             ("tanimoto_nn_distance", "random"),
             ("tanimoto_nn_distance", "unfiltered"),
             ("yhat_mag", "random"), ("yhat_mag", "unfiltered")]
CONTRAST_R = [0.50, 0.30]


def _rankdata(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(1, len(x) + 1)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    if (counts > 1).any():                       # average ties
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    return ranks


def _spearman(a, b):
    ra, rb = _rankdata(np.asarray(a, float)), _rankdata(np.asarray(b, float))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan


def collect():
    """One row per eligible cell, carrying every per-cell quantity we report."""
    rng = np.random.RandomState(SEED)
    rows = []
    for fi, path in enumerate(FOLDS):
        d = pd.read_csv(path)
        d = d.assign(yhat_mag=-d.y_pred.abs())
        for pair, g in d.groupby("metal_species_pair"):
            n = len(g)
            sd = g.y_true.std(ddof=1)
            if n < MIN_N or not np.isfinite(sd) or sd <= 0:
                continue
            res = (g.y_true - g.y_pred).values
            e2 = res ** 2
            rec = {"fold": fi, "pair": pair, "n": n, "sd": sd,
                   "unfiltered": float(np.sqrt(e2.mean()))}
            for s in SCORES + ["yhat_mag"]:
                if s not in g.columns or g[s].nunique() < MIN_DISTINCT:
                    continue
                se = e2[np.argsort(g[s].values, kind="mergesort")]
                for r in GRID:
                    rec[f"{s}@{r:.2f}"] = float(
                        np.sqrt(se[:max(2, int(round(r * n)))].mean()))
                rec[f"rho::{s}"] = _spearman(g[s].values, np.abs(res))
            tot = np.zeros(len(GRID))
            for _ in range(NPERM):
                se = e2[rng.permutation(n)]
                tot += [np.sqrt(se[:max(2, int(round(r * n)))].mean()) for r in GRID]
            for j, r in enumerate(GRID):
                rec[f"random@{r:.2f}"] = float(tot[j] / NPERM)
            rows.append(rec)
    return pd.DataFrame(rows)


def _col(cells, score, r, scale):
    c = "unfiltered" if score == "unfiltered" else f"{score}@{r:.2f}"
    v = cells[c]
    return v / cells.sd if scale else v


def retention_table(cells):
    out = []
    for scale, unit in ((False, "absolute_dlogK"), (True, "over_cell_sd")):
        for s in SCORES + ["yhat_mag", "random"]:
            if f"{s}@{GRID[0]:.2f}" not in cells:
                continue
            for r in GRID:
                v = _col(cells, s, r, scale)
                m = cells.assign(v=v).dropna(subset=["v"]).groupby("fold").v.mean()
                out.append(dict(unit=unit, score=s, retention=r,
                                fold_mean=m.mean(), fold_sd=m.std(ddof=1),
                                n_valid_folds=int(m.notna().sum()),
                                n_cells=int(v.notna().sum())))
    return pd.DataFrame(out)


def _bootstrap_diff(cells, a, b, r, scale, rng):
    ca, cb = _col(cells, a, r, scale), _col(cells, b, r, scale)
    frame = pd.DataFrame({"pair": cells.pair.values, "a": ca.values, "b": cb.values}).dropna()
    pairs = frame.pair.unique()
    by_pair = {p: g[["a", "b"]].values for p, g in frame.groupby("pair")}
    boots = np.empty(NBOOT)
    for i in range(NBOOT):
        draw = rng.choice(pairs, len(pairs), replace=True)
        stack = np.vstack([by_pair[p] for p in draw])
        boots[i] = stack[:, 0].mean() - stack[:, 1].mean()
    return frame, np.percentile(boots, [2.5, 97.5])


def contrast_table(cells):
    rng = np.random.RandomState(BOOT_SEED)
    out = []
    for scale, unit in ((False, "absolute_dlogK"), (True, "over_cell_sd")):
        for a, b in CONTRASTS:
            for r in CONTRAST_R:
                if f"{a}@{r:.2f}" not in cells and a != "unfiltered":
                    continue
                ca, cb = _col(cells, a, r, scale), _col(cells, b, r, scale)
                fm = cells.assign(a=ca, b=cb).dropna(subset=["a", "b"]).groupby("fold")[["a", "b"]].mean()
                dif = (fm.a - fm.b).values
                frame, ci = _bootstrap_diff(cells, a, b, r, scale, rng)
                out.append(dict(
                    unit=unit, score_a=a, score_b=b, retention=r,
                    mean_a=fm.a.mean(), mean_b=fm.b.mean(),
                    paired_diff=dif.mean(), paired_sd=dif.std(ddof=1),
                    folds_same_sign=bool((np.sign(dif) == np.sign(dif[0])).all()),
                    ci_low=ci[0], ci_high=ci[1],
                    ci_excludes_zero=bool(ci[0] > 0 or ci[1] < 0),
                    cell_win_fraction=float((frame.a < frame.b).mean()),
                    n_cells=len(frame)))
    return pd.DataFrame(out)


def rho_table(cells):
    rng = np.random.RandomState(BOOT_SEED)
    out = []
    for s in SCORES:
        c = f"rho::{s}"
        frame = (pd.DataFrame({"pair": cells.pair.values, "fold": cells.fold.values,
                               "rho": cells[c].values}).dropna()
                 if c in cells else pd.DataFrame(columns=["pair", "fold", "rho"]))
        if frame.empty:
            out.append(dict(score=s, n_cells=0, fold_mean=np.nan, fold_sd=np.nan,
                            ci_low=np.nan, ci_high=np.nan, frac_cells_negative=np.nan,
                            status="undefined_within_cell"))
            continue
        m = frame.groupby("fold").rho.mean()
        pairs = frame.pair.unique()
        by_pair = {p: g.rho.values for p, g in frame.groupby("pair")}
        boots = np.empty(NBOOT)
        for i in range(NBOOT):
            draw = rng.choice(pairs, len(pairs), replace=True)
            boots[i] = np.concatenate([by_pair[p] for p in draw]).mean()
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.append(dict(score=s, n_cells=len(frame), fold_mean=m.mean(),
                        fold_sd=m.std(ddof=1), ci_low=lo, ci_high=hi,
                        frac_cells_negative=float((frame.rho < 0).mean()),
                        status="defined"))
    return pd.DataFrame(out)


MOP_MIN_ROWS = 3
DECOMP_SCORES = ["sigma_total", "euclidean_feature_median",
                 "mahalanobis_feature_centroid", "tanimoto_median_distance",
                 "tanimoto_nn_distance", "pair_density_risk", "yhat_mag"]


def _mop(df):
    vals = [np.sqrt((((g.y_true - g.y_pred) ** 2)).mean())
            for _, g in df.groupby("metal_species_pair") if len(g) >= MOP_MIN_ROWS]
    return (float(np.mean(vals)), len(vals)) if vals else (np.nan, 0)


def decomposition_table(cells):
    """Why the pooled comparison and the decision-level comparison differ.

    Protocol A is the pooled, mean-of-pairs retention used in the earlier
    analysis of this dataset: rank every held-out row of a fold, keep the
    lowest-risk half, then average pair-level RMSE over whichever pairs survive.
    A score that assigns high risk to whole metal pairs removes those pairs from
    the average, so part of its apparent advantage is a change of population
    rather than an ordering of candidates; the retained pair count makes that
    visible. Protocol B is the same thing restricted to eligible cells, which
    isolates the effect of the eligibility rule. Protocol C is the decision-level
    statistic used in the paper, in which every cell contributes its own
    lowest-risk fraction, so the pair population is identical for every score.
    """
    out = []
    for fi, path in enumerate(FOLDS):
        d = pd.read_csv(path).assign(yhat_mag=lambda x: -x.y_pred.abs())
        keep_pairs = [p for p, g in d.groupby("metal_species_pair")
                      if len(g) >= MIN_N and np.isfinite(g.y_true.std(ddof=1))
                      and g.y_true.std(ddof=1) > 0]
        for label, frame in (("A_pooled_all_pairs", d),
                             ("B_pooled_eligible_cells",
                              d[d.metal_species_pair.isin(keep_pairs)])):
            m, k = _mop(frame)
            out.append(dict(protocol=label, score="unfiltered", fold=fi,
                            mop_rmse=m, pairs_retained=k))
            n = len(frame)
            for s in DECOMP_SCORES:
                if s not in frame or frame[s].nunique() < 2:
                    continue
                sub = frame.iloc[np.argsort(frame[s].values, kind="mergesort")[:n // 2]]
                m, k = _mop(sub)
                out.append(dict(protocol=label, score=s, fold=fi,
                                mop_rmse=m, pairs_retained=k))
            rs = np.random.RandomState(fi)
            sub = frame.iloc[rs.permutation(n)[:n // 2]]
            m, k = _mop(sub)
            out.append(dict(protocol=label, score="random", fold=fi,
                            mop_rmse=m, pairs_retained=k))
    df = pd.DataFrame(out).groupby(["protocol", "score"]).agg(
        fold_mean=("mop_rmse", "mean"), fold_sd=("mop_rmse", lambda v: v.std(ddof=1)),
        pairs_retained=("pairs_retained", "mean")).reset_index()

    cell_pairs = cells.groupby("fold").pair.nunique().mean()
    for s in DECOMP_SCORES + ["random", "unfiltered"]:
        col = "unfiltered" if s == "unfiltered" else f"{s}@0.50"
        if col not in cells:
            continue
        m = cells.groupby("fold")[col].mean()
        df.loc[len(df)] = dict(protocol="C_within_cell_decision_level", score=s,
                               fold_mean=m.mean(), fold_sd=m.std(ddof=1),
                               pairs_retained=cell_pairs)
    return df


def pair_loss_table():
    """Why pooled retention leaves a different number of pairs for each score.

    Pooled retention keeps the globally lowest-risk half of *rows*, not half of
    each pair's rows. A pair therefore leaves the mean-of-pairs average in one of
    two ways: every one of its rows falls above the global cut, or fewer than
    MOP_MIN_ROWS of them survive. How often that happens is governed by how much
    of a score's variation lies between pairs rather than within them, which is
    reported here as the share of the score's rank variance explained by pair
    identity. A score that is constant within a pair keeps or drops whole pairs;
    a score that varies mostly within pairs thins every pair by about the same
    fraction and keeps almost all of them.
    """
    rows = []
    for fi, path in enumerate(FOLDS):
        d = pd.read_csv(path).assign(yhat_mag=lambda x: -x.y_pred.abs())
        n = len(d)
        sizes = d.groupby("metal_species_pair").size()
        eligible = sizes >= MOP_MIN_ROWS
        rng = np.random.RandomState(fi)
        variants = [(s, d.iloc[np.argsort(d[s].values, kind="mergesort")[:n // 2]])
                    for s in DECOMP_SCORES if s in d]
        variants.append(("random", d.iloc[rng.permutation(n)[:n // 2]]))
        for s, keep in variants:
            k = keep.groupby("metal_species_pair").size().reindex(sizes.index, fill_value=0)
            frac = k / sizes
            if s == "random":
                eta = np.nan
            else:
                r = d[s].rank()
                rg = r.groupby(d.metal_species_pair.values).transform("mean")
                eta = 1 - float(((r - rg) ** 2).sum() / ((r - r.mean()) ** 2).sum())
            rows.append(dict(
                fold=fi, score=s,
                pairs_eligible_unfiltered=int(eligible.sum()),
                pairs_losing_every_row=int(((k == 0) & eligible).sum()),
                pairs_below_mop_minimum=int(((k >= 1) & (k < MOP_MIN_ROWS) & eligible).sum()),
                pairs_counted=int((k >= MOP_MIN_ROWS).sum()),
                median_retained_fraction=float(frac[eligible].median()),
                between_pair_rank_variance_share=eta))
    return pd.DataFrame(rows).groupby("score").mean(numeric_only=True).drop(
        columns=["fold"]).reset_index()


def _provenance(name, extra):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = "unavailable"
    json.dump(dict(table=name, inputs=FOLDS, git_commit=commit,
                   cell_rule=dict(min_ligands=MIN_N, min_distinct_score_values=MIN_DISTINCT,
                                  requires_nonzero_target_sd=True),
                   aggregation="cell-equal mean within fold, then fold mean +/- s.d. (ddof=1)",
                   random_baseline_permutations=NPERM, random_seed=SEED,
                   cluster_bootstrap=dict(resamples=NBOOT, unit="metal pair", seed=BOOT_SEED),
                   **extra),
              open(T + name + ".provenance.json", "w"), indent=2)


if __name__ == "__main__":
    cells = collect()
    print(f"eligible cells: {len(cells)}  per fold: {cells.groupby('fold').size().tolist()}")

    ret = retention_table(cells)
    ret.to_csv(T + "decision_level_retention.csv", index=False)
    _provenance("decision_level_retention.csv", dict(units=["absolute_dlogK", "over_cell_sd"]))

    con = contrast_table(cells)
    con.to_csv(T + "decision_level_contrasts.csv", index=False)
    _provenance("decision_level_contrasts.csv",
                dict(note="paired per-fold differences and a cluster bootstrap over metal pairs"))

    rho = rho_table(cells)
    rho.to_csv(T + "decision_level_rho.csv", index=False)
    _provenance("decision_level_rho.csv", dict(note="within-cell Spearman with |residual|"))

    dec = decomposition_table(cells)
    dec.to_csv(T + "aggregation_decomposition.csv", index=False)
    _provenance("aggregation_decomposition.csv",
                dict(note="pooled mean-of-pairs retention vs the decision-level statistic",
                     mop_min_rows_per_pair=MOP_MIN_ROWS))
    loss = pair_loss_table()
    loss.to_csv(T + "pooled_retention_pair_loss.csv", index=False)
    _provenance("pooled_retention_pair_loss.csv",
                dict(note="how pooled 50% retention changes the pair population, and why"))
    print("\n=== pooled 50% retention: where the pairs go ===")
    for _, r in loss.sort_values("pairs_counted").iterrows():
        eta = "  n/a" if np.isnan(r.between_pair_rank_variance_share) else \
            f"{r.between_pair_rank_variance_share:5.3f}"
        print(f"  {r.score:30s} of {r.pairs_eligible_unfiltered:5.1f} pairs: "
              f"{r.pairs_losing_every_row:5.1f} lose every row, "
              f"{r.pairs_below_mop_minimum:5.1f} fall below {MOP_MIN_ROWS} rows, "
              f"{r.pairs_counted:5.1f} counted | median kept {r.median_retained_fraction:.2f} "
              f"| between-pair variance share {eta}")

    print("\n=== aggregation decomposition, 50% retention ===")
    for proto, g in dec.groupby("protocol"):
        print(f"  {proto}")
        for _, r in g.sort_values("fold_mean").iterrows():
            print(f"    {r.score:30s} {r.fold_mean:6.3f} +/- {r.fold_sd:.3f}   "
                  f"pairs {r.pairs_retained:6.1f}")

    for unit in ("absolute_dlogK", "over_cell_sd"):
        print(f"\n=== {unit}: retained error at r = 0.50 ===")
        sub = ret[(ret.unit == unit) & (ret.retention == 0.50)].sort_values("fold_mean")
        unf = ret[(ret.unit == unit) & (ret.score == "random") & (ret.retention == 1.00)].fold_mean.iloc[0]
        print(f"  {'unfiltered':32s} {unf:6.3f}")
        for _, r in sub.iterrows():
            print(f"  {r.score:32s} {r.fold_mean:6.3f} +/- {r.fold_sd:.3f}")
        print(f"\n=== {unit}: paired contrasts ===")
        for _, r in con[(con.unit == unit) & (con.retention == 0.50)].iterrows():
            flag = "*" if r.ci_excludes_zero else " "
            print(f"  r=0.50 {r.score_a:28s} vs {r.score_b:26s} "
                  f"{r.paired_diff:+.3f} CI[{r.ci_low:+.3f},{r.ci_high:+.3f}]{flag} "
                  f"win {r.cell_win_fraction:.0%}")
    print("\n=== within-cell rho ===")
    for _, r in rho.sort_values("fold_mean").iterrows():
        if r.status != "defined":
            print(f"  {r.score:32s} {r.status}")
        else:
            print(f"  {r.score:32s} {r.fold_mean:+.3f} +/- {r.fold_sd:.3f} "
                  f"CI[{r.ci_low:+.3f},{r.ci_high:+.3f}]  negative in {r.frac_cells_negative:.0%} of cells")
