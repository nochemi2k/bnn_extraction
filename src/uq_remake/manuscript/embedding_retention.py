"""Retention on a second, architecturally unrelated model, and two small tables.

Three outputs, all computed from frozen fold-level score files. No model is
loaded, trained or rescored.

  embedding_retention.csv     Pooled retention on the evidential D-MPNN's own
                              held-out errors, comparing that model's own
                              predictive uncertainty, the classic molecular
                              distances, and distances measured in its own
                              learned embedding. The embedding distances exist
                              only for this model, so this is the only place
                              they can be put on a retention curve rather than
                              a correlation.

  regime_normalized_error.csv The workhorse's uncertainty and error by
                              selectivity regime, in normalized units. Inside
                              an eligible (metal pair, fold) cell the regime's
                              rows give an RMSE, which is divided by that
                              cell's own SD of the observed target; cells are
                              averaged with equal weight, then folds. The
                              normalized uncertainty divides the regime's mean
                              sigma_total by the same quantity. A row-level
                              ratio is not used: it is dominated by the few
                              pairs whose spread is close to zero.

The retention protocol is the one used throughout: within a fold, rank every
prediction by the score, keep the lowest-risk fraction, and recompute
mean-of-pairs RMSE over the metal pairs that keep at least MOP_MIN_ROWS rows.
"""
import json
import os
import subprocess

import numpy as np
import pandas as pd

T = os.environ.get("BNN_TABLES", os.path.join("manuscripts", "jacs", "tables")) + os.sep
_FS = os.path.join("results", "downstream_cv", "ad_retention", "fold_scores")
DMPNN = [os.path.join(_FS, "DS2_GRAPH_DMPNN_EVID", f"fold{i}.csv") for i in range(5)]
WORKHORSE = [os.path.join(_FS, "DS2_RDKIT_MLPENS5_HETERO", f"fold{i}.csv")
             for i in range(5)]

GRID = [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
MOP_MIN_ROWS = 3
NPERM, SEED = 20, 0
HIGH_SEL_TAU = 1.0
CELL_MIN_N = 20        # the cell-eligibility rule used throughout the paper
REGIME_MIN_ROWS = 5    # a regime must be represented in the cell to contribute

# The model's own uncertainty, the two classic distances available for this
# model at their best aggregation, and two learned-embedding distances: the
# standardised Euclidean nearest neighbour, which is the embedding analogue of
# the usual fingerprint criterion, and the Mahalanobis nearest neighbour, the
# most strongly negative score in the whole screen.
EMB_SCORES = ["sigma_total", "euclidean_feature_median", "tanimoto_median_distance",
              "chemprop_std_euclidean_nn", "chemprop_mahalanobis_nn"]


def _pair_rmse(frame):
    out = {}
    for p, g in frame.groupby("metal_species_pair"):
        if len(g) >= MOP_MIN_ROWS:
            out[p] = float(np.sqrt((((g.y_true - g.y_pred) ** 2)).mean()))
    return out


def embedding_retention():
    rng = np.random.RandomState(SEED)
    rows, unf = [], []
    acc = {}
    for fi, path in enumerate(DMPNN):
        d = pd.read_csv(path)
        n = len(d)
        unf.append(np.mean(list(_pair_rmse(d).values())))
        for s in EMB_SCORES:
            order = np.argsort(d[s].values, kind="mergesort")
            for r in GRID:
                t = _pair_rmse(d.iloc[order[:int(round(r * n))]])
                acc.setdefault((s, r), []).append((np.mean(list(t.values())), len(t)))
        for r in GRID:
            k = int(round(r * n))
            mops, sizes = [], []
            for _ in range(NPERM):
                t = _pair_rmse(d.iloc[rng.permutation(n)[:k]])
                mops.append(np.mean(list(t.values())))
                sizes.append(len(t))
            acc.setdefault(("random", r), []).append((np.mean(mops), np.mean(sizes)))
    for (s, r), v in acc.items():
        vals = np.array([x[0] for x in v])
        rows.append(dict(model="DS2_GRAPH_DMPNN_EVID", score=s, retention=r,
                         fold_mean=float(vals.mean()),
                         fold_sd=float(vals.std(ddof=1)),
                         pairs_retained=float(np.mean([x[1] for x in v]))))
    rows.append(dict(model="DS2_GRAPH_DMPNN_EVID", score="unfiltered", retention=1.00,
                     fold_mean=float(np.mean(unf)),
                     fold_sd=float(np.std(unf, ddof=1)), pairs_retained=float("nan")))
    df = pd.DataFrame(rows)
    auc = []
    for s in EMB_SCORES + ["random"]:
        c = df[df.score == s].sort_values("retention")
        auc.append(dict(score=s, auc=float(np.trapezoid(c.fold_mean, c.retention))))
    return df.merge(pd.DataFrame(auc), on="score", how="left")


def regime_normalized_error():
    """Normalized error and normalized uncertainty by selectivity regime.

    Computed the way the normalized error is defined everywhere else in this
    paper: inside an eligible (metal pair, fold) cell -- at least CELL_MIN_N
    held-out ligands and a non-zero spread of the observed target -- the
    regime's rows give an RMSE, that RMSE is divided by the cell's own
    SD of the target, and the ratios are averaged over cells with equal
    weight, then over folds. A row-level ratio would be dominated by the few
    pairs whose spread is close to zero.
    """
    rows, per_fold = [], {}
    for fi, path in enumerate(WORKHORSE):
        d = pd.read_csv(path, usecols=["y_true", "y_pred", "sigma_total",
                                       "metal_species_pair"])
        d = d.assign(abs_error=(d.y_true - d.y_pred).abs())
        for regime, mask in (("high_selectivity", d.y_true.abs() > HIGH_SEL_TAU),
                             ("near_zero", d.y_true.abs() < HIGH_SEL_TAU)):
            e_norm, s_norm, ncell = [], [], 0
            for _, cell in d.groupby("metal_species_pair"):
                if len(cell) < CELL_MIN_N:
                    continue
                sd = float(cell.y_true.std(ddof=1))
                if not np.isfinite(sd) or sd <= 0:
                    continue
                g = cell[mask.loc[cell.index]]
                if len(g) < REGIME_MIN_ROWS:
                    continue
                e_norm.append(float(np.sqrt((g.abs_error ** 2).mean()) / sd))
                s_norm.append(float(g.sigma_total.mean() / sd))
                ncell += 1
            per_fold.setdefault((regime, "normalized_error"), []).append(np.mean(e_norm))
            per_fold.setdefault((regime, "normalized_sigma_total"), []).append(np.mean(s_norm))
            per_fold.setdefault((regime, "cells"), []).append(float(ncell))
    for (regime, metric), v in per_fold.items():
        rows.append(dict(regime=regime,
                         regime_definition=("|y_true| > " if regime == "high_selectivity"
                                            else "|y_true| < ") + str(HIGH_SEL_TAU),
                         metric_name=metric, fold_mean=float(np.mean(v)),
                         fold_sd=float(np.std(v, ddof=1)), n_valid_folds=len(v),
                         cell_min_n=CELL_MIN_N, regime_min_rows=REGIME_MIN_ROWS))
    return pd.DataFrame(rows)


def _provenance(name, inputs, note):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = "unavailable"
    json.dump(dict(table=name, inputs=inputs, git_commit=commit, note=note,
                   random_baseline_permutations=NPERM, random_seed=SEED,
                   mop_min_rows=MOP_MIN_ROWS),
              open(T + name + ".provenance.json", "w"), indent=2)


if __name__ == "__main__":
    e = embedding_retention()
    e.to_csv(T + "embedding_retention.csv", index=False)
    _provenance("embedding_retention.csv", DMPNN,
                "pooled retention on the evidential D-MPNN's own held-out errors")
    print("=== D-MPNN pooled retention, r = 0.50 ===")
    u = float(e[e.score == "unfiltered"].fold_mean.iloc[0])
    print(f"  unfiltered {u:.3f}")
    for _, r in e[e.retention == 0.50].sort_values("fold_mean").iterrows():
        print(f"  {r.score:32s} {r.fold_mean:6.3f} +/- {r.fold_sd:.3f}   "
              f"pairs {r.pairs_retained:6.1f}   AUC {r.auc:.3f}")

    g = regime_normalized_error()
    g.to_csv(T + "regime_normalized_error.csv", index=False)
    _provenance("regime_normalized_error.csv", WORKHORSE,
                "row-level normalized error and normalized sigma by selectivity regime")
    print("\n=== workhorse, normalized units by regime ===")
    for _, r in g.iterrows():
        print(f"  {r.regime:18s} {r.metric_name:24s} {r.fold_mean:8.3f} +/- {r.fold_sd:.3f}")
