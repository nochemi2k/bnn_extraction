"""Pooled risk ordering on held-out predictions, with the matched-population control.

The protocol is the one used in the earlier analysis of this dataset: within each
held-out fold, rank every prediction by a risk score, retain the lowest-risk
fraction, and recompute mean-of-pairs RMSE over the metal pairs that keep at
least MOP_MIN_ROWS rows. No model is loaded, trained or rescored.

Retention across a whole fold changes which pairs are averaged over, because a
score that assigns high risk to entire pairs removes them. Two populations are
therefore reported at every retention fraction:

  free     every pair that keeps at least MOP_MIN_ROWS rows under that score
  matched  only the pairs kept by *all* scores in MATCHED_SET at that fraction,
           so every score is scored on an identical population

Emitted tables:
  pooled_retention.csv            free-population curves, fold mean and s.d.,
                                  pairs retained, and the area under the curve
  pooled_retention_matched.csv    matched-population curves and the shared
                                  population size
  pooled_contrasts.csv            paired differences on the matched population
                                  at r = 0.50 and r = 0.30, with a cluster
                                  bootstrap over metal pairs and the fraction of
                                  pairs in which one score retains lower error
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
MOP_MIN_ROWS = 3
NPERM, SEED = 20, 0
NBOOT, BOOT_SEED = 2000, 1

SCORES = ["sigma_total", "sigma_aleatoric", "pair_density_risk",
          "euclidean_feature_median", "euclidean_feature_nn",
          "mahalanobis_feature_centroid", "mahalanobis_feature_median",
          "mahalanobis_feature_nn",
          "mahalanobis_feature_knn_mean_5", "mahalanobis_feature_knn_mean_10",
          "tanimoto_median_distance", "tanimoto_nn_distance",
          "tanimoto_knn_mean_5", "tanimoto_knn_mean_10", "yhat_mag"]

# mahalanobis_feature_median is not a column of the fold-score files; its
# per-row values are deposited separately by the earlier analysis and are
# joined on (record_id, outer_fold_id). No distance is recomputed here.
MAHAL_MEDIAN = os.path.join("manuscripts", "final", "tables",
                            "iz_mahalanobis_feature_median_scores.csv")

# The five scores compared on a common population. pair_density_risk is excluded
# because its retained set is so narrow that including it would shrink the shared
# population to a third of the next-narrowest score; it is controlled separately.
MATCHED_SET = ["sigma_total", "euclidean_feature_median",
               "mahalanobis_feature_centroid", "tanimoto_median_distance",
               "tanimoto_nn_distance"]
MATCHED_SET_WITH_COUNT = MATCHED_SET + ["pair_density_risk"]

CONTRASTS = [("sigma_total", "euclidean_feature_median"),
             ("sigma_total", "mahalanobis_feature_centroid"),
             ("sigma_total", "tanimoto_nn_distance"),
             ("sigma_total", "random"),
             ("euclidean_feature_median", "random"),
             ("mahalanobis_feature_centroid", "random"),
             ("tanimoto_median_distance", "random"),
             ("tanimoto_nn_distance", "random")]
CONTRAST_R = [0.50, 0.30]


def _pair_rmse(frame):
    """Pair-level RMSE for every pair keeping at least MOP_MIN_ROWS rows."""
    out = {}
    for p, g in frame.groupby("metal_species_pair"):
        if len(g) >= MOP_MIN_ROWS:
            out[p] = float(np.sqrt((((g.y_true - g.y_pred) ** 2)).mean()))
    return out


def per_fold_pair_tables():
    """{(fold, score, r): {pair: rmse}}, the unfiltered reference, and the
    free-population random baseline as a per-permutation fold statistic."""
    rng = np.random.RandomState(SEED)
    tables, unfiltered, random_free = {}, {}, {}
    for fi, path in enumerate(FOLDS):
        d = pd.read_csv(path).assign(yhat_mag=lambda x: -x.y_pred.abs())
        if os.path.exists(MAHAL_MEDIAN):
            mm = pd.read_csv(MAHAL_MEDIAN)
            before = len(d)
            d = d.merge(mm, on=["record_id", "outer_fold_id"], how="left")
            assert len(d) == before, "mahalanobis median join changed the row count"
            assert d.mahalanobis_feature_median.notna().all(), \
                "mahalanobis median join left missing values"
        n = len(d)
        unfiltered[fi] = _pair_rmse(d)
        for s in SCORES:
            if s not in d:
                continue
            order = np.argsort(d[s].values, kind="mergesort")
            for r in GRID:
                k = int(round(r * n))
                tables[(fi, s, r)] = _pair_rmse(d.iloc[order[:k]])
        for r in GRID:                                     # random baseline
            k = int(round(r * n))
            acc, mops, sizes = {}, [], []
            for _ in range(NPERM):
                t = _pair_rmse(d.iloc[rng.permutation(n)[:k]])
                mops.append(np.mean(list(t.values())))
                sizes.append(len(t))
                for p, v in t.items():
                    acc.setdefault(p, []).append(v)
            # per-pair means are used only for the matched population, where the
            # population is fixed by the scores; the free-population value is the
            # mean over permutations of the fold statistic, so that it is computed
            # exactly as every score's is rather than over a wider pair set.
            tables[(fi, "random", r)] = {p: float(np.mean(v)) for p, v in acc.items()}
            random_free[(fi, r)] = (float(np.mean(mops)), float(np.mean(sizes)))
    return tables, unfiltered, random_free


ALL_SCORES = SCORES + ["random"]


def free_table(tables, unfiltered, random_free):
    rows = []
    for s in ALL_SCORES:
        for r in GRID:
            vals, npairs = [], []
            for fi in range(5):
                if s == "random":
                    if (fi, r) not in random_free:
                        continue
                    v, npr = random_free[(fi, r)]
                    vals.append(v); npairs.append(npr)
                    continue
                t = tables.get((fi, s, r))
                if not t:
                    continue
                vals.append(np.mean(list(t.values())))
                npairs.append(len(t))
            if vals:
                rows.append(dict(score=s, retention=r, population="free",
                                 fold_mean=float(np.mean(vals)),
                                 fold_sd=float(np.std(vals, ddof=1)),
                                 pairs_retained=float(np.mean(npairs))))
    u = [np.mean(list(unfiltered[fi].values())) for fi in range(5)]
    rows.append(dict(score="unfiltered", retention=1.00, population="free",
                     fold_mean=float(np.mean(u)), fold_sd=float(np.std(u, ddof=1)),
                     pairs_retained=float(np.mean([len(unfiltered[fi]) for fi in range(5)]))))
    df = pd.DataFrame(rows)
    auc = []
    for s in ALL_SCORES:
        c = df[(df.score == s) & (df.population == "free")].sort_values("retention")
        if len(c) == len(GRID):
            # plain trapezoid over the retention grid, the convention used in
            # the earlier analysis of this dataset, so the values are comparable
            auc.append(dict(score=s, auc=float(np.trapezoid(c.fold_mean, c.retention))))
    return df, pd.DataFrame(auc)


def _shared(tables, fi, r, score_set):
    """Pairs retained by every score in score_set, in a deterministic order.

    The order matters: the cluster bootstrap draws from this sequence, so an
    unordered set would make the interval endpoints depend on the interpreter's
    string hash seed. Point estimates are unaffected either way.
    """
    keys = [set(tables[(fi, s, r)]) for s in score_set if (fi, s, r) in tables]
    return sorted(set.intersection(*keys)) if keys else []


def matched_table(tables, score_set, label):
    rows = []
    for r in GRID:
        for s in list(score_set) + ["random"]:
            vals, sizes = [], []
            for fi in range(5):
                shared = _shared(tables, fi, r, score_set)
                t = tables.get((fi, s, r))
                if not t or not shared:
                    continue
                v = [t[p] for p in shared if p in t]
                if v:
                    vals.append(np.mean(v))
                    sizes.append(len(shared))
            if vals:
                rows.append(dict(score=s, retention=r, population=label,
                                 fold_mean=float(np.mean(vals)),
                                 fold_sd=float(np.std(vals, ddof=1)),
                                 pairs_shared=float(np.mean(sizes))))
    return pd.DataFrame(rows)


def matched_contrasts(tables):
    rng = np.random.RandomState(BOOT_SEED)
    rows = []
    for r in CONTRAST_R:
        per_fold = {fi: _shared(tables, fi, r, MATCHED_SET) for fi in range(5)}
        for a, b in CONTRASTS:
            da, db, pair_a, pair_b, pair_id = [], [], [], [], []
            for fi in range(5):
                shared = per_fold[fi]
                ta, tb = tables.get((fi, a, r)), tables.get((fi, b, r))
                if not ta or not tb or not shared:
                    continue
                common = [p for p in shared if p in ta and p in tb]
                da.append(np.mean([ta[p] for p in common]))
                db.append(np.mean([tb[p] for p in common]))
                pair_a += [ta[p] for p in common]
                pair_b += [tb[p] for p in common]
                pair_id += [f"{p}" for p in common]
            if not da:
                continue
            dif = np.array(da) - np.array(db)
            frame = pd.DataFrame({"pair": pair_id, "a": pair_a, "b": pair_b})
            uniq = frame.pair.unique()
            by_pair = {p: g[["a", "b"]].values for p, g in frame.groupby("pair")}
            boots = np.empty(NBOOT)
            for i in range(NBOOT):
                draw = rng.choice(uniq, len(uniq), replace=True)
                st = np.vstack([by_pair[p] for p in draw])
                boots[i] = st[:, 0].mean() - st[:, 1].mean()
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append(dict(
                score_a=a, score_b=b, retention=r,
                mean_a=float(np.mean(da)), mean_b=float(np.mean(db)),
                paired_diff=float(dif.mean()), paired_sd=float(dif.std(ddof=1)),
                folds_same_sign=bool((np.sign(dif) == np.sign(dif[0])).all()),
                ci_low=float(lo), ci_high=float(hi),
                ci_excludes_zero=bool(lo > 0 or hi < 0),
                pair_win_fraction=float((frame.a < frame.b).mean()),
                n_pair_fold=len(frame)))
    return pd.DataFrame(rows)


def _provenance(name, extra):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = "unavailable"
    json.dump(dict(table=name, inputs=FOLDS, git_commit=commit,
                   protocol="pooled retention within a fold, mean-of-pairs over "
                            f"pairs keeping >= {MOP_MIN_ROWS} rows",
                   random_baseline_permutations=NPERM, random_seed=SEED,
                   cluster_bootstrap=dict(resamples=NBOOT, unit="metal pair",
                                          seed=BOOT_SEED),
                   matched_set=MATCHED_SET, **extra),
              open(T + name + ".provenance.json", "w"), indent=2)


if __name__ == "__main__":
    tables, unfiltered, random_free = per_fold_pair_tables()

    free, auc = free_table(tables, unfiltered, random_free)
    free = free.merge(auc, on="score", how="left")
    free.to_csv(T + "pooled_retention.csv", index=False)
    _provenance("pooled_retention.csv", dict(population="free"))

    m5 = matched_table(tables, MATCHED_SET, "matched_five_scores")
    m6 = matched_table(tables, MATCHED_SET_WITH_COUNT, "matched_including_count")
    pd.concat([m5, m6], ignore_index=True).to_csv(T + "pooled_retention_matched.csv",
                                                  index=False)
    _provenance("pooled_retention_matched.csv",
                dict(populations=["matched_five_scores", "matched_including_count"]))

    con = matched_contrasts(tables)
    con.to_csv(T + "pooled_contrasts.csv", index=False)
    _provenance("pooled_contrasts.csv", dict(note="paired on the shared population"))

    unf = float(free[free.score == "unfiltered"].fold_mean.iloc[0])
    print(f"unfiltered mean-of-pairs RMSE {unf:.3f} "
          f"over {free[free.score == 'unfiltered'].pairs_retained.iloc[0]:.1f} pairs\n")
    print("=== free population, r = 0.50 ===")
    for _, r in free[(free.retention == 0.50)].sort_values("fold_mean").iterrows():
        print(f"  {r.score:32s} {r.fold_mean:6.3f} +/- {r.fold_sd:.3f}   "
              f"pairs {r.pairs_retained:6.1f}   AUC {r.auc:.3f}")
    print("\n=== matched population (five scores), r = 0.50 ===")
    for _, r in m5[m5.retention == 0.50].sort_values("fold_mean").iterrows():
        print(f"  {r.score:32s} {r.fold_mean:6.3f} +/- {r.fold_sd:.3f}   "
              f"shared pairs {r.pairs_shared:6.1f}")
    print("\n=== matched population including the count, r = 0.50 ===")
    for _, r in m6[m6.retention == 0.50].sort_values("fold_mean").iterrows():
        print(f"  {r.score:32s} {r.fold_mean:6.3f} +/- {r.fold_sd:.3f}   "
              f"shared pairs {r.pairs_shared:6.1f}")
    print("\n=== paired contrasts on the shared population ===")
    for _, r in con.iterrows():
        flag = "*" if r.ci_excludes_zero else " "
        print(f"  r={r.retention:.2f}  {r.score_a:28s} vs {r.score_b:26s} "
              f"{r.paired_diff:+.3f} CI[{r.ci_low:+.3f},{r.ci_high:+.3f}]{flag} "
              f"pairs won {r.pair_win_fraction:.0%}")
