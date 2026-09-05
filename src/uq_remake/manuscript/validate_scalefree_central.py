"""Independent validator for the paper's central statistic.

Recomputes, from the frozen fold-level predictions and by a code path that does
not import the module that produced them, every headline value in the main text:
the cell-eligibility rule, the unfiltered baseline, the retention curve for each
score, the undefined-by-construction result, and the scale-free coverage grid.
Exits non-zero on any mismatch and prints the offending row.
"""
import json, sys
import numpy as np
import pandas as pd

FOLDS = [f"results/downstream_cv/ad_retention/fold_scores/DS2_RDKIT_MLPENS5_HETERO/fold{i}.csv"
         for i in range(5)]
T = "manuscripts/jacs/tables/"
GRID = [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
MIN_N, MIN_DISTINCT, NPERM, SEED = 20, 5, 20, 0
checks = []


def chk(name, ok, detail=""):
    checks.append(dict(check=name, passed=bool(ok), detail=str(detail)))


def eligible(g, score):
    sd = g.y_true.std(ddof=1)
    if len(g) < MIN_N or not np.isfinite(sd) or sd <= 0:
        return False, sd
    if score is not None and g[score].nunique() < MIN_DISTINCT:
        return False, sd
    return True, sd


def recompute():
    rng = np.random.RandomState(SEED)
    per_fold, undef = {}, {}
    for fi, path in enumerate(FOLDS):
        d = pd.read_csv(path)
        d = d.assign(yhat_mag=-d.y_pred.abs())
        scores = [c for c in d.columns if c in SCORES] + ["yhat_mag", "random"]
        acc = {s: [] for s in scores}
        for _, g in d.groupby("metal_species_pair"):
            res = (g.y_true - g.y_pred).values
            e2 = res ** 2
            n = len(g)
            for s in scores:
                sc = None if s == "random" else s
                ok, sd = eligible(g, sc)
                if not ok:
                    if sc is not None and len(g) >= MIN_N and np.isfinite(sd) and sd > 0:
                        undef[s] = undef.get(s, 0) + 1
                    continue
                if s == "random":
                    tot = np.zeros(len(GRID))
                    for _ in range(NPERM):
                        se = e2[rng.permutation(n)]
                        tot += [np.sqrt(se[:max(2, int(round(r * n)))].mean()) / sd for r in GRID]
                    acc[s].append(tot / NPERM)
                else:
                    se = e2[np.argsort(g[s].values, kind="mergesort")]
                    acc[s].append(np.array(
                        [np.sqrt(se[:max(2, int(round(r * n)))].mean()) / sd for r in GRID]))
        for s, v in acc.items():
            if v:
                per_fold.setdefault(s, []).append(np.mean(np.vstack(v), axis=0))
    return per_fold, undef


SCORES = ["sigma_total", "sigma_aleatoric", "euclidean_feature_median", "euclidean_feature_nn",
          "mahalanobis_feature_centroid", "mahalanobis_feature_nn",
          "mahalanobis_feature_knn_mean_5", "mahalanobis_feature_knn_mean_10",
          "tanimoto_median_distance", "tanimoto_nn_distance",
          "tanimoto_knn_mean_5", "tanimoto_knn_mean_10", "pair_density_risk"]

per_fold, undef = recompute()
stored = pd.read_csv(T + "within_pair_retention_scalefree.csv")

# 1. every stored (score, retention) value reproduces to 3 dp
mismatch = []
for s, mat in per_fold.items():
    M = np.vstack(mat)
    for j, r in enumerate(GRID):
        row = stored[(stored.score == s) & (np.isclose(stored.retention.fillna(-1), r))]
        if not len(row):
            mismatch.append((s, r, "absent from table")); continue
        got, exp = M[:, j].mean(), row.fold_mean.iloc[0]
        if abs(got - exp) > 5e-4:
            mismatch.append((s, r, f"recomputed={got:.4f} stored={exp:.4f}"))
        gsd, esd = M[:, j].std(ddof=1), row.fold_sd.iloc[0]
        if abs(gsd - esd) > 5e-4:
            mismatch.append((s, r, f"sd recomputed={gsd:.4f} stored={esd:.4f}"))
chk("all_scalefree_retention_values_reproduce_to_3dp", not mismatch,
    f"checked={sum(len(GRID) for _ in per_fold)} mismatched={len(mismatch)} {mismatch[:3]}")

# 2. the unfiltered baseline is identical for every score (r = 1.0 retains everything)
base = {s: np.vstack(m)[:, -1].mean() for s, m in per_fold.items()}
chk("unfiltered_baseline_identical_across_scores",
    max(base.values()) - min(base.values()) < 1e-9, f"spread={max(base.values())-min(base.values()):.2e}")
chk("unfiltered_baseline_matches_manuscript_0974",
    abs(base["sigma_total"] - 0.974) < 6e-4, f"recomputed={base['sigma_total']:.4f} manuscript=0.974")

# 3. headline retention values quoted in the main text
for s, r, exp in [("sigma_total", 0.5, 0.516), ("euclidean_feature_median", 0.5, 0.608),
                  ("tanimoto_nn_distance", 0.5, 0.859), ("random", 0.5, 0.928),
                  ("yhat_mag", 0.5, 1.167), ("yhat_mag", 0.3, 1.302)]:
    got = np.vstack(per_fold[s])[:, GRID.index(r)].mean()
    chk(f"manuscript_value:{s}@r={r}", abs(got - exp) < 6e-4, f"recomputed={got:.4f} manuscript={exp}")

# 4. the 47% reduction claim
red = 100 * (1 - np.vstack(per_fold["sigma_total"])[:, GRID.index(0.5)].mean() / base["sigma_total"])
chk("manuscript_reduction_is_47pct", abs(red - 47) < 0.5, f"recomputed={red:.1f}%")

# 5. the undefined-by-construction result
chk("pair_density_undefined_in_every_eligible_cell",
    "pair_density_risk" not in per_fold and undef.get("pair_density_risk", 0) > 0,
    f"undefined_cells={undef.get('pair_density_risk', 0)} (manuscript: 1,124)")
chk("pair_density_undefined_cell_count_is_1124",
    undef.get("pair_density_risk", 0) == 1124, f"count={undef.get('pair_density_risk', 0)}")

# 6. fingerprint NN is not better than random where the manuscript says so
for r in (0.3,):
    fp = np.vstack(per_fold["tanimoto_nn_distance"])[:, GRID.index(r)].mean()
    rd = np.vstack(per_fold["random"])[:, GRID.index(r)].mean()
    chk(f"fingerprint_nn_not_better_than_random@r={r}", fp >= rd - 1e-9,
        f"fingerprint={fp:.4f} random={rd:.4f}")

# 7. predicted magnitude is worse than not filtering
for r in (0.5, 0.3):
    y = np.vstack(per_fold["yhat_mag"])[:, GRID.index(r)].mean()
    chk(f"yhat_worse_than_unfiltered@r={r}", y > base["sigma_total"],
        f"yhat={y:.4f} unfiltered={base['sigma_total']:.4f}")

# 8. scale-free coverage grid: marginal spans and the two highlighted cells
g = pd.read_csv(T + "scale_free_coverage_grid.csv")
sm = g[g.section == "sigma_marginal"].sort_values("sigma_quartile").fold_mean.values
pm = g[g.section == "support_marginal"].sort_values("support_quartile").fold_mean.values
ov = g[g.section == "overall"].fold_mean.iloc[0]
gr = g[g.section == "grid"]
cell = lambda p, s: gr[(gr.support_quartile == p) & (gr.sigma_quartile == s)].fold_mean.iloc[0]
chk("sigma_marginal_monotone_increasing", bool(np.all(np.diff(sm) > 0)), np.round(sm, 3).tolist())
chk("sigma_marginal_span_is_2.1", abs(sm.max() / sm.min() - 2.1) < 0.05, f"{sm.max()/sm.min():.2f}")
chk("support_marginal_span_is_1.2", abs(pm.max() / pm.min() - 1.2) < 0.05, f"{pm.max()/pm.min():.2f}")
chk("dense_high_sigma_is_1.49x_overall", abs(cell(4, 4) / ov - 1.49) < 0.01, f"{cell(4,4)/ov:.3f}")
chk("sparse_low_sigma_is_0.68x_overall", abs(cell(1, 1) / ov - 0.68) < 0.01, f"{cell(1,1)/ov:.3f}")
chk("coverage_interaction_low_sigma_halves", abs(cell(4, 1) / cell(1, 1) - 0.50) < 0.02,
    f"dense/sparse at low sigma = {cell(4,1)/cell(1,1):.3f} (manuscript: halves)")
chk("coverage_interaction_high_sigma_no_benefit", cell(4, 4) >= cell(1, 4),
    f"dense={cell(4,4):.3f} sparse={cell(1,4):.3f} at high sigma")

n_pass = sum(c["passed"] for c in checks)
out = dict(**{"pass": n_pass == len(checks)}, n_checks=len(checks), n_pass=n_pass, checks=checks)
json.dump(out, open(T + "validate_scalefree_central.json", "w"), indent=2)
for c in checks:
    print(("OK  " if c["passed"] else "FAIL"), c["check"], "|", c["detail"])
print(f"\n{n_pass}/{len(checks)} passed")
sys.exit(0 if n_pass == len(checks) else 1)
