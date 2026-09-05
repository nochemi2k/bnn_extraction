"""Two robustness analyses requested at review.

(A) Cell-threshold sensitivity: does the ordering of reliability estimates survive
    admitting sparser cells (n >= 20, 10, 5)?
(B) Model-skill conditionality: does it survive on models of different skill,
    including one excluded by the calibration screen despite lower point error,
    and one on an architecturally unrelated representation?

Frozen fold scores only; no retraining. Outputs a single tidy table.
"""
import json, subprocess
import numpy as np
import pandas as pd

ROOT = "results/downstream_cv/ad_retention/fold_scores/"
MODELS = ["DS2_RDKIT_MLPENS5_HETERO", "DS2_RDKIT_MLP_MCDROPOUT",
          "DS2_RDKIT_MLPENS5_EVID", "DS2_GRAPH_DMPNN_EVID"]
SCORES = ["sigma_total", "euclidean_feature_median", "mahalanobis_feature_centroid",
          "tanimoto_nn_distance", "tanimoto_knn_mean_5", "tanimoto_knn_mean_10"]
MIN_DISTINCT = 5


def srho(a, b):
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def analyse(model, min_n, nperm=20, seed=0):
    rng = np.random.RandomState(seed)
    rho = {s: [] for s in SCORES}
    ret = {s: [] for s in SCORES + ["random"]}
    base, ncells = [], []
    for i in range(5):
        d = pd.read_csv(f"{ROOT}{model}/fold{i}.csv")
        r_f = {s: [] for s in SCORES}
        t_f = {s: [] for s in SCORES + ["random"]}
        b_f, cnt = [], 0
        for _, g in d.groupby("metal_species_pair"):
            n = len(g); sd = g.y_true.std(ddof=1)
            if n < min_n or not np.isfinite(sd) or sd <= 0:
                continue
            res = (g.y_true - g.y_pred).values; e2 = res ** 2
            b_f.append(np.sqrt(e2.mean()) / sd); cnt += 1
            k = max(2, int(round(0.5 * n)))
            for s in SCORES:
                if s not in g.columns or g[s].nunique() < MIN_DISTINCT:
                    continue
                r_f[s].append(srho(g[s].values, np.abs(res)))
                t_f[s].append(np.sqrt(e2[np.argsort(g[s].values, kind="mergesort")][:k].mean()) / sd)
            acc = 0.0
            for _ in range(nperm):
                acc += np.sqrt(e2[rng.permutation(n)][:k].mean()) / sd
            t_f["random"].append(acc / nperm)
        base.append(np.mean(b_f)); ncells.append(cnt)
        for s in SCORES:
            if r_f[s]: rho[s].append(np.nanmean(r_f[s])); ret[s].append(np.mean(t_f[s]))
        ret["random"].append(np.mean(t_f["random"]))
    out = dict(model=model, min_cell_n=min_n, cells_per_fold=float(np.mean(ncells)),
               unfiltered=float(np.mean(base)), unfiltered_sd=float(np.std(base, ddof=1)))
    for s in SCORES:
        out[f"rho_{s}"] = float(np.mean(rho[s])) if rho[s] else np.nan
        out[f"ret50_{s}"] = float(np.mean(ret[s])) if ret[s] else np.nan
    out["ret50_random"] = float(np.mean(ret["random"]))
    return out


rows = []
print("=== (A) cell-threshold sensitivity, workhorse ===")
for m in [20, 10, 5]:
    r = analyse(MODELS[0], m); rows.append(r)
    print(f"  n>={m:2d}  cells/fold {r['cells_per_fold']:5.0f}  unfiltered {r['unfiltered']:.3f}  "
          f"rho(sigma) {r['rho_sigma_total']:+.3f}  rho(fpNN) {r['rho_tanimoto_nn_distance']:+.3f}  "
          f"ret50 sigma {r['ret50_sigma_total']:.3f} vs random {r['ret50_random']:.3f}")

print("\n=== (B) model-skill conditionality, n>=20 ===")
for mod in MODELS[1:]:
    r = analyse(mod, 20); rows.append(r)
    print(f"  {mod:26s} unfiltered {r['unfiltered']:.3f}  rho(sigma) {r['rho_sigma_total']:+.3f}  "
          f"rho(fpNN) {r['rho_tanimoto_nn_distance']:+.3f}  rho(euclid) {r['rho_euclidean_feature_median']:+.3f}  "
          f"ret50 sigma {r['ret50_sigma_total']:.3f} random {r['ret50_random']:.3f}")

df = pd.DataFrame(rows)
df["source"] = ROOT + "{model}/fold{0..4}.csv"
df.to_csv("manuscripts/jacs/tables/robustness_threshold_and_model.csv", index=False)
try: c = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception: c = "unavailable"
json.dump(dict(git_commit=c, note=(
    "Robustness. (A) cell-threshold sensitivity at n>=20/10/5 on the workhorse. "
    "(B) the same analysis on three further frozen DS2 models, including "
    "DS2_RDKIT_MLP_MCDROPOUT, which the calibration screen excluded despite lower "
    "point error, and DS2_GRAPH_DMPNN_EVID on an unrelated graph representation. "
    "Scale-free statistic throughout; 20 random permutations per cell, seed 0. No retraining.")),
    open("manuscripts/jacs/tables/robustness_threshold_and_model.csv.provenance.json", "w"), indent=2)
