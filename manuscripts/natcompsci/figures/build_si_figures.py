"""Nature Computational Science supplementary figures (Fig. S1-S10).

Reads the deposited derived tables and the frozen fold-level prediction files.
No model is loaded, trained or rescored: Fig. S1 reads the stored predictions of
the workhorse and recomputes descriptive statistics from them; every other panel
is drawn from a table listed in the artifact inventory (Supplementary S9.2).

Palette matches the main-text figures:
  #0072B2 blue       model uncertainty
  #D55E00 vermillion predicted magnitude, the harmful comparator
  #009E73 green      descriptor-space distances
  #CC79A7 pink       fingerprint distances
  #E69F00 orange     training-frequency count
  #6E7079 gray       random baseline / neutral reference
Every series is additionally direct-labelled or separated by linestyle.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(OUT, os.pardir, os.pardir, os.pardir))
T = os.environ.get("BNN_TABLES", os.path.join(_REPO, "manuscripts", "jacs", "tables")) + os.sep
TF = os.environ.get("BNN_TABLES_FINAL", os.path.join(_REPO, "manuscripts", "final", "tables")) + os.sep
PRED = os.environ.get(
    "BNN_PREDICTIONS",
    os.path.join(_REPO, "results", "cv_predictions", "fold_test_predictions",
                 "DS2_RDKIT_MLPENS5_HETERO"))

C_SIG, C_YHAT = "#0072B2", "#D55E00"
C_DESC, C_FP, C_CNT, C_RAND = "#009E73", "#CC79A7", "#E69F00", "#6E7079"
INK, INK2, INK3 = "#1a1a1a", "#4a4a4a", "#8a8a8a"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.labelsize": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.6,
    "axes.edgecolor": INK3, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK, "figure.dpi": 200, "savefig.dpi": 400,
    "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
})

CMAP = LinearSegmentedColormap.from_list("bnn", ["#f7f9fb", "#9ecae1", "#3182bd", "#08306b"])


def grid(ax, axis="y"):
    ax.grid(axis=axis, color="#e6e6e6", lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def tag(fig, ax, s, dx=0.052, dy=0.045):
    p = ax.get_position()
    fig.text(p.x0 - dx, p.y1 + dy, s, fontsize=11, fontweight="bold",
             va="top", ha="left", color=INK)


def save(fig, name):
    for e in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{e}")
    plt.close(fig)
    print("wrote", name)


def load_folds():
    frames = []
    for k in range(5):
        p = os.path.join(PRED, f"fold{k}", "fold_test_predictions.csv")
        d = pd.read_csv(p, usecols=["fold_id", "y_true", "y_pred", "sigma_total",
                                    "uncertainty_status"])
        frames.append(d[d.uncertainty_status == "valid"])
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------- Fig. S1  the model
def figS2():
    d = load_folds()
    d = d.dropna(subset=["y_true", "y_pred", "sigma_total"])
    resid = d.y_pred - d.y_true

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.4, 2.75))

    lim = 15
    hb = a.hexbin(d.y_true, d.y_pred, gridsize=52, extent=(-lim, lim, -lim, lim),
                  bins="log", cmap=CMAP, linewidths=0)
    a.plot([-lim, lim], [-lim, lim], color=C_YHAT, lw=1.0, ls="--", zorder=5)
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
    a.set_xlabel("observed $\\Delta\\log K$")
    a.set_ylabel("predicted $\\Delta\\log K$")
    a.set_title("Held-out predictions", loc="left", pad=8)
    cb = fig.colorbar(hb, ax=a, fraction=0.046, pad=0.03)
    cb.set_label("rows", fontsize=7.4)
    cb.ax.tick_params(labelsize=7)
    rmse = float(np.sqrt((resid ** 2).mean()))
    a.text(0.04, 0.96, f"n = {len(d):,} rows\nrow-level RMSE {rmse:.2f}",
           transform=a.transAxes, va="top", ha="left", fontsize=7.2, color=INK,
           bbox=dict(fc="white", ec="#dcdcdc", lw=0.6, pad=3.2))

    b.hist(resid, bins=np.linspace(-10, 10, 81), color="#9ecae1",
           edgecolor="#3182bd", lw=0.4)
    b.axvline(0, color=C_YHAT, lw=1.0, ls="--")
    b.set_xlabel("prediction − observation ($\\Delta\\log K$)")
    b.set_ylabel("held-out rows")
    b.set_title("Residual distribution", loc="left", pad=8)
    b.set_xlim(-10, 10)
    grid(b)
    q = np.percentile(resid, [5, 50, 95])
    b.text(0.03, 0.96, f"median {q[1]:+.2f}\n5–95% [{q[0]:+.1f}, {q[2]:+.1f}]",
           transform=b.transAxes, va="top", ha="left", fontsize=7.0, color=INK2,
           bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.6))

    from math import erf, sqrt
    levels = np.arange(0.1, 0.95, 0.1)
    zc = np.array([np.sqrt(2) * _erfinv(l) for l in levels])
    z = np.abs(resid.to_numpy()) / d.sigma_total.to_numpy()
    emp = np.array([(z <= zz).mean() for zz in zc])
    per_fold = []
    for f, g in d.groupby("fold_id"):
        zz = np.abs((g.y_pred - g.y_true).to_numpy()) / g.sigma_total.to_numpy()
        per_fold.append([(zz <= t).mean() for t in zc])
    per_fold = np.array(per_fold)
    c.plot([0, 1], [0, 1], color=INK3, lw=0.9, ls="--", zorder=2)
    c.fill_between(levels, per_fold.min(0), per_fold.max(0), color=C_SIG,
                   alpha=0.16, lw=0, zorder=3)
    c.plot(levels, emp, "o-", color=C_SIG, lw=1.5, ms=4.2, zorder=4)
    c.set_xlim(0.05, 0.95); c.set_ylim(0.05, 0.95)
    c.set_xlabel("nominal coverage")
    c.set_ylabel("empirical coverage")
    c.set_title("Interval calibration", loc="left", pad=8)
    grid(c)
    ace = float(np.abs(emp - levels).mean())
    i90 = np.argmin(np.abs(levels - 0.9))
    c.text(0.97, 0.10, f"row-level ACE {ace:.3f}\n{np.interp(0.9, levels, emp):.3f} "
                       f"at 0.90 nominal",
           transform=c.transAxes, va="bottom", ha="right", fontsize=7.0, color=INK,
           bbox=dict(fc="white", ec="#dcdcdc", lw=0.6, pad=2.8))

    fig.tight_layout(w_pad=2.4)
    for ax, t in ((a, "a"), (b, "b"), (c, "c")):
        tag(fig, ax, t, dx=0.048, dy=0.040)
    save(fig, "figS2_model_predictions")
    return dict(n=len(d), rmse=rmse, ace=ace,
                cov90=float(np.interp(0.9, levels, emp)))


def _erfinv(y):
    # Winitzki approximation refined by two Newton steps; sufficient for plotting
    a = 0.147
    ln = np.log(1 - y ** 2)
    t1 = 2 / (np.pi * a) + ln / 2
    x = np.sign(y) * np.sqrt(np.sqrt(t1 ** 2 - ln / a) - t1)
    for _ in range(3):
        from math import erf
        x = x - (erf(x) - y) / (2 / np.sqrt(np.pi) * np.exp(-x ** 2))
    return x


# ------------------------------------------- Fig. S2  dataset breakdown
BLOCK_ORDER = ["4f", "4d-5d", "p_block", "3d", "s", "5f"]
BLOCK_LABEL = {"4f": "4f", "4d-5d": "4d–5d", "p_block": "p", "3d": "3d",
               "s": "s", "5f": "5f"}
HIST_BINS = ["hist_(-inf,-1]", "hist_(-1,0]", "hist_(0,1]", "hist_(1,2]",
             "hist_(2,3]", "hist_(3,5]", "hist_(5,10]", "hist_(10,20]",
             "hist_(20,inf)"]
HIST_TICK = ["≤−1", "−1–0", "0–1", "1–2", "2–3", "3–5", "5–10", "10–20", ">20"]


def figS1():
    d = load_folds()
    lig = (pd.concat([pd.read_csv(os.path.join(PRED, f"fold{k}",
                                               "fold_test_predictions.csv"),
                                  usecols=["ligand_smiles"])
                      for k in range(5)], ignore_index=True)
           .ligand_smiles.value_counts())
    cen = pd.read_csv(T + "logk_censoring.csv")
    cen = cen[cen.dataset == "DS2_endpoints_pooled"]

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.5, 2.9),
                                  gridspec_kw=dict(width_ratios=[1.0, 1.10, 0.95]))

    a.hist(d.y_true, bins=np.linspace(-20, 20, 81), color="#9ecae1",
           edgecolor="#3182bd", lw=0.4)
    a.set_xlabel("$\\Delta\\log K$ (the target)")
    a.set_ylabel("held-out records")
    a.set_xlim(-20, 20)
    a.set_title("The selectivity target", loc="left", pad=8)
    q = np.percentile(d.y_true, [5, 50, 95])
    a.text(0.03, 0.96, f"n = {len(d):,}\nmedian {q[1]:+.2f}\n"
                       f"5–95% [{q[0]:+.1f}, {q[2]:+.1f}]",
           transform=a.transAxes, va="top", ha="left", fontsize=7.0, color=INK2)
    grid(a)

    tot = cen[cen.metal_block == "all"]
    x = np.arange(len(HIST_BINS))
    share_all = np.array([float(tot[h].iloc[0]) for h in HIST_BINS])
    share_all = share_all / share_all.sum() * 100
    b.bar(x, share_all, 0.78, color="#c9d2d8", edgecolor=INK3, lw=0.4, zorder=3,
          label="all metals")
    for blk, col in (("5f", C_YHAT), ("4f", C_SIG)):
        row = cen[cen.metal_block == blk]
        v = np.array([float(row[h].iloc[0]) for h in HIST_BINS])
        v = v / v.sum() * 100
        b.plot(x, v, "o-", ms=3.6, lw=1.4, color=col, zorder=5,
               label=f"{BLOCK_LABEL[blk]}-block metals")
    b.axvspan(-0.5, 2.5, color="#fbeaea", alpha=0.7, zorder=1)
    b.set_xticks(x); b.set_xticklabels(HIST_TICK, fontsize=6.6, rotation=45,
                                       ha="right")
    b.set_xlim(-0.5, len(HIST_BINS) - 0.5)
    b.set_xlabel("measured $\\log K$ of one metal")
    b.set_ylabel("share of measurements (%)")
    b.set_title("What the record contains", loc="left", pad=8)
    b.legend(frameon=False, loc="upper right", fontsize=6.8)
    grid(b)

    share = np.cumsum(lig.values) / lig.values.sum() * 100
    rank = np.arange(1, len(share) + 1)
    c.plot(rank, share, "-", color=C_SIG, lw=1.8, zorder=4)
    for r in (10, 100):
        c.plot([r, r], [0, share[r - 1]], ls=":", lw=0.9, color=INK3, zorder=3)
        c.plot([1, r], [share[r - 1]] * 2, ls=":", lw=0.9, color=INK3, zorder=3)
        c.text(1.15, share[r - 1] + 2.5, f"top {r}: {share[r - 1]:.0f}%",
               fontsize=7.0, color=INK2, ha="left", va="bottom")
    c.set_xscale("log")
    c.set_xlim(1, len(share))
    c.set_ylim(0, 104)
    c.set_xlabel("ligands, ranked by frequency")
    c.set_ylabel("cumulative share of records (%)")
    c.set_title("The ligand axis", loc="left", pad=8)
    c.text(0.96, 0.06, f"{len(lig):,} distinct ligands", transform=c.transAxes,
           va="bottom", ha="right", fontsize=7.0, color=INK2)
    grid(c)

    fig.tight_layout(w_pad=2.2)
    for ax, s in ((a, "a"), (b, "b"), (c, "c")):
        tag(fig, ax, s, dx=0.055, dy=0.048)
    save(fig, "figS1_dataset_breakdown")


# ------------------------------------------- Fig. S3  uncertainty by regime
def figS3():
    u = pd.read_csv(T + "uncertainty_by_regime.csv")
    r = pd.read_csv(T + "regime_normalized_error.csv")
    g = lambda reg, m: u[(u.regime == reg) & (u.metric_name == m)].iloc[0]
    h = lambda reg, m: r[(r.regime == reg) & (r.metric_name == m)].iloc[0]

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.9))
    regimes = ["near_zero", "high_selectivity"]
    labels = ["$|\\Delta\\log K| < 1$\n(near zero)",
              "$|\\Delta\\log K| > 1$\n(high selectivity)"]
    x = np.arange(2); w = 0.34

    for ax, getter, sig_m, err_m, ylab, ttl, ylim in (
            (a, g, "sigma_total", "abs_error", "$\\Delta\\log K$",
             "In the units of the measurement", 3.05),
            (b, h, "normalized_sigma_total", "normalized_error",
             "multiples of the pair's spread",
             "In normalized units", 1.65)):
        sig = [getter(reg, sig_m).fold_mean for reg in regimes]
        sig_sd = [getter(reg, sig_m).fold_sd for reg in regimes]
        err = [getter(reg, err_m).fold_mean for reg in regimes]
        err_sd = [getter(reg, err_m).fold_sd for reg in regimes]
        ax.bar(x - w / 2, sig, w, yerr=sig_sd, capsize=3, color=C_SIG,
               label="$\\sigma_{\\mathrm{total}}$", zorder=3)
        ax.bar(x + w / 2, err, w, yerr=err_sd, capsize=3, color="#b8c4cc",
               edgecolor=INK3, lw=0.5, label="error", zorder=3)
        for xi, v, e in list(zip(x - w / 2, sig, sig_sd)) + list(zip(x + w / 2, err, err_sd)):
            ax.text(xi, v + e + ylim * 0.03, f"{v:.2f}", ha="center", fontsize=7.4,
                    color=INK)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, ylim)
        ax.set_title(ttl, loc="left", pad=8)
        grid(ax)
    b.axhline(1.0, color=INK3, lw=0.8, ls=":", zorder=2)
    a.legend(frameon=False, loc="upper left")

    fig.tight_layout(w_pad=2.6)
    tag(fig, a, "a", dx=0.070); tag(fig, b, "b", dx=0.070)
    save(fig, "figS3_uncertainty_by_regime")


# --------------------------------- Fig. S5  raw versus normalized error
def figS4():
    """Every model that produces a pairwise selectivity prediction, both routes.

    The main-text panel shows a readable subset; this is the whole grid, with
    the point-only ad hoc models on the accuracy axis alone, since they produce
    no interval to calibrate.
    """
    s = pd.read_csv(T + "selectivity_construction.csv")
    AH = "ad hoc, difference of two absolute predictions"
    PT = "ad hoc, point models without an interval"
    DI = "direct, modelled on the pairwise target"

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 4.4),
                               gridspec_kw=dict(width_ratios=[1.25, 1.0]))

    # ---- a: every model, named, ordered by calibration
    d = s[s.construction != PT].sort_values("mop_ace", ascending=False)
    y = np.arange(len(d))
    col = [C_FP if c == AH else C_SIG for c in d.construction]
    a.barh(y, d.mop_ace, height=0.66, color=col, zorder=3,
           xerr=d.mop_ace_sd, error_kw=dict(ecolor=INK2, elinewidth=0.7, capsize=1.8))
    a.set_yticks(y)
    a.set_yticklabels([f"{m}" for m in d.source_model], fontsize=6.4)
    for yi, wk in zip(y, d.is_workhorse):
        if wk:
            a.text(0.009, yi, "\u2605", fontsize=8, color="white", ha="left",
                   va="center", zorder=5)
    a.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_SIG,
                                    label="direct"),
                      plt.Rectangle((0, 0), 1, 1, color=C_FP,
                                    label="ad hoc, composed"),
                      plt.Line2D([], [], marker="$\u2605$", ls="", color=INK2, ms=7,
                                 label="selected model")],
             frameon=False, loc="upper right", fontsize=6.4, labelspacing=0.4,
             handlelength=1.1, handletextpad=0.5, borderaxespad=0.6)
    best_ah = float(s[s.construction == AH].mop_ace.min())
    a.axvline(best_ah, color=INK3, ls=(0, (5, 3)), lw=1.0, zorder=4)
    a.text(best_ah + 0.006, len(d) - 0.45, "best composed", fontsize=6.5,
           color=INK2, ha="left", va="center")
    a.set_xlabel("average coverage error  (pair-equal)")
    a.set_xlim(0, 0.42)
    a.set_ylim(-0.8, len(d) - 0.2)
    a.set_title("Every interval-bearing model", loc="left", pad=8)
    grid(a, "x")

    # ---- b: accuracy against calibration, the panel the main text subsets
    for key, c2, mk, lab in ((AH, C_FP, "o", "ad hoc, composed"),
                             (DI, C_SIG, "s", "direct")):
        g = s[s.construction == key]
        b.errorbar(g.mop_rmse, g.mop_ace, xerr=g.mop_rmse_sd, yerr=g.mop_ace_sd,
                   fmt=mk, ms=4.4, color=c2, ecolor=c2, elinewidth=0.6,
                   capsize=1.4, lw=0, zorder=4, alpha=0.9, label=lab,
                   markeredgecolor="white", markeredgewidth=0.4)
    w = s[s.is_workhorse].iloc[0]
    b.plot(w.mop_rmse, w.mop_ace, "*", ms=13, color=C_SIG, zorder=6,
           markeredgecolor="white", markeredgewidth=0.7, label="selected model")
    for _, r in s[s.construction == PT].iterrows():
        b.axvline(float(r.mop_rmse), color=INK3, ls=":", lw=0.8, zorder=2)
    pt = s[s.construction == PT].sort_values("mop_rmse")
    b.axvline(float(pt.mop_rmse.iloc[0]), color=INK3, ls=(0, (5, 3)), lw=1.0,
              zorder=3)
    b.set_xlabel("mean-of-pairs RMSE  ($\\Delta\\log K$)")
    b.set_ylabel("average coverage error  (pair-equal)")
    b.set_title("Accuracy against calibration", loc="left", pad=8)
    b.legend(frameon=False, loc="upper left", fontsize=6.6, labelspacing=0.4,
             handletextpad=0.5, borderaxespad=0.25)
    b.text(0.99, 0.015,
           "vertical rules:\npoint-only ad hoc models\n(best of them dashed)",
           transform=b.transAxes, fontsize=6.3, color=INK2, ha="right", va="bottom",
           linespacing=1.35)
    grid(b, "both")

    fig.tight_layout(w_pad=2.0)
    tag(fig, a, "a"); tag(fig, b, "b")
    save(fig, "figS4_selectivity_construction")


def figS5():
    p = pd.read_csv(T + "per_pair_difficulty.csv")

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    a.scatter(p.sd_p, p.mean_abs_error, s=13, color="#8fbcdb", edgecolor="none",
              alpha=0.75, zorder=3)
    hi = max(p.sd_p.max(), p.mean_abs_error.max()) * 1.02
    a.plot([0, hi], [0, hi], color=INK3, lw=0.9, ls="--", zorder=2)
    a.text(hi * 0.62, hi * 0.68, "error = pair spread", fontsize=7.0, color=INK2,
           rotation=38, ha="center")
    a.set_xlabel("spread of the pair's observed selectivity, $\\mathrm{SD}_p$")
    a.set_ylabel("mean absolute error  ($\\Delta\\log K$)")
    a.set_title("Per-pair error against pair spread", loc="left", pad=8)
    grid(a, axis="both")

    b.scatter(p.mean_abs_error, p.mean_e_norm, s=13, color="#8fbcdb",
              edgecolor="none", alpha=0.75, zorder=3)
    b.axhline(1.0, color=INK3, lw=0.9, ls=":", zorder=2)
    b.text(p.mean_abs_error.max() * 0.98, 1.04, "normalized error = 1",
           fontsize=7.0, color=INK2, ha="right", va="bottom")
    b.set_xlabel("mean absolute error  ($\\Delta\\log K$)")
    b.set_ylabel("normalized error")
    b.set_title("Absolute against normalized error", loc="left", pad=8)
    grid(b, axis="both")

    fig.tight_layout(w_pad=2.6)
    tag(fig, a, "a", dx=0.085); tag(fig, b, "b", dx=0.085)
    save(fig, "figS5_raw_versus_normalized")


# ------------------------------- Fig. S6  the full 23-score inventory
INV_FAMILY_COLOR = {"uncertainty": C_SIG, "descriptor": C_DESC,
                    "fingerprint": C_FP, "learned-embedding": "#7b3294",
                    "count": C_CNT}
INV_NAME = {
    "sigma_total": "predictive uncertainty $\\sigma_{\\mathrm{total}}$",
    "sigma_aleatoric": "$\\sigma_{\\mathrm{aleatoric}}$",
    "sigma_epistemic_within": "$\\sigma_{\\mathrm{epistemic,\\,within}}$",
    "sigma_epistemic_between": "$\\sigma_{\\mathrm{epistemic,\\,between}}$",
    "pair_density_risk": "training-frequency count",
    "euclidean_feature_median": "descriptor Euclid. median",
    "euclidean_feature_nn": "descriptor Euclid. NN",
    "mahalanobis_feature_median": "descriptor Mahal. median",
    "mahalanobis_feature_centroid": "descriptor Mahal. centroid",
    "mahalanobis_feature_nn": "descriptor Mahal. NN",
    "mahalanobis_feature_knn_mean_5": "descriptor Mahal. $k$NN-5",
    "mahalanobis_feature_knn_mean_10": "descriptor Mahal. $k$NN-10",
    "tanimoto_median_distance": "fingerprint median",
    "tanimoto_nn_distance": "fingerprint NN",
    "tanimoto_knn_mean_5": "fingerprint $k$NN-5",
    "tanimoto_knn_mean_10": "fingerprint $k$NN-10",
    "chemprop_euclidean_nn": "embedding Euclid. NN, unscaled",
    "chemprop_std_euclidean_nn": "embedding Euclid. NN",
    "chemprop_std_euclidean_knn_5": "embedding Euclid. $k$NN-5",
    "chemprop_std_euclidean_knn_10": "embedding Euclid. $k$NN-10",
    "chemprop_cosine_nn": "embedding cosine NN",
    "chemprop_mahalanobis_centroid": "embedding Mahal. centroid",
    "chemprop_mahalanobis_nn": "embedding Mahal. NN",
}
MEDIANS = {"euclidean_feature_median", "mahalanobis_feature_median",
           "tanimoto_median_distance"}


def _inv_family(score):
    if score.startswith("sigma"):
        return "uncertainty"
    if score.startswith("tanimoto"):
        return "fingerprint"
    if score.startswith("chemprop"):
        return "learned-embedding"
    if "pair_density" in score:
        return "count"
    return "descriptor"


def figS6():
    z = pd.read_csv(TF + "iz_risk_score_spearman.csv").dropna(subset=["fold_mean"])
    z = z.assign(family=lambda x: x.score.map(_inv_family)).sort_values("fold_mean")
    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    y = np.arange(len(z))
    ax.axvline(0, color=INK, lw=1.0, zorder=4)
    for yi, (_, r) in zip(y, z.iterrows()):
        col = INV_FAMILY_COLOR[r.family]
        ax.plot([0, r.fold_mean], [yi, yi], color=col, lw=1.0, alpha=0.5, zorder=3)
        ax.errorbar(r.fold_mean, yi, xerr=r.fold_sd, fmt="o", ms=4.4, color=col,
                    ecolor=col, elinewidth=0.9, capsize=2.2, zorder=5)
    ax.set_yticks(y)
    labels = ax.set_yticklabels([INV_NAME.get(s, s) for s in z.score], fontsize=6.8)
    for lab, s in zip(labels, z.score):
        if s in MEDIANS or s == "sigma_total" or s == "pair_density_risk":
            lab.set_fontweight("bold")
    ax.set_ylim(-0.8, len(z) - 0.2)
    ax.set_xlabel("Spearman ρ with $|$held-out residual$|$")
    grid(ax, "x")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f, ms=5.0)
               for f, c in INV_FAMILY_COLOR.items()]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=6.8)
    fig.tight_layout()
    save(fig, "figS6_pooled_score_inventory")


# ------------------------------- Fig. S8  a second model and its own embedding
EMB_DISPLAY = [
    ("sigma_total", "predictive uncertainty $\\sigma_{\\mathrm{total}}$", C_SIG, "-", 2.4),
    ("euclidean_feature_median", "descriptor Euclid. median", C_DESC, "-", 1.7),
    ("tanimoto_median_distance", "fingerprint median", C_FP, "--", 1.7),
    ("random", "random retention", C_RAND, "--", 2.0),
    ("chemprop_std_euclidean_nn", "embedding Euclid. NN", "#7b3294", "-", 1.7),
    ("chemprop_mahalanobis_nn", "embedding Mahal. NN", "#b07aa1", "-.", 1.7),
]


def figS8():
    e = pd.read_csv(T + "embedding_retention.csv")
    unf = float(e[e.score == "unfiltered"].fold_mean.iloc[0])

    fig, ax = plt.subplots(figsize=(6.0, 4.3))
    handles = []
    for score, lab, col, ls, lw in EMB_DISPLAY:
        d = e[e.score == score].sort_values("retention")
        ax.fill_between(d.retention, d.fold_mean - d.fold_sd,
                        d.fold_mean + d.fold_sd, color=col, alpha=0.11, lw=0,
                        zorder=2)
        ln, = ax.plot(d.retention, d.fold_mean, ls, color=col, lw=lw, zorder=4,
                      label=lab)
        handles.append(ln)
    ax.axhline(unf, color=INK3, lw=0.9, ls=":", zorder=3)
    ax.text(1.0, unf + 0.04, f"{unf:.2f}, no filtering", fontsize=7.4, color=INK2,
            va="bottom", ha="right")
    ax.set_xlim(0.075, 1.03); ax.set_ylim(0.25, 2.35)
    ax.set_xticks([0.1, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["10%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("held-out predictions retained")
    ax.set_ylabel("mean-of-pairs RMSE  ($\\Delta\\log K$)")
    grid(ax)
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=7.4,
              handlelength=2.4, labelspacing=0.34)
    fig.tight_layout()
    save(fig, "figS8_embedding_retention")


# ----------------------------------------------------- Fig. S9  robustness
MODEL_LABEL = {
    "DS2_RDKIT_MLPENS5_HETERO": "MLP ens.\nhetero.\n(workhorse)",
    "DS2_RDKIT_MLP_MCDROPOUT": "MLP\nMC-dropout",
    "DS2_RDKIT_MLPENS5_EVID": "MLP ens.\nevidential",
    "DS2_GRAPH_DMPNN_EVID": "D-MPNN\nevidential",
}


def figS9():
    b0 = pd.read_csv(T + "robustness_threshold_and_model.csv")
    thr = b0[b0.model == "DS2_RDKIT_MLPENS5_HETERO"].sort_values("min_cell_n", ascending=False)
    mod = b0[b0.min_cell_n == 20].copy()
    mod["ord"] = mod.model.map({k: i for i, k in enumerate(MODEL_LABEL)})
    mod = mod.sort_values("ord")

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0))

    x = np.arange(len(thr)); w = 0.36
    a.bar(x - w / 2, thr.rho_sigma_total, w, color=C_SIG,
          label="$\\sigma_{\\mathrm{total}}$", zorder=3)
    a.bar(x + w / 2, thr.rho_tanimoto_nn_distance, w, color=C_FP,
          label="fingerprint NN", zorder=3)
    for xi, v in zip(x - w / 2, thr.rho_sigma_total):
        a.text(xi, v + 0.012, f"+{v:.3f}", ha="center", fontsize=7.2, color=INK)
    for xi, v in zip(x + w / 2, thr.rho_tanimoto_nn_distance):
        a.text(xi, v - 0.022, f"{v:.3f}", ha="center", va="top", fontsize=7.2, color=INK)
    a.axhline(0, color=INK, lw=1.0)
    a.set_xticks(x)
    a.set_xticklabels([f"n ≥ {int(n)}\n{c:.0f} cells/fold"
                       for n, c in zip(thr.min_cell_n, thr.cells_per_fold)])
    a.set_ylabel("Spearman ρ with absolute residual")
    a.set_ylim(-0.16, 0.52)
    a.set_title("Relaxing the cell threshold", loc="left", pad=8)
    a.legend(frameon=False, loc="upper right")
    grid(a)

    x = np.arange(len(mod))
    a2 = b.twinx()
    a2.plot(x, mod.unfiltered, "s--", color=INK3, lw=1.0, ms=4.6, zorder=5,
            label="unfiltered error")
    a2.set_ylim(0.70, 1.10)
    a2.set_ylabel("unfiltered normalized error", color=INK2, fontsize=8.4)
    a2.tick_params(labelsize=7.4)
    a2.spines["top"].set_visible(False)
    b.bar(x - w / 2, mod.rho_sigma_total, w, color=C_SIG, zorder=3)
    b.bar(x + w / 2, mod.rho_tanimoto_nn_distance, w, color=C_FP, zorder=3)
    for xi, v in zip(x - w / 2, mod.rho_sigma_total):
        b.text(xi, v + 0.012, f"+{v:.3f}", ha="center", fontsize=7.2, color=INK)
    for xi, v in zip(x + w / 2, mod.rho_tanimoto_nn_distance):
        b.text(xi, v - 0.022, f"{v:.3f}", ha="center", va="top", fontsize=7.2, color=INK)
    b.axhline(0, color=INK, lw=1.0)
    b.set_xticks(x)
    b.set_xticklabels([MODEL_LABEL[m] for m in mod.model], fontsize=7.0)
    b.set_ylabel("Spearman ρ with absolute residual")
    b.set_ylim(-0.16, 0.52)
    b.set_title("Four frozen models", loc="left", pad=8)
    b.legend(*a2.get_legend_handles_labels(), frameon=False, loc="upper center",
             fontsize=7.0)
    grid(b)

    fig.tight_layout(w_pad=3.4)
    tag(fig, a, "a", dx=0.058); tag(fig, b, "b", dx=0.048)
    save(fig, "figS9_robustness")


# --------------------------------- Fig. S7  pooled versus decision-level
DECOMP_ORDER = ["sigma_total", "euclidean_feature_median", "mahalanobis_feature_centroid",
                "tanimoto_median_distance", "tanimoto_nn_distance", "random", "unfiltered"]
DECOMP_LABEL = {"sigma_total": "$\\sigma_{\\mathrm{total}}$",
                "euclidean_feature_median": "descriptor Euclid. median",
                "mahalanobis_feature_centroid": "descriptor Mahal. centroid",
                "tanimoto_median_distance": "fingerprint median",
                "tanimoto_nn_distance": "fingerprint NN",
                "random": "random retention", "unfiltered": "no filtering"}
DECOMP_COLOR = {"sigma_total": C_SIG, "euclidean_feature_median": C_DESC,
                "mahalanobis_feature_centroid": C_DESC,
                "tanimoto_median_distance": C_FP, "tanimoto_nn_distance": C_FP,
                "random": C_RAND, "unfiltered": INK3}
PROTO = ["A_pooled_all_pairs", "B_pooled_eligible_cells", "C_within_cell_decision_level"]
PROTO_LABEL = ["A  pooled, all pairs", "B  pooled, eligible", "C  within cell"]


def figS7():
    d = pd.read_csv(T + "aggregation_decomposition.csv")
    L = pd.read_csv(T + "pooled_retention_pair_loss.csv").set_index("score")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.6, 3.6),
                                  gridspec_kw=dict(width_ratios=[1.42, 0.70, 0.62]))

    y = np.arange(len(DECOMP_ORDER))[::-1]
    h = 0.24
    for k, proto in enumerate(PROTO):
        sub = d[d.protocol == proto].set_index("score")
        for yi, sc in zip(y, DECOMP_ORDER):
            if sc not in sub.index:
                continue
            v = sub.loc[sc, "fold_mean"]
            a.barh(yi + (1 - k) * h, v, height=h * 0.92, color=DECOMP_COLOR[sc],
                   alpha=[0.45, 0.72, 1.0][k], edgecolor="white", lw=0.4, zorder=3)
            a.text(v + 0.03, yi + (1 - k) * h, f"{v:.2f}", va="center", fontsize=6.4,
                   color=INK2)
            if yi == y[0]:
                a.text(0.035, yi + (1 - k) * h, "ABC"[k], va="center", ha="left",
                       fontsize=6.6, color="white", fontweight="bold")
    a.set_yticks(y)
    a.set_yticklabels([DECOMP_LABEL[s] for s in DECOMP_ORDER], fontsize=7.4)
    a.set_xlim(0, 2.55)
    a.set_xlabel("mean-of-pairs RMSE at 50% retention  ($\\Delta\\log K$)")
    a.set_title("Three retention protocols", loc="left", pad=8)
    a.text(0.99, 0.985, "A  pooled, all pairs\nB  pooled, eligible cells\n"
                        "C  within cell", transform=a.transAxes, ha="right",
           va="top", fontsize=6.6, color=INK2, linespacing=1.5)
    grid(a, "x")

    tot = float(L.pairs_eligible_unfiltered.iloc[0])
    for yi, sc in zip(y, DECOMP_ORDER):
        if sc not in L.index:
            continue
        r = L.loc[sc]
        b.barh(yi, r.pairs_counted, height=0.52, color=DECOMP_COLOR[sc], zorder=4)
        b.barh(yi, r.pairs_below_mop_minimum, height=0.52, left=r.pairs_counted,
               color=DECOMP_COLOR[sc], alpha=0.42, zorder=4)
        b.barh(yi, r.pairs_losing_every_row, height=0.52,
               left=r.pairs_counted + r.pairs_below_mop_minimum,
               color="#e3e7ea", edgecolor=INK3, lw=0.4, zorder=3)
        b.text(r.pairs_counted / 2, yi, f"{r.pairs_counted:.0f}", ha="center",
               va="center", fontsize=6.6, color="white", fontweight="bold", zorder=6)
    b.axvline(tot, color=INK3, lw=0.8, ls=":", zorder=2)
    b.set_yticks(y); b.set_yticklabels([])
    b.set_ylim(-0.8, len(DECOMP_ORDER) - 0.2)
    b.set_xlim(0, tot * 1.06)
    b.set_xticks([0, 400, 800])
    b.set_xlabel("metal pairs")
    b.set_title("Pairs left under A", loc="left", pad=8)
    hb = [plt.Rectangle((0, 0), 1, 1, fc=INK3, ec="white"),
          plt.Rectangle((0, 0), 1, 1, fc=INK3, alpha=0.42, ec="white"),
          plt.Rectangle((0, 0), 1, 1, fc="#e3e7ea", ec=INK3, lw=0.4)]
    b.legend(hb, ["counted", "1–2 rows left", "no rows left"], frameon=False,
             loc="lower right", fontsize=6.2, handlelength=1.2, labelspacing=0.25)
    grid(b, "x")

    for yi, sc in zip(y, DECOMP_ORDER):
        if sc not in L.index:
            continue
        v = L.loc[sc, "between_pair_rank_variance_share"]
        if np.isnan(v):
            continue
        c.barh(yi, v, height=0.52, color=DECOMP_COLOR[sc], zorder=3)
        c.text(v + 0.03, yi, f"{v:.2f}", va="center", fontsize=6.8, color=INK2)
    c.set_yticks(y); c.set_yticklabels([])
    c.set_ylim(-0.8, len(DECOMP_ORDER) - 0.2)
    c.set_xlim(0, 1.28); c.set_xticks([0, 0.5, 1.0])
    c.set_xlabel("between-pair share")
    c.set_title("Variance between pairs", loc="left", pad=8)
    grid(c, "x")

    fig.tight_layout(w_pad=1.2)
    tag(fig, a, "a", dx=0.075, dy=0.048)
    tag(fig, b, "b", dx=0.026, dy=0.048)
    tag(fig, c, "c", dx=0.026, dy=0.048)
    save(fig, "figS7_aggregation_decomposition")


# ---------------------------------------------------------------- Fig. S10
# The acquisition benchmark repeated at stricter event definitions, in the same
# layout as main-text Fig. 4 so the two can be read against each other.
_RULES = [("expected_improvement", "standardised margin", C_SIG),
          ("lcb_1sigma", "lower bound, 1\u03c3", "#C4622D"),
          ("lcb_2sigma", "lower bound, 2\u03c3", "#E8A87C"),
          ("lcb_3sigma", "lower bound, 3\u03c3", "#F2D3BC"),
          ("y_pred", "predicted value only", "#5a6472"),
          ("random", "random shortlist", "#c3c8cd")]
_KS = [1, 5, 10]
_METRICS = [("event_precision_at_{}", "event precision", False),
            ("any_hit_at_{}", "any hit", False),
            ("regret_at_{}", "regret  ($\\Delta\\log K$)", True)]
_OBJ = [("high_selectivity", "both", "High selectivity  ($\\Delta\\log K > 2$)"),
        ("low_selectivity", "near_zero", "Near zero  ($|\\Delta\\log K| < 1/2$)")]


def figS10():
    h = pd.read_csv(TF + "pair_acquisition_tau2_half_hit_and_precision.csv")
    g = pd.read_csv(TF + "pair_acquisition_tau2_half_regret.csv")

    fig, axes = plt.subplots(3, 2, figsize=(7.5, 6.4))
    w = 0.135
    off = (np.arange(len(_RULES)) - (len(_RULES) - 1) / 2) * w
    for ri, (tmpl, ylab, from_regret) in enumerate(_METRICS):
        for ci, (task, orient, obj_label) in enumerate(_OBJ):
            ax = axes[ri, ci]
            src = g if from_regret else h
            f = src[(src.task == task) & (src.orientation == orient)].set_index("strategy")
            x = np.arange(len(_KS))
            for bi, (key, lab, col) in enumerate(_RULES):
                v = [float(f.loc[key, tmpl.format(k)]) for k in _KS]
                s = [float(f.loc[key, tmpl.format(k) + "_sd"]) for k in _KS]
                ax.bar(x + off[bi], v, width=w * 0.92, color=col, zorder=3,
                       label=lab if (ri == 0 and ci == 0) else None,
                       edgecolor="white", linewidth=0.3)
                ax.errorbar(x + off[bi], v, yerr=s, fmt="none", ecolor=INK2,
                            elinewidth=0.7, capsize=1.4, zorder=4)
            ax.set_xticks(x)
            ax.set_xticklabels([f"$k$ = {k}" for k in _KS])
            ax.set_xlim(-0.55, len(_KS) - 0.45)
            if ci == 0:
                ax.set_ylabel(ylab)
            if not from_regret:
                ax.set_ylim(0, 1.06)
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            else:
                ax.set_ylim(0)
            if ri == 0:
                ax.set_title(obj_label, loc="left", pad=8)
            grid(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, fontsize=8.0,
               loc="lower center", bbox_to_anchor=(0.5, -0.008),
               handlelength=1.9, columnspacing=2.4, labelspacing=0.4)
    fig.tight_layout(w_pad=2.4, h_pad=2.0, rect=(0, 0.085, 1, 1))
    for ax, s in zip(axes.ravel(), ["a", "d", "b", "e", "c", "f"]):
        tag(fig, ax, s, dx=0.056, dy=0.026)
    save(fig, "figS10_acquisition_threshold_sensitivity")



if __name__ == "__main__":
    figS1()
    stats = figS2()
    figS3(); figS4(); figS5(); figS6(); figS7(); figS8(); figS9(); figS10()
    print("Fig. S2 recomputed statistics:", stats)
    print("supplementary figures written to", OUT)
