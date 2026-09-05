"""Nature Computational Science main-text figures (Fig. 1-3).

Reads validated derived tables only; no model is loaded, trained or rescored.

One retention protocol throughout the main text: within each held-out fold,
predictions are ranked by a risk score, the lowest-risk fraction is retained,
and mean-of-pairs RMSE is recomputed on the pairs that keep at least three
retained rows. Errors are in dlog K. Bands are the fold s.d.

Palette (light surface):
  #0072B2 blue       model uncertainty
  #D55E00 vermillion predicted magnitude, the harmful comparator
  #009E73 green      descriptor-space distances
  #CC79A7 pink       fingerprint distances
  #E69F00 orange     training-frequency count
  #6E7079 gray       random baseline / neutral reference
Green<->pink is the deutan CVD warn pair; every series is additionally
direct-labelled and separated by linestyle.

No panel carries interpretive text: annotation is limited to axis labels,
reference lines and printed values.
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
# Derived tables live under manuscripts/jacs/tables/ for historical reasons; the
# validator modules read the same directory. Override with BNN_TABLES.
T = os.environ.get("BNN_TABLES", os.path.join(_REPO, "manuscripts", "jacs", "tables")) + os.sep
TF = os.environ.get("BNN_TABLES_FINAL",
                    os.path.join(_REPO, "manuscripts", "final", "tables")) + os.sep

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

LABEL = {
    "sigma_total": "predictive uncertainty $\\sigma_{\\mathrm{total}}$",
    "sigma_aleatoric": "$\\sigma_{\\mathrm{aleatoric}}$",
    "euclidean_feature_median": "descriptor Euclid. median",
    "euclidean_feature_nn": "descriptor Euclid. NN",
    "mahalanobis_feature_centroid": "descriptor Mahal. centroid",
    "mahalanobis_feature_median": "descriptor Mahal. median",
    "mahalanobis_feature_nn": "descriptor Mahal. NN",
    "mahalanobis_feature_knn_mean_5": "descriptor Mahal. $k$NN-5",
    "mahalanobis_feature_knn_mean_10": "descriptor Mahal. $k$NN-10",
    "tanimoto_median_distance": "fingerprint median",
    "tanimoto_nn_distance": "fingerprint NN",
    "tanimoto_knn_mean_5": "fingerprint $k$NN-5",
    "tanimoto_knn_mean_10": "fingerprint $k$NN-10",
    "pair_density_risk": "training-frequency count",
    "yhat_mag": "predicted magnitude $|\\hat y|$",
    "random": "random retention",
    "unfiltered": "no filtering",
}
COLOR = {
    "sigma_total": C_SIG, "sigma_aleatoric": C_SIG,
    "euclidean_feature_median": C_DESC, "euclidean_feature_nn": C_DESC,
    "mahalanobis_feature_centroid": C_DESC, "mahalanobis_feature_median": C_DESC,
    "mahalanobis_feature_nn": C_DESC,
    "mahalanobis_feature_knn_mean_5": C_DESC, "mahalanobis_feature_knn_mean_10": C_DESC,
    "tanimoto_median_distance": C_FP, "tanimoto_nn_distance": C_FP,
    "tanimoto_knn_mean_5": C_FP, "tanimoto_knn_mean_10": C_FP,
    "pair_density_risk": C_CNT, "yhat_mag": C_YHAT,
    "random": C_RAND, "unfiltered": INK3,
}


CMAP = LinearSegmentedColormap.from_list("bnn", ["#f7f9fb", "#9ecae1", "#3182bd", "#08306b"])


def grid(ax, axis="y"):
    ax.grid(axis=axis, color="#e6e6e6", lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def tag(fig, ax, s, dx=0.052, dy=0.045):
    p = ax.get_position()
    fig.text(p.x0 - dx, p.y1 + dy, s, fontsize=11, fontweight="bold",
             va="top", ha="left", color=INK)


# ---------------------------------------------------------------- Figure 0
def figure0():
    """Graphical abstract. Panel a is a schematic; panel b is measured data."""
    rng = np.random.RandomState(7)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.2),
                               gridspec_kw=dict(width_ratios=[1.0, 1.06]))

    # ---- a: the domain is assumed, and its boundary is not determined
    centres = [(-1.55, 0.95), (1.30, 1.15), (0.10, -1.25)]
    spreads = [0.42, 0.34, 0.46]
    for (cx, cy), sd in zip(centres, spreads):
        pts = rng.normal((cx, cy), sd, size=(90, 2))
        a.scatter(pts[:, 0], pts[:, 1], s=9, color="#c9d2d8", edgecolor="none",
                  zorder=2)
    inside = np.array([[-1.45, 0.80], [-1.70, 1.15], [1.22, 1.05], [1.45, 1.30],
                       [0.05, -1.10], [0.25, -1.40]])
    outside = np.array([[-0.30, 1.85], [0.75, 0.15], [-0.95, -0.35],
                        [2.05, -0.55], [-2.25, -0.10]])
    a.scatter(inside[:, 0], inside[:, 1], s=34, color=C_SIG, edgecolor="white",
              lw=0.6, zorder=5)
    a.scatter(outside[:, 0], outside[:, 1], s=34, marker="D", color=C_YHAT,
              edgecolor="white", lw=0.6, zorder=5)

    from matplotlib.patches import Circle
    for (cx, cy), sd in zip(centres, spreads):
        a.add_patch(Circle((cx, cy), sd * 2.15, fill=False, ls="--", lw=1.0,
                           ec="#7f8c94", zorder=3))
        a.add_patch(Circle((cx, cy), sd * 3.40, fill=False, ls=":", lw=1.0,
                           ec="#b0bcc4", zorder=3))
    a.annotate("in?", xy=(0.75, 0.15), xytext=(1.05, 0.52), fontsize=8,
               color=C_YHAT, ha="left",
               arrowprops=dict(arrowstyle="->", color=C_YHAT, lw=0.9))
    a.annotate("out?", xy=(-0.95, -0.35), xytext=(-1.95, -0.85), fontsize=8,
               color=C_YHAT, ha="left",
               arrowprops=dict(arrowstyle="->", color=C_YHAT, lw=0.9))
    a.set_xlim(-2.9, 2.9); a.set_ylim(-2.4, 3.7)
    a.set_xticks([]); a.set_yticks([])
    a.set_xlabel("?", fontsize=15, color=INK2, labelpad=-2)
    a.set_ylabel("?", fontsize=15, color=INK2, rotation=0, labelpad=8, va="center")
    a.set_title("An applicability domain, assumed", loc="left", pad=8)
    for sp in a.spines.values():
        sp.set_visible(True)
        sp.set_color("#dcdcdc")
    hs = [plt.Line2D([], [], marker="o", ls="", color="#c9d2d8", ms=5,
                     label="training data"),
          plt.Line2D([], [], marker="o", ls="", color=C_SIG, ms=6,
                     label="test, near training"),
          plt.Line2D([], [], marker="D", ls="", color=C_YHAT, ms=6,
                     label="test, far from training"),
          plt.Line2D([], [], ls="--", color="#7f8c94", lw=1.0,
                     label="two candidate boundaries")]
    a.legend(handles=hs, frameon=False, loc="upper left", fontsize=6.6,
             handletextpad=0.5, labelspacing=0.30, borderaxespad=0.2)

    # ---- b: the domain measured against error
    free = pd.read_csv(T + "pooled_retention.csv")
    free = free[free.population == "free"]
    unf = float(free[free.score == "unfiltered"].fold_mean.iloc[0])
    show = [("sigma_total", "predictive uncertainty", C_SIG, "-"),
            ("euclidean_feature_median", "best descriptor distance", C_DESC, "-"),
            ("tanimoto_nn_distance", "fingerprint distance", C_FP, "--"),
            ("random", "random", C_RAND, "--")]
    for sc, lab, col, ls in show:
        d = free[free.score == sc].sort_values("retention")
        b.plot(d.retention, d.fold_mean, ls, color=col, lw=2.2, zorder=4)
        yv = float(d[d.retention == 0.10].fold_mean.iloc[0])
        b.text(0.085, max(yv, 0.34), lab, fontsize=7.4, color=col, ha="right",
               va="center")
    b.axhline(unf, color=INK3, lw=0.9, ls=":", zorder=3)
    b.text(1.0, unf + 0.05, "no filtering", fontsize=7.2, color=INK2, ha="right")
    b.set_xlim(-0.32, 1.03); b.set_ylim(0.18, 2.42)
    b.set_xticks([0.25, 0.5, 0.75, 1.0])
    b.set_xticklabels(["25%", "50%", "75%", "100%"])
    b.set_xlabel("held-out predictions retained")
    b.set_ylabel("error of the retained set  ($\\Delta\\log K$)")
    b.set_title("An applicability domain, measured", loc="left", pad=8)
    b.spines["left"].set_bounds(0.25, 2.4)
    grid(b)

    fig.tight_layout(w_pad=2.2)
    tag(fig, a, "a", dx=0.048, dy=0.046); tag(fig, b, "b", dx=0.058, dy=0.046)
    for e in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig0_graphical_abstract.{e}")
    plt.close(fig)
    print("wrote fig0")


# ---------------------------------------------------------------- Figure 1
# What the published benchmark measures, and why the uncertainty has to be
# attached to the pairwise target. Both panels read deposited tables only.
def figure1():
    b = pd.read_csv(T + "benchmark_reproduction.csv")
    s = pd.read_csv(T + "selectivity_construction.csv")

    fig, (a, c) = plt.subplots(1, 2, figsize=(7.5, 4.0),
                               gridspec_kw=dict(width_ratios=[1.0, 1.0]))

    # ---- a: the same architecture under three regimes
    lab = ["published split,\ntest partition", "published split,\nexternal holdout",
           "ligand-disjoint\ncross-validation"]
    y = np.arange(3)[::-1]
    val = [float(b.released_weights_mae.iloc[0]), float(b.released_weights_mae.iloc[1]),
           float(b.retrained_mae.iloc[2])]
    err = [0.0, 0.0, float(b.retrained_mae_fold_sd.iloc[2])]
    pub = [float(b.published_mae.iloc[0]), float(b.published_mae.iloc[1]), np.nan]
    over = ["93.9% of test ligands\nalso in training", "no ligand overlap\n(40 ligands)",
            "no ligand overlap\nby construction"]
    cols = ["#b8c4cc", "#8fbcdb", C_SIG]
    for yi, v, e, col in zip(y, val, err, cols):
        a.barh(yi, v, height=0.55, color=col, zorder=3,
               xerr=e if e else None, error_kw=dict(ecolor=INK2, elinewidth=0.9,
                                                   capsize=2.5))
    for yi, v, pv in zip(y, val, pub):
        if not np.isnan(pv):
            a.plot(pv, yi, "D", ms=5.0, color=INK, zorder=5,
                   markeredgecolor="white", markeredgewidth=0.6)
    # Annotations clear the bar, its error bar and the published diamond, so
    # nothing is written over a drawn element.
    for yi, v, e, pv, o in zip(y, val, err, pub, over):
        x0 = max(v + e, 0.0 if np.isnan(pv) else pv)
        a.text(x0 + 0.07, yi, o, fontsize=6.8, color=INK2, va="center", ha="left")
    a.plot([], [], "D", ms=5.0, color=INK, label="published value")
    a.set_yticks(y); a.set_yticklabels(lab, fontsize=8.0)
    a.set_xlim(0, 2.75)
    a.set_xlabel("mean absolute error of $\\log K$")
    a.set_ylim(-0.55, 2.75)
    a.set_title("The same architecture, three regimes", loc="left", pad=8)
    a.legend(frameon=False, loc="upper right", fontsize=6.9, borderaxespad=0.2)
    grid(a, "x")

    # ---- c: accuracy overlaps, calibration does not
    AH = "ad hoc, difference of two absolute predictions"
    PT = "ad hoc, point models without an interval"
    DI = "direct, modelled on the pairwise target"
    # A readable subset rather than the whole grid: within each construction the
    # best-calibrated model, the most accurate one, and the evidential graph
    # network, which is the same architecture on both sides of the comparison.
    # Supplementary Fig. S3 draws all twenty-three.
    SHOW = {AH: ["DS1_RDKIT_MLP_MCDROPOUT", "DS1_RDKIT_MLPENS5_MCDROPOUT",
                 "DS1_GRAPH_DMPNN_EVID"],
            DI: ["DS2_RDKIT_MLPENS5_HETERO", "DS2_RDKIT_MLPENS5_MCDROPOUT",
                 "DS2_RDKIT_MLPENS5_EVID", "DS2_GRAPH_DMPNN_EVID"]}
    # Short forms, and where each label is set. The selected model
    # carries no inline label: it is named in the legend, beside its own symbol.
    # The interval-calibration axis separates these models by less than a label
    # is tall, so the crowded ones are set in clear space on a leader line.
    SHORT = {"DS1_RDKIT_MLP_MCDROPOUT": ("MC-dropout", 2.250, 0.190, "left"),
             "DS1_RDKIT_MLPENS5_MCDROPOUT": ("MC-dropout ens.", 1.755, 0.262, "left"),
             "DS1_GRAPH_DMPNN_EVID": ("Chemprop evid.", 2.172, 0.300, "center"),
             "DS2_RDKIT_MLPENS5_MCDROPOUT": ("MC-dropout ens.", 1.755, 0.148, "left"),
             "DS2_RDKIT_MLPENS5_EVID": ("Evidential ens.", 2.250, 0.135, "left"),
             "DS2_GRAPH_DMPNN_EVID": ("Chemprop evid.", 2.300, 0.283, "left")}

    # The best point-only ad hoc model: no interval, so it fixes the accuracy
    # this construction reaches when nothing is asked of calibration.
    pt = s[s.construction == PT].sort_values("mop_rmse").iloc[0]
    c.axvline(float(pt.mop_rmse), color=INK3, ls=(0, (5, 3)), lw=1.0, zorder=2)

    for key, col, mk, lab2 in ((AH, "#CC79A7", "o", "ad hoc, difference of two absolute predictions"),
                               (DI, C_SIG, "s", "direct, modelled on the pairwise target")):
        d = s[(s.construction == key) & (s.source_model_id.isin(SHOW[key]))]
        c.errorbar(d.mop_rmse, d.mop_ace, xerr=d.mop_rmse_sd, yerr=d.mop_ace_sd,
                   fmt=mk, ms=5.4, color=col, ecolor=col, elinewidth=0.8,
                   capsize=1.8, lw=0, zorder=4, label=lab2,
                   markeredgecolor="white", markeredgewidth=0.5)
        for _, r in d.iterrows():
            if r.source_model_id not in SHORT:
                continue
            nm, tx, ty, ha = SHORT[r.source_model_id]
            c.annotate(nm, xy=(r.mop_rmse, r.mop_ace), xytext=(tx, ty),
                       fontsize=6.7, color=col, ha=ha, va="center", zorder=7,
                       arrowprops=dict(arrowstyle="-", color=col, lw=0.6,
                                       alpha=0.65, shrinkA=1.5, shrinkB=4.0))
    w = s[s.is_workhorse].iloc[0]
    c.plot(w.mop_rmse, w.mop_ace, "*", ms=15, color=C_SIG, zorder=6,
           markeredgecolor="white", markeredgewidth=0.8,
           label="selected model: heteroscedastic MLP ensemble")

    c.set_ylim(0.118, 0.520)
    c.set_xlim(1.74, 2.62)
    c.text(float(pt.mop_rmse) - 0.014, 0.432,
           "best point-only\nad hoc model", fontsize=6.7, color=INK2,
           ha="right", va="top", linespacing=1.35)
    c.set_xlabel("mean-of-pairs RMSE  ($\\Delta\\log K$)")
    c.set_ylabel("average coverage error  (pair-equal)")
    c.set_title("Accuracy against calibration", loc="left", pad=8)
    c.legend(frameon=False, loc="upper left", fontsize=6.8, labelspacing=0.45,
             handletextpad=0.5, borderaxespad=0.25)
    grid(c, "both")

    fig.tight_layout(w_pad=2.2)
    tag(fig, a, "a", dx=0.105, dy=0.048); tag(fig, c, "b", dx=0.070, dy=0.048)
    for e in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig1.{e}")
    plt.close(fig)
    print("wrote fig1")


# ---------------------------------------------------------------- Figure 2
# Each distance family is shown at its own best aggregation, which is the median
# in all three families, so the comparison is the most favourable one available
# to the distances. The full inventory of aggregations is Supplementary Fig. S5.
CURVES = ["sigma_total", "pair_density_risk", "euclidean_feature_median",
          "mahalanobis_feature_median", "tanimoto_median_distance", "random"]
STYLE = {"sigma_total": "-", "pair_density_risk": "-",
         "euclidean_feature_median": "-", "mahalanobis_feature_median": "-.",
         "tanimoto_median_distance": "--", "random": "--"}
RHO_SCORES = ["sigma_total", "pair_density_risk", "euclidean_feature_median",
              "mahalanobis_feature_median", "tanimoto_median_distance"]
SHORT_NAME = {"sigma_total": "$\\sigma_{\\mathrm{total}}$",
              "pair_density_risk": "training-frequency count",
              "euclidean_feature_median": "descriptor Euclid. median",
              "mahalanobis_feature_median": "descriptor Mahal. median",
              "tanimoto_median_distance": "fingerprint median"}
FAMILY_COLOR = {"uncertainty": C_SIG, "descriptor": C_DESC, "fingerprint": C_FP,
                "learned-embedding": "#7b3294", "count": C_CNT}


SCORE_NAME = {
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


def _family(score):
    if score.startswith("sigma"):
        return "uncertainty"
    if score.startswith("tanimoto"):
        return "fingerprint"
    if score.startswith("chemprop"):
        return "learned-embedding"
    if "pair_density" in score:
        return "count"
    return "descriptor"


def figure2():
    free = pd.read_csv(T + "pooled_retention.csv")
    free = free[free.population == "free"]
    z = pd.read_csv(TF + "iz_risk_score_spearman.csv").dropna(subset=["fold_mean"])
    z = z.set_index("score")
    unf = float(free[free.score == "unfiltered"].fold_mean.iloc[0])

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.5, 4.2),
                               gridspec_kw=dict(width_ratios=[1.06, 0.94]))

    handles = []
    for sc in CURVES:
        d = free[free.score == sc].sort_values("retention")
        lw = 2.4 if sc == "sigma_total" else 1.7
        a.fill_between(d.retention, d.fold_mean - d.fold_sd, d.fold_mean + d.fold_sd,
                       color=COLOR[sc], alpha=0.13, lw=0, zorder=2)
        ln, = a.plot(d.retention, d.fold_mean, STYLE[sc], color=COLOR[sc], lw=lw,
                     zorder=4, label=LABEL[sc])
        handles.append(ln)
    a.axhline(unf, color=INK3, lw=0.9, ls=":", zorder=3)
    a.text(1.0, unf + 0.045, f"{unf:.2f}, no filtering", fontsize=7.8, color=INK2,
           va="bottom", ha="right")
    a.set_xlim(0.075, 1.03); a.set_ylim(0.15, 2.32)
    a.set_xticks([0.1, 0.25, 0.5, 0.75, 1.0])
    a.set_xticklabels(["10%", "25%", "50%", "75%", "100%"])
    a.set_xlabel("held-out predictions retained")
    a.set_ylabel("mean-of-pairs RMSE  ($\\Delta\\log K$)")
    a.set_title("Retained error under each score", loc="left", pad=8)
    grid(a)

    order = sorted(RHO_SCORES, key=lambda s: float(z.loc[s, "fold_mean"]))
    y = np.arange(len(order))
    b.axvline(0, color=INK, lw=1.1, zorder=4)
    for yi, sc in zip(y, order):
        col = FAMILY_COLOR[_family(sc)]
        m, s = float(z.loc[sc, "fold_mean"]), float(z.loc[sc, "fold_sd"])
        b.plot([0, m], [yi, yi], color=col, lw=6.0, alpha=0.30, zorder=3,
               solid_capstyle="butt")
        b.errorbar(m, yi, xerr=s, fmt="o", ms=6.4, color=col, ecolor=col,
                   elinewidth=1.2, capsize=2.8, zorder=5)
        b.text(m + s + 0.028, yi, f"{m:+.3f}", fontsize=8.0, color=INK2,
               va="center", ha="left")
    b.set_yticks(y)
    b.set_yticklabels([SHORT_NAME[s] for s in order], fontsize=8.2)
    b.set_ylim(-0.55, len(order) - 0.45)
    b.set_xlim(-0.04, 0.86)
    b.set_xticks([0, 0.25, 0.5, 0.75])
    b.set_xlabel("rank correlation with absolute error")
    b.set_title("Association with error", loc="left", pad=8)
    grid(b, "x")

    fig.legend(handles=handles, frameon=False, ncol=3, fontsize=8.0,
               loc="lower center", bbox_to_anchor=(0.5, -0.015),
               handlelength=2.6, columnspacing=1.8, labelspacing=0.4)
    fig.tight_layout(w_pad=2.2, rect=(0, 0.135, 1, 1))
    tag(fig, a, "a", dx=0.058, dy=0.040); tag(fig, b, "b", dx=0.108, dy=0.040)
    for e in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig2.{e}")
    plt.close(fig)
    print("wrote fig2")


# ---------------------------------------------------------------- Figure 3
# Panels a and b read the deposited fold-level score files, which is the only
# place the within-fold quartile boundaries can be recovered from; panels c and
# d read the two deposited grids. No model is loaded.
FOLD_SCORES = [os.path.join(_REPO, "results", "downstream_cv", "ad_retention",
                            "fold_scores", "DS2_RDKIT_MLPENS5_HETERO",
                            f"fold{i}.csv") for i in range(5)]
Q_FILL = ["#e3edf5", "#c3dbeb", "#8fbcdb", "#4a90c4"]


def _quartile_edges(frame, col):
    """Within-fold quartile boundaries, averaged over folds for display."""
    e = np.array([np.percentile(g[col].values, [25, 50, 75])
                  for _, g in frame.groupby("fold_id")])
    return e.mean(axis=0)


def _heat(fig, cell, G, sig, sup, fmt, ylabel=True):
    """A 4x4 grid with its two axis marginals, drawn into one gridspec cell."""
    gs = cell.subgridspec(2, 3, height_ratios=[0.26, 1.0],
                          width_ratios=[1.0, 0.30, 0.055],
                          hspace=0.08, wspace=0.08)
    top = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])
    right = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[1, 2])

    im = ax.imshow(G, cmap=CMAP, aspect="auto")
    lo, hi = np.nanmin(G), np.nanmax(G)
    for i in range(4):
        for j in range(4):
            rel = (G[i, j] - lo) / (hi - lo)
            ax.text(j, i, format(G[i, j], fmt), ha="center", va="center",
                    fontsize=8.2, color="white" if rel > 0.55 else INK)
    ax.set_xticks(range(4)); ax.set_xticklabels(["Q1\nlow", "Q2", "Q3", "Q4\nhigh"])
    ax.set_yticks(range(4))
    ax.set_yticklabels(["Q1\nsparse", "Q2", "Q3", "Q4\ndense"])
    ax.set_xlabel("uncertainty, quartile within fold", fontsize=8.2)
    if ylabel:
        ax.set_ylabel("training support, quartile within fold", fontsize=8.2)
    ax.spines[:].set_visible(False)

    top.bar(np.arange(4), sig, width=0.76, color=C_SIG, zorder=3)
    for k, v in enumerate(sig):
        top.text(k, v * 1.03, format(v, fmt), ha="center", va="bottom",
                 fontsize=7.0, color=INK2)
    top.set_xlim(-0.5, 3.5); top.set_ylim(0, max(sig) * 1.40)
    top.set_xticks([]); top.set_yticks([]); top.spines[:].set_visible(False)

    right.barh(np.arange(4), sup, height=0.76, color="#8a9aa5", zorder=3)
    for k, v in enumerate(sup):
        right.text(v * 1.04, k, format(v, fmt), va="center", fontsize=7.0,
                   color=INK2)
    right.set_ylim(3.5, -0.5); right.set_xlim(0, max(sup) * 1.55)
    right.set_xticks([]); right.set_yticks([]); right.spines[:].set_visible(False)

    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=6.8)
    return ax, top


def figure3():
    d = pd.concat([pd.read_csv(f, usecols=["fold_id", "sigma_total",
                                           "n_training_ligands_for_pair"])
                   for f in FOLD_SCORES], ignore_index=True)
    o = pd.read_csv(T + "coverage_override_grid.csv")
    o = o[o.aggregation == "fold_mean_plus_sd"]
    g = pd.read_csv(T + "scale_free_coverage_grid.csv")

    def _abs(section, sup=None, sig=None):
        f = o[o.table_section == section]
        if sup is not None:
            f = f[f.support_quartile == sup]
        if sig is not None:
            f = f[f.sigma_quartile == sig]
        return float(f.mean_abs_error.iloc[0])

    def _nrm(section, sup=None, sig=None):
        f = g[g.section == section]
        if sup is not None:
            f = f[f.support_quartile == sup]
        if sig is not None:
            f = f[f.sigma_quartile == sig]
        return float(f.fold_mean.iloc[0])

    ks = [1, 2, 3, 4]
    GA = np.array([[_abs("grid", i, j) for j in ks] for i in ks])
    GN = np.array([[_nrm("grid", i, j) for j in ks] for i in ks])

    fig = plt.figure(figsize=(7.5, 6.6))
    outer = fig.add_gridspec(2, 2, height_ratios=[0.52, 1.0], hspace=0.42,
                             wspace=0.30)
    a = fig.add_subplot(outer[0, 0])
    b = fig.add_subplot(outer[0, 1])

    for ax, col, lab, ttl in (
            (a, "sigma_total",
             "predictive uncertainty $\\sigma_{\\mathrm{total}}$  ($\\Delta\\log K$)",
             "Uncertainty, and its quartiles"),
            (b, "n_training_ligands_for_pair",
             "training ligands carried by the metal pair",
             "Training support, and its quartiles")):
        v = d[col].values
        v = v[v > 0]
        edges = _quartile_edges(d, col)
        bins = np.logspace(np.log10(v.min()), np.log10(v.max()), 61)
        cuts = np.concatenate([[v.min()], edges, [v.max()]])
        h, _ = np.histogram(v, bins=bins)
        for qi in range(4):
            sel = (bins[:-1] >= cuts[qi]) & (bins[:-1] < cuts[qi + 1])
            ax.bar(bins[:-1][sel], h[sel], width=np.diff(bins)[sel],
                   align="edge", color=Q_FILL[qi], lw=0, zorder=3)
        for e in edges:
            ax.axvline(e, color=INK2, lw=0.8, ls="--", zorder=4)
        ax.set_xscale("log")
        ymax = ax.get_ylim()[1]
        ax.set_ylim(0, ymax * 1.18)
        for qi, xc in enumerate(np.sqrt(cuts[:-1] * cuts[1:])):
            ax.text(xc, ymax * 1.06, f"Q{qi + 1}", fontsize=7.6, ha="center",
                    va="center", color=INK2)
        ax.set_xlabel(lab, fontsize=8.4)
        ax.set_ylabel("held-out predictions", fontsize=8.4)
        ax.set_title(ttl, loc="left", pad=8)
        grid(ax)

    c_ax, c_top = _heat(fig, outer[1, 0], GA,
                        [_abs("sigma_marginal", sig=q) for q in ks],
                        [_abs("support_marginal", sup=q) for q in ks], ".2f")
    d_ax, d_top = _heat(fig, outer[1, 1], GN,
                        [_nrm("sigma_marginal", sig=q) for q in ks],
                        [_nrm("support_marginal", sup=q) for q in ks], ".2f",
                        ylabel=False)
    c_top.set_title("Mean $|$error$|$  ($\\Delta\\log K$)", loc="left", pad=6)
    d_top.set_title("The same, as normalized error", loc="left", pad=6)

    for ax, s, dx in ((a, "a", 0.052), (b, "b", 0.052),
                      (c_top, "c", 0.052), (d_top, "d", 0.052)):
        tag(fig, ax, s, dx=dx, dy=0.030)
    for e in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig3.{e}")
    plt.close(fig)
    print("wrote fig3")


# ---------------------------------------------------------------- Figure 4
# All six ranking rules, all three metrics, both objectives, as grouped bars.
# The metrics are defined in Methods and in Supplementary Section S8; the axes
# carry only their names and units.
RULES = [("expected_improvement", "standardised margin", C_SIG),
         ("lcb_1sigma", "lower bound, 1σ", "#C4622D"),
         ("lcb_2sigma", "lower bound, 2σ", "#E8A87C"),
         ("lcb_3sigma", "lower bound, 3σ", "#F2D3BC"),
         ("y_pred", "predicted value only", "#5a6472"),
         ("random", "random shortlist", "#c3c8cd")]
KS = [1, 5, 10]
METRICS = [("event_precision_at_{}", "event precision", False),
           ("any_hit_at_{}", "any hit", False),
           ("regret_at_{}", "regret  ($\\Delta\\log K$)", True)]
OBJECTIVES = [("high_selectivity", "both",
               "High selectivity  ($\\Delta\\log K > 1$)"),
              ("low_selectivity", "near_zero",
               "Near zero  ($|\\Delta\\log K| < 1$)")]


def figure4():
    h = pd.read_csv(TF + "pair_acquisition_hit_and_precision.csv")
    g = pd.read_csv(TF + "pair_acquisition_regret.csv")

    fig, axes = plt.subplots(3, 2, figsize=(7.5, 6.4))
    w = 0.135
    off = (np.arange(len(RULES)) - (len(RULES) - 1) / 2) * w
    for ri, (tmpl, ylab, from_regret) in enumerate(METRICS):
        for ci, (task, orient, obj_label) in enumerate(OBJECTIVES):
            ax = axes[ri, ci]
            src = g if from_regret else h
            f = src[(src.task == task) & (src.orientation == orient)].set_index("strategy")
            x = np.arange(len(KS))
            for bi, (key, lab, col) in enumerate(RULES):
                v = [float(f.loc[key, tmpl.format(k)]) for k in KS]
                s = [float(f.loc[key, tmpl.format(k) + "_sd"]) for k in KS]
                ax.bar(x + off[bi], v, width=w * 0.92, color=col, zorder=3,
                       label=lab if (ri == 0 and ci == 0) else None,
                       edgecolor="white", linewidth=0.3)
                ax.errorbar(x + off[bi], v, yerr=s, fmt="none", ecolor=INK2,
                            elinewidth=0.7, capsize=1.4, zorder=4)
            ax.set_xticks(x)
            ax.set_xticklabels([f"$k$ = {k}" for k in KS])
            ax.set_xlim(-0.55, len(KS) - 0.45)
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
    for e in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig4.{e}")
    plt.close(fig)
    print("wrote fig4")


if __name__ == "__main__":
    figure0()
    figure1()
    figure2()
    figure3()
    figure4()
    print("figures written to", OUT)
