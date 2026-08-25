"""Redrawn manuscript figures.

Rebuilds the figures whose original layout obscured the result they exist to show.
Every panel reads a stored summary CSV; nothing here retrains or rescores, and no
displayed number changes. Run after `generate_figures.py` /
`iz_risk_ordering_figure.py` and before `finalize_main_figures.py` is not required —
this module writes the manuscript ids directly.

    python -m uq_remake.manuscript.redraw_figures            # all
    python -m uq_remake.manuscript.redraw_figures --only 1 3 # selected

What changed and why
--------------------
Figure 1  Fold-SD error bars added on both axes. The figure's argument is that the
          candidates are *not* separable on accuracy; without error bars it read as a
          leaderboard. Panel A y-limits now bracket the data (the old 0.00-0.25 range
          left 60% of the panel empty and made the reference band fill it). Reference
          bands redrawn as light guide lines, not filled regions, because they are not
          the screen and were being read as a pass/fail area.

Figure 4  Old panels A and B were the same curve twice - error and uncertainty against
          predicted magnitude, in the same units, on separate axes, which hid the
          coupling that is the point. Merged into one twin-series panel. The freed
          panel now shows the magnitude-adjusted partial correlation, previously
          reported only in prose although it is the number that proves sigma_total is
          not a relabelling of |yhat|.

Figure 5  LCB-1/2/3 recoloured as an ordered ramp of one hue (they are a monotone
          penalty, previously orange/red/purple). Y-axes cropped to the data. The dead
          true-hit panel is dropped and its one live column (k=1) folded into the
          precision panel. The objective inversion is annotated directly. Regret panels
          carry an explicit "lower is better" arrow.

Figure 3  Promoted from the Supporting Information and replaces the 23-row score-residual
          table that used to sit in Section 3.4: the claim is a sign pattern across two
          representations, which a sorted axis shows and a table makes the reader assemble.
          Exact values moved to Supplementary Table S8.6.

Figure S10 Three series were near-identical teal across two unrelated score families.
          Recoloured by family with distinct lightness. Pair-density risk promoted to a
          heavy line: it is the second-strongest score and the main text now argues from
          it.

Figure S11 Panels A and C (error and uncertainty against pair support) overlaid on one
          twin axis, same reasoning as Figure 4. PICP panel keeps its own axis and gains
          the nominal-coverage reference.

Figure S12 Both panels were an unreadable pile at the origin; now log-scaled. Panel B's
          colourbar encoded the same variable as its own y-axis, so colour now carries
          high-selectivity fraction instead.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "results"
TABLES = REPO / "manuscripts" / "final" / "tables"
FIGDIR = REPO / "manuscripts" / "final" / "figures"
WORKHORSE = "DS2_RDKIT_MLPENS5_HETERO"

# Fallback search path: files staged from the device during an interactive session.
_ALT = Path("/mnt/user-data/uploads/BNN Extraction")


def _read(rel: str) -> pd.DataFrame:
    """Read a repo-relative CSV, falling back to a staged copy."""
    for base in (REPO, _ALT):
        p = base / rel
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(f"{rel} not found under {REPO} or {_ALT}")


# --------------------------------------------------------------------------- style
INK = "#1A1A1A"
MUTED = "#6B7280"
GRID = "#E5E7EB"

C_SIGMA = "#C43D3D"      # uncertainty family - red
C_MAHAL = "#2F6DB5"      # descriptor-space distance - blue
C_MAHAL2 = "#7C3AED"     # descriptor-space, median variant - violet
C_TANI = "#15803D"       # fingerprint distance - dark green
C_TANI2 = "#84CC16"      # fingerprint, median variant - lime
C_DENS = "#B45309"       # pair-density support - amber
C_RANDOM = "#111111"

# LCB is a monotone penalty (z = 1, 2, 3): one hue, increasing darkness.
LCB_RAMP = ["#93C5FD", "#3B82F6", "#1E3A8A"]
C_VALUE = "#0F9D8F"
C_MARGIN = "#B45309"
C_RAND_BAR = "#9CA3AF"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _despine(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig, stem: str) -> list[Path]:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("png", "pdf"):
        p = FIGDIR / f"{stem}.{ext}"
        fig.savefig(p)
        out.append(p)
    plt.close(fig)
    print(f"wrote {stem}.png / .pdf", flush=True)
    return out


# ------------------------------------------------------------------------ figure 0
def figure0() -> list[Path]:
    """Graphical abstract: what the paper does, and the two results it turns on.

    Left half is the schematic (collection -> task choice -> fixed workhorse).
    Right half embeds the two headline results as small live plots drawn from the
    same summary CSVs as Figures 2 and 4, so the abstract cannot drift from the
    paper.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    apply_style()
    fig = plt.figure(figsize=(13.4, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.32, 1.0], height_ratios=[1, 1],
                          wspace=0.22, hspace=0.42)
    schem = fig.add_subplot(gs[:, 0]); schem.axis("off")
    schem.set_xlim(0, 10); schem.set_ylim(0, 10)

    def box(x, y, w, h, title, body, edge, fill, tsize=10.2, bsize=8.5):
        schem.add_patch(FancyBboxPatch((x, y), w, h,
                        boxstyle="round,pad=0.14,rounding_size=0.22",
                        linewidth=1.7, edgecolor=edge, facecolor=fill, zorder=2))
        schem.text(x + w / 2, y + h - 0.42, title, ha="center", va="top",
                   fontsize=tsize, fontweight="bold", color=edge, zorder=3)
        schem.text(x + w / 2, y + h - 1.02, body, ha="center", va="top",
                   fontsize=bsize, color=INK, linespacing=1.45, zorder=3)

    def arrow(x1, y1, x2, y2, c=MUTED, rad=0.0):
        schem.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=15, linewidth=1.5, color=c,
                        connectionstyle=f"arc3,rad={rad}", zorder=1))

    box(0.15, 8.05, 9.7, 1.80, "Heterogeneous literature measurements",
        "CMI M4 collection: 43,388 records, 4,347 ligands\n"
        "ligand + metal + oxidation state + medium + temperature",
        INK, "#F3F4F6", 10.6, 8.6)
    arrow(2.6, 8.00, 2.2, 7.25); arrow(7.3, 8.00, 7.7, 7.25)

    box(0.15, 5.05, 4.5, 2.15, "DS1  absolute binding",
        "predict $\\log K$, then subtract\ntwo predictions per pair\n"
        "$\\sigma_\\Delta=\\sqrt{\\sigma_X^2+\\sigma_Y^2}$",
        MUTED, "#FFFFFF", 10.2, 8.4)
    box(5.35, 5.05, 4.5, 2.15, "DS2  pairwise selectivity",
        "predict $\\Delta\\log K$ directly\n88,525 rows, 2,503 species pairs\n"
        "native uncertainty on the target",
        C_SIGMA, "#FEF2F2", 10.2, 8.4)

    schem.text(2.4, 4.62, "MoP ACE 0.378", ha="center", va="center", fontsize=9.2,
               color=MUTED, fontweight="bold")
    schem.text(7.6, 4.62, "MoP ACE 0.153", ha="center", va="center", fontsize=9.2,
               color=C_SIGMA, fontweight="bold")
    schem.text(5.0, 4.06, "2.5$\\times$ better calibrated on the quantity the experiment uses",
               ha="center", va="center", fontsize=8.6, color=INK, style="italic")
    arrow(7.6, 3.72, 7.6, 3.08, C_SIGMA)

    box(2.5, 1.10, 7.35, 1.95, "Fixed direct-UQ workhorse",
        "heteroscedastic MLP ensemble, chosen once\n"
        "five-fold ligand-disjoint CV, pair-equal (MoP) metrics",
        C_SIGMA, "#FEF2F2", 10.2, 8.4)
    schem.text(0.10, 2.10, "ligand-disjoint:\nevery test ligand\nunseen in training",
               ha="left", va="center", fontsize=8.3, color=MUTED, linespacing=1.5)
    schem.text(6.2, 0.42, "Two questions follow  " + r"$\longrightarrow$",
               ha="center", va="center", fontsize=9.8, color=INK, fontweight="bold")

    # ---- result 1: uncertainty orders error, distance does not ----
    ax = fig.add_subplot(gs[0, 1])
    s = _read("results/downstream_cv/ad_retention/retention_summary.csv")
    s = s[(s.model_id == WORKHORSE) & (s.scope == "all_pairs") & (s.metric_name == "mop_rmse")]
    rnd = _read("results/downstream_cv/ad_retention/random_retention_summary.csv")
    rnd = rnd[(rnd.model_id == WORKHORSE) & (rnd.scope == "all_pairs") & (rnd.metric_name == "mop_rmse")]
    # Same four scores as Figure 2, so the abstract cannot drift from the paper.
    fm = _read("manuscripts/final/tables/iz_mahalanobis_feature_median_retention_summary.csv")
    fm = fm[(fm.scope == "all_pairs") & (fm.metric_name == "mop_rmse")]
    for nm, lab, c, lw in [("sigma_total", r"$\sigma_{\mathrm{total}}$", C_SIGMA, 2.8),
                           ("pair_density_risk", "pair-support count", C_DENS, 2.3),
                           ("mahalanobis_feature_median", "descriptor distance", C_MAHAL, 2.0),
                           ("tanimoto_median_distance", "fingerprint distance", C_TANI, 2.0)]:
        g = (fm if nm == "mahalanobis_feature_median"
             else s[s.score_name == nm]).sort_values("retention_fraction")
        ax.plot(g.retention_fraction, g.fold_mean, color=c, lw=lw, label=lab, zorder=3)
    g = rnd.sort_values("retention_fraction")
    ax.plot(g.retention_fraction, g.fold_mean, color=C_RANDOM, ls=(0, (4, 2)), lw=1.9,
            label="no filter", zorder=2)
    ax.set_xlabel("Fraction of predictions kept", fontsize=9)
    ax.set_ylabel(r"MoP RMSE ($\Delta\log K$)", fontsize=9)
    ax.set_title("Which predictions can you trust?", loc="left", fontsize=10.4)
    ax.tick_params(labelsize=8.4)
    ax.set_ylim(0.05, 2.45)
    ax.legend(fontsize=7.5, loc="upper left", ncol=2, columnspacing=0.9,
              handlelength=1.4, bbox_to_anchor=(0.0, 1.03))
    ax.annotate("keep the half the model is\nsurest of: error drops 65%",
                xy=(0.62, 0.90), xytext=(0.60, 0.17), fontsize=8.2, color=C_SIGMA,
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", color=C_SIGMA, lw=1.1,
                                connectionstyle="arc3,rad=0.25"))
    ax.annotate("a support count with no chemistry\nin it beats both distances",
                xy=(0.47, 1.00), xytext=(0.115, 0.20), fontsize=8.2, color=C_DENS,
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", color=C_DENS, lw=1.0,
                                connectionstyle="arc3,rad=-0.25"))
    _despine(ax)

    # ---- result 2: the objective inverts the value of an uncertainty penalty ----
    ax = fig.add_subplot(gs[1, 1])
    hp = _read("manuscripts/final/tables/pair_acquisition_hit_and_precision.csv")
    def prec(task, orient, strat):
        r = hp[(hp.task == task) & (hp.orientation == orient) & (hp.strategy == strat)]
        return float(r["event_precision_at_1"].iloc[0])
    groups = [("High selectivity\n" + r"$\Delta\log K>1$", "high_selectivity", "both"),
              ("Near-zero\n" + r"$|\Delta\log K|<1$", "low_selectivity", "near_zero")]
    rules = [("y_pred", "value only", C_VALUE), ("lcb_1sigma", "LCB-1", LCB_RAMP[0]),
             ("lcb_3sigma", "LCB-3", LCB_RAMP[2]),
             ("expected_improvement", "threshold margin", C_MARGIN)]
    w = 0.19
    for gi, (glab, task, orient) in enumerate(groups):
        for ri, (key, rlab, c) in enumerate(rules):
            ax.bar(gi + (ri - 1.5) * w, prec(task, orient, key), w, color=c,
                   edgecolor="white", linewidth=0.6,
                   label=rlab if gi == 0 else None, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels([g[0] for g in groups], fontsize=8.6)
    ax.set_ylim(0.55, 1.03); ax.set_ylabel("Event precision@1", fontsize=9)
    ax.set_title("How should uncertainty enter the ranking?", loc="left", fontsize=10.4)
    ax.tick_params(labelsize=8.4)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9])
    ax.legend(fontsize=7.8, ncol=4, loc="lower center", columnspacing=0.7,
              handlelength=1.0, bbox_to_anchor=(0.5, -0.44))
    ax.annotate("subtract $\\sigma$:\nhurts badly", xy=(0.095, 0.678), xytext=(0.055, 0.925),
                fontsize=8.3, color=LCB_RAMP[2], fontweight="bold", ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=LCB_RAMP[2], lw=1.2,
                                connectionstyle="arc3,rad=0.15"))
    ax.annotate("divide by $\\sigma$:\nbest here", xy=(0.285, 0.840), xytext=(0.53, 0.960),
                fontsize=8.3, color=C_MARGIN, fontweight="bold", ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=C_MARGIN, lw=1.2,
                                connectionstyle="arc3,rad=-0.2"))
    ax.annotate("both help, equally", xy=(1.09, 0.812), xytext=(1.25, 0.945),
                fontsize=8.3, color=INK, fontweight="bold", ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1,
                                connectionstyle="arc3,rad=0.2"))
    ax.set_xlim(-0.55, 1.60)
    _despine(ax)

    fig.suptitle("Uncertainty-based risk ordering and objective-specific ligand ranking "
                 "for metal\u2013ligand selectivity", y=1.015, fontsize=12.4)
    return _save(fig, "fig00_graphical_abstract")


# ------------------------------------------------------------------------ figure 1
def figure1() -> list[Path]:
    """Pure pair-equal (MoP) model landscape, plus the row-level-vs-MoP gap.

    The manuscript reports MoP throughout. The earlier version of this figure
    plotted MoP coordinates but coloured markers by a *row-level* screen outcome,
    which mixed the two aggregations and could not be read off the axes. Panels A
    and B are now MoP only. Panel C reports the row-level values in the one role
    the paper gives them: showing that row weighting flatters calibration.
    """
    d = _read("results/manuscript_final/workhorse_selection_accuracy_calibration.csv")
    d = d[d["all_pair_mop_picp_90_mean"].notna()].copy()

    selected = d["selection_status"].astype(str).str.startswith("selected")
    partial = d["uq_status"].astype(str).str.contains("partial", na=False)
    others = ~selected & ~partial

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0))

    panels = [
        ("all_pair_mop_ace_mean", "all_pair_mop_ace_sd", "All-pair MoP ACE",
         "A. Pair-equal accuracy versus calibration error", None),
        ("all_pair_mop_picp_90_mean", "all_pair_mop_picp_90_sd", "All-pair MoP PICP@90",
         "B. Pair-equal accuracy versus coverage", (0.90, "nominal 0.90")),
    ]
    for ax, (ycol, ysd, ylab, title, ref) in zip(axes[:2], panels):
        x, xs = d["all_pair_mop_rmse_mean"], d["all_pair_mop_rmse_sd"]
        y, ys = d[ycol], d[ysd]
        ax.errorbar(x, y, xerr=xs, yerr=ys, fmt="none", ecolor=MUTED,
                    elinewidth=0.9, capsize=2.0, alpha=0.75, zorder=1)
        ax.scatter(x[others], y[others], marker="o", s=58, facecolors="#FFFFFF",
                   edgecolors=C_MAHAL, linewidths=1.8, zorder=3)
        if partial.any():
            ax.scatter(x[partial], y[partial], marker="D", s=58, facecolors="#FFFFFF",
                       edgecolors=C_TANI, linewidths=1.8, zorder=3)
        ax.scatter(x[selected], y[selected], marker="*", s=340,
                   facecolors=C_SIGMA, edgecolors=C_SIGMA, linewidths=1.2, zorder=4)
        if ref:
            ax.axhline(ref[0], color=MUTED, ls=(0, (4, 3)), lw=1.0, zorder=0)
            ax.annotate(ref[1], xy=(0.985, ref[0]), xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=8.5, color=MUTED)
        lo = float((y - ys).min()); hi = float((y + ys).max())
        if ref: hi = max(hi, ref[0])
        pad = 0.10 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        star = d[selected]
        ax.annotate("selected workhorse",
                    xy=(float(star["all_pair_mop_rmse_mean"].iloc[0]), float(star[ycol].iloc[0])),
                    xytext=(14, -26), textcoords="offset points",
                    fontsize=9, color=C_SIGMA, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=C_SIGMA, lw=1.0))
        ax.set_xlabel(r"All-pair MoP RMSE ($\Delta\log K$)")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
        _despine(ax)

    handles = [
        Line2D([], [], marker="*", ls="none", ms=17, mfc=C_SIGMA, mec=C_SIGMA,
               label="selected workhorse (heteroscedastic MLP ensemble)"),
        Line2D([], [], marker="o", ls="none", ms=8, mfc="white", mec=C_MAHAL, mew=1.8,
               label="other direct-UQ models"),
        Line2D([], [], marker="D", ls="none", ms=8, mfc="white", mec=C_TANI, mew=1.8,
               label="Chemprop evidential (partial native UQ)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.09), fontsize=9)
    fig.suptitle("All-pair direct-UQ model context and workhorse choice", y=1.02)
    fig.text(
        0.5, -0.135,
        "Both panels are pair-equal (MoP) on both axes; error bars are fold SD "
        "($n=5$, $\\mathrm{ddof}=1$). The horizontal bars overlap heavily: neighboring models\n"
        "differ in accuracy by well under one fold standard deviation, so the candidates "
        "separate on calibration rather than on RMSE.",
        ha="center", va="top", fontsize=8.4, color=MUTED,
    )
    fig.tight_layout()
    return _save(fig, "fig01_workhorse_selection")


# ------------------------------------------------------------------------ figure 2
_F2 = [
    ("sigma_total", r"$\sigma_{\mathrm{total}}$ (model uncertainty)", C_SIGMA, 2.9, "-", 6),
    ("pair_density_risk", "pair-density risk (training support count)", C_DENS, 2.7, "-", 5),
    ("mahalanobis_feature_median", "RDKit feature-median Mahalanobis distance", C_MAHAL, 2.5, "-", 4),
    ("tanimoto_median_distance", "Tanimoto median distance", C_TANI, 2.5, "-", 3),
]


def figure2() -> list[Path]:
    """Fixed-retention risk ordering: four risk scores against random retention.

    The four are chosen to separate *what each score can see* rather than to
    inventory aggregation variants: model uncertainty, a pure training-support
    count that uses no structure and no model output, a descriptor-space distance
    that encodes metal and medium, and a ligand-fingerprint distance. Every one
    carries a fold-SD band, because the notable result -- pair-density risk
    ordering error better than either distance above 30% retention -- has to be
    readable against fold spread to be believed.

    Aggregation variants of the two distances (nearest-neighbor and centroid) are
    in Supplementary Tables S7.1/S7.4 and behave like the median curves shown.
    """
    s = _read("results/downstream_cv/ad_retention/retention_summary.csv")
    s = s[(s.model_id == WORKHORSE) & (s.scope == "all_pairs")]
    fm = _read("manuscripts/final/tables/iz_mahalanobis_feature_median_retention_summary.csv")
    fm = fm[fm.scope == "all_pairs"]
    rnd = _read("results/downstream_cv/ad_retention/random_retention_summary.csv")
    rnd = rnd[(rnd.model_id == WORKHORSE) & (rnd.scope == "all_pairs")]

    def series(name, metric):
        src = fm if name == "mahalanobis_feature_median" else s
        g = src[src.metric_name == metric]
        if "score_name" in g.columns and name != "mahalanobis_feature_median":
            g = g[g.score_name == name]
        g = g.sort_values("retention_fraction")
        return (g.retention_fraction.to_numpy(float), g.fold_mean.to_numpy(float),
                g.fold_sd.to_numpy(float))

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))

    for ax, metric, ylab, title in [
        (axes[0], "mop_rmse", r"Retained-subset all-pair MoP RMSE ($\Delta\log K$)",
         "A. Retained-subset error"),
        (axes[1], "n_pairs_used", r"Supported pairs ($n\geq3$ retained rows)",
         "B. Supported-pair count"),
    ]:
        for name, lab, c, lw, ls, z in _F2:
            x, y, sd = series(name, metric)
            if not len(x):
                continue
            m = np.isfinite(x) & np.isfinite(y)
            # every retention strategy carries its fold-SD band
            ax.fill_between(x[m], (y - sd)[m], (y + sd)[m], color=c, alpha=0.13,
                            lw=0, zorder=z - 4)
            ax.plot(x[m], y[m], color=c, lw=lw, ls=ls, label=lab if ax is axes[0] else None,
                    zorder=z, solid_capstyle="round")
        g = rnd[rnd.metric_name == metric].sort_values("retention_fraction")
        if len(g):
            ax.plot(g.retention_fraction, g.fold_mean, color=C_RANDOM, ls=(0, (5, 2.2)),
                    lw=2.1, label="random retention" if ax is axes[0] else None, zorder=7)
        ax.set_xlabel("Retention fraction")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
        _despine(ax)

    ax = axes[0]
    ax.set_ylim(0.0, 2.42)
    ax.annotate("a count of training ligands per metal pair —\n"
                "no structure, no model output — sits below\n"
                "both distances above 30% retention (but see B)",
                xy=(0.55, 1.02), xytext=(0.40, 0.13), fontsize=8.6, color=C_DENS,
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", color=C_DENS, lw=1.1,
                                connectionstyle="arc3,rad=-0.24"))
    ax.annotate("but it fails in the tail:\nat 10% retention it is\nworse than random",
                xy=(0.113, 1.27), xytext=(0.145, 1.86), fontsize=8.6, color=C_DENS,
                arrowprops=dict(arrowstyle="->", color=C_DENS, lw=1.0,
                                connectionstyle="arc3,rad=0.22"))
    ax.annotate("the two distances converge above 30%\nretention and sit only modestly below random",
                xy=(0.72, 1.83), xytext=(0.40, 2.26), fontsize=8.6, color=C_TANI,
                ha="left",
                arrowprops=dict(arrowstyle="->", color=C_TANI, lw=1.0,
                                connectionstyle="arc3,rad=-0.18"))

    axes[1].annotate("pair-density risk keeps a third as many pairs\n"
                     "as anything else: its advantage in A is\n"
                     "selection, not ordering (Table S7.6)",
                     xy=(0.50, 96), xytext=(0.10, 880), fontsize=8.6, color=C_DENS,
                     ha="left", va="top",
                     arrowprops=dict(arrowstyle="->", color=C_DENS, lw=1.0,
                                     connectionstyle="arc3,rad=0.30"))

    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.15), fontsize=9)
    fig.suptitle("DS2 workhorse all-pair fixed-retention risk ordering", y=1.01)
    fig.text(0.5, -0.215,
             "Each score retains the lowest-risk fraction of held-out rows within every fold; "
             "MoP RMSE is recomputed on pairs keeping at least three rows.\nBands are fold SD "
             r"($n=5$, $\mathrm{ddof}=1$) on all four scores. Panel B makes the changing pair "
             "population explicit and is not a performance metric; on the pairs all scores share,\n"
             "the ordering of the three non-uncertainty scores reverses (Supplementary Table S7.6).",
             ha="center", va="top", fontsize=8.4, color=MUTED)
    fig.tight_layout()
    return _save(fig, "fig02_risk_ordering")


# ------------------------------------------------------------------------ figure 3
_BINS = ["abs_yhat_0-0.5", "abs_yhat_0.5-1", "abs_yhat_1-2",
         "abs_yhat_2-4", "abs_yhat_4-8", "abs_yhat_>8"]
_BINLAB = ["[0, 0.5)", "[0.5, 1)", "[1, 2)", "[2, 4)", "[4, 8)", r"[8, $\infty$)"]


def figure4() -> list[Path]:
    b = _read("results/downstream_cv/pair_acquisition/pair_acquisition_calibration_bin_summary.csv")
    b = b[b.model_id == WORKHORSE]
    r = _read("results/downstream_cv/pair_acquisition/magnitude_adjusted_risk_summary.csv")
    r = r[r.model_id == WORKHORSE].set_index("metric_name")

    def series(metric):
        s = b[b.metric_name == metric].set_index("bin_id").reindex(_BINS)
        return s["fold_mean"].to_numpy(float), s["fold_sd"].to_numpy(float)

    rmse, rmse_sd = series("rmse")
    sig, sig_sd = series("mean_sigma")

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})

    # --- A: error and uncertainty on ONE axis (same units) ---
    ax = axes[0]
    xs = np.arange(len(_BINS))
    ax.errorbar(xs, rmse, yerr=rmse_sd, color=C_MAHAL, lw=2.4, marker="o", ms=6.5,
                capsize=3, label=r"held-out RMSE")
    ax.errorbar(xs, sig, yerr=sig_sd, color=C_SIGMA, lw=2.4, marker="s", ms=6,
                capsize=3, ls=(0, (5, 2.2)), label=r"mean $\sigma_{\mathrm{total}}$")
    ax.fill_between(xs, np.minimum(rmse, sig), np.maximum(rmse, sig),
                    color="#9CA3AF", alpha=0.16, zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels(_BINLAB, rotation=25, ha="right")
    ax.set_xlabel(r"Predicted selectivity magnitude $|\widehat y|$ ($\Delta\log K$)")
    ax.set_ylabel(r"$\Delta\log K$")
    ax.set_title("A. Error and uncertainty rise together with predicted magnitude", loc="left")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.annotate(
        "the model is least accurate\nexactly where it is least confident",
        xy=(4.05, 0.5 * (rmse[4] + sig[4])), xytext=(1.45, 4.35),
        fontsize=9, color=MUTED, ha="left",
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0,
                        connectionstyle="arc3,rad=-0.2"),
    )
    _despine(ax)

    # --- B: does sigma survive controlling for magnitude? ---
    ax = axes[1]
    rows = [
        (r"$\rho(\sigma_{\mathrm{total}},\,|e|)$", "rho_sigma_abs_error", C_SIGMA, "raw"),
        (r"$\rho(|\widehat y|,\,|e|)$", "rho_abs_yhat_abs_error", MUTED, "raw"),
        (r"$\rho(\sigma_{\mathrm{total}},\,|e| \mid |\widehat y|)$",
         "rho_partial_sigma_error_given_abs_yhat", C_SIGMA, "partial"),
    ]
    ys = np.arange(len(rows))[::-1]
    for yi, (lab, key, col, kind) in zip(ys, rows):
        m = float(r.loc[key, "fold_mean"])
        s = float(r.loc[key, "fold_sd"])
        ax.barh(yi, m, xerr=s, height=0.5,
                color=col if kind == "raw" else "white",
                edgecolor=col, linewidth=1.8,
                hatch="" if kind == "raw" else "///",
                error_kw=dict(ecolor=INK, capsize=3, lw=1.0), zorder=2)
        ax.text(m + s + 0.018, yi, f"{m:.3f}", va="center", fontsize=9.5, color=INK)

    ax.set_yticks(ys)
    ax.set_yticklabels([lab for lab, *_ in rows], fontsize=10)
    ax.set_xlim(0, 0.82)
    ax.set_xlabel("Fold-mean Spearman rank correlation with absolute residual")
    ax.set_title("B. Uncertainty survives controlling for magnitude", loc="left")
    ax.grid(axis="y", visible=False)
    ax.annotate(
        "still 0.428 after removing the magnitude effect",
        xy=(0.40, ys[-1] - 0.27), xytext=(0.035, ys[-1] - 0.56),
        fontsize=9, color=C_SIGMA, va="center", ha="left",
        arrowprops=dict(arrowstyle="->", color=C_SIGMA, lw=1.0,
                        connectionstyle="arc3,rad=-0.25"),
    )
    ax.set_ylim(ys[-1] - 0.80, ys[0] + 0.45)
    _despine(ax)

    fig.suptitle("Predicted magnitude, error, and uncertainty", y=1.02)
    fig.text(
        0.5, -0.11,
        "Points and bars are fold means; error bars are fold SD ($n=5$). Panel A is a within-bin row-level "
        "RMSE, not pair-level MoP RMSE.\nPanel B: solid bars are raw rank correlations, the hatched bar is "
        "the fold-local partial correlation controlling for $|\\widehat y|$.",
        ha="center", va="top", fontsize=8.5, color=MUTED,
    )
    fig.tight_layout()
    return _save(fig, "fig04_predicted_magnitude")


# ------------------------------------------------------------------------ figure 4
# Released tables use the legacy `strategy` vocabulary: `y_pred` is ranking by predicted
# selectivity and `expected_improvement` is the standardized threshold margin (never a
# Gaussian expected-improvement integral). See pair_acquisition_metric_migration.json.
_STRAT = [
    ("y_pred", "predicted selectivity", C_VALUE),
    ("lcb_1sigma", "LCB-1", LCB_RAMP[0]),
    ("lcb_2sigma", "LCB-2", LCB_RAMP[1]),
    ("lcb_3sigma", "LCB-3", LCB_RAMP[2]),
    ("expected_improvement", "threshold margin", C_MARGIN),
    ("random", "random", C_RAND_BAR),
]
# High-selectivity rows are reported on the orientation-pooled `both` view; the
# near-zero objective is orientation-invariant and carries `near_zero`.
_TASKS = [("high_selectivity", "both", r"High selectivity ($\Delta\log K > 1$)"),
          ("low_selectivity", "near_zero", r"Near-zero selectivity ($|\Delta\log K| < 1$)")]


def _acq(df, task, orient, stem, k):
    """Pull (mean, sd) for every strategy at shortlist size k from the wide table."""
    col = f"{stem}_at_{k}"
    out = []
    for key, _, _ in _STRAT:
        r = df[(df.task == task) & (df.orientation == orient) & (df.strategy == key)]
        out.append((float(r[col].iloc[0]), float(r[col + "_sd"].iloc[0]))
                   if len(r) else (np.nan, np.nan))
    return np.array(out)


def figure5() -> list[Path]:
    """Objective-specific pair acquisition, three metrics x two objectives.

    True hit@k is restored: it answers a different experimental question from
    precision ("will this shortlist contain anything at all" versus "what fraction
    of it works"), and it is the only metric on which the most aggressive
    uncertainty penalty falls below random ranking.

    The figure is ordered so that the threshold-margin rule reads as the one
    uncertainty-integrated score that never underperforms the value-only baseline.
    """
    hp = _read("manuscripts/final/tables/pair_acquisition_hit_and_precision.csv")
    rg = _read("manuscripts/final/tables/pair_acquisition_regret.csv")

    apply_style()
    fig, axes = plt.subplots(3, 2, figsize=(12.4, 11.4))
    width = 0.13
    ks = [1, 5, 10]

    specs = [
        (hp, "any_hit", "True hit@$k$", "higher is better", (0.55, 1.02), False),
        (hp, "event_precision", "Event precision@$k$", "higher is better", (0.35, 0.95), False),
        (rg, None, r"Retrospective regret@$k$ ($\Delta\log K$)", "lower is better  \u2193", None, True),
    ]

    for row, (src, stem, ylab, hint, ylim, is_regret) in enumerate(specs):
        for col, (task, orient, tlab) in enumerate(_TASKS):
            ax = axes[row][col]
            for j2, (key, lab, c) in enumerate(_STRAT):
                s = "regret" if is_regret else stem
                m = np.array([_acq(src, task, orient, s, k)[j2] for k in ks])
                xs = np.arange(len(ks)) + (j2 - 2.5) * width
                ax.bar(xs, m[:, 0], width, yerr=m[:, 1], color=c, edgecolor="white",
                       linewidth=0.5, label=lab if (row == 0 and col == 0) else None,
                       error_kw=dict(ecolor=INK, capsize=2, lw=0.8), zorder=2)
            ax.set_xticks(np.arange(len(ks)))
            ax.set_xticklabels([f"$k = {k}$" for k in ks])
            if ylim:
                ax.set_ylim(*ylim)
            if col == 0:
                ax.set_ylabel(ylab)
            ax.set_title(f"{'ABCDEF'[row * 2 + col]}. {tlab}", loc="left", fontsize=10.6)
            ax.annotate(hint, xy=(0.015, 0.965), xycoords="axes fraction",
                        fontsize=8.4, color=MUTED, ha="left", va="top",
                        fontweight="bold" if is_regret else "normal")
            _despine(ax)

    # the two results the figure exists to show
    # LCB-3 sits above random at k=1 and below it at k=5 and k=10; annotate where it is true.
    axes[0][0].annotate("at $k\\geq5$ the strongest penalty\nfinds an event less often than random",
                        xy=(2.065, 0.828), xytext=(1.20, 0.632),
                        fontsize=8.8, color=LCB_RAMP[2], fontweight="bold", ha="center",
                        arrowprops=dict(arrowstyle="-|>", color=LCB_RAMP[2], lw=1.4,
                                        connectionstyle="arc3,rad=-0.22"))
    axes[0][0].annotate("", xy=(2.325, 0.888), xytext=(1.20, 0.652),
                        arrowprops=dict(arrowstyle="-|>", color=C_RAND_BAR, lw=1.2,
                                        connectionstyle="arc3,rad=-0.18"))
    axes[1][0].annotate("threshold margin leads\nwhere LCB collapses",
                        xy=(0.195, 0.835), xytext=(0.95, 0.885),
                        fontsize=9.2, color=C_MARGIN, fontweight="bold", ha="center",
                        arrowprops=dict(arrowstyle="-|>", color=C_MARGIN, lw=1.5,
                                        connectionstyle="arc3,rad=0.18"))
    axes[1][1].annotate("and matches the best rule here",
                        xy=(0.195, 0.804), xytext=(0.92, 0.885),
                        fontsize=9.2, color=C_MARGIN, fontweight="bold", ha="center",
                        arrowprops=dict(arrowstyle="-|>", color=C_MARGIN, lw=1.5,
                                        connectionstyle="arc3,rad=0.18"))

    handles = [Line2D([], [], marker="s", ls="none", ms=10, color=c, label=lab)
               for _, lab, c in _STRAT]
    fig.legend(handles=handles, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.028), fontsize=9.5)
    fig.suptitle("Objective-specific pair acquisition", y=1.0)
    fig.text(
        0.5, -0.062,
        "LCB-1/2/3 are one rule at increasing uncertainty penalty, shaded light to dark. "
        "The threshold-margin rule divides the distance past the event threshold by\n"
        "$\\sigma_{\\mathrm{total}}$ instead of subtracting it, and is the only "
        "uncertainty-integrated rule that never falls materially below the value-only baseline on an event "
        "metric in either objective.\nBars are fold means, error bars fold SD ($n=5$), after a "
        "pair-equal mean within each fold.",
        ha="center", va="top", fontsize=8.5, color=MUTED,
    )
    fig.tight_layout()
    return _save(fig, "fig05_pair_acquisition")


# ----------------------------------------------------------------------- figure S10
_S10 = [
    ("sigma_total", r"$\sigma_{\mathrm{total}}$", C_SIGMA, 2.8, 1.0),
    ("sigma_aleatoric", r"$\sigma_{\mathrm{aleatoric}}$", "#E8887F", 1.5, 0.9),
    ("sigma_epistemic_within", r"$\sigma_{\mathrm{epistemic,within}}$", "#F0B0A8", 1.3, 0.85),
    ("sigma_epistemic_between", r"$\sigma_{\mathrm{epistemic,between}}$", "#F6D3CE", 1.3, 0.85),
    ("pair_density_risk", "pair-density risk", C_DENS, 2.6, 1.0),
    ("mahalanobis_feature_centroid", "RDKit feature-centroid Mahalanobis", C_MAHAL, 2.4, 1.0),
    ("tanimoto_median_distance", "Tanimoto median distance", C_TANI2, 1.8, 0.95),
    ("tanimoto_nn_distance", "Tanimoto nearest-neighbor distance", C_TANI, 2.6, 1.0),
]


def figureS10() -> list[Path]:
    s = _read("results/downstream_cv/ad_retention/retention_summary.csv")
    s = s[(s.model_id == WORKHORSE) & (s.scope == "all_pairs") & (s.metric_name == "mop_rmse")]
    rnd = _read("results/downstream_cv/ad_retention/random_retention_summary.csv")
    rnd = rnd[(rnd.model_id == WORKHORSE) & (rnd.scope == "all_pairs") & (rnd.metric_name == "mop_rmse")]

    apply_style()
    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    for name, lab, c, lw, alpha in _S10:
        g = s[s.score_name == name].sort_values("retention_fraction")
        if not len(g):
            continue
        x, y, sd = (g.retention_fraction.to_numpy(float), g.fold_mean.to_numpy(float),
                    g.fold_sd.to_numpy(float))
        ax.plot(x, y, color=c, lw=lw, alpha=alpha, label=lab, zorder=3 if lw > 2 else 2)
        if lw > 2:
            ax.fill_between(x, y - sd, y + sd, color=c, alpha=0.13, zorder=1)

    if len(rnd):
        g = rnd.sort_values("retention_fraction")
        ax.plot(g.retention_fraction, g.fold_mean, color=C_RANDOM, ls=(0, (5, 2.2)),
                lw=2.4, label="random retention", zorder=4)

    ax.set_xlabel("Retention fraction (fold-local)")
    ax.set_ylabel(r"Retained-subset all-pair MoP RMSE ($\Delta\log K$)")
    ax.set_title("All-pair MoP retention error curves, registered inventory", loc="left")
    ax.annotate(
        "fingerprint distance is worse than\nno filter at all across this range",
        xy=(0.32, 1.86), xytext=(0.36, 1.30), fontsize=9, color=C_TANI,
        arrowprops=dict(arrowstyle="->", color=C_TANI, lw=1.2,
                        connectionstyle="arc3,rad=0.25"),
    )
    ax.annotate(
        "counting training ligands per metal pair beats\nevery molecular distance from 30% retention up",
        xy=(0.5, 0.956), xytext=(0.54, 0.42), fontsize=9, color=C_DENS,
        arrowprops=dict(arrowstyle="->", color=C_DENS, lw=1.2,
                        connectionstyle="arc3,rad=-0.2"),
    )
    _despine(ax)
    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.14), fontsize=9)
    fig.text(0.5, -0.20,
             "Shaded bands are fold SD on the emphasized scores. All scores oriented so that larger = greater "
             "predicted risk; retention keeps the lowest-scoring rows.",
             ha="center", va="top", fontsize=8.5, color=MUTED)
    return _save(fig, "figS10_retention_all_pairs")


# ----------------------------------------------------------------------- figure S11
_DBINS = ["0", "1-5", "6-10", "11-25", "26-50", ">50"]
_DLAB = ["0", "1–5", "6–10", "11–25", "26–50", ">50"]


def figureS11() -> list[Path]:
    d = _read("results/downstream_cv/pair_density/pair_density_summary.csv")
    d = d[d.scope == "all_pairs"]

    def g(metric):
        s = d[d.metric_name == metric].set_index("bin").reindex(_DBINS)
        return s.fold_mean.to_numpy(float), s.fold_sd.to_numpy(float)

    rmse, rmse_sd = g("rmse")
    sig, sig_sd = g("mean_sigma_total")
    picp, picp_sd = g("picp_90")

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    xs = np.arange(len(_DBINS))

    ax = axes[0]
    ax.errorbar(xs, rmse, yerr=rmse_sd, color=C_MAHAL, lw=2.4, marker="o", ms=6.5,
                capsize=3, label="held-out row-level RMSE")
    ax.errorbar(xs, sig, yerr=sig_sd, color=C_SIGMA, lw=2.4, marker="s", ms=6,
                capsize=3, ls=(0, (5, 2.2)), label=r"mean $\sigma_{\mathrm{total}}$")
    ax.fill_between(xs, np.minimum(rmse, sig), np.maximum(rmse, sig),
                    color="#9CA3AF", alpha=0.16, zorder=0)
    ax.set_xticks(xs); ax.set_xticklabels(_DLAB)
    ax.set_xlabel("Training ligands available for the held-out row's metal pair")
    ax.set_ylabel(r"$\Delta\log K$")
    ax.set_title("A. Error and uncertainty both fall with pair support", loc="left")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.annotate("uncertainty tracks support\nwithout being given it",
                xy=(1.05, 0.5 * (rmse[1] + sig[1])), xytext=(1.8, 1.35),
                fontsize=9, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0,
                                connectionstyle="arc3,rad=0.22"))
    _despine(ax)

    ax = axes[1]
    ax.errorbar(xs, picp, yerr=picp_sd, color=C_DENS, lw=2.4, marker="D", ms=6, capsize=3)
    ax.axhline(0.90, color=MUTED, ls=(0, (4, 3)), lw=1.0)
    ax.annotate("nominal 0.90", xy=(0.985, 0.901), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=8.5, color=MUTED)
    ax.fill_between(xs, picp, 0.90, where=picp < 0.90, color=C_DENS, alpha=0.12)
    ax.set_xticks(xs); ax.set_xticklabels(_DLAB)
    ax.set_xlabel("Training ligands available for the held-out row's metal pair")
    ax.set_ylabel("Row-level PICP@90")
    ax.set_title("B. Coverage degrades where support is thin", loc="left")
    ax.annotate("intervals undercover by ~10 points\nwhere the metal pair is unseen",
                xy=(0.15, picp[0]), xytext=(1.1, 0.775), fontsize=9, color=C_DENS,
                arrowprops=dict(arrowstyle="->", color=C_DENS, lw=1.2))
    _despine(ax)

    fig.suptitle("Pair-support density and workhorse behavior", y=1.02)
    fig.text(0.5, -0.10,
             "Fold means with fold SD ($n=5$). The zero-training-ligand bin is a metal-pair extrapolation "
             "regime, not merely an unseen ligand.",
             ha="center", va="top", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    return _save(fig, "figS11_pair_density")


# ----------------------------------------------------------------------- figure S12
def figureS12() -> list[Path]:
    d = _read("results/downstream_cv/pair_event_support/pair_event_support_pooled.csv")
    d = d[d.model_id == WORKHORSE].copy()

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0))

    ax = axes[0]
    x = d.n_heldout_rows.to_numpy(float)
    y = d.n_high_selectivity_events.to_numpy(float) + 0.5  # log-safe
    sc = ax.scatter(x, y, c=d.high_selectivity_fraction, cmap="viridis",
                    s=16, alpha=0.6, linewidths=0, vmin=0, vmax=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Held-out rows per species-level pair")
    ax.set_ylabel("Observed high-selectivity events (+0.5)")
    ax.set_title("A. Pair support versus event support", loc="left")
    for name, off in [("Cu:2.0--Ni:2.0", (-6, 10)), ("Cu:2.0--Zn:2.0", (8, -12)),
                      ("Ca:2.0--Mg:2.0", (-4, -16)), ("Co:2.0--Zn:2.0", (8, 4))]:
        r = d[d.metal_species_pair == name]
        if len(r):
            ax.annotate(name.replace(":2.0", "").replace("--", "/"),
                        xy=(float(r.n_heldout_rows.iloc[0]),
                            float(r.n_high_selectivity_events.iloc[0]) + 0.5),
                        xytext=off, textcoords="offset points", fontsize=8, color=INK,
                        ha="right" if off[0] < 0 else "left")
    ax.annotate("upper edge: pairs where almost every\nheld-out row is a high-selectivity event",
                xy=(60, 55), xytext=(1.6, 620), fontsize=8.5, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0,
                                connectionstyle="arc3,rad=-0.25"))
    _despine(ax)

    ax = axes[1]
    ok = d.out_of_fold_pooled_per_pair_rmse.notna()
    sc2 = ax.scatter(d.loc[ok, "n_heldout_rows"], d.loc[ok, "out_of_fold_pooled_per_pair_rmse"],
                     c=d.loc[ok, "high_selectivity_fraction"], cmap="viridis",
                     s=16, alpha=0.6, linewidths=0, vmin=0, vmax=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Held-out rows per species-level pair")
    ax.set_ylabel(r"Pooled per-pair RMSE ($\Delta\log K$)")
    ax.set_title("B. Per-pair error versus pair support", loc="left")
    ax.annotate("well-sampled pairs converge\nto ~1–2 $\\Delta\\log K$",
                xy=(700, 1.6), xytext=(12, 0.020), fontsize=9, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0,
                                connectionstyle="arc3,rad=0.25"))
    _despine(ax)
    cb = fig.colorbar(sc2, ax=axes.tolist(), pad=0.015, fraction=0.035)
    cb.set_label("Observed high-selectivity fraction of the pair", fontsize=9.5)

    fig.suptitle("Held-out pair support versus high-selectivity event support", y=1.02)
    fig.text(0.5, -0.06,
             "Both axes are logarithmic; the pair inventory spans three orders of magnitude in support. "
             "Colour is the observed high-selectivity fraction,\nnot per-pair RMSE. Descriptive and "
             "label-informed; not used for selection or ranking.",
             ha="center", va="top", fontsize=8.5, color=MUTED)
    return _save(fig, "figS12_pair_event_support")





# ---------------------------------------------------------------- figure S19
_FAMILY = [
    ("uncertainty", C_SIGMA,
     ("sigma_total", "sigma_aleatoric", "sigma_epistemic_within", "sigma_epistemic_between")),
    ("pair support (no structure, no model output)", C_DENS, ("pair_density_risk",)),
    ("RDKit descriptor space (ligand + metal + medium)", C_MAHAL,
     ("mahalanobis_feature_centroid", "mahalanobis_feature_median", "mahalanobis_feature_nn",
      "mahalanobis_feature_knn_mean_5", "mahalanobis_feature_knn_mean_10",
      "euclidean_feature_nn", "euclidean_feature_median")),
    ("ligand fingerprint space", C_TANI,
     ("tanimoto_nn_distance", "tanimoto_median_distance",
      "tanimoto_knn_mean_5", "tanimoto_knn_mean_10")),
    ("Chemprop learned representation", C_MAHAL2,
     ("chemprop_std_euclidean_nn", "chemprop_std_euclidean_knn_5",
      "chemprop_std_euclidean_knn_10", "chemprop_euclidean_nn", "chemprop_cosine_nn",
      "chemprop_mahalanobis_centroid", "chemprop_mahalanobis_nn")),
]


def figure3() -> list[Path]:
    """Every score in Table 3 on one axis, sorted, with a rule at rho = 0.

    Table 3's central claim is a *sign pattern* across 23 correlations -- every
    nearest-neighbor and every embedding score lands on the negative side of zero,
    across two unrelated representations. That is a statement about the shape of a
    distribution, and it was previously only recoverable by scanning a 23-row table.
    """
    d = _read("manuscripts/final/tables/iz_risk_score_spearman.csv").copy()
    colour, famlab = {}, {}
    for lab, c, keys in _FAMILY:
        for k in keys:
            colour[k] = c
            famlab[k] = lab
    d = d[d.score.isin(colour)].copy()
    d["c"] = d.score.map(colour)
    d = d.sort_values("fold_mean").reset_index(drop=True)

    apply_style()
    fig, ax = plt.subplots(figsize=(10.4, 8.2))
    y = np.arange(len(d))
    ax.axvline(0.0, color=C_RANDOM, lw=1.5, zorder=2)
    ax.axvspan(ax.get_xlim()[0], 0.0, color="#111111", alpha=0.035, zorder=0)
    for yi, r in d.iterrows():
        ax.plot([0.0, r.fold_mean], [yi, yi], color=r.c, lw=2.0, alpha=0.55, zorder=3)
        ax.errorbar(r.fold_mean, yi, xerr=r.fold_sd, color=r.c, fmt="o", ms=7.0,
                    capsize=3.0, lw=1.6, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(d.score_display.tolist(), fontsize=9)
    ax.set_xlabel(r"Fold-mean Spearman $\rho$ with the absolute held-out residual $|y-\widehat y|$")
    ax.set_ylabel("")
    ax.set_ylim(-0.8, len(d) - 0.2)
    ax.set_title("Score--residual association across the full score inventory", loc="left")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, marker="o", ls="-", lw=2.0, ms=7, label=lab)
               for lab, c, _ in _FAMILY]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 0.68),
              fontsize=8.8, frameon=True, facecolor="white", edgecolor=GRID)
    n_neg = int((d.fold_mean < 0).sum())
    ax.annotate(f"{n_neg} of {len(d)} scores land on the negative\n"
                "side of zero, and they are exactly the\n"
                "fingerprint nearest-neighbor and the\n"
                "learned-embedding scores: two unrelated\n"
                "representations, no positives among them",
                xy=(-0.03, 5.4), xytext=(0.235, 1.4), fontsize=8.8, color=MUTED, ha="left",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0,
                                connectionstyle="arc3,rad=0.2"))
    _despine(ax)
    ax.grid(axis="y", visible=False)
    fig.text(0.5, -0.045,
             "Points are fold means, whiskers fold SD ($n=5$, $\\mathrm{ddof}=1$), on 17,705 held-out rows per fold. "
             "Exact values are Supplementary Table S8.6.\nScores are oriented so that larger means greater predicted risk, "
             "so a positive $\\rho$ is the sign a usable risk score must have.",
             ha="center", va="top", fontsize=8.4, color=MUTED)
    fig.tight_layout()
    return _save(fig, "fig03_score_residual_inventory")


BUILDERS = {"0": figure0, "1": figure1, "2": figure2, "3": figure3, "4": figure4,
            "5": figure5, "S10": figureS10, "S11": figureS11, "S12": figureS12}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None,
                    help=f"subset of {sorted(BUILDERS)}")
    a = ap.parse_args(argv)
    keys = a.only or list(BUILDERS)
    bad = [k for k in keys if k not in BUILDERS]
    if bad:
        print(f"unknown figure ids: {bad}; valid: {sorted(BUILDERS)}", file=sys.stderr)
        return 2
    for k in keys:
        BUILDERS[k]()
    print(f"\n{len(keys)} figure(s) rebuilt into {FIGDIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
