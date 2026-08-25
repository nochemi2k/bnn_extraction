"""Iz-only DS2 risk-ordering figure with Tanimoto median and Mahalanobis median.

Does not overwrite the official three-score fig03_risk_ordering artifact.
Mahalanobis median is the median Ledoit-Wolf Mahalanobis distance to all
outer-training rows (median Euclidean distance in the same whitened space
used for centroid Mahalanobis). It is not stored in the frozen AD inventory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402

from uq_remake.downstream.analyses import risk_coverage_fold, summarize_fold_metrics  # noqa: E402
from uq_remake.manuscript import generate_figures as gf  # noqa: E402
from uq_remake.manuscript.generate_figures import (  # noqa: E402
    _ad_retention_panel,
    apply_style,
    load_config,
    new_figure,
    place_legend_below,
    save_figure,
    short_title,
)

SCORE_COL = "mahalanobis_feature_median"
MODEL_ID = "DS2_RDKIT_MLPENS5_HETERO"
FOLDS = tuple(f"fold{i}" for i in range(5))
SCORE_DIR = REPO / "results" / "downstream_cv" / "ad_retention" / "fold_scores" / MODEL_ID
TABLES = REPO / "manuscripts" / "final" / "tables"
CACHE_SCORES = TABLES / "iz_mahalanobis_feature_median_scores.csv"
CACHE_FOLD = TABLES / "iz_mahalanobis_feature_median_fold_retention.csv"
CACHE_SUM = TABLES / "iz_mahalanobis_feature_median_retention_summary.csv"
CACHE_SPEARMAN = TABLES / "iz_mahalanobis_feature_median_spearman.csv"
FIGURE_ID = "fig02_risk_ordering"

PLOT_SCORES = [
    "sigma_total",
    "mahalanobis_feature_centroid",
    "mahalanobis_feature_median",
    "tanimoto_nn_distance",
    "tanimoto_median_distance",
]
# Family colors: Tanimoto = greens; Mahalanobis = blue/violet. Keep sigma red.
IZ_SCORE_COLORS = {
    "sigma_total": "#C44E52",
    "mahalanobis_feature_centroid": "#2F6DB5",
    "mahalanobis_feature_median": "#7C3AED",
    "tanimoto_nn_distance": "#15803D",
    "tanimoto_median_distance": "#84CC16",
}

# The centroid and median Mahalanobis curves nearly coincide (50% retained-subset
# MoP RMSE 1.586 vs 1.580), so colour alone leaves the centroid series completely
# hidden beneath the median series, which is drawn later and on top. Dash the
# upper (median) curve so the centroid curve shows through the gaps. Applied
# before the legend is collected, so the legend keys carry the same dash pattern.
IZ_SCORE_LINESTYLES = {
    "mahalanobis_feature_median": (0, (5.0, 2.6)),
}


def fit_mahalanobis_ledoit(x_train: np.ndarray) -> dict:
    x_train = np.asarray(x_train, dtype=float)
    est = LedoitWolf().fit(x_train)
    return {
        "mean": np.asarray(est.location_, dtype=float),
        "covariance": np.asarray(est.covariance_, dtype=float),
    }


def _whiten_ledoit(fit: dict, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.asarray(fit["mean"], dtype=float)
    cov = np.asarray(fit["covariance"], dtype=float)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 1e-12)
    w = evecs * (1.0 / np.sqrt(evals))
    return (x - mu) @ w


def _median_euclidean_to_all(x_ref: np.ndarray, x_query: np.ndarray, batch: int = 64) -> np.ndarray:
    x_ref = np.asarray(x_ref, dtype=np.float32)
    x_query = np.asarray(x_query, dtype=np.float32)
    ref_norm = np.sum(x_ref * x_ref, axis=1)
    out = np.empty(len(x_query), dtype=np.float64)
    step = max(int(batch), 1)
    for i in range(0, len(x_query), step):
        q = x_query[i : i + step]
        q_norm = np.sum(q * q, axis=1, keepdims=True)
        d2 = q_norm + ref_norm - 2.0 * (q @ x_ref.T)
        np.maximum(d2, 0.0, out=d2)
        out[i : i + step] = np.sqrt(np.median(d2, axis=1))
    return out


def feature_dir(fold: str) -> Path:
    return REPO / "data" / "processed" / "features" / "rdkit_ds2" / fold


def load_frozen_fold_arrays(fold: str) -> dict[str, np.ndarray]:
    fold_dir = feature_dir(fold)
    manifest = json.loads((fold_dir / "manifest.json").read_text(encoding="utf-8"))
    z = np.load(fold_dir / "arrays.npz")
    arrays = {k: np.asarray(z[k]) for k in z.files}
    dim = int(manifest["feature_dimension"])
    for key in ("X_train", "X_val", "X_test"):
        if key in arrays and int(arrays[key].shape[1]) != dim:
            raise RuntimeError(f"{fold_dir}: {key} width {arrays[key].shape[1]} != manifest {dim}")
    return arrays


def compute_mahalanobis_median(fold: str, n_query: int) -> np.ndarray:
    arrays = load_frozen_fold_arrays(fold)
    x_train = np.concatenate([arrays["X_train"], arrays["X_val"]], axis=0)
    x_test = arrays["X_test"]
    if len(x_test) != n_query:
        raise RuntimeError(f"{fold}: feature rows {len(x_test)} != fold_scores {n_query}")
    fit = fit_mahalanobis_ledoit(x_train)
    zw_train = _whiten_ledoit(fit, x_train)
    zw_query = _whiten_ledoit(fit, x_test)
    return _median_euclidean_to_all(zw_train, zw_query, batch=64)


def load_or_compute_median_scores() -> pd.DataFrame:
    if CACHE_SCORES.exists():
        cached = pd.read_csv(CACHE_SCORES)
        n_ok = int((cached.outer_fold_id.nunique() == 5) and (len(cached) == 5 * 17705))
        if n_ok and cached[SCORE_COL].notna().all():
            print(f"Using cached {CACHE_SCORES}", flush=True)
            return cached
    rows = []
    for fold in FOLDS:
        scores = pd.read_csv(SCORE_DIR / f"{fold}.csv")
        print(f"Computing {SCORE_COL} for {fold} (n={len(scores)})", flush=True)
        vals = compute_mahalanobis_median(fold, len(scores))
        part = scores[["record_id", "outer_fold_id"]].copy()
        part[SCORE_COL] = vals
        rows.append(part)
        print(f"  {fold} median range {float(np.nanmin(vals)):.4g}–{float(np.nanmax(vals)):.4g}", flush=True)
    out = pd.concat(rows, ignore_index=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(CACHE_SCORES, index=False)
    return out


def retention_from_fold_scores(median_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    spearman_rows = []
    for fold in FOLDS:
        scores = pd.read_csv(SCORE_DIR / f"{fold}.csv")
        add = median_scores[median_scores.outer_fold_id.astype(str) == fold][["record_id", SCORE_COL]]
        merged = scores.merge(add, on="record_id", how="left", validate="one_to_one")
        if merged[SCORE_COL].isna().any():
            raise RuntimeError(f"{fold}: missing {SCORE_COL} after merge")
        fold_rows.extend(
            risk_coverage_fold(
                merged,
                SCORE_COL,
                fold,
                experiment_id="iz_ad_retention",
                dataset="DS2",
                model_id=MODEL_ID,
                family="workhorse",
                score_role="sensitivity",
            )
        )
        yt = pd.to_numeric(merged["y_true"], errors="coerce")
        yp = pd.to_numeric(merged["y_pred"], errors="coerce")
        sc = pd.to_numeric(merged[SCORE_COL], errors="coerce")
        abs_err = (yt - yp).abs()
        ok = np.isfinite(sc) & np.isfinite(abs_err)
        rho, _ = spearmanr(sc[ok], abs_err[ok])
        spearman_rows.append(
            {
                "outer_fold_id": fold,
                "score": SCORE_COL,
                "spearman_rho": float(rho),
                "n_rows": int(ok.sum()),
            }
        )
    fold_df = pd.DataFrame(fold_rows)
    keys = [
        "dataset",
        "model_id",
        "family",
        "score_role",
        "experiment_id",
        "scope",
        "score_name",
        "retention_fraction",
        "metric_name",
        "baseline",
    ]
    summary = summarize_fold_metrics(fold_df, keys)
    fold_df.to_csv(CACHE_FOLD, index=False)
    summary.to_csv(CACHE_SUM, index=False)
    sp = pd.DataFrame(spearman_rows)
    sp.to_csv(CACHE_SPEARMAN, index=False)
    print(
        "Mahalanobis median Spearman "
        f"{sp.spearman_rho.mean():.3f} ± {sp.spearman_rho.std(ddof=1):.3f}",
        flush=True,
    )
    return fold_df, summary


def fmt_pm(mean: float, sd: float, nd: int = 3) -> str:
    return f"{mean:.{nd}f} ± {sd:.{nd}f}"


def print_key_numbers(summary: pd.DataFrame) -> None:
    mop = summary[
        (summary.scope == "all_pairs")
        & (summary.metric_name == "mop_rmse")
        & (summary.retention_fraction == 0.5)
    ]
    auc = summary[
        (summary.scope == "all_pairs") & (summary.metric_name == "retention_curve_auc_mop_rmse")
    ]
    pairs = summary[
        (summary.scope == "all_pairs")
        & (summary.metric_name == "n_pairs_used")
        & (summary.retention_fraction == 0.5)
    ]
    if len(mop):
        print("50% MoP RMSE", fmt_pm(float(mop.fold_mean.iloc[0]), float(mop.fold_sd.iloc[0])), flush=True)
    if len(auc):
        print("AURC MoP RMSE", fmt_pm(float(auc.fold_mean.iloc[0]), float(auc.fold_sd.iloc[0])), flush=True)
    if len(pairs):
        print("50% supported pairs", fmt_pm(float(pairs.fold_mean.iloc[0]), float(pairs.fold_sd.iloc[0]), 1), flush=True)


def _restyle_overlapping_series(axes, labels: dict) -> None:
    """Dash the curves listed in IZ_SCORE_LINESTYLES so co-located series stay readable.

    Applied after the panels are drawn but before the legend is collected, so the
    legend keys show the same dash pattern as the plotted curves.
    """
    wanted = {labels.get(score, score): style for score, style in IZ_SCORE_LINESTYLES.items()}
    for ax in axes:
        for line in ax.get_lines():
            style = wanted.get(line.get_label())
            if style is not None:
                line.set_linestyle(style)


def draw_figure(extra_sum: pd.DataFrame) -> list[Path]:
    cfg = load_config()
    apply_style(cfg)
    cfg = dict(cfg)
    colors = dict(cfg.get("score_color_map", {}))
    labels = dict(cfg.get("score_labels", {}))
    colors.update(IZ_SCORE_COLORS)
    labels[SCORE_COL] = "RDKit feature-median Mahalanobis distance"
    labels["mahalanobis_feature_centroid"] = "RDKit feature-centroid Mahalanobis distance"
    labels["tanimoto_nn_distance"] = "Tanimoto NN distance"
    labels["tanimoto_median_distance"] = "Tanimoto median distance"
    cfg["score_color_map"] = colors
    cfg["score_labels"] = labels
    gf.PRIMARY_RETENTION_COLORS = dict(IZ_SCORE_COLORS)

    root = REPO / "results" / "downstream_cv" / "ad_retention"
    summary = pd.read_csv(root / "retention_summary.csv")
    summary = pd.concat([summary, extra_sum], ignore_index=True, sort=False)
    random_sum = pd.read_csv(root / "random_retention_summary.csv")
    random_fold = pd.read_csv(root / "random_fold_retention.csv")

    fig, axes = new_figure((10.4, 4.8), 1, 2)
    n0 = _ad_retention_panel(
        axes[0],
        summary,
        random_sum,
        "DS2",
        MODEL_ID,
        "all_pairs",
        "mop_rmse",
        PLOT_SCORES,
        cfg,
        "Retained-subset MoP RMSE",
        "A. Retained-subset MoP RMSE",
        primary=PLOT_SCORES,
    )
    n1 = _ad_retention_panel(
        axes[1],
        summary,
        pd.DataFrame(),
        "DS2",
        MODEL_ID,
        "all_pairs",
        "n_pairs_used",
        PLOT_SCORES,
        cfg,
        r"Supported pairs ($n\geq 3$ retained rows)",
        "B. Supported-pair count",
        primary=PLOT_SCORES,
    )
    _restyle_overlapping_series(axes, labels)
    rf = random_fold[
        (random_fold.dataset.astype(str) == "DS2")
        & (random_fold.model_id.astype(str) == MODEL_ID)
        & (random_fold.scope.astype(str) == "all_pairs")
        & (random_fold.metric_name.astype(str) == "mop_rmse")
    ].copy()
    if len(rf) and "n_pairs_used" in rf.columns:
        g = rf.groupby("retention_fraction", as_index=False)["n_pairs_used"].agg(
            fold_mean="mean", fold_sd="std"
        )
        axes[1].plot(
            g.retention_fraction,
            g.fold_mean,
            color="#111111",
            ls=(0, (5, 2.2)),
            lw=2.2,
            label="random",
            zorder=5,
        )
        axes[1].fill_between(
            g.retention_fraction,
            g.fold_mean - g.fold_sd.fillna(0),
            g.fold_mean + g.fold_sd.fillna(0),
            color="#111111",
            alpha=0.08,
        )
        marks = {0.10, 0.25, 0.50, 1.00}
        rx = g.retention_fraction.to_numpy(dtype=float)
        ry = g.fold_mean.to_numpy(dtype=float)
        keep = np.array([round(float(v), 2) in marks for v in rx])
        if keep.any():
            axes[1].scatter(rx[keep], ry[keep], color="#111111", s=18, zorder=4)
        n1 += 1
    handles, labels_out = axes[0].get_legend_handles_labels()
    place_legend_below(fig, handles, labels_out, ncol=3)
    short_title(fig, "DS2 selected-workhorse all-pair fixed-retention risk ordering")
    paths = save_figure(fig, FIGURE_ID, cfg)
    print(f"Drew {n0}+{n1} series; saved {paths}", flush=True)
    return paths


def main() -> None:
    if CACHE_SUM.exists():
        extra_sum = pd.read_csv(CACHE_SUM)
    else:
        median_scores = load_or_compute_median_scores()
        _fold_df, extra_sum = retention_from_fold_scores(median_scores)
        print_key_numbers(extra_sum)
    draw_figure(extra_sum)


if __name__ == "__main__":
    main()
