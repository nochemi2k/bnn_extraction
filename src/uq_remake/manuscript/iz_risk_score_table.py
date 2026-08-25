"""Rebuild main-text Table 3: fold-mean Spearman rank correlation between each risk score
and the row-level absolute residual.

Why this module exists. Table 3 was assembled by hand during the "iz" revision and shipped
as `manuscripts/final/tables/iz_risk_score_spearman.csv` with no generator in the
repository, which made a displayed main-text table unreproducible. This module closes that
gap. It recomputes every row from registered artifacts and, by default, verifies the result
against the shipped CSV rather than overwriting it.

Sources, by row group:

* Workhorse uncertainty components, pair-density risk, and every RDKit/Tanimoto
  descriptor-space score are computed here from the frozen fold-local scores in
  ``results/downstream_cv/ad_retention/fold_scores/DS2_RDKIT_MLPENS5_HETERO/fold*.csv``.
* The post-hoc ``mahalanobis_feature_median`` sensitivity score is merged from
  ``manuscripts/final/tables/iz_mahalanobis_feature_median_scores.csv`` (see SI S7 for its
  provenance; it is deliberately not part of the frozen AD inventory).
* The Chemprop learned-representation distances are read from the registered
  ``results/downstream_cv/ad_distance_error_corr/ad_distance_error_corr_summary.csv``,
  because they are computed against the Chemprop evidential checkpoint's embedding space
  and are not present in the workhorse fold-score files.

Aggregation matches the rest of the manuscript: fold-local Spearman on all finite held-out
rows, then fold mean ± SD with ``ddof=1``.

    python -m uq_remake.manuscript.iz_risk_score_table            # verify against shipped CSV
    python -m uq_remake.manuscript.iz_risk_score_table --write    # regenerate it
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from uq_remake.manuscript.manuscript_paths import DOWNSTREAM, TABLES  # noqa: E402

WORKHORSE = "DS2_RDKIT_MLPENS5_HETERO"
CHEMPROP = "DS2_GRAPH_DMPNN_EVID"
FOLDS = tuple(f"fold{i}" for i in range(5))

FOLD_SCORES = DOWNSTREAM / "ad_retention" / "fold_scores" / WORKHORSE
DIST_CORR = DOWNSTREAM / "ad_distance_error_corr" / "ad_distance_error_corr_summary.csv"
MEDIAN_CACHE = TABLES / "iz_mahalanobis_feature_median_scores.csv"
OUT = TABLES / "iz_risk_score_spearman.csv"

MEDIAN_SCORE = "mahalanobis_feature_median"

# score -> (display name, role). Order here is the order of Table 3.
WORKHORSE_SCORES: dict[str, tuple[str, str]] = {
    "sigma_total": (r"$\sigma_{\mathrm{total}}$", "primary"),
    "mahalanobis_feature_centroid": ("RDKit feature-centroid Mahalanobis distance", "primary"),
    "tanimoto_nn_distance": ("Tanimoto nearest-neighbor distance", "primary"),
    "sigma_aleatoric": (r"$\sigma_{\mathrm{aleatoric}}$", "sensitivity"),
    "sigma_epistemic_within": (r"$\sigma_{\mathrm{epistemic,within}}$", "sensitivity"),
    "sigma_epistemic_between": (r"$\sigma_{\mathrm{epistemic,between}}$", "sensitivity"),
    "pair_density_risk": ("pair-density risk", "sensitivity"),
    "euclidean_feature_median": ("RDKit feature median Euclidean distance", "sensitivity"),
    MEDIAN_SCORE: ("RDKit feature-median Mahalanobis distance", "sensitivity"),
    "euclidean_feature_nn": ("RDKit feature Euclidean-NN", "sensitivity"),
    "mahalanobis_feature_knn_mean_10": (r"RDKit feature Mahalanobis $k$NN mean, $k=10$", "sensitivity"),
    "mahalanobis_feature_knn_mean_5": (r"RDKit feature Mahalanobis $k$NN mean, $k=5$", "sensitivity"),
    "tanimoto_median_distance": ("Tanimoto median distance", "sensitivity"),
    "mahalanobis_feature_nn": ("RDKit feature Mahalanobis-NN", "sensitivity"),
    "tanimoto_knn_mean_5": (r"Tanimoto $k$NN mean, $k=5$", "sensitivity"),
    "tanimoto_knn_mean_10": (r"Tanimoto $k$NN mean, $k=10$", "sensitivity"),
}

CHEMPROP_SCORES: tuple[str, ...] = (
    "chemprop_std_euclidean_nn",
    "chemprop_std_euclidean_knn_5",
    "chemprop_std_euclidean_knn_10",
    "chemprop_cosine_nn",
    "chemprop_mahalanobis_centroid",
    "chemprop_euclidean_nn",
    "chemprop_mahalanobis_nn",
)

MODEL_LABEL = {
    WORKHORSE: "workhorse (direct DS2)",
    CHEMPROP: "Chemprop evidential (learned representation)",
}


def _fmt(mean: float, sd: float) -> str:
    return f"{float(mean):.3f} ± {float(sd):.3f}"


def _load_fold(fold: str, median: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(FOLD_SCORES / f"{fold}.csv")
    add = median[median.outer_fold_id.astype(str) == fold][["record_id", MEDIAN_SCORE]]
    merged = df.merge(add, on="record_id", how="left", validate="one_to_one")
    if MEDIAN_SCORE in merged and merged[MEDIAN_SCORE].isna().all():
        raise RuntimeError(f"{fold}: {MEDIAN_SCORE} did not merge; check {MEDIAN_CACHE}")
    return merged


def workhorse_rows() -> pd.DataFrame:
    """Fold-local Spearman of each workhorse-side score against |y - yhat|."""
    median = pd.read_csv(MEDIAN_CACHE)
    per_fold: list[dict] = []
    for fold in FOLDS:
        df = _load_fold(fold, median)
        abs_err = (pd.to_numeric(df["y_true"], errors="coerce") - pd.to_numeric(df["y_pred"], errors="coerce")).abs()
        for score in WORKHORSE_SCORES:
            if score not in df.columns:
                raise RuntimeError(f"{fold}: missing score column {score}")
            values = pd.to_numeric(df[score], errors="coerce")
            ok = np.isfinite(values) & np.isfinite(abs_err)
            per_fold.append(
                {
                    "outer_fold_id": fold,
                    "score": score,
                    "rho": float(spearmanr(values[ok], abs_err[ok]).statistic),
                    "n_rows": int(ok.sum()),
                }
            )
    long = pd.DataFrame(per_fold)
    grouped = long.groupby("score", sort=False).agg(
        fold_mean=("rho", "mean"),
        fold_sd=("rho", lambda s: s.std(ddof=1)),
        n_valid_folds=("rho", "size"),
        n_rows=("n_rows", "mean"),
    )
    grouped = grouped.reindex(list(WORKHORSE_SCORES)).reset_index()
    grouped.insert(0, "model_id", WORKHORSE)
    return grouped


def chemprop_rows() -> pd.DataFrame:
    """Chemprop embedding-space Spearman values, read from the registered artifact."""
    src = pd.read_csv(DIST_CORR)
    sel = src[
        (src.model_id.astype(str) == CHEMPROP)
        & (src.scope.astype(str) == "all_pairs")
        & (src.metric_name.astype(str) == "spearman_abs_residual")
        & (src.score_name.astype(str).isin(CHEMPROP_SCORES))
    ].copy()
    missing = set(CHEMPROP_SCORES) - set(sel.score_name.astype(str))
    if missing:
        raise RuntimeError(f"{DIST_CORR}: missing Chemprop scores {sorted(missing)}")
    sel["_ord"] = sel.score_name.map({s: i for i, s in enumerate(CHEMPROP_SCORES)})
    sel = sel.sort_values("_ord")
    return pd.DataFrame(
        {
            "model_id": CHEMPROP,
            "score": sel.score_name.astype(str).to_numpy(),
            "fold_mean": sel.fold_mean.to_numpy(dtype=float),
            "fold_sd": sel.fold_sd.to_numpy(dtype=float),
            "n_valid_folds": sel.n_valid_folds.to_numpy(dtype=int),
            "n_rows": sel.n_rows_used_fold_mean.to_numpy(dtype=float),
            "score_display": sel.score_display.astype(str).to_numpy(),
        }
    )


def build() -> pd.DataFrame:
    wh = workhorse_rows()
    wh["score_display"] = wh.score.map(lambda s: WORKHORSE_SCORES[s][0])
    wh["role"] = wh.score.map(lambda s: WORKHORSE_SCORES[s][1])
    cp = chemprop_rows()
    cp["role"] = "sensitivity"
    out = pd.concat([wh, cp], ignore_index=True)
    out["model"] = out.model_id.map(MODEL_LABEL)
    out["Spearman ρ"] = [_fmt(m, s) for m, s in zip(out.fold_mean, out.fold_sd)]
    out["_ord"] = range(len(out))
    out["role_ord"] = (out.role != "primary").astype(int)
    cols = [
        "model_id", "score", "fold_mean", "fold_sd", "n_valid_folds", "n_rows",
        "score_display", "role", "model", "Spearman ρ", "_ord", "role_ord",
    ]
    return out[cols]


def verify(built: pd.DataFrame) -> list[str]:
    """Compare the rebuilt table against the shipped CSV. Returns a list of complaints."""
    if not OUT.exists():
        return [f"{OUT} does not exist"]
    shipped = pd.read_csv(OUT)
    problems: list[str] = []
    ship_map = dict(zip(shipped.score.astype(str), shipped["Spearman ρ"].astype(str)))
    for score, value in zip(built.score.astype(str), built["Spearman ρ"].astype(str)):
        if score not in ship_map:
            problems.append(f"{score}: absent from shipped table")
        elif ship_map[score] != value:
            problems.append(f"{score}: rebuilt {value} != shipped {ship_map[score]}")
    for score in ship_map:
        if score not in set(built.score.astype(str)):
            problems.append(f"{score}: in shipped table but not rebuilt")
    return problems


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--write", action="store_true", help="Overwrite the shipped CSV instead of verifying it.")
    args = p.parse_args(argv)

    built = build()
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        built.to_csv(OUT, index=False)
        print(f"wrote {OUT} ({len(built)} rows)", flush=True)
        return 0

    problems = verify(built)
    if problems:
        print(f"iz_risk_score_table: {len(problems)} mismatch(es) against {OUT}", flush=True)
        for line in problems:
            print(f"  {line}", flush=True)
        return 1
    print(f"iz_risk_score_table: {len(built)} rows reproduce {OUT.name} exactly", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
