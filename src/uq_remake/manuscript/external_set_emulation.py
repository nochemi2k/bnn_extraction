"""JACS Task 9: emulate the benchmark's external-set design on our own
ligand-disjoint held-out predictions.

The benchmark (ref 9, SI p. S3) selected its 20-ligand "external set" by:
PCA on normalized RDKit descriptors -> first 10 components (~60% variance)
-> Mahalanobis distance of each ligand from the AD centre in that reduced
space -> sort by distance -> take twenty ligands at *approximately equal*
distance (a shell, not the most-distant tail). No retraining is needed:
every ligand already has held-out predictions from the ligand-disjoint
folds, so an "external set" is just a subset of rows already computed.

Geometry (per fold, held-out ligands only)
--------------------------------------------
Reuses the frozen, already-fold-consistent inputs rather than recomputing
RDKit descriptors or refitting a scaler:

  * `data/processed/features/rdkit_ds2/fold{N}/arrays.npz` -- `X_test` is
    the joint (RDKit + metal-pair + medium) feature matrix, already
    train-fold-only StandardScaled (see `data.rdkit_features.fit_fold_preprocessor`).
    Columns 0:187 are the pure RDKit block (`manifest.json["feature_names"]`
    starting with `rdkit_`); columns 187: carry metal/medium covariates that
    do not belong to "normalized RDKit descriptors of the ligand" and are
    dropped here.
  * `data/splits/ds2_project_folds.json[fold]["test_idx"]` positionally
    aligns `X_test` rows to `data/processed/ds2.csv` rows (verified: 100%
    match against fold-score `ligand_smiles` for fold0). A ligand can repeat
    across DS2 rows (different metal pairs / media); the RDKit-only block is
    bit-identical across a ligand's repeats (verified), so one row per
    unique `ligand_smiles` is kept before PCA.

PCA is fit **on each fold's own held-out ligand set** (not on training) --
this is the natural reading of "select N ligands from the pool you have,
by their position in that pool's own reduced-descriptor space," and it
needs no training-set access. sklearn's `PCA.fit_transform` mean-centers,
so the "distance from the AD centre" is Mahalanobis distance from the
PCA-space origin, covariance estimated the same way as the rest of this
repo's AD scores (Ledoit-Wolf, `protocol.COVARIANCE_ESTIMATOR`).

Three N in {20, 30, 40} selections per fold
---------------------------------------------
  * shell: the N *consecutive* ligands in ascending-distance sorted order
    whose distance range (max - min) is smallest -- the tightest possible
    run of N, i.e. the benchmark's "approximately equal distance" rule
    made precise. Found by sliding a fixed-size window over the sorted
    array (O(n)).
  * tail: the N most distant ligands.
  * random: N ligands drawn uniformly without replacement, >=200 repeats,
    seed 0, fold-seeded (seed + fold_index * 1009, matching the repo's
    existing random-baseline convention).

Every set's rows are scored with `metrics.fold_metrics.mop_block` (the
repo's standard MoP RMSE, >=3 retained rows/pair) -- the same function used
to produce the frozen `results/cv_metrics` numbers, so the full
ligand-disjoint reference computed here must reproduce 2.078 +/- 0.130
(`protocol.WORKHORSE_EXPECTED["mop_rmse"]`) or this module has a bug, not a
finding.

Decomposition (the actual result; the headline ratio alone is not
interpretable): for each fold, gap(shell) = shell - full is split into
gap_size = random - full (subset-size / pair-coverage effect, isolated by
the random baseline) and gap_selection = shell - random (what shell
selection specifically adds on top of just being small), so
gap(shell) = gap_size + gap_selection identically; same for tail.

Emits `manuscripts/jacs/tables/external_set_emulation.csv`.

    python -m uq_remake.manuscript.external_set_emulation
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

from uq_remake.downstream.protocol import FOLDS, PAIR_COLUMN
from uq_remake.manuscript.jacs_common import FOLD_SCORES_DIR, JACS_TABLES, REMAKE_ROOT, fold_mean_sd_row, load_all_folds, with_abs_residual, write_provenance
from uq_remake.metrics.cv_protocol import fold_mean_sd
from uq_remake.metrics.fold_metrics import mop_block

OUT = JACS_TABLES / "external_set_emulation.csv"

DS2_PATH = REMAKE_ROOT / "data" / "processed" / "ds2.csv"
SPLITS_PATH = REMAKE_ROOT / "data" / "splits" / "ds2_project_folds.json"
FEATURES_DIR = REMAKE_ROOT / "data" / "processed" / "features" / "rdkit_ds2"

N_VALUES = (20, 30, 40)
PCA_N_COMPONENTS = 10
BENCHMARK_TARGET_VARIANCE = 0.60
RANDOM_REPEATS = 200
RANDOM_SEED = 0
KINDS = ("shell", "tail", "random")
WORKHORSE_EXPECTED_MOP_RMSE = (2.078, 0.130)


def _nan() -> float:
    return float("nan")


# --------------------------------------------------------------- geometry ---
def _load_fold_geometry(fold: str) -> dict:
    with open(SPLITS_PATH, encoding="utf-8") as f:
        splits = json.load(f)[fold]
    test_idx = splits["test_idx"]
    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]

    ds2 = pd.read_csv(DS2_PATH, usecols=["ligand_smiles"])
    ligand_test = ds2.iloc[test_idx]["ligand_smiles"].reset_index(drop=True)
    train_ligands = set(ds2.iloc[train_idx]["ligand_smiles"]) | set(ds2.iloc[val_idx]["ligand_smiles"])

    with open(FEATURES_DIR / fold / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    feature_names = manifest["feature_names"]
    rdkit_cols = [i for i, n in enumerate(feature_names) if str(n).startswith("rdkit_")]

    arrays = np.load(FEATURES_DIR / fold / "arrays.npz")
    X_test = arrays["X_test"][:, rdkit_cols]
    if len(X_test) != len(ligand_test):
        raise RuntimeError(f"{fold}: X_test rows {len(X_test)} != test_idx rows {len(ligand_test)}")

    df = pd.DataFrame(X_test)
    df.insert(0, "ligand_smiles", ligand_test.to_numpy())
    dedup = df.drop_duplicates(subset="ligand_smiles", keep="first").reset_index(drop=True)
    X_ligand = dedup.drop(columns=["ligand_smiles"]).to_numpy(dtype=float)
    ligand_ids = dedup["ligand_smiles"].to_numpy()

    return {
        "ligand_ids": ligand_ids,
        "X_ligand": X_ligand,
        "n_rdkit_cols": len(rdkit_cols),
        "train_ligands": train_ligands,
    }


def _pca_mahalanobis(X: np.ndarray, n_components: int = PCA_N_COMPONENTS) -> dict:
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(X)  # already mean-centered by PCA
    explained = float(np.sum(pca.explained_variance_ratio_))
    fit = LedoitWolf().fit(scores)
    d2 = np.maximum(np.asarray(fit.mahalanobis(scores), dtype=float), 0.0)
    dist = np.sqrt(d2)
    return {"distances": dist, "explained_variance": explained, "n_components_used": n_components}


def _shell_window(sorted_vals: np.ndarray, n: int) -> tuple[int, int]:
    """Start index of the N consecutive sorted values with the smallest (max - min) spread."""
    if len(sorted_vals) < n:
        raise ValueError(f"pool of {len(sorted_vals)} ligands smaller than N={n}")
    spreads = sorted_vals[n - 1 :] - sorted_vals[: len(sorted_vals) - n + 1]
    start = int(np.argmin(spreads))
    return start, start + n


# ---------------------------------------------------------- set scoring ---
def _score_rows(rows: pd.DataFrame) -> dict:
    n_rows = int(len(rows))
    n_pairs = int(rows[PAIR_COLUMN].nunique())
    if n_rows == 0:
        return {"mop_rmse": _nan(), "mean_abs_error": _nan(), "n_rows": 0, "n_pairs_covered": 0}
    mop = mop_block(rows)
    return {
        "mop_rmse": mop["mop_rmse"],
        "mean_abs_error": float(rows["_abs_residual"].mean()),
        "n_rows": n_rows,
        "n_pairs_covered": n_pairs,
        "n_pairs_retained_mop": int(mop["n_pairs_retained"]),
    }


def _rows_for_ligands(df: pd.DataFrame, ligands: np.ndarray) -> pd.DataFrame:
    return df[df["ligand_smiles"].isin(set(ligands))]


def main() -> dict:
    fold_frames_raw = load_all_folds()
    fold_frames = {fold: with_abs_residual(df) for fold, df in fold_frames_raw.items()}

    geometry_rows: list[dict] = []
    fold_level_rows: list[dict] = []
    overlap_violations: list[dict] = []
    fold_gap_records: list[dict] = []

    for i, fold in enumerate(FOLDS):
        geo = _load_fold_geometry(fold)
        pm = _pca_mahalanobis(geo["X_ligand"])
        geometry_rows.append(
            {
                "outer_fold_id": fold,
                "n_ligands_in_pool": int(len(geo["ligand_ids"])),
                "n_rdkit_cols": geo["n_rdkit_cols"],
                "n_components_used": pm["n_components_used"],
                "explained_variance_10_components": pm["explained_variance"],
            }
        )

        order = np.argsort(pm["distances"], kind="mergesort")
        sorted_ligands = geo["ligand_ids"][order]
        sorted_dist = pm["distances"][order]

        df = fold_frames[fold]
        full = _score_rows(df)
        fold_level_rows.append({"outer_fold_id": fold, "N": "", "kind": "full_ligand_disjoint", "repeat": "", **full})

        rng = np.random.default_rng(int(RANDOM_SEED) + i * 1009)
        for N in N_VALUES:
            if len(sorted_ligands) < N:
                continue
            start, end = _shell_window(sorted_dist, N)
            shell_ligands = sorted_ligands[start:end]
            tail_ligands = sorted_ligands[-N:]

            if not set(shell_ligands).issubset(set(geo["ligand_ids"])) or set(shell_ligands) & geo["train_ligands"]:
                overlap_violations.append({"outer_fold_id": fold, "N": N, "kind": "shell", "issue": "overlaps training ligands"})
            if set(tail_ligands) & geo["train_ligands"]:
                overlap_violations.append({"outer_fold_id": fold, "N": N, "kind": "tail", "issue": "overlaps training ligands"})

            shell_score = _score_rows(_rows_for_ligands(df, shell_ligands))
            tail_score = _score_rows(_rows_for_ligands(df, tail_ligands))
            fold_level_rows.append({"outer_fold_id": fold, "N": N, "kind": "shell", "repeat": "", "shell_distance_spread": float(sorted_dist[end - 1] - sorted_dist[start]), **shell_score})
            fold_level_rows.append({"outer_fold_id": fold, "N": N, "kind": "tail", "repeat": "", **tail_score})

            random_scores = []
            for rep in range(int(RANDOM_REPEATS)):
                sel = rng.choice(geo["ligand_ids"], size=N, replace=False)
                random_scores.append(_score_rows(_rows_for_ligands(df, sel)))
            rand_mop = np.array([r["mop_rmse"] for r in random_scores], dtype=float)
            rand_mae = np.array([r["mean_abs_error"] for r in random_scores], dtype=float)
            rand_npairs = np.array([r["n_pairs_covered"] for r in random_scores], dtype=float)
            random_avg = {
                "mop_rmse": float(np.nanmean(rand_mop)),
                "mean_abs_error": float(np.nanmean(rand_mae)),
                "n_rows": int(np.mean([r["n_rows"] for r in random_scores])),
                "n_pairs_covered": float(np.nanmean(rand_npairs)),
                "n_pairs_retained_mop": float(np.nanmean([r.get("n_pairs_retained_mop", np.nan) for r in random_scores])),
            }
            fold_level_rows.append({"outer_fold_id": fold, "N": N, "kind": "random", "repeat": RANDOM_REPEATS, **random_avg})

            fold_gap_records.append(
                {
                    "outer_fold_id": fold,
                    "N": N,
                    "full_mop_rmse": full["mop_rmse"],
                    "shell_mop_rmse": shell_score["mop_rmse"],
                    "tail_mop_rmse": tail_score["mop_rmse"],
                    "random_mop_rmse": random_avg["mop_rmse"],
                }
            )

    geometry_df = pd.DataFrame(geometry_rows)
    fold_level_df = pd.DataFrame(fold_level_rows)

    # Full ligand-disjoint reference, aggregated the same way as the reference figure.
    full_vals = fold_level_df[fold_level_df.kind == "full_ligand_disjoint"]["mop_rmse"].tolist()
    full_stats = fold_mean_sd(full_vals)
    reproduces_reference = (
        abs(full_stats["fold_mean"] - WORKHORSE_EXPECTED_MOP_RMSE[0]) <= 5e-4
        and abs(full_stats["fold_sd"] - WORKHORSE_EXPECTED_MOP_RMSE[1]) <= 5e-4
    )
    if not reproduces_reference:
        print(
            f"external_set_emulation: STOP -- full ligand-disjoint MoP RMSE "
            f"{full_stats['fold_mean']:.4f} +/- {full_stats['fold_sd']:.4f} does not reproduce "
            f"the frozen {WORKHORSE_EXPECTED_MOP_RMSE[0]} +/- {WORKHORSE_EXPECTED_MOP_RMSE[1]}",
            flush=True,
        )

    summary_rows: list[dict] = [
        {
            "table_section": "full_reference",
            "N": "",
            "kind": "full_ligand_disjoint",
            "metric_name": "mop_rmse",
            "fold_mean": full_stats["fold_mean"],
            "fold_sd": full_stats["fold_sd"],
            "n_valid_folds": full_stats["n_valid_folds"],
            "cv_fraction": full_stats["fold_sd"] / abs(full_stats["fold_mean"]) if full_stats["fold_mean"] else _nan(),
            "reproduces_frozen_2078_130": reproduces_reference,
        }
    ]

    for N in N_VALUES:
        for kind in KINDS:
            part = fold_level_df[(fold_level_df.N == N) & (fold_level_df.kind == kind)]
            if part.empty:
                continue
            for metric in ("mop_rmse", "mean_abs_error", "n_pairs_covered"):
                fm = fold_mean_sd_row(part[metric].tolist())
                summary_rows.append(
                    {
                        "table_section": "external_set_summary",
                        "N": N,
                        "kind": kind,
                        "metric_name": metric,
                        "fold_mean": fm["fold_mean"],
                        "fold_sd": fm["fold_sd"],
                        "n_valid_folds": fm["n_valid_folds"],
                        "cv_fraction": fm["fold_sd"] / abs(fm["fold_mean"]) if fm["fold_mean"] else _nan(),
                    }
                )

    # Decomposition: gap(shell/tail) = gap_size (random - full) + gap_selection (shell/tail - random).
    gap_df = pd.DataFrame(fold_gap_records)
    decomposition_rows: list[dict] = []
    for N in N_VALUES:
        part = gap_df[gap_df.N == N]
        if part.empty:
            continue
        gap_size = (part["random_mop_rmse"] - part["full_mop_rmse"]).tolist()
        for kind, col in (("shell", "shell_mop_rmse"), ("tail", "tail_mop_rmse")):
            gap_total = (part[col] - part["full_mop_rmse"]).tolist()
            gap_selection = (part[col] - part["random_mop_rmse"]).tolist()
            fm_total = fold_mean_sd_row(gap_total)
            fm_size = fold_mean_sd_row(gap_size)
            fm_sel = fold_mean_sd_row(gap_selection)
            identity_ok = abs(fm_total["fold_mean"] - (fm_size["fold_mean"] + fm_sel["fold_mean"])) <= 1e-6
            decomposition_rows.append(
                {
                    "table_section": "decomposition",
                    "N": N,
                    "kind": kind,
                    "gap_total_vs_full": fm_total["fold_mean"],
                    "gap_total_sd": fm_total["fold_sd"],
                    "gap_size_component_vs_random": fm_size["fold_mean"],
                    "gap_size_sd": fm_size["fold_sd"],
                    "gap_selection_component_vs_random": fm_sel["fold_mean"],
                    "gap_selection_sd": fm_sel["fold_sd"],
                    "identity_holds": identity_ok,
                }
            )

    geometry_df["table_section"] = "pca_geometry"
    fold_level_df["table_section"] = "external_set_fold_level"
    summary_df = pd.DataFrame(summary_rows)
    decomposition_df = pd.DataFrame(decomposition_rows)
    overlap_df = pd.DataFrame(overlap_violations) if overlap_violations else pd.DataFrame([{"table_section": "overlap_check", "n_violations": 0}])
    if overlap_violations:
        overlap_df["table_section"] = "overlap_violation"
        print(f"external_set_emulation: WARNING {len(overlap_violations)} overlap violations found", flush=True)
    else:
        print("external_set_emulation: no external set overlaps its fold's training ligands", flush=True)

    out = pd.concat([geometry_df, summary_df, fold_level_df, decomposition_df, overlap_df], ignore_index=True)
    out["source"] = "results/downstream_cv/ad_retention/fold_scores/DS2_RDKIT_MLPENS5_HETERO/fold{0..4}.csv + data/processed/features/rdkit_ds2/fold{0..4}/arrays.npz + data/splits/ds2_project_folds.json"
    JACS_TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    write_provenance(
        OUT,
        module="external_set_emulation.main",
        inputs=[FOLD_SCORES_DIR / f"{f}.csv" for f in FOLDS] + [SPLITS_PATH, DS2_PATH] + [FEATURES_DIR / f / "arrays.npz" for f in FOLDS],
        notes={
            "n_values": list(N_VALUES),
            "random_repeats": RANDOM_REPEATS,
            "shell_rule": "N consecutive ligands in ascending-distance sorted order with the smallest (max-min) spread",
            "reproduces_frozen_workhorse_mop_rmse": reproduces_reference,
            "explained_variance_target": BENCHMARK_TARGET_VARIANCE,
            "explained_variance_achieved_fold_mean": float(geometry_df["explained_variance_10_components"].mean()),
        },
    )
    print(f"wrote {OUT} ({len(out)} rows)", flush=True)
    return {"table": out, "reproduces_reference": reproduces_reference}


if __name__ == "__main__":
    main()
