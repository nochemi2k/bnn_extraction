"""Shared constants and helpers for the JACS repositioning analyses.

Everything here reads frozen artifacts under ``results/`` and ``data/processed/``.
Nothing retrains, rescores, or writes outside ``manuscripts/jacs/``. See
``manuscripts/jacs/CODE_PROMPT_analyses.md`` for the task specification and
``manuscripts/jacs/analysis/chemical_sanity_checks.json`` for the prototype
numbers each analysis module is a regression test against.

Pair-pooled-across-folds vs. (pair, fold) cells
------------------------------------------------
The task prose describes the primary analysis unit as a "(species pair, fold)
cell." Empirically, the headline prototype numbers (Task 1's per-score mean/
median/frac-positive with n_pairs=647; Task 3's REE/non-REE ratios) only
reproduce when a pair's held-out rows are pooled **across all five folds**
before computing one Spearman correlation or one within-pair SD per pair —
not when each (pair, fold) combination is treated as its own cell (which
gives a different sample size, 267 pairs / 1124 cells, and different numbers).
This is legitimate: DS2 folds are ligand-disjoint, so a given metal-species
pair's held-out rows across different folds never share a ligand and were
never seen by the model that produced them — pooling them is pooling a
pair's own test predictions, not pooling OOF metrics across models or folds.

Both readings are kept, at the two places they are actually needed:

* The Task 1 headline distribution (`within_pair_score_residual.csv`,
  aggregation "distribution_over_pairs") uses the pair-pooled-across-folds
  reading, because that is what reproduces the prototype figures.
* The genuinely fold-local aggregation ("fold_mean_plus_sd") and the
  retention curves (Task 2) use true (pair, fold) cells, so the reported
  fold mean ± SD (ddof=1) is a real fold-to-fold spread, matching the
  fold-locality convention used everywhere else in this repository.
* `pair_density_risk` is asserted constant within every (pair, fold) cell
  (the literal, narrower claim in the prompt) independent of which reading
  is used downstream.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from uq_remake import REMAKE_ROOT
from uq_remake.downstream.protocol import (
    AD_RETENTION_DIR,
    DS2_WORKHORSE,
    FOLDS,
    PAIR_COLUMN,
    RANDOM_N_REPEATS,
    RANDOM_SEED,
    RETENTION_FRACTIONS,
    WORKHORSE_PRIMARY_SCORES,
)
from uq_remake.metrics.cv_protocol import EVALUATION_PARTITION, FOLD_SD_DDOF, assert_single_heldout_fold, fold_mean_sd

JACS_ROOT = REMAKE_ROOT / "manuscripts" / "jacs"
JACS_TABLES = JACS_ROOT / "tables"
JACS_ANALYSIS = JACS_ROOT / "analysis"
PROTOTYPE_JSON = JACS_ANALYSIS / "chemical_sanity_checks.json"

WORKHORSE = DS2_WORKHORSE
FOLD_SCORES_DIR = AD_RETENTION_DIR / "fold_scores" / WORKHORSE
DS1_PATH = REMAKE_ROOT / "data" / "processed" / "ds1.csv"
DS2_PATH = REMAKE_ROOT / "data" / "processed" / "ds2.csv"
CMI_EXTERNAL_PATH = REMAKE_ROOT / "data" / "processed" / "m4_external.csv"

# Every candidate risk score present in the workhorse fold-score files.
CANDIDATE_SCORES = (
    "sigma_total",
    "sigma_aleatoric",
    "sigma_epistemic_within",
    "sigma_epistemic_between",
    "tanimoto_nn_distance",
    "tanimoto_median_distance",
    "tanimoto_knn_mean_5",
    "tanimoto_knn_mean_10",
    "mahalanobis_feature_centroid",
    "mahalanobis_feature_nn",
    "mahalanobis_feature_knn_mean_5",
    "mahalanobis_feature_knn_mean_10",
    "euclidean_feature_nn",
    "euclidean_feature_median",
    "pair_density_risk",
)
PRIMARY_SCORES = WORKHORSE_PRIMARY_SCORES  # (sigma_total, tanimoto_nn_distance, mahalanobis_feature_centroid)

MIN_CELL_N = 20
MIN_CELL_DISTINCT = 5
MIN_PAIR_N_SCALE_FREE = 10
HIGH_SEL_TAU = 1.0

# Score constant within every (pair, fold) cell by construction: a training-set
# support count cannot rank two ligands held out from the same (pair, fold).
CONSTANT_WITHIN_CELL_SCORES = ("pair_density_risk",)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REMAKE_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REMAKE_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(p)


def write_provenance(out_csv: Path, *, module: str, inputs: list[Path | str], notes: dict | None = None) -> Path:
    """Sidecar `<name>.provenance.json` naming exact inputs and the git commit."""
    payload = {
        "table": rel(out_csv),
        "module": module,
        "git_commit": git_commit(),
        "inputs": [rel(p) for p in inputs],
        "generated_by": "manuscripts/jacs analysis pipeline (--jacs)",
    }
    if notes:
        payload.update(notes)
    side = out_csv.with_suffix(out_csv.suffix + ".provenance.json")
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return side


def load_fold(fold: str) -> pd.DataFrame:
    path = FOLD_SCORES_DIR / f"{fold}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing frozen fold scores: {path}")
    df = pd.read_csv(path)
    assert_single_heldout_fold(df, context=f"jacs:{fold}")
    part = df["evaluation_partition"].astype(str)
    if not part.eq(EVALUATION_PARTITION).all():
        raise RuntimeError(f"{fold}: expected evaluation_partition={EVALUATION_PARTITION}")
    return df


def load_all_folds() -> dict[str, pd.DataFrame]:
    return {fold: load_fold(fold) for fold in FOLDS}


def concat_folds(fold_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pair-pooled-across-folds view. See module docstring for when this is valid."""
    parts = []
    for fold, df in fold_frames.items():
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def with_abs_residual(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    yt = pd.to_numeric(out["y_true"], errors="coerce")
    yp = pd.to_numeric(out["y_pred"], errors="coerce")
    out["_abs_residual"] = (yt - yp).abs()
    return out


def assert_pair_density_constant_within_cells(fold_frames: dict[str, pd.DataFrame]) -> dict:
    """The literal, narrow claim: pair_density_risk never varies inside a (pair, fold) cell."""
    n_cells = 0
    violations = []
    for fold, df in fold_frames.items():
        for pair, g in df.groupby(PAIR_COLUMN, sort=False):
            n_cells += 1
            nun = pd.to_numeric(g["pair_density_risk"], errors="coerce").nunique()
            if nun > 1:
                violations.append({"fold": fold, "pair": pair, "n_distinct": int(nun)})
    if violations:
        raise AssertionError(
            f"pair_density_risk is not constant within {len(violations)} (pair, fold) cells: "
            f"{violations[:10]}"
        )
    return {"n_cells_total": n_cells, "n_violations": 0}


# --- Periodic-table block classifier (s / 3d / 4d-5d / 4f / 5f / p_block) ------------
_S_BLOCK_Z = frozenset({1, 3, 4, 11, 12, 19, 20, 37, 38, 55, 56, 87, 88})


def block_from_atomic_number(z) -> str:
    try:
        z = int(z)
    except (TypeError, ValueError):
        return "unknown"
    if z in _S_BLOCK_Z:
        return "s"
    if 21 <= z <= 30:
        return "3d"
    if (39 <= z <= 48) or (72 <= z <= 80):
        return "4d-5d"
    if 57 <= z <= 71:
        return "4f"
    if 89 <= z <= 103:
        return "5f"
    return "p_block"


DS2_CHEMICAL_JOIN_COLS = [
    "pair_id",
    "logK_x",
    "logK_y",
    "selectivity_logK",
    "metal_Sanderson Electronegativity_x",
    "metal_Sanderson Electronegativity_y",
    "metal_covalent_radius_by_pyykko_x",
    "metal_covalent_radius_by_pyykko_y",
    "metal_oxidation_state_x",
    "metal_oxidation_state_y",
    "temperature_C",
    "ionic_strength_M",
    "ligand_smiles",
    "metal_species_pair",
]


def load_ds2_chemical_axes() -> tuple[pd.DataFrame, dict]:
    """Per-row chemical-axis columns derived from ds2.csv, keyed by `record_id`.

    See `sigma_chemical_axes` module docstring: the task-specified join key
    `(ligand_smiles, metal_species_pair)` is not unique in ds2.csv (11,127 /
    70,390 groups have >1 row); `record_id` (fold scores) == `pair_id`
    (ds2.csv) is the exact 1:1 key actually intended, verified by 100% match.
    """
    ds2 = pd.read_csv(DS2_PATH, usecols=DS2_CHEMICAL_JOIN_COLS)
    dup_groups = ds2.groupby(["ligand_smiles", "metal_species_pair"]).size()
    diag = {
        "n_ligand_pair_groups": int(len(dup_groups)),
        "n_ligand_pair_groups_with_gt1_row": int((dup_groups > 1).sum()),
        "join_key_used": "record_id == pair_id (see sigma_chemical_axes module docstring)",
    }
    out = pd.DataFrame(
        {
            "record_id": ds2["pair_id"],
            "favoured_logK": ds2[["logK_x", "logK_y"]].max(axis=1),
            "weaker_logK": ds2[["logK_x", "logK_y"]].min(axis=1),
            "abs_dlogK": ds2["selectivity_logK"].abs(),
            "d_electronegativity": (ds2["metal_Sanderson Electronegativity_x"] - ds2["metal_Sanderson Electronegativity_y"]).abs(),
            "d_covalent_radius": (ds2["metal_covalent_radius_by_pyykko_x"] - ds2["metal_covalent_radius_by_pyykko_y"]).abs(),
            "d_oxidation_state": (ds2["metal_oxidation_state_x"] - ds2["metal_oxidation_state_y"]).abs(),
            "temperature_C": ds2["temperature_C"],
            "ionic_strength_M": ds2["ionic_strength_M"],
        }
    )
    return out, diag


def join_ds2_chemical_axes(df: pd.DataFrame, ds2_derived: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(ds2_derived, on="record_id", how="left", validate="one_to_one")
    n_unmatched = int(merged["favoured_logK"].isna().sum())
    if n_unmatched:
        raise RuntimeError(f"{n_unmatched} fold-score rows failed to join to ds2.csv on record_id==pair_id")
    merged["abs_error"] = pd.to_numeric(merged["y_true"], errors="coerce").sub(pd.to_numeric(merged["y_pred"], errors="coerce")).abs()
    merged["log1p_support"] = np.log1p(pd.to_numeric(merged["n_training_ligands_for_pair"], errors="coerce"))
    return merged


def fold_mean_sd_row(values, **extra) -> dict:
    stats = fold_mean_sd(values)
    row = {
        "fold_mean": stats["fold_mean"],
        "fold_sd": stats["fold_sd"],
        "n_folds": stats["n_folds"],
        "n_valid_folds": stats["n_valid_folds"],
        "fold_sd_ddof": FOLD_SD_DDOF,
    }
    row.update(extra)
    return row
