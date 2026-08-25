"""Within-fold ligand-cluster bootstrap. Never pool folds. Never y_true/y_pred-only for MoP.

NOT ON THE PRODUCTION PATH. `data/run_hplus_decision.py` uses the equivalent
implementation in `metrics/point_metrics.py` instead. The two resample identically; this
one differs in two respects that make it the safer of the pair, and is kept as the
reference implementation and as the contract the tests assert:

  * it reports `bootstrap_sign_probability` with `hypothesis_test: False` rather than a
    key named like a p-value, and
  * it calls `assert_single_heldout_fold`, so it refuses pooled out-of-fold input.

If the H+ decision is ever recomputed, migrate it to this module rather than the other way
round. Do not delete one without reconciling the other.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from uq_remake.metrics.cv_protocol import PooledFoldError, assert_single_heldout_fold


def ligand_cluster_bootstrap(
    df: pd.DataFrame,
    metric_fn,
    ligand_col: str = "ligand_smiles",
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Resample ligand clusters inside one held-out fold; reconstruct the fold dataframe."""
    assert_single_heldout_fold(df, "bootstrap")
    if ligand_col not in df.columns:
        raise KeyError(f"bootstrap requires {ligand_col}")
    rng = np.random.default_rng(int(seed))
    ligands = df[ligand_col].astype(str).to_numpy()
    uniq, inv = np.unique(ligands, return_inverse=True)
    group_rows = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    point = metric_fn(df)
    boots = []
    n_g = len(uniq)
    for _ in range(int(n_boot)):
        chosen = rng.integers(0, n_g, size=n_g)
        sel = np.concatenate([group_rows[i] for i in chosen])
        part = df.iloc[sel].copy()
        val = metric_fn(part)
        if np.isfinite(val):
            boots.append(float(val))
    n_valid = int(len(boots))
    if n_valid == 0:
        lo = hi = float("nan")
    else:
        lo = float(np.quantile(boots, alpha / 2))
        hi = float(np.quantile(boots, 1 - alpha / 2))
    return {
        "value": float(point) if np.isfinite(point) else float("nan"),
        "ci_low": lo,
        "ci_high": hi,
        "bootstrap_unit": "ligand",
        "n_bootstrap_requested": int(n_boot),
        "n_bootstrap_valid": n_valid,
        "ci_method": "percentile_within_heldout_fold",
        "n_clusters": int(n_g),
    }


def paired_ligand_bootstrap(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metric_fn,
    ligand_col: str = "ligand_smiles",
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Same resampled ligand clusters for both models. Label is bootstrap_sign_probability, not a p-value."""
    assert_single_heldout_fold(df_a, "bootstrap_a")
    assert_single_heldout_fold(df_b, "bootstrap_b")
    if str(df_a["fold_id"].iloc[0]) != str(df_b["fold_id"].iloc[0]):
        raise PooledFoldError("paired bootstrap requires the same held-out fold")
    lig_a = set(df_a[ligand_col].astype(str))
    lig_b = set(df_b[ligand_col].astype(str))
    common = sorted(lig_a & lig_b)
    if not common:
        raise ValueError("no shared ligands for paired bootstrap")
    rng = np.random.default_rng(int(seed))
    a = df_a[df_a[ligand_col].astype(str).isin(common)].copy()
    b = df_b[df_b[ligand_col].astype(str).isin(common)].copy()
    uniq = np.asarray(common)
    rows_a = {g: np.flatnonzero(a[ligand_col].astype(str).to_numpy() == g) for g in uniq}
    rows_b = {g: np.flatnonzero(b[ligand_col].astype(str).to_numpy() == g) for g in uniq}
    diffs = []
    n_g = len(uniq)
    for _ in range(int(n_boot)):
        chosen = rng.choice(uniq, size=n_g, replace=True)
        sel_a = np.concatenate([rows_a[g] for g in chosen])
        sel_b = np.concatenate([rows_b[g] for g in chosen])
        va = metric_fn(a.iloc[sel_a])
        vb = metric_fn(b.iloc[sel_b])
        if np.isfinite(va) and np.isfinite(vb):
            diffs.append(float(va - vb))
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0:
        sign_prob = float("nan")
    else:
        sign_prob = float(np.mean(diffs > 0))
    return {
        "diff_mean": float(np.mean(diffs)) if len(diffs) else float("nan"),
        "bootstrap_sign_probability": sign_prob,
        "n_bootstrap_requested": int(n_boot),
        "n_bootstrap_valid": int(len(diffs)),
        "bootstrap_unit": "ligand",
        "ci_method": "percentile_within_heldout_fold",
        "hypothesis_test": False,
    }
