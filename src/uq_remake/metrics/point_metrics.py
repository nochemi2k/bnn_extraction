"""Accuracy metrics + ligand cluster bootstrap (Stage 1: no ACE/PICP)."""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def compute_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return {"r2": float("nan"), "mae": float("nan"), "n": float(len(y_true))}
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n": float(len(y_true)),
    }


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Resample unique groups (ligands); return (point, ci_low, ci_high)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    # Precompute row indices per group — avoids O(N) np.isin on every bootstrap draw.
    group_rows = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    point = metric_fn(y_true, y_pred)
    boots = []
    n_g = len(uniq)
    for _ in range(n_boot):
        chosen = rng.integers(0, n_g, size=n_g)
        sel = np.concatenate([group_rows[i] for i in chosen])
        if sel.size < 2:
            continue
        boots.append(metric_fn(y_true[sel], y_pred[sel]))
    if not boots:
        return point, float("nan"), float("nan")
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(point), lo, hi


def paired_cluster_bootstrap(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    groups: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired difference metric(A) - metric(B) under ligand resampling."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred_a = np.asarray(y_pred_a, dtype=float)
    y_pred_b = np.asarray(y_pred_b, dtype=float)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    group_rows = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    diff0 = metric_fn(y_true, y_pred_a) - metric_fn(y_true, y_pred_b)
    diffs = []
    n_g = len(uniq)
    for _ in range(n_boot):
        chosen = rng.integers(0, n_g, size=n_g)
        sel = np.concatenate([group_rows[i] for i in chosen])
        if sel.size < 2:
            continue
        diffs.append(metric_fn(y_true[sel], y_pred_a[sel]) - metric_fn(y_true[sel], y_pred_b[sel]))
    diffs = np.asarray(diffs, dtype=float)
    return {
        "diff": float(diff0),
        "ci95_low": float(np.quantile(diffs, 0.025)) if len(diffs) else float("nan"),
        "ci95_high": float(np.quantile(diffs, 0.975)) if len(diffs) else float("nan"),
        # Twice the smaller bootstrap tail fraction. This is a resampling sign
        # probability, NOT a calibrated p-value: no null model is fitted and no
        # multiplicity correction is applied. `p_two_sided` is retained because it is
        # already written into results/canonical/hplus_decision.json; prefer the
        # honestly named synonym below and do not report either as a hypothesis test.
        # metrics/bootstrap_fold.py implements the same resampling with only the
        # honest label, plus a guard against pooled out-of-fold input.
        "p_two_sided": float(2 * min(np.mean(diffs >= 0), np.mean(diffs <= 0))) if len(diffs) else float("nan"),
        "bootstrap_sign_probability": float(2 * min(np.mean(diffs >= 0), np.mean(diffs <= 0))) if len(diffs) else float("nan"),
        "hypothesis_test": False,
    }


def format_metric_line(
    model_id: str,
    metric_name: str,
    value: float,
    fold_sd: float | None = None,
    ci_low: float | None = None,
    ci_high: float | None = None,
    dispersion_applies: bool = True,
) -> str:
    if not dispersion_applies:
        return f"{model_id} | {metric_name} = {value:.6f} (no fold_sd / ci95_cluster; cmisplit single split)"
    return (
        f"{model_id} | {metric_name} = {value:.6f} ± {fold_sd:.6f} "
        f"[95% CI: {ci_low:.6f}, {ci_high:.6f}]"
    )
