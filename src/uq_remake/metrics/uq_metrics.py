"""Fold-local interval helpers. ACE is a fixed 9-level grid, not adaptive binning.

Primary interval metrics live in fold_metrics.interval_metrics_fold and must be
computed inside a single held-out fold.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from uq_remake.metrics.cv_protocol import ACE_LEVELS

CONFIDENCE_LEVELS = np.asarray(ACE_LEVELS, dtype=float)


def _z_for_confidence(p: float) -> float:
    return float(stats.norm.ppf(0.5 + 0.5 * p))


def interval_coverage(y_true, y_pred, sigma_total, confidence: float) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sigma_total = np.asarray(sigma_total, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(sigma_total) & (sigma_total > 0)
    n_invalid = int((~mask).sum())
    if n_invalid:
        # Do not drop invalid uncertainty to inflate coverage.
        return float("nan")
    if mask.sum() == 0:
        return float("nan")
    z = _z_for_confidence(confidence)
    lo = y_pred - z * sigma_total
    hi = y_pred + z * sigma_total
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def adaptive_calibration_error(y_true, y_pred, sigma_total, levels=None) -> float:
    """ACE = mean_q |C(q)-q| for q in {0.1,...,0.9}. Name kept for callers; not adaptive bins."""
    levels = CONFIDENCE_LEVELS if levels is None else np.asarray(levels, dtype=float)
    if tuple(np.round(levels.astype(float), 1)) != ACE_LEVELS:
        raise ValueError("ACE must use the nine-level grid 0.1..0.9")
    errs = [abs(interval_coverage(y_true, y_pred, sigma_total, float(p)) - float(p)) for p in levels]
    if not all(np.isfinite(errs)):
        return float("nan")
    return float(np.mean(errs))


def picp_at_90(y_true, y_pred, sigma_total) -> float:
    return interval_coverage(y_true, y_pred, sigma_total, 0.9)

