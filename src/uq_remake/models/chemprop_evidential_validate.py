"""Inference-time validation of native Chemprop evidential (NIG) outputs.

Chemprop 1.6.1 `MoleculeModel` already applies the native inverse transforms
before `predict(..., return_unc_parameters=True)` returns values:

  lambda = softplus(raw_lambda)          # NIG nu / v
  alpha  = softplus(raw_alpha) + 1
  beta   = softplus(raw_beta)

Do not clip or replace invalid native parameters. Variances are computed only
for rows that already satisfy nu>0, alpha>1, beta>0 using Chemprop's formulas:

  var_aleatoric = beta / (alpha - 1)
  var_epistemic = beta / (lambda * (alpha - 1))

See chemprop.models.model.MoleculeModel (evidential branch) and
chemprop.uncertainty.uncertainty_predictor.EvidentialAleatoricPredictor /
EvidentialEpistemicPredictor.
"""
from __future__ import annotations

import numpy as np

NATIVE_PARAM_MEANING = {
    "mu": "predictive mean (gamma); Chemprop preds after inverse target scaling",
    "nu": "Chemprop lambda from return_unc_parameters; NIG evidence/precision v",
    "alpha": "Chemprop alpha after softplus(raw)+1",
    "beta": "Chemprop beta after softplus(raw), then * scaler.stds**2",
}

NATIVE_VAR_ALEATORIC = "beta / (alpha - 1)"
NATIVE_VAR_EPISTEMIC = "beta / (nu * (alpha - 1))"


def validate_native_evidential_output(native_output: dict) -> dict:
    """Validate native Chemprop evidential parameters. Never repair them."""
    mu = np.asarray(native_output["mu"], dtype=float).reshape(-1)
    nu = np.asarray(native_output["nu"], dtype=float).reshape(-1)
    alpha = np.asarray(native_output["alpha"], dtype=float).reshape(-1)
    beta = np.asarray(native_output["beta"], dtype=float).reshape(-1)
    n = int(len(mu))
    if not (len(nu) == len(alpha) == len(beta) == n):
        raise ValueError(
            f"NIG length mismatch: mu={n} nu={len(nu)} alpha={len(alpha)} beta={len(beta)}"
        )

    point_valid = np.isfinite(mu)
    finite_params = (
        np.isfinite(nu) & np.isfinite(alpha) & np.isfinite(beta)
    )
    uq_valid = (
        point_valid
        & finite_params
        & (nu > 0.0)
        & (alpha > 1.0)
        & (beta > 0.0)
    )

    nan = np.full(n, np.nan, dtype=float)
    var_a = nan.copy()
    var_e = nan.copy()
    var_a[uq_valid] = beta[uq_valid] / (alpha[uq_valid] - 1.0)
    var_e[uq_valid] = beta[uq_valid] / (nu[uq_valid] * (alpha[uq_valid] - 1.0))
    var_t = var_a + var_e

    status = np.full(n, "invalid_native_nig", dtype=object)
    status[uq_valid] = "valid_native_nig"
    status[point_valid & ~uq_valid] = "invalid_native_nig"
    status[~point_valid] = "invalid_point"

    invalid_nu = ~np.isfinite(nu) | (nu <= 0.0)
    invalid_alpha = ~np.isfinite(alpha) | (alpha <= 1.0)
    invalid_beta = ~np.isfinite(beta) | (beta <= 0.0)
    n_nonfinite = int(
        (~np.isfinite(mu) | ~np.isfinite(nu) | ~np.isfinite(alpha) | ~np.isfinite(beta)).sum()
    )

    def _min_finite(arr):
        finite = arr[np.isfinite(arr)]
        return float(finite.min()) if len(finite) else float("nan")

    return {
        "mu": mu,
        "nu": nu,
        "alpha": alpha,
        "beta": beta,
        "point_valid": point_valid,
        "uq_valid": uq_valid,
        "uncertainty_status": status,
        "var_aleatoric": var_a,
        "var_epistemic": var_e,
        "var_total": var_t,
        "n_rows_total": n,
        "n_point_valid": int(point_valid.sum()),
        "n_point_invalid": int((~point_valid).sum()),
        "n_uq_valid": int(uq_valid.sum()),
        "n_uq_invalid": int((~uq_valid).sum()),
        "n_invalid_nu": int(invalid_nu.sum()),
        "n_invalid_alpha": int(invalid_alpha.sum()),
        "n_invalid_beta": int(invalid_beta.sum()),
        "n_nonfinite_native_values": n_nonfinite,
        "min_nu": _min_finite(nu),
        "min_alpha": _min_finite(alpha),
        "min_beta": _min_finite(beta),
        "native_var_aleatoric_formula": NATIVE_VAR_ALEATORIC,
        "native_var_epistemic_formula": NATIVE_VAR_EPISTEMIC,
        "native_param_meaning": NATIVE_PARAM_MEANING,
    }


def fold_uq_status(*, n_point_invalid: int, n_uq_invalid: int, n_rows: int, aligned: bool) -> str:
    if not aligned or n_rows <= 0:
        return "invalid_fold"
    if n_point_invalid == n_rows:
        return "invalid_fold"
    if n_uq_invalid == 0 and n_point_invalid == 0:
        return "complete"
    if n_uq_invalid > 0:
        return "partial_invalid_rows"
    return "complete"
