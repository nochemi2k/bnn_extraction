"""Fold-local DS1/DS2 metrics. Never concatenate held-out folds."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from uq_remake.metrics.cv_protocol import (
    ACE_LEVELS,
    EVALUATION_PARTITION,
    HIGH_SEL_TAU,
    MIN_PAIR_N,
    PAIR_COLUMN,
    RETENTION_LEVELS,
    PooledFoldError,
    assert_single_heldout_fold,
    finite_mask,
    metric_row,
    require_species_pair,
)

_NAN = float("nan")


def _base(df: pd.DataFrame, scope: str, context: str) -> tuple[pd.DataFrame, str]:
    fold = assert_single_heldout_fold(df, context)
    out = df.copy()
    if "evaluation_partition" not in out.columns:
        out["evaluation_partition"] = EVALUATION_PARTITION
    if str(out["evaluation_partition"].iloc[0]) != EVALUATION_PARTITION:
        raise PooledFoldError(f"{context}: evaluation_partition must be {EVALUATION_PARTITION}")
    return out, fold


def filter_finite_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    mask = finite_mask(df["y_true"], df["y_pred"])
    used = df.loc[mask].copy()
    diag = {
        "n_rows_total": int(len(df)),
        "n_rows_used": int(mask.sum()),
        "n_rows_invalid": int((~mask).sum()),
    }
    return used, diag


def _rmse(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) == 0:
        return _NAN
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def row_point_metrics(y_true, y_pred) -> dict:
    """Unweighted row-level RMSE and R² over individual held-out rows.

    RMSE_row = sqrt(mean_i (y_i - ŷ_i)²).
    R²_row = 1 - Σ_i (y_i - ŷ_i)² / Σ_i (y_i - ȳ)², with ȳ the unweighted
    mean of the supplied rows. Constant-target sets return NaN. This is not a
    pair-level application metric.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) == 0:
        return {"rmse": _NAN, "r2": _NAN}
    rmse = _rmse(yt, yp)
    if len(yt) < 2:
        return {"rmse": rmse, "r2": _NAN}
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot <= 1e-18:
        r2 = _NAN
    else:
        r2 = float(1.0 - np.sum((yt - yp) ** 2) / ss_tot)
    return {"rmse": rmse, "r2": r2}


def sign_metrics(y_true, y_pred) -> dict:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) == 0:
        return {
            "sign_accuracy_all": _NAN,
            "n_exact_zero_observed": 0,
            "sign_accuracy_nonzero_observed": _NAN,
        }
    zero = np.abs(yt) <= 0.0
    all_acc = float(np.mean(np.sign(yt) == np.sign(yp)))
    n_zero = int(zero.sum())
    if (~zero).any():
        nz_acc = float(np.mean(np.sign(yt[~zero]) == np.sign(yp[~zero])))
    else:
        nz_acc = _NAN
    return {
        "sign_accuracy_all": all_acc,
        "n_exact_zero_observed": n_zero,
        "sign_accuracy_nonzero_observed": nz_acc,
    }


def _pair_groups(df: pd.DataFrame, min_n: int):
    col = require_species_pair(df, "mop")
    n_total = int(df[col].nunique())
    sizes = df.groupby(col, sort=False).size()
    keep = sizes[sizes >= int(min_n)].index
    retained = df[df[col].isin(keep)].copy()
    return col, n_total, retained, int(len(keep))


def per_pair_metric(g: pd.DataFrame, name: str) -> float:
    yt = g["y_true"].to_numpy(dtype=float)
    yp = g["y_pred"].to_numpy(dtype=float)
    if name == "rmse":
        return _rmse(yt, yp)
    if name == "r2":
        if len(g) < 2:
            return _NAN
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        if ss_tot <= 1e-18:
            return _NAN
        return float(1.0 - np.sum((yt - yp) ** 2) / ss_tot)
    raise ValueError(name)


def mop_block(df: pd.DataFrame, min_n: int = MIN_PAIR_N) -> dict:
    """Mean-of-pairs RMSE and mean of within-pair R².

    For each retained pair p with n_p ≥ min_n ligands:
      RMSE_p = sqrt(mean_i (y_pi - ŷ_pi)²)
      R²_p  = 1 - Σ_i (y_pi - ŷ_pi)² / Σ_i (y_pi - ȳ_p)²
    then mop_rmse = mean_p RMSE_p and mop_r2 = mean_p R²_p over pairs with
    finite R²_p. Constant-target pairs make R²_p invalid (NaN) and are
    excluded from mop_r2 only. This is not R² of pair means.
    """
    assert_single_heldout_fold(df, "mop_block")
    col, n_total, retained, n_retained = _pair_groups(df, min_n)
    n_rows_ret = int(len(retained))
    if n_retained == 0:
        return {
            "pair_column": col,
            "min_pair_size": int(min_n),
            "n_pairs_total": n_total,
            "n_pairs_retained": 0,
            "n_pairs_valid": {"rmse": 0, "r2": 0},
            "n_pairs_invalid": {"rmse": 0, "r2": 0},
            "n_rows_in_retained_pairs": 0,
            "mop_rmse": _NAN,
            "mop_r2": _NAN,
            "pair_weights_sum_ok": True,
        }
    vals = {"rmse": [], "r2": []}
    invalid = {"rmse": 0, "r2": 0}
    for _, g in retained.groupby(col, sort=False):
        for name in ("rmse", "r2"):
            v = per_pair_metric(g, name)
            if np.isfinite(v):
                vals[name].append(v)
            else:
                invalid[name] += 1
    mop = {
        "mop_rmse": float(np.mean(vals["rmse"])) if vals["rmse"] else _NAN,
        "mop_r2": float(np.mean(vals["r2"])) if vals["r2"] else _NAN,
    }
    return {
        "pair_column": col,
        "min_pair_size": int(min_n),
        "n_pairs_total": n_total,
        "n_pairs_retained": n_retained,
        "n_pairs_valid": {k: int(len(vals[k])) for k in vals},
        "n_pairs_invalid": invalid,
        "n_rows_in_retained_pairs": n_rows_ret,
        **mop,
        "pair_weights_sum_ok": True,
    }


def high_selectivity_frames(df: pd.DataFrame, tau: float = HIGH_SEL_TAU, min_n: int = MIN_PAIR_N) -> dict:
    """Two named estimands; never an unlabeled 'high-selectivity' result."""
    assert_single_heldout_fold(df, "high_selectivity")
    col = require_species_pair(df, "high_selectivity")
    hs_rows = df.loc[df["y_true"].abs() > float(tau)].copy()
    sizes_hs = hs_rows.groupby(col, sort=False).size() if len(hs_rows) else pd.Series(dtype=int)
    keep_hs = sizes_hs[sizes_hs >= int(min_n)].index
    row_estimand = hs_rows[hs_rows[col].isin(keep_hs)].copy()

    sizes_all = df.groupby(col, sort=False).size()
    keep_all = sizes_all[sizes_all >= int(min_n)].index
    membership = df[df[col].isin(keep_all)]
    membership_hs = membership.loc[membership["y_true"].abs() > float(tau)].copy()
    return {
        "high_selectivity_row": row_estimand,
        "high_selectivity_all_pair_membership": membership_hs,
        "n_high_selectivity_rows": int(len(hs_rows)),
        "n_pairs_high_selectivity_total": int(hs_rows[col].nunique()) if len(hs_rows) else 0,
        "tau": float(tau),
    }


def nig_student_t_interval(gamma, nu, alpha, beta, confidence: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NIG predictive: Student-t with df=2α, scale=sqrt(β(1+ν)/(ν α))."""
    g = np.asarray(gamma, dtype=float)
    nu = np.asarray(nu, dtype=float)
    a = np.asarray(alpha, dtype=float)
    b = np.asarray(beta, dtype=float)
    valid = np.isfinite(g) & np.isfinite(nu) & np.isfinite(a) & np.isfinite(b) & (nu > 0) & (a > 1) & (b > 0)
    lo = np.full_like(g, _NAN, dtype=float)
    hi = np.full_like(g, _NAN, dtype=float)
    df = 2.0 * a
    scale = np.sqrt(b * (1.0 + nu) / (nu * a))
    tcrit = np.full_like(g, _NAN, dtype=float)
    if valid.any():
        tcrit[valid] = stats.t.ppf(0.5 + 0.5 * float(confidence), df[valid])
        half = tcrit[valid] * scale[valid]
        lo[valid] = g[valid] - half
        hi[valid] = g[valid] + half
    return lo, hi, valid


def normal_interval(mu, sigma_total, confidence: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.asarray(mu, dtype=float)
    sig = np.asarray(sigma_total, dtype=float)
    valid = np.isfinite(mu) & np.isfinite(sig) & (sig > 0)
    z = float(stats.norm.ppf(0.5 + 0.5 * float(confidence)))
    lo = np.full_like(mu, _NAN, dtype=float)
    hi = np.full_like(mu, _NAN, dtype=float)
    lo[valid] = mu[valid] - z * sig[valid]
    hi[valid] = mu[valid] + z * sig[valid]
    return lo, hi, valid


def interval_method_for_model(model_id: str, df: pd.DataFrame) -> str:
    mid = str(model_id).upper()
    has_nig = all(c in df.columns for c in ("alpha", "beta", "nu", "gamma"))
    if has_nig and ("EVID" in mid):
        return "student_t_nig_predictive"
    if "BAYES_FULL" in mid:
        return "empirical_posterior_predictive"
    if "SNGP" in mid or "GPR_SPARSE" in mid or mid.endswith("_GPR"):
        return "gaussian_predictive_moment"
    if "HETERO" in mid or "MCDROPOUT" in mid:
        return "normal_moment"
    if "POINT" in mid or "XGB" in mid:
        return "unavailable"
    return "normal_moment"


def coverage_at(y_true, lo, hi, valid) -> float:
    yt = np.asarray(y_true, dtype=float)
    ok = valid & np.isfinite(yt) & np.isfinite(lo) & np.isfinite(hi)
    if not ok.any():
        return _NAN
    return float(np.mean((yt[ok] >= lo[ok]) & (yt[ok] <= hi[ok])))


def interval_metrics_fold(df: pd.DataFrame, model_id: str) -> dict:
    method = interval_method_for_model(model_id, df)
    n_total = int(len(df))
    if method == "unavailable" or str(df.get("uncertainty_status", pd.Series(["unavailable"])).iloc[0]) == "unavailable":
        return {
            "interval_method": method,
            "n_total": n_total,
            "n_valid_intervals": 0,
            "n_invalid_intervals": n_total,
            "marginal_picp_90": _NAN,
            "marginal_ace": _NAN,
            "interval_status": "unavailable",
            "coverages": {},
        }
    yt = df["y_true"].to_numpy(dtype=float)
    yp = df["y_pred"].to_numpy(dtype=float)
    if method == "student_t_nig_predictive":
        lo90, hi90, valid = nig_student_t_interval(df["gamma"], df["nu"], df["alpha"], df["beta"], 0.9)
        coverages = {}
        for q in ACE_LEVELS:
            lo, hi, vq = nig_student_t_interval(df["gamma"], df["nu"], df["alpha"], df["beta"], float(q))
            coverages[float(q)] = coverage_at(yt, lo, hi, vq)
            valid = valid & vq
    elif method == "empirical_posterior_predictive" and "lower_90" in df.columns and "upper_90" in df.columns:
        lo90 = df["lower_90"].to_numpy(dtype=float)
        hi90 = df["upper_90"].to_numpy(dtype=float)
        valid = np.isfinite(lo90) & np.isfinite(hi90) & (hi90 >= lo90)
        coverages = {}
        for q in ACE_LEVELS:
            lo, hi, vq = normal_interval(yp, df["sigma_total"], float(q))
            coverages[float(q)] = coverage_at(yt, lo, hi, vq)
            valid = valid & vq
    else:
        if "sigma_total" not in df.columns:
            valid = np.zeros(n_total, dtype=bool)
            coverages = {float(q): _NAN for q in ACE_LEVELS}
            lo90 = hi90 = np.full(n_total, _NAN)
        else:
            lo90, hi90, valid = normal_interval(yp, df["sigma_total"], 0.9)
            coverages = {}
            for q in ACE_LEVELS:
                lo, hi, vq = normal_interval(yp, df["sigma_total"], float(q))
                coverages[float(q)] = coverage_at(yt, lo, hi, vq)
                valid = valid & vq
    n_valid = int(valid.sum())
    n_invalid = n_total - n_valid
    if len(ACE_LEVELS) != 9:
        raise RuntimeError("ACE must use the nine-level grid 0.1..0.9")
    picp = coverage_at(yt, lo90, hi90, valid) if n_valid > 0 else _NAN
    ace_vals = [abs(coverages[float(q)] - float(q)) for q in ACE_LEVELS]
    ace = float(np.mean(ace_vals)) if n_valid > 0 and all(np.isfinite(ace_vals)) else _NAN
    if n_invalid > 0 and n_valid == 0:
        status = "invalid_uncertainty_present"
    elif n_invalid > 0:
        status = "partial_invalid_rows"
    else:
        status = "ok"
    return {
        "interval_method": method,
        "n_total": n_total,
        "n_valid_intervals": n_valid,
        "n_invalid_intervals": n_invalid,
        "marginal_picp_90": picp,
        "marginal_ace": ace,
        "interval_status": status,
        "coverages": coverages,
        "nominal_grid": list(ACE_LEVELS),
    }


def _row_covered_flags(df: pd.DataFrame, model_id: str) -> tuple[np.ndarray, np.ndarray, dict[float, np.ndarray], str]:
    """Per-row 90% coverage, joint interval validity, and ACE-grid coverage flags."""
    method = interval_method_for_model(model_id, df)
    n = int(len(df))
    none = np.zeros(n, dtype=bool)
    empty = {float(q): none for q in ACE_LEVELS}
    if n == 0 or method == "unavailable":
        return none, none, empty, method
    if "uncertainty_status" in df.columns:
        status0 = str(df["uncertainty_status"].iloc[0])
        if status0 == "unavailable":
            return none, none, empty, method
    yt = df["y_true"].to_numpy(dtype=float)
    yp = df["y_pred"].to_numpy(dtype=float)
    coverages: dict[float, np.ndarray] = {}
    valid = np.ones(n, dtype=bool)
    if method == "student_t_nig_predictive":
        lo90, hi90, v90 = nig_student_t_interval(df["gamma"], df["nu"], df["alpha"], df["beta"], 0.9)
        for q in ACE_LEVELS:
            lo, hi, vq = nig_student_t_interval(df["gamma"], df["nu"], df["alpha"], df["beta"], float(q))
            coverages[float(q)] = (yt >= lo) & (yt <= hi) & vq
            valid = valid & vq
        covered90 = (yt >= lo90) & (yt <= hi90) & v90
        valid = valid & v90
    elif method == "empirical_posterior_predictive" and "lower_90" in df.columns and "upper_90" in df.columns:
        lo90 = pd.to_numeric(df["lower_90"], errors="coerce").to_numpy(dtype=float)
        hi90 = pd.to_numeric(df["upper_90"], errors="coerce").to_numpy(dtype=float)
        v90 = np.isfinite(lo90) & np.isfinite(hi90) & (hi90 >= lo90)
        covered90 = (yt >= lo90) & (yt <= hi90) & v90
        valid = v90.copy()
        if "sigma_total" not in df.columns:
            return none, none, empty, method
        for q in ACE_LEVELS:
            lo, hi, vq = normal_interval(yp, df["sigma_total"], float(q))
            coverages[float(q)] = (yt >= lo) & (yt <= hi) & vq
            valid = valid & vq
        covered90 = covered90 & valid
    else:
        if "sigma_total" not in df.columns:
            return none, none, empty, method
        lo90, hi90, v90 = normal_interval(yp, df["sigma_total"], 0.9)
        for q in ACE_LEVELS:
            lo, hi, vq = normal_interval(yp, df["sigma_total"], float(q))
            coverages[float(q)] = (yt >= lo) & (yt <= hi) & vq
            valid = valid & vq
        covered90 = (yt >= lo90) & (yt <= hi90) & v90
        valid = valid & v90
    return covered90, valid, coverages, method


def mop_interval_block(df: pd.DataFrame, model_id: str, min_n: int = MIN_PAIR_N) -> dict:
    """Mean of within-pair PICP@90 and ACE on the same retained pairs as MoP RMSE.

    A pair is kept only if every finite-prediction row has a valid interval, matching
    the previous pair-equal diagnostic. This is not row-weighted coverage.
    """
    assert_single_heldout_fold(df, "mop_interval_block")
    empty = {
        "pair_column": PAIR_COLUMN if PAIR_COLUMN in df.columns else "",
        "min_pair_size": int(min_n),
        "n_pairs_total": int(df[PAIR_COLUMN].nunique()) if PAIR_COLUMN in df.columns else 0,
        "n_pairs_retained": 0,
        "n_pairs_valid": {"picp_90": 0, "ace": 0},
        "n_pairs_invalid": {"picp_90": 0, "ace": 0},
        "n_rows_in_retained_pairs": 0,
        "mop_picp_90": _NAN,
        "mop_ace": _NAN,
        "interval_method": interval_method_for_model(model_id, df) if len(df) else "unavailable",
    }
    if PAIR_COLUMN not in df.columns or len(df) == 0:
        return empty
    col, n_total, retained, n_retained = _pair_groups(df, min_n)
    empty["pair_column"] = col
    empty["n_pairs_total"] = n_total
    empty["n_pairs_retained"] = n_retained
    empty["n_rows_in_retained_pairs"] = int(len(retained))
    if n_retained == 0:
        return empty
    covered90, valid, coverages, method = _row_covered_flags(retained, model_id)
    work = retained.copy()
    work["_ok"] = valid
    work["_c90"] = covered90
    for q in ACE_LEVELS:
        work[f"_c{q:.1f}"] = coverages[float(q)]
    gsize = work.groupby(col, sort=False).size()
    gok = work.groupby(col, sort=False)["_ok"].sum()
    keep_uq = gsize.index[(gok.to_numpy() == gsize.to_numpy())]
    good = work[work[col].isin(keep_uq)]
    n_valid = int(len(keep_uq))
    n_invalid = int(n_retained - n_valid)
    if n_valid == 0:
        return {
            **empty,
            "n_pairs_invalid": {"picp_90": n_invalid, "ace": n_invalid},
            "interval_method": method,
        }
    means = good.groupby(col, sort=False)[["_c90"] + [f"_c{q:.1f}" for q in ACE_LEVELS]].mean()
    qs = np.asarray(ACE_LEVELS, dtype=float)
    ace_grid = means[[f"_c{q:.1f}" for q in ACE_LEVELS]].to_numpy(dtype=float)
    pair_ace = np.abs(ace_grid - qs).mean(axis=1)
    return {
        "pair_column": col,
        "min_pair_size": int(min_n),
        "n_pairs_total": n_total,
        "n_pairs_retained": n_retained,
        "n_pairs_valid": {"picp_90": n_valid, "ace": n_valid},
        "n_pairs_invalid": {"picp_90": n_invalid, "ace": n_invalid},
        "n_rows_in_retained_pairs": int(len(retained)),
        "mop_picp_90": float(means["_c90"].mean()),
        "mop_ace": float(pair_ace.mean()),
        "interval_method": method,
    }


def risk_ordering_fold(df: pd.DataFrame, score_col: str = "sigma_total") -> list[dict]:
    """Uncertainty-based risk ordering inside one held-out fold. Not an OOD detector."""
    rows = []
    if score_col not in df.columns:
        return rows
    work, diag = filter_finite_rows(df)
    score = pd.to_numeric(work[score_col], errors="coerce")
    valid_u = np.isfinite(score.to_numpy(dtype=float)) & (score.to_numpy(dtype=float) > 0)
    n_valid_u = int(valid_u.sum())
    ranked = work.loc[valid_u].copy()
    ranked["_score"] = score.loc[valid_u].to_numpy(dtype=float)
    ranked = ranked.sort_values("_score", ascending=True)  # keep lowest uncertainty first
    n = len(ranked)
    for ret in RETENTION_LEVELS:
        k = max(1, int(np.floor(ret * n))) if n else 0
        part = ranked.iloc[:k] if k else ranked.iloc[0:0]
        rmse = _rmse(part["y_true"], part["y_pred"]) if len(part) else _NAN
        rows.append(
            {
                "risk_score_definition": "sigma_total_ascending_keep_least_uncertain",
                "retention": float(ret),
                "n_retained": int(len(part)),
                "n_valid_uncertainty": n_valid_u,
                "n_rows_used": diag["n_rows_used"],
                "rmse_retained": rmse,
            }
        )
    return rows


def apply_scope(df: pd.DataFrame, dataset: str, scope: str, tau: float = HIGH_SEL_TAU) -> pd.DataFrame:
    dataset = dataset.upper()
    if dataset == "DS1":
        if scope == "all_heldout":
            return df
        if scope == "metal_complex":
            if "family" not in df.columns:
                raise KeyError("DS1 metal_complex scope requires family")
            return df.loc[df["family"].astype(str) == "metal_complex"].copy()
        raise KeyError(f"Unknown DS1 scope {scope}")
    if scope == "all_pairs":
        return df
    if scope == "ree_pairs":
        if "is_ree_pair" not in df.columns:
            raise KeyError("ree_pairs scope requires is_ree_pair")
        flag = df["is_ree_pair"]
        if flag.dtype == object:
            flag = flag.astype(str).str.lower().isin({"true", "1", "yes"})
        return df.loc[flag.astype(bool)].copy()
    hs = high_selectivity_frames(df, tau=tau)
    if scope in hs:
        return hs[scope]
    raise KeyError(f"Unknown DS2 scope {scope}")


def ds1_scopes() -> tuple[str, ...]:
    return ("all_heldout", "metal_complex")


def ds2_scopes() -> tuple[str, ...]:
    return (
        "all_pairs",
        "ree_pairs",
        "high_selectivity_row",
        "high_selectivity_all_pair_membership",
    )


def emit_point_rows(
    meta: dict,
    diag: dict,
    mop: dict | None,
    signs: dict,
    row_m: dict,
    mop_int: dict | None = None,
) -> list[dict]:
    rows = []
    shared = dict(
        dataset=meta["dataset"],
        model_id=meta["model_id"],
        fold_id=meta["fold_id"],
        scope=meta["scope"],
        pair_column=meta.get("pair_column", ""),
        min_pair_size=MIN_PAIR_N if mop else "",
        high_selectivity_threshold=meta.get("tau", ""),
        high_selectivity_estimand=meta.get("hs_estimand", ""),
        n_rows_total=diag["n_rows_total"],
        n_rows_used=diag["n_rows_used"],
        n_rows_invalid=diag["n_rows_invalid"],
        evaluation_partition=EVALUATION_PARTITION,
    )
    def add(name, value, aggregation, extra=None):
        payload = dict(shared, metric_name=name, aggregation=aggregation, value=value)
        if extra:
            payload.update(extra)
        rows.append(metric_row(**payload))

    add("rmse", row_m["rmse"], "row_level")
    add("r2", row_m["r2"], "row_level")
    add("sign_accuracy_all", signs["sign_accuracy_all"], "row_level")
    add("n_exact_zero_observed", signs["n_exact_zero_observed"], "count")
    if mop is not None:
        extra = dict(
            n_pairs_total=mop["n_pairs_total"],
            n_pairs_retained=mop["n_pairs_retained"],
            n_rows_in_retained_pairs=mop["n_rows_in_retained_pairs"],
            pair_column=mop["pair_column"],
            min_pair_size=mop["min_pair_size"],
        )
        add("mop_rmse", mop["mop_rmse"], "mean_of_pairs", {**extra, "n_pairs_valid": mop["n_pairs_valid"]["rmse"], "n_pairs_invalid": mop["n_pairs_invalid"]["rmse"]})
        add("mop_r2", mop["mop_r2"], "mean_of_pairs", {**extra, "n_pairs_valid": mop["n_pairs_valid"]["r2"], "n_pairs_invalid": mop["n_pairs_invalid"]["r2"]})
        if mop_int is not None:
            extra_int = dict(
                extra,
                n_pairs_valid=mop_int["n_pairs_valid"]["picp_90"],
                n_pairs_invalid=mop_int["n_pairs_invalid"]["picp_90"],
                interval_method=mop_int.get("interval_method", ""),
            )
            add("mop_picp_90", mop_int["mop_picp_90"], "mean_of_pairs", extra_int)
            add("mop_ace", mop_int["mop_ace"], "mean_of_pairs", extra_int)
    return rows


def evaluate_one_fold(df: pd.DataFrame, dataset: str, model_id: str) -> dict:
    """Compute all fold-local metrics for one model on one held-out fold."""
    work, fold = _base(df, "all", f"{dataset} {model_id}")
    if "member_id" in work.columns:
        mids = set(pd.to_numeric(work["member_id"], errors="coerce").dropna().astype(int))
        if -1 in mids:
            work = work.loc[pd.to_numeric(work["member_id"], errors="coerce") == -1].copy()
        elif mids and mids != {0} and len(mids) > 1:
            raise PooledFoldError(f"{model_id} {fold}: multiple member_id values without ensemble aggregate")
    dataset = dataset.upper()
    scopes = ds1_scopes() if dataset == "DS1" else ds2_scopes()
    metric_rows: list[dict] = []
    interval_rows: list[dict] = []
    risk_rows: list[dict] = []
    coverage_rows: list[dict] = []
    hs_meta = None
    if dataset == "DS2":
        hs_meta = high_selectivity_frames(work)

    for scope in scopes:
        scoped = apply_scope(work, dataset, scope)
        used, diag = filter_finite_rows(scoped)
        row_m = row_point_metrics(used["y_true"], used["y_pred"]) if len(used) else {"rmse": _NAN, "r2": _NAN}
        signs = sign_metrics(used["y_true"], used["y_pred"]) if len(used) else sign_metrics([], [])
        mop = None
        mop_int = None
        hs_estimand = ""
        tau = ""
        if dataset == "DS2" and len(used):
            mop = mop_block(used)
            mop_int = mop_interval_block(used, model_id)
            if scope.startswith("high_selectivity"):
                hs_estimand = scope
                tau = HIGH_SEL_TAU
        meta = dict(dataset=dataset, model_id=model_id, fold_id=fold, scope=scope, tau=tau, hs_estimand=hs_estimand)
        before = len(metric_rows)
        metric_rows.extend(emit_point_rows(meta, diag, mop, signs, row_m, mop_int=mop_int))
        if hs_meta is not None and scope.startswith("high_selectivity"):
            for r in metric_rows[before:]:
                r["n_high_selectivity_rows"] = hs_meta["n_high_selectivity_rows"]
                r["n_pairs_high_selectivity_total"] = hs_meta["n_pairs_high_selectivity_total"]

        im = interval_metrics_fold(used if len(used) else scoped, model_id)
        interval_rows.append(
            metric_row(
                dataset=dataset,
                model_id=model_id,
                fold_id=fold,
                scope=scope,
                metric_name="marginal_picp_90",
                aggregation="fold_local_interval",
                value=im["marginal_picp_90"],
                interval_method=im["interval_method"],
                nominal_level=0.9,
                n_rows_total=diag["n_rows_total"],
                n_rows_used=diag["n_rows_used"],
                n_rows_invalid=diag["n_rows_invalid"],
                n_valid_intervals=im["n_valid_intervals"],
                n_invalid_intervals=im["n_invalid_intervals"],
            )
        )
        interval_rows.append(
            metric_row(
                dataset=dataset,
                model_id=model_id,
                fold_id=fold,
                scope=scope,
                metric_name="marginal_ace",
                aggregation="fold_local_interval",
                value=im["marginal_ace"],
                interval_method=im["interval_method"],
                nominal_level="0.1:0.9:9",
                n_rows_total=diag["n_rows_total"],
                n_rows_used=diag["n_rows_used"],
                n_rows_invalid=diag["n_rows_invalid"],
                n_valid_intervals=im["n_valid_intervals"],
                n_invalid_intervals=im["n_invalid_intervals"],
            )
        )
        for q, c in im.get("coverages", {}).items():
            coverage_rows.append(
                {
                    "dataset": dataset,
                    "model_id": model_id,
                    "fold_id": fold,
                    "scope": scope,
                    "nominal_level": q,
                    "empirical_coverage": c,
                    "interval_method": im["interval_method"],
                    "interval_status": im["interval_status"],
                    "n_valid_intervals": im["n_valid_intervals"],
                    "n_invalid_intervals": im["n_invalid_intervals"],
                    "evaluation_partition": EVALUATION_PARTITION,
                }
            )
        for rr in risk_ordering_fold(used if len(used) else scoped):
            risk_rows.append(
                {
                    "dataset": dataset,
                    "model_id": model_id,
                    "fold_id": fold,
                    "scope": scope,
                    "evaluation_partition": EVALUATION_PARTITION,
                    **rr,
                }
            )

    return {
        "fold_metrics": metric_rows,
        "interval_metrics": interval_rows,
        "coverage_diagnostics": coverage_rows,
        "risk_metrics": risk_rows,
    }
