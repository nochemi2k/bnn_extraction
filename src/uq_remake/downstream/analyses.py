"""Fold-local downstream analyses: risk-coverage, pair-density, sign-stability."""
from __future__ import annotations

import numpy as np
import pandas as pd

from uq_remake.downstream.protocol import (
    MIN_PAIR_N,
    PAIR_COLUMN,
    RANDOM_N_REPEATS,
    RANDOM_SEED,
    RETENTION_FRACTIONS,
    Z90,
)
from uq_remake.metrics.cv_protocol import EVALUATION_PARTITION, fold_mean_sd
from uq_remake.metrics.fold_metrics import (
    apply_scope,
    ds1_scopes,
    ds2_scopes,
    filter_finite_rows,
    mop_block,
    sign_metrics,
)


def _nan() -> float:
    return float("nan")


def curve_aurc(retention, values) -> float:
    """Trapezoidal AURC on a fold-local retention curve. Requires ≥2 finite points."""
    finite = [
        (float(r), float(v))
        for r, v in zip(retention, values)
        if np.isfinite(r) and np.isfinite(v)
    ]
    if len(finite) < 2:
        return _nan()
    rr, mm = zip(*sorted(finite))
    trapz_fn = getattr(np, "trapezoid", None)
    if trapz_fn is None:
        trapz_fn = getattr(np, "trapz", None)
    if trapz_fn is None:
        raise AttributeError("Neither np.trapezoid nor np.trapz is available in this NumPy version.")
    return float(trapz_fn(mm, rr))


def deterministic_rank(df: pd.DataFrame, score_col: str, higher_is_risk: bool = True) -> pd.DataFrame:
    work = df.copy()
    work["_score"] = pd.to_numeric(work[score_col], errors="coerce")
    work["_tid"] = work["record_id"].astype(str) if "record_id" in work.columns else work.index.astype(str)
    work = work.loc[np.isfinite(work["_score"].to_numpy(dtype=float))].copy()
    work = work.sort_values(["_score", "_tid"], ascending=[True, True], kind="mergesort")
    if not higher_is_risk:
        work = work.sort_values(["_score", "_tid"], ascending=[False, True], kind="mergesort")
    return work.reset_index(drop=True)


def coverage_at_normal(y_true, y_pred, sigma, z: float = Z90) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    sig = np.asarray(sigma, dtype=float)
    ok = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(sig) & (sig > 0)
    if not ok.any():
        return _nan()
    lo = yp[ok] - z * sig[ok]
    hi = yp[ok] + z * sig[ok]
    return float(np.mean((yt[ok] >= lo) & (yt[ok] <= hi)))


def pair_coverage_stats(df: pd.DataFrame, min_n: int = MIN_PAIR_N) -> dict:
    if PAIR_COLUMN not in df.columns or len(df) == 0:
        return {
            "n_pairs_total": 0,
            "n_pairs_used": 0,
            "n_pairs_with_at_least_min": 0,
            "fraction_pairs_with_at_least_3_rows": _nan(),
        }
    sizes = df.groupby(PAIR_COLUMN, sort=False).size()
    n_total = int(len(sizes))
    n_ok = int((sizes >= int(min_n)).sum())
    return {
        "n_pairs_total": n_total,
        "n_pairs_used": n_ok,
        "n_pairs_with_at_least_min": n_ok,
        "fraction_pairs_with_at_least_3_rows": float(n_ok / n_total) if n_total else _nan(),
    }


def retained_metrics(df: pd.DataFrame, dataset: str = "DS2") -> dict:
    used, diag = filter_finite_rows(df)
    if len(used) == 0:
        signs = sign_metrics([], [])
        return {
            "rmse": _nan(),
            "sign_accuracy": _nan(),
            "mop_rmse": _nan(),
            "mop_r2": _nan(),
            "n_rows_used": 0,
            "n_rows_total": diag["n_rows_total"],
            "n_pairs_total": 0,
            "n_pairs_used": 0,
            "n_pairs_valid_rmse": 0,
            "n_pairs_with_at_least_min": 0,
            "metric_status": "no_valid_rows",
        }
    yt = used["y_true"].to_numpy(dtype=float)
    yp = used["y_pred"].to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    signs = sign_metrics(used["y_true"], used["y_pred"])
    if str(dataset).upper() == "DS1":
        return {
            "rmse": rmse,
            "sign_accuracy": signs["sign_accuracy_all"],
            "mop_rmse": _nan(),
            "mop_r2": _nan(),
            "n_rows_used": diag["n_rows_used"],
            "n_rows_total": diag["n_rows_total"],
            "n_pairs_total": 0,
            "n_pairs_used": 0,
            "n_pairs_valid_rmse": 0,
            "n_pairs_with_at_least_min": 0,
            "metric_status": "ok",
        }
    mop = mop_block(used)
    pc = pair_coverage_stats(used)
    mop_ok = mop["n_pairs_valid"]["rmse"] > 0
    return {
        "rmse": rmse,
        "sign_accuracy": signs["sign_accuracy_all"],
        "mop_rmse": mop["mop_rmse"] if mop_ok else _nan(),
        "mop_r2": mop["mop_r2"] if mop["n_pairs_valid"]["r2"] > 0 else _nan(),
        "n_rows_used": diag["n_rows_used"],
        "n_rows_total": diag["n_rows_total"],
        "n_pairs_total": mop["n_pairs_total"],
        "n_pairs_used": mop["n_pairs_retained"],
        "n_pairs_valid_rmse": mop["n_pairs_valid"]["rmse"],
        "metric_status": "ok" if mop_ok else "insufficient_pair_coverage",
        **pc,
    }


def scopes_for_dataset(dataset: str) -> tuple[str, ...]:
    if str(dataset).upper() == "DS1":
        return ds1_scopes()
    return ("ree_pairs", "all_pairs")


def k_at_retention(n: int, ret: float) -> int:
    """Retained count at fraction r: floor(r n), except r=1 uses all n rows.

    Ties at the cutoff are resolved by deterministic_rank (score, then record_id).
    Ranking is performed *within the analysis scope* after apply_scope, not by
    ranking all outer-test rows and then subsetting REE rows.
    """
    if n <= 0:
        return 0
    if float(ret) >= 1.0:
        return int(n)
    return int(np.floor(float(ret) * n))


def risk_coverage_fold(
    df: pd.DataFrame,
    score_col: str,
    fold: str,
    experiment_id: str,
    dataset: str = "DS2",
    model_id: str = "",
    family: str = "",
    score_role: str = "",
    retentions: tuple[float, ...] = RETENTION_FRACTIONS,
) -> list[dict]:
    rows = []
    dataset = str(dataset).upper()
    for scope in scopes_for_dataset(dataset):
        scoped = apply_scope(df, dataset, scope)
        ranked = deterministic_rank(scoped, score_col, higher_is_risk=True)
        n = len(ranked)
        metric_series: dict[str, list[tuple[float, float]]] = {
            "rmse": [],
            "mop_rmse": [],
        }
        empty_met = {
            "rmse": _nan(),
            "sign_accuracy": _nan(),
            "mop_rmse": _nan(),
            "mop_r2": _nan(),
            "n_rows_used": 0,
            "n_rows_total": n,
            "n_pairs_total": 0,
            "n_pairs_used": 0,
            "n_pairs_valid_rmse": 0,
            "n_pairs_with_at_least_min": 0,
            "metric_status": "empty_retention",
        }
        for ret in retentions:
            k = k_at_retention(n, ret)
            part = ranked.iloc[:k] if k else ranked.iloc[0:0]
            met = retained_metrics(part, dataset=dataset) if len(part) else empty_met
            metric_series["rmse"].append((float(ret), met["rmse"]))
            metric_series["mop_rmse"].append((float(ret), met.get("mop_rmse", _nan())))
            metric_items = [
                ("rmse", met["rmse"]),
                ("sign_accuracy", met["sign_accuracy"]),
                ("mop_rmse", met.get("mop_rmse", _nan())),
                ("n_rows_retained", float(len(part))),
                ("n_pairs_total", float(met.get("n_pairs_total", 0) or 0)),
                ("n_pairs_used", float(met.get("n_pairs_used", 0) or 0)),
                ("n_pairs_with_at_least_min", float(met.get("n_pairs_with_at_least_min", met.get("n_pairs_used", 0)) or 0)),
            ]
            for metric_name, value in metric_items:
                status = met["metric_status"]
                if metric_name == "mop_rmse" and not np.isfinite(met.get("mop_rmse", _nan())):
                    status = "insufficient_pair_coverage"
                if dataset == "DS1" and metric_name.startswith("mop"):
                    status = "not_applicable_ds1"
                    value = _nan()
                rows.append(
                    {
                        "dataset": dataset,
                        "model_id": model_id,
                        "family": family,
                        "score_role": score_role,
                        "experiment_id": experiment_id,
                        "outer_fold_id": fold,
                        "scope": scope,
                        "score_name": score_col,
                        "retention_fraction": float(ret),
                        "metric_name": metric_name,
                        "value": value,
                        "n_rows_total": int(n),
                        "n_rows_used": int(len(part)),
                        "n_pairs_total": met.get("n_pairs_total", 0),
                        "n_pairs_used": met.get("n_pairs_used", 0),
                        "n_pairs_with_at_least_min": met.get("n_pairs_with_at_least_min", met.get("n_pairs_used", 0)),
                        "metric_status": status,
                        "evaluation_partition": EVALUATION_PARTITION,
                        "tie_break": "score_asc_then_record_id",
                        "score_orientation": "higher_is_greater_risk",
                        "baseline": "score",
                    }
                )
        for metric_name, series in metric_series.items():
            if dataset == "DS1" and metric_name.startswith("mop"):
                continue
            aurc = curve_aurc([r for r, _ in series], [v for _, v in series])
            rows.append(
                {
                    "dataset": dataset,
                    "model_id": model_id,
                    "family": family,
                    "score_role": score_role,
                    "experiment_id": experiment_id,
                    "outer_fold_id": fold,
                    "scope": scope,
                    "score_name": score_col,
                    "retention_fraction": "",
                    "metric_name": f"retention_curve_auc_{metric_name}",
                    "value": aurc,
                    "n_rows_total": int(n),
                    "n_rows_used": int(n),
                    "n_pairs_total": "",
                    "n_pairs_used": "",
                    "n_pairs_with_at_least_min": "",
                    "metric_status": "ok" if np.isfinite(aurc) else "undefined_metric",
                    "evaluation_partition": EVALUATION_PARTITION,
                    "tie_break": "score_asc_then_record_id",
                    "score_orientation": "higher_is_greater_risk",
                    "baseline": "score",
                }
            )
    return rows


def summarize_fold_metrics(fold_df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    if fold_df is None or len(fold_df) == 0:
        return pd.DataFrame()
    for key, g in fold_df.groupby(keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        stats = fold_mean_sd(pd.to_numeric(g["value"], errors="coerce").to_numpy())
        rec = {k: v for k, v in zip(keys, key)}
        n_att = int(g["outer_fold_id"].nunique()) if "outer_fold_id" in g.columns else int(stats["n_folds"])
        rec.update(
            {
                "fold_mean": stats["fold_mean"],
                "fold_sd": stats["fold_sd"],
                "n_folds_attempted": n_att,
                "n_valid_folds": stats["n_valid_folds"],
                "n_fold_bin_observations": stats["n_valid_folds"],
                "metric_aggregation": "fold_mean_plus_sd",
                "sd_definition": "ddof_1",
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def random_retention_fold(
    df: pd.DataFrame,
    fold: str,
    dataset: str = "DS2",
    model_id: str = "",
    family: str = "",
    n_repeats: int = RANDOM_N_REPEATS,
    seed: int = RANDOM_SEED,
    retentions: tuple[float, ...] = RETENTION_FRACTIONS,
) -> list[dict]:
    """Fold-local random retention. Repeated subsets; never pooled across folds."""
    rows = []
    dataset = str(dataset).upper()
    fold_seed = int(seed) + (int(str(fold).replace("fold", "") or 0) * 1009)
    rng = np.random.default_rng(fold_seed)
    for scope in scopes_for_dataset(dataset):
        scoped = apply_scope(df, dataset, scope).reset_index(drop=True)
        n = len(scoped)
        if n == 0:
            continue
        collected: dict[tuple[float, str], list[float]] = {}
        support: dict[tuple[float, str], list[float]] = {}
        for _rep in range(int(n_repeats)):
            perm = rng.permutation(n)
            ranked = scoped.iloc[perm]
            for ret in retentions:
                k = k_at_retention(n, ret)
                part = ranked.iloc[:k] if k else ranked.iloc[0:0]
                met = retained_metrics(part, dataset=dataset) if len(part) else {
                    "rmse": _nan(),
                    "mop_rmse": _nan(),
                    "n_pairs_used": 0,
                    "n_pairs_total": 0,
                    "n_pairs_with_at_least_min": 0,
                    "metric_status": "empty_retention",
                }
                for metric_name in ("rmse", "mop_rmse"):
                    if dataset == "DS1" and metric_name.startswith("mop"):
                        continue
                    collected.setdefault((float(ret), metric_name), []).append(met.get(metric_name, _nan()))
                support.setdefault((float(ret), "n_pairs_used"), []).append(float(met.get("n_pairs_used", 0) or 0))
                support.setdefault((float(ret), "n_rows_used"), []).append(float(len(part)))
        for (ret, metric_name), vals in collected.items():
            arr = np.asarray(vals, dtype=float)
            finite = arr[np.isfinite(arr)]
            rows.append(
                {
                    "dataset": dataset,
                    "model_id": model_id,
                    "family": family,
                    "score_role": "random_baseline",
                    "experiment_id": "ad_random_retention",
                    "outer_fold_id": fold,
                    "scope": scope,
                    "score_name": "random_retention",
                    "retention_fraction": float(ret),
                    "metric_name": metric_name,
                    "value": float(np.mean(finite)) if len(finite) else _nan(),
                    "repeat_sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else _nan(),
                    "n_repeats": int(n_repeats),
                    "n_valid_repeats": int(len(finite)),
                    "n_rows_total": int(n),
                    "n_rows_used": int(np.nanmean(support.get((ret, "n_rows_used"), [0]))),
                    "n_pairs_used": int(np.nanmean(support.get((ret, "n_pairs_used"), [0]))),
                    "metric_status": "ok" if len(finite) else "undefined_metric",
                    "evaluation_partition": EVALUATION_PARTITION,
                    "baseline": "random",
                    "random_seed": int(fold_seed),
                }
            )
        for metric_name in ("rmse", "mop_rmse"):
            if dataset == "DS1" and metric_name.startswith("mop"):
                continue
            series = []
            for ret in retentions:
                hit = [r for r in rows if r["scope"] == scope and r["metric_name"] == metric_name and r["retention_fraction"] == float(ret)]
                if hit:
                    series.append((float(ret), hit[-1]["value"]))
            aurc = curve_aurc([r for r, _ in series], [v for _, v in series])
            rows.append(
                {
                    "dataset": dataset,
                    "model_id": model_id,
                    "family": family,
                    "score_role": "random_baseline",
                    "experiment_id": "ad_random_retention",
                    "outer_fold_id": fold,
                    "scope": scope,
                    "score_name": "random_retention",
                    "retention_fraction": "",
                    "metric_name": f"retention_curve_auc_{metric_name}",
                    "value": aurc,
                    "n_repeats": int(n_repeats),
                    "n_rows_total": int(n),
                    "n_rows_used": int(n),
                    "metric_status": "ok" if np.isfinite(aurc) else "undefined_metric",
                    "evaluation_partition": EVALUATION_PARTITION,
                    "baseline": "random",
                    "random_seed": int(fold_seed),
                }
            )
    return rows


def utility_overlap_fold(
    df: pd.DataFrame,
    score_col: str,
    fold: str,
    dataset: str,
    scope: str,
    model_id: str = "",
    family: str = "",
    score_role: str = "",
    utility_fracs: tuple[float, ...] = (0.10, 0.20),
    ad_fracs: tuple[float, ...] = (0.10, 0.50),
) -> list[dict]:
    """Predicted-utility shortlist vs AD reliability. Separate from risk-coverage."""
    dataset = str(dataset).upper()
    scoped = apply_scope(df, dataset, scope).copy()
    if len(scoped) == 0:
        return []
    if dataset == "DS1":
        scoped["_utility"] = pd.to_numeric(scoped["y_pred"], errors="coerce")
        utility_name = "predicted_logK"
    else:
        scoped["_utility"] = np.abs(pd.to_numeric(scoped["y_pred"], errors="coerce").to_numpy(dtype=float))
        utility_name = "predicted_abs_delta_logK"
    scoped["_tid"] = scoped["record_id"].astype(str) if "record_id" in scoped.columns else scoped.index.astype(str)
    util_ranked = scoped.loc[np.isfinite(scoped["_utility"].to_numpy(dtype=float))].sort_values(
        ["_utility", "_tid"], ascending=[False, True], kind="mergesort"
    )
    ad_ranked = deterministic_rank(scoped, score_col, higher_is_risk=True)
    n = len(scoped)
    rows = []
    for ufrac in utility_fracs:
        uk = k_at_retention(len(util_ranked), ufrac)
        util_ids = set(util_ranked.iloc[:uk]["_tid"].astype(str))
        util_part = util_ranked.iloc[:uk]
        true_mean = float(np.mean(np.abs(util_part["y_true"].to_numpy(dtype=float)))) if len(util_part) else _nan()
        for afrac in ad_fracs:
            ak = k_at_retention(len(ad_ranked), afrac)
            ad_ids = set(ad_ranked.iloc[:ak]["_tid"].astype(str))
            both = util_ids & ad_ids
            inter = util_part.loc[util_part["_tid"].astype(str).isin(both)]
            true_both = float(np.mean(np.abs(inter["y_true"].to_numpy(dtype=float)))) if len(inter) else _nan()
            rows.append(
                {
                    "dataset": dataset,
                    "model_id": model_id,
                    "family": family,
                    "score_role": score_role,
                    "experiment_id": "ad_utility_overlap",
                    "outer_fold_id": fold,
                    "scope": scope,
                    "score_name": score_col,
                    "utility_name": utility_name,
                    "utility_fraction": float(ufrac),
                    "ad_retention_fraction": float(afrac),
                    "n_utility": int(uk),
                    "n_ad": int(ak),
                    "n_overlap": int(len(both)),
                    "overlap_fraction_of_utility": float(len(both) / uk) if uk else _nan(),
                    "mean_abs_y_true_utility": true_mean,
                    "mean_abs_y_true_overlap": true_both,
                    "n_rows_total": int(n),
                    "evaluation_partition": EVALUATION_PARTITION,
                    "analysis": "predicted_utility_then_reliability",
                    "not_merged_with_risk_coverage": True,
                }
            )
    return rows


def sign_stability_fold(df: pd.DataFrame, fold: str, dataset: str = "DS2") -> list[dict]:
    rows = []
    sigma = pd.to_numeric(df["sigma_total"], errors="coerce").to_numpy(dtype=float)
    yp = df["y_pred"].to_numpy(dtype=float)
    lo = yp - Z90 * sigma
    hi = yp + Z90 * sigma
    cat = np.full(len(df), "unavailable", dtype=object)
    valid = np.isfinite(lo) & np.isfinite(hi) & (hi >= lo)
    cat[valid & (lo > 0)] = "entirely_positive"
    cat[valid & (hi < 0)] = "entirely_negative"
    cat[valid & (lo <= 0) & (hi >= 0)] = "overlaps_zero"
    work = df.copy()
    work["_cat"] = cat
    for scope in ("all_pairs", "ree_pairs"):
        scoped = apply_scope(work, dataset, scope)
        n_scope = len(scoped)
        for name in ("entirely_positive", "entirely_negative", "overlaps_zero"):
            part = scoped.loc[scoped["_cat"] == name]
            if len(part) == 0:
                rows.append(
                    {
                        "dataset": dataset,
                        "experiment_id": "sign_stability",
                        "outer_fold_id": fold,
                        "scope": scope,
                        "score_name": "predictive_interval_90",
                        "retention_fraction": "",
                        "metric_name": "fraction_rows",
                        "value": _nan(),
                        "n_rows_total": n_scope,
                        "n_rows_used": 0,
                        "n_pairs_total": 0,
                        "n_pairs_used": 0,
                        "metric_status": "unavailable_in_fold",
                        "category": name,
                        "evaluation_partition": EVALUATION_PARTITION,
                    }
                )
                continue
            met = retained_metrics(part)
            frac = float(len(part) / n_scope) if n_scope else _nan()
            for metric_name, value in (
                ("fraction_rows", frac),
                ("sign_accuracy", met["sign_accuracy"]),
                ("rmse", met["rmse"]),
                ("mop_rmse", met["mop_rmse"] if scope == "ree_pairs" else _nan()),
            ):
                if scope != "ree_pairs" and metric_name.startswith("mop"):
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "experiment_id": "sign_stability",
                        "outer_fold_id": fold,
                        "scope": scope,
                        "score_name": "predictive_interval_90",
                        "retention_fraction": "",
                        "metric_name": metric_name,
                        "value": value,
                        "n_rows_total": n_scope,
                        "n_rows_used": int(len(part)),
                        "n_pairs_total": met.get("n_pairs_total", 0),
                        "n_pairs_used": met.get("n_pairs_used", 0),
                        "metric_status": "ok" if np.isfinite(value) or metric_name == "fraction_rows" else met["metric_status"],
                        "category": name,
                        "evaluation_partition": EVALUATION_PARTITION,
                    }
                )
    return rows
