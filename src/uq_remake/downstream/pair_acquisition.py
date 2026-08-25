"""Fold-local sigma-only pair acquisition for high- and low-selectivity events."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from uq_remake.downstream.protocol import (
    PAIR_ACQ_EVENT_MIN_EXCLUSIVE,
    PAIR_ACQ_RANDOM_STRATEGY,
    PAIR_ACQ_STRATEGIES,
    PAIR_ACQ_THRESHOLD,
    PAIR_ACQ_TOP_K,
    Z90,
)
from uq_remake.metrics.uq_metrics import adaptive_calibration_error, picp_at_90

TASK_HIGH = "high_selectivity"
TASK_LOW = "low_selectivity"
ORIENT_CANONICAL = "canonical"
ORIENT_REVERSED = "reversed"
ORIENT_NEAR_ZERO = "near_zero"
SELECTION_FORBIDDEN = (
    "y_true",
    "selectivity_logK",
    "observed_event",
    "target_observed",
    "absolute_error",
)
ABS_YHAT_BINS = (
    ("0-0.5", 0.0, 0.5),
    ("0.5-1", 0.5, 1.0),
    ("1-2", 1.0, 2.0),
    ("2-4", 2.0, 4.0),
    ("4-8", 4.0, 8.0),
    (">8", 8.0, np.inf),
)
CALIBRATION_MIN_N = 10
REGRET_TOL = 1e-12
RHO_PAIRS = (
    ("rho_sigma_abs_error", "sigma_total", "abs_error"),
    ("rho_sigma_abs_y", "sigma_total", "abs_y"),
    ("rho_sigma_abs_yhat", "sigma_total", "abs_yhat"),
    ("rho_abs_yhat_abs_error", "abs_yhat", "abs_error"),
)
STRATEGY_LABELS = {
    "y_pred": r"$\widehat y$",
    "lcb_1sigma": r"LCB-1",
    "lcb_2sigma": r"LCB-2",
    "lcb_3sigma": r"LCB-3",
    "expected_improvement": "threshold-margin",
    "random": "random",
}


def _nan() -> float:
    return float("nan")


def assert_no_outcome_leakage(kwargs: dict) -> None:
    leaked = [k for k in kwargs if k in SELECTION_FORBIDDEN]
    if leaked:
        raise RuntimeError(f"selection received outcome columns: {leaked}")


def target_observed(y: np.ndarray, orientation: str) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if orientation == ORIENT_CANONICAL:
        return y
    if orientation == ORIENT_REVERSED:
        return -y
    if orientation == ORIENT_NEAR_ZERO:
        return -np.abs(y)
    raise ValueError(orientation)


def target_predicted(yhat: np.ndarray, orientation: str) -> np.ndarray:
    return target_observed(yhat, orientation)


def resolved_event_thresholds(
    high_threshold: float | None = None,
    low_threshold: float | None = None,
) -> tuple[float, float]:
    high = float(PAIR_ACQ_THRESHOLD if high_threshold is None else high_threshold)
    low = float(PAIR_ACQ_THRESHOLD if low_threshold is None else low_threshold)
    if high <= 0.0 or low <= 0.0:
        raise ValueError(f"event thresholds must be positive, got high={high} low={low}")
    return high, low


def task_event_threshold(
    task: str,
    high_threshold: float | None = None,
    low_threshold: float | None = None,
) -> float:
    high, low = resolved_event_thresholds(high_threshold, low_threshold)
    if task == TASK_HIGH:
        return high
    if task == TASK_LOW:
        return low
    raise ValueError(task)


def observed_event(
    y: np.ndarray,
    task: str,
    orientation: str,
    threshold: float | None = None,
    *,
    high_threshold: float | None = None,
    low_threshold: float | None = None,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    tau = float(threshold) if threshold is not None else task_event_threshold(task, high_threshold, low_threshold)
    if task == TASK_HIGH:
        return target_observed(y, orientation) > tau
    if task == TASK_LOW:
        return np.abs(y) < tau
    raise ValueError(task)


def ranking_score(
    yhat: np.ndarray,
    sigma: np.ndarray,
    strategy: str,
    task: str,
    orientation: str,
    *,
    high_threshold: float | None = None,
    low_threshold: float | None = None,
) -> np.ndarray:
    """Score using predicted value and uncertainty only. `expected_improvement` is a standardized threshold-margin, not Gaussian EI."""
    assert_no_outcome_leakage({"strategy": strategy})
    mu = np.asarray(yhat, dtype=float)
    sig = np.asarray(sigma, dtype=float)
    high_tau, low_tau = resolved_event_thresholds(high_threshold, low_threshold)
    if task == TASK_HIGH:
        directed = target_predicted(mu, orientation)
        if strategy == "y_pred":
            return directed
        if strategy == "lcb_1sigma":
            return directed - sig
        if strategy == "lcb_2sigma":
            return directed - 2.0 * sig
        if strategy == "lcb_3sigma":
            return directed - 3.0 * sig
        if strategy == "expected_improvement":
            return _standardized_margin(directed - high_tau, sig, directed > high_tau)
        raise ValueError(strategy)
    if task == TASK_LOW:
        mag = np.abs(mu)
        if strategy == "y_pred":
            return -mag
        if strategy == "lcb_1sigma":
            return -(mag + sig)
        if strategy == "lcb_2sigma":
            return -(mag + 2.0 * sig)
        if strategy == "lcb_3sigma":
            return -(mag + 3.0 * sig)
        if strategy == "expected_improvement":
            return _standardized_margin(low_tau - mag, sig, mag < low_tau)
        raise ValueError(strategy)
    raise ValueError(task)


def _standardized_margin(margin: np.ndarray, sig: np.ndarray, predicted_event: np.ndarray) -> np.ndarray:
    out = np.full(margin.shape, np.nan, dtype=float)
    zero = np.isfinite(sig) & (sig == 0.0)
    pos = np.isfinite(sig) & (sig > 0.0)
    finite_m = np.isfinite(margin)
    out[pos & finite_m] = margin[pos & finite_m] / sig[pos & finite_m]
    out[zero & finite_m] = np.where(predicted_event[zero & finite_m], np.inf, -np.inf)
    return out


def rank_rows(
    df: pd.DataFrame,
    strategy: str,
    task: str,
    orientation: str,
    *,
    high_threshold: float | None = None,
    low_threshold: float | None = None,
) -> pd.DataFrame:
    work = df.copy()
    yhat = pd.to_numeric(work["y_pred"], errors="coerce").to_numpy(dtype=float)
    sigma = pd.to_numeric(work["sigma_total"], errors="coerce").to_numpy(dtype=float)
    work["_score"] = ranking_score(
        yhat,
        sigma,
        strategy,
        task,
        orientation,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
    )
    keep = np.isfinite(work["_score"]) | np.isposinf(work["_score"]) | np.isneginf(work["_score"])
    work = work.loc[keep].copy()
    work["_tie"] = work["record_id"].astype(str)
    work = work.sort_values(["_score", "_tie"], ascending=[False, True], kind="mergesort")
    work["_rank"] = np.arange(1, len(work) + 1)
    return work


def random_rank_rows(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    work = df.copy()
    work["_score"] = rng.random(len(work))
    work["_tie"] = work["record_id"].astype(str)
    work = work.sort_values(["_score", "_tie"], ascending=[False, True], kind="mergesort")
    work["_rank"] = np.arange(1, len(work) + 1)
    return work


def event_eligible(n_event: int) -> bool:
    return int(n_event) > int(PAIR_ACQ_EVENT_MIN_EXCLUSIVE)


def pair_event_counts(
    df: pd.DataFrame,
    *,
    high_threshold: float | None = None,
    low_threshold: float | None = None,
) -> pd.DataFrame:
    high_tau, low_tau = resolved_event_thresholds(high_threshold, low_threshold)
    rows = []
    for pair, g in df.groupby("metal_species_pair", sort=False):
        y = pd.to_numeric(g["y_true"], errors="coerce").to_numpy(dtype=float)
        n = int(np.isfinite(y).sum())
        n_pos = int(np.sum(y > high_tau))
        n_neg = int(np.sum(y < -high_tau))
        n_low = int(np.sum(np.abs(y) < low_tau))
        rows.append(
            {
                "metal_species_pair": pair,
                "n_samples": n,
                "n_high_canonical": n_pos,
                "n_high_reversed": n_neg,
                "n_low": n_low,
                "high_canonical_eligible": event_eligible(n_pos),
                "high_reversed_eligible": event_eligible(n_neg),
                "low_eligible": event_eligible(n_low),
                "n_unique_ligands": int(g["ligand_smiles"].nunique()),
                "is_ree_pair": bool(g["is_ree_pair"].iloc[0]) if "is_ree_pair" in g.columns else False,
            }
        )
    return pd.DataFrame(rows)


def eligible_tasks(
    df: pd.DataFrame,
    *,
    high_threshold: float | None = None,
    low_threshold: float | None = None,
) -> list[dict]:
    counts = pair_event_counts(df, high_threshold=high_threshold, low_threshold=low_threshold)
    tasks = []
    for rec in counts.itertuples(index=False):
        if rec.high_canonical_eligible:
            tasks.append(
                {
                    "task": TASK_HIGH,
                    "metal_species_pair": rec.metal_species_pair,
                    "orientation": ORIENT_CANONICAL,
                    "n_samples": rec.n_samples,
                    "n_event": rec.n_high_canonical,
                    "n_unique_ligands": rec.n_unique_ligands,
                    "is_ree_pair": rec.is_ree_pair,
                }
            )
        if rec.high_reversed_eligible:
            tasks.append(
                {
                    "task": TASK_HIGH,
                    "metal_species_pair": rec.metal_species_pair,
                    "orientation": ORIENT_REVERSED,
                    "n_samples": rec.n_samples,
                    "n_event": rec.n_high_reversed,
                    "n_unique_ligands": rec.n_unique_ligands,
                    "is_ree_pair": rec.is_ree_pair,
                }
            )
        if rec.low_eligible:
            tasks.append(
                {
                    "task": TASK_LOW,
                    "metal_species_pair": rec.metal_species_pair,
                    "orientation": ORIENT_NEAR_ZERO,
                    "n_samples": rec.n_samples,
                    "n_event": rec.n_low,
                    "n_unique_ligands": rec.n_unique_ligands,
                    "is_ree_pair": rec.is_ree_pair,
                }
            )
    return tasks


def _replace_inf_for_rank(values: np.ndarray) -> np.ndarray:
    s = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(s)
    if finite.any():
        lo = float(np.min(s[finite]))
        hi = float(np.max(s[finite]))
        s[np.isneginf(s)] = lo - 1.0
        s[np.isposinf(s)] = hi + 1.0
    else:
        s[np.isneginf(s)] = -1.0
        s[np.isposinf(s)] = 1.0
    return s


def rank_corr(score: np.ndarray, target: np.ndarray) -> float:
    x = _replace_inf_for_rank(score)
    y = np.asarray(target, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return _nan()
    xs = x[mask]
    ys = y[mask]
    if np.unique(xs).size < 2 or np.unique(ys).size < 2:
        return _nan()
    rho = pd.Series(xs).corr(pd.Series(ys), method="spearman")
    return float(rho) if pd.notna(rho) else _nan()


def evaluate_topk(
    ranked: pd.DataFrame,
    task: str,
    orientation: str,
    n_event_pool: int,
    *,
    high_threshold: float | None = None,
    low_threshold: float | None = None,
) -> dict:
    y = pd.to_numeric(ranked["y_true"], errors="coerce").to_numpy(dtype=float)
    target = target_observed(y, orientation)
    event = observed_event(y, task, orientation, high_threshold=high_threshold, low_threshold=low_threshold)
    mag = np.abs(y)
    score = ranked["_score"].to_numpy(dtype=float)
    n = int(len(ranked))
    out = {
        "rank_corr_score_target_all": rank_corr(score, target),
        "n_rank_corr_all": int(np.isfinite(target).sum()),
    }
    for k in PAIR_ACQ_TOP_K:
        if n < k:
            out[f"n_selected_at_{k}"] = _nan()
            out[f"n_hits_at_{k}"] = _nan()
            out[f"hit_rate_at_{k}"] = _nan()
            out[f"event_precision_at_{k}"] = _nan()
            out[f"recall_at_{k}"] = _nan()
            out[f"any_hit_at_{k}"] = _nan()
            out[f"hit_at_{k}"] = _nan()
            out[f"mean_target_observed_at_{k}"] = _nan()
            out[f"mean_abs_selectivity_at_{k}"] = _nan()
            out[f"n_unique_ligands_selected_at_{k}"] = _nan()
            out[f"rank_corr_score_target_at_{k}"] = _nan()
            continue
        top = ranked.iloc[:k]
        n_hits = int(event[:k].sum())
        out[f"n_selected_at_{k}"] = k
        out[f"n_hits_at_{k}"] = n_hits
        out[f"hit_rate_at_{k}"] = n_hits / k
        out[f"event_precision_at_{k}"] = n_hits / k
        out[f"recall_at_{k}"] = n_hits / n_event_pool if n_event_pool else _nan()
        out[f"any_hit_at_{k}"] = float(n_hits > 0)
        out[f"hit_at_{k}"] = float(n_hits > 0)
        out[f"mean_target_observed_at_{k}"] = float(np.mean(target[:k]))
        out[f"mean_abs_selectivity_at_{k}"] = float(np.mean(mag[:k]))
        out[f"n_unique_ligands_selected_at_{k}"] = int(top["ligand_smiles"].nunique())
        out[f"rank_corr_score_target_at_{k}"] = rank_corr(score[:k], target[:k]) if k >= 2 else _nan()
    out.update(evaluate_regret(ranked, task, orientation))
    assert_hit_monotone_values(out)
    return out


def pool_objective(y: np.ndarray, task: str, orientation: str) -> tuple[np.ndarray, str]:
    """Objective values and sense for retrospective regret. Higher-better is 'max'."""
    y = np.asarray(y, dtype=float)
    if task == TASK_HIGH:
        return target_observed(y, orientation), "max"
    if task == TASK_LOW:
        return np.abs(y), "min"
    raise ValueError(task)


def evaluate_regret(ranked: pd.DataFrame, task: str, orientation: str) -> dict:
    """Retrospective regret@k on an already-ranked pair-task. Does not rerank."""
    out = {
        "pool_optimum": _nan(),
        "pool_optimum_tied": _nan(),
        "n_pool_optima": _nan(),
    }
    for k in PAIR_ACQ_TOP_K:
        out[f"regret_at_{k}"] = _nan()
        out[f"batch_best_at_{k}"] = _nan()
        out[f"selected_contains_optimum_at_{k}"] = _nan()
    if ranked is None or getattr(ranked, "empty", True) or "y_true" not in ranked.columns:
        return out
    if "record_id" not in ranked.columns:
        raise RuntimeError("regret requires record_id")
    y = pd.to_numeric(ranked["y_true"], errors="coerce").to_numpy(dtype=float)
    ids = ranked["record_id"].astype(str).to_numpy()
    pool_ids = set(ids.tolist())
    obj, sense = pool_objective(y, task, orientation)
    finite = np.isfinite(obj)
    if not finite.any():
        return out
    pool_obj = obj[finite]
    pool_id_arr = ids[finite]
    if sense == "max":
        pool_opt = float(np.max(pool_obj))
        is_opt = np.abs(pool_obj - pool_opt) <= REGRET_TOL
    else:
        pool_opt = float(np.min(pool_obj))
        is_opt = np.abs(pool_obj - pool_opt) <= REGRET_TOL
    n_opt = int(is_opt.sum())
    opt_ids = set(pool_id_arr[is_opt].tolist())
    out["pool_optimum"] = pool_opt
    out["pool_optimum_tied"] = float(n_opt > 1)
    out["n_pool_optima"] = float(n_opt)
    n = int(len(ranked))
    for k in PAIR_ACQ_TOP_K:
        if n < k:
            continue
        batch = ranked.iloc[:k]
        batch_ids = [str(x) for x in batch["record_id"].tolist()]
        extra = set(batch_ids) - pool_ids
        if extra:
            raise RuntimeError(f"selected batch is not a subset of the candidate pool: {sorted(extra)}")
        batch_obj = obj[:k]
        batch_finite = batch_obj[np.isfinite(batch_obj)]
        if batch_finite.size == 0:
            continue
        if sense == "max":
            batch_best = float(np.max(batch_finite))
            regret = pool_opt - batch_best
        else:
            batch_best = float(np.min(batch_finite))
            regret = batch_best - pool_opt
        if regret < -REGRET_TOL:
            raise RuntimeError(f"regret@{k}={regret} is below tolerance {REGRET_TOL}")
        contains = bool(set(batch_ids) & opt_ids)
        if contains:
            if regret > REGRET_TOL:
                raise RuntimeError(f"batch contains a pool optimum but regret@{k}={regret}")
            regret = 0.0
        else:
            regret = float(max(regret, 0.0))
        out[f"regret_at_{k}"] = regret
        out[f"batch_best_at_{k}"] = batch_best
        out[f"selected_contains_optimum_at_{k}"] = float(contains)
    return out


def assert_hit_monotone_values(rec: dict) -> None:
    """True hit@k must be nondecreasing in k on a single pair-task."""
    vals = []
    for k in PAIR_ACQ_TOP_K:
        key = f"any_hit_at_{k}" if f"any_hit_at_{k}" in rec else f"hit_at_{k}"
        v = rec.get(key)
        if v is None or not np.isfinite(v):
            continue
        vals.append((k, float(v)))
    for (k0, a), (k1, b) in zip(vals, vals[1:]):
        if a > b + 1e-12:
            raise RuntimeError(f"hit@{k0}={a} > hit@{k1}={b}")


def assert_fold_hit_monotone(df: pd.DataFrame) -> None:
    """Check task-level hit@k monotonicity on stored fold metrics (no reranking)."""
    for rec in df.to_dict(orient="records"):
        assert_hit_monotone_values(rec)


def n_tasks_defined_columns(part: pd.DataFrame) -> dict[str, float]:
    """Per-k pair-task counts with at least k held-out rows (`n_tasks_defined`)."""
    rec: dict[str, float] = {}
    if part.empty or "metric_name" not in part.columns or "k" not in part.columns:
        return rec
    names = part.metric_name.astype(str)
    kvals = pd.to_numeric(part.k, errors="coerce")
    for k in PAIR_ACQ_TOP_K:
        ndef = part[(names == "n_tasks_defined") & (kvals == k)]
        if len(ndef):
            rec[f"n_tasks_defined_at_{k}"] = float(ndef.iloc[0].fold_mean)
            rec[f"n_tasks_defined_at_{k}_sd"] = float(ndef.iloc[0].fold_sd)
    return rec


def hit_and_precision_wide(summary: pd.DataFrame) -> pd.DataFrame:
    """Register event precision@k (legacy hit_rate) and true hit@k (any_hit) as a wide table."""
    rows = []
    keys = ["task", "orientation", "strategy"]
    for (task, orientation, strategy), part in summary.groupby(keys, dropna=False):
        rec = {
            "task": task,
            "orientation": orientation,
            "strategy": strategy,
            "n_tasks_fold_mean": float(part.n_tasks_fold_mean.iloc[0]),
            "n_tasks_fold_sd": float(part.n_tasks_fold_sd.iloc[0]),
            "n_valid_folds": int(part.n_valid_folds.iloc[0]) if pd.notna(part.n_valid_folds.iloc[0]) else 5,
            "aggregation": "pair_equal_then_fold_mean_plus_sd",
            "event_precision_source_metric": "hit_rate",
            "hit_source_metric": "any_hit",
            "legacy_hit_rate_is_event_precision": True,
        }
        rec.update(n_tasks_defined_columns(part))
        for k in PAIR_ACQ_TOP_K:
            prec = part[(part.metric_name.astype(str) == "hit_rate") & (pd.to_numeric(part.k, errors="coerce") == k)]
            hit = part[(part.metric_name.astype(str) == "any_hit") & (pd.to_numeric(part.k, errors="coerce") == k)]
            if len(prec):
                rec[f"event_precision_at_{k}"] = float(prec.iloc[0].fold_mean)
                rec[f"event_precision_at_{k}_sd"] = float(prec.iloc[0].fold_sd)
                rec[f"hit_rate_at_{k}"] = float(prec.iloc[0].fold_mean)
                rec[f"hit_rate_at_{k}_sd"] = float(prec.iloc[0].fold_sd)
            if len(hit):
                rec[f"hit_at_{k}"] = float(hit.iloc[0].fold_mean)
                rec[f"hit_at_{k}_sd"] = float(hit.iloc[0].fold_sd)
                rec[f"any_hit_at_{k}"] = float(hit.iloc[0].fold_mean)
                rec[f"any_hit_at_{k}_sd"] = float(hit.iloc[0].fold_sd)
        rows.append(rec)
    return pd.DataFrame(rows)


def register_hit_and_precision(
    summary: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    dest: Path,
) -> dict[str, Path]:
    """Write true hit@k and event precision@k from existing fold summaries. Does not rerank."""
    assert_fold_hit_monotone(fold_metrics)
    wide = hit_and_precision_wide(summary)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    csv_path = dest / "pair_acquisition_hit_and_precision.csv"
    wide.to_csv(csv_path, index=False)
    payload = {
        "legacy_column": "hit_rate_at_k",
        "legacy_displayed_as_through_2026_08_17": "hit@k",
        "correct_name": "event_precision_at_k",
        "event_precision_definition": "n_hits / k",
        "true_hit_column": "any_hit_at_k",
        "true_hit_displayed_as": "hit@k",
        "true_hit_definition": "1[sum_{j=1}^k I_j > 0]",
        "source_fold_metrics_unchanged": True,
        "ranking_unchanged": True,
        "ranking_inputs": ["y_pred", "sigma_total"],
        "held_out_labels_used_for_ranking": False,
        "n_fold_metric_rows": int(len(fold_metrics)),
        "n_wide_rows": int(len(wide)),
        "aggregation": "pair_equal_then_fold_mean_plus_sd",
        "note": "Legacy hit_rate_at_k remains the machine-readable event-precision field. True hit@k is any_hit_at_k.",
    }
    json_path = dest / "pair_acquisition_metric_migration.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"csv": csv_path, "json": json_path}


def regret_wide(summary: pd.DataFrame) -> pd.DataFrame:
    """Wide regret@k table from the long pair-acquisition summary. Does not rerank."""
    rows = []
    keys = ["task", "orientation", "strategy"]
    for (task, orientation, strategy), part in summary.groupby(keys, dropna=False):
        rec = {
            "task": task,
            "orientation": orientation,
            "strategy": strategy,
            "n_tasks_fold_mean": float(part.n_tasks_fold_mean.iloc[0]),
            "n_tasks_fold_sd": float(part.n_tasks_fold_sd.iloc[0]),
            "n_valid_folds": int(part.n_valid_folds.iloc[0]) if pd.notna(part.n_valid_folds.iloc[0]) else 5,
            "aggregation": "pair_equal_then_fold_mean_plus_sd",
            "lower_is_better": True,
        }
        rec.update(n_tasks_defined_columns(part))
        opt = part[part.metric_name.astype(str) == "pool_optimum"]
        if len(opt):
            rec["pool_optimum"] = float(opt.iloc[0].fold_mean)
            rec["pool_optimum_sd"] = float(opt.iloc[0].fold_sd)
        tied = part[part.metric_name.astype(str) == "pool_optimum_tied"]
        if len(tied):
            rec["pool_optimum_tied_rate"] = float(tied.iloc[0].fold_mean)
            rec["pool_optimum_tied_rate_sd"] = float(tied.iloc[0].fold_sd)
        for k in PAIR_ACQ_TOP_K:
            rg = part[(part.metric_name.astype(str) == "regret") & (pd.to_numeric(part.k, errors="coerce") == k)]
            if len(rg):
                rec[f"regret_at_{k}"] = float(rg.iloc[0].fold_mean)
                rec[f"regret_at_{k}_sd"] = float(rg.iloc[0].fold_sd)
            bb = part[(part.metric_name.astype(str) == "batch_best") & (pd.to_numeric(part.k, errors="coerce") == k)]
            if len(bb):
                rec[f"batch_best_at_{k}"] = float(bb.iloc[0].fold_mean)
                rec[f"batch_best_at_{k}_sd"] = float(bb.iloc[0].fold_sd)
            sc = part[(part.metric_name.astype(str) == "selected_contains_optimum") & (pd.to_numeric(part.k, errors="coerce") == k)]
            if len(sc):
                rec[f"selected_contains_optimum_at_{k}"] = float(sc.iloc[0].fold_mean)
                rec[f"selected_contains_optimum_at_{k}_sd"] = float(sc.iloc[0].fold_sd)
        rows.append(rec)
    return pd.DataFrame(rows)


def fold_regret_table(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Pair-equal fold-level regret and pool optimum. Does not rerank."""
    rows = []
    keys = ["outer_fold", "task", "orientation", "strategy"]
    if fold_metrics.empty or any(c not in fold_metrics.columns for c in keys):
        return pd.DataFrame()
    for (outer_fold, task, orientation, strategy), g in fold_metrics.groupby(keys, dropna=False):
        rec_base = {
            "outer_fold": outer_fold,
            "task": task,
            "orientation": orientation,
            "strategy": strategy,
            "aggregation": "pair_equal_mean",
            "n_pair_tasks": int(len(g)),
        }
        if "pool_optimum" in g.columns:
            rec_base["pool_optimum"] = float(np.nanmean(pd.to_numeric(g["pool_optimum"], errors="coerce")))
        if "pool_optimum_tied" in g.columns:
            rec_base["pool_optimum_tied_rate"] = float(np.nanmean(pd.to_numeric(g["pool_optimum_tied"], errors="coerce")))
        if "n_pool_optima" in g.columns:
            rec_base["n_pool_optima_mean"] = float(np.nanmean(pd.to_numeric(g["n_pool_optima"], errors="coerce")))
        for k in PAIR_ACQ_TOP_K:
            rec = dict(rec_base)
            rec["k"] = k
            col = f"regret_at_{k}"
            if col in g.columns:
                vals = pd.to_numeric(g[col], errors="coerce")
                rec["regret"] = float(np.nanmean(vals))
                rec["n_tasks_defined"] = int(np.isfinite(vals).sum())
            bb = f"batch_best_at_{k}"
            if bb in g.columns:
                rec["batch_best"] = float(np.nanmean(pd.to_numeric(g[bb], errors="coerce")))
            sc = f"selected_contains_optimum_at_{k}"
            if sc in g.columns:
                rec["selected_contains_optimum"] = float(np.nanmean(pd.to_numeric(g[sc], errors="coerce")))
            rows.append(rec)
    return pd.DataFrame(rows)


def register_regret(
    summary: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    dest: Path,
) -> dict[str, Path]:
    """Write regret@k tables from stored rankings. Does not change hit@k or precision@k."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    wide = regret_wide(summary)
    fold_tbl = fold_regret_table(fold_metrics)
    csv_path = dest / "pair_acquisition_regret.csv"
    fold_path = dest / "pair_acquisition_regret_fold.csv"
    wide.to_csv(csv_path, index=False)
    fold_tbl.to_csv(fold_path, index=False)
    payload = {
        "metric": "retrospective_regret_at_k",
        "lower_is_better": True,
        "high_selectivity": "max_pool target-direction Delta log K minus max_batch",
        "near_zero": "min_batch |Delta log K| minus min_pool |Delta log K|",
        "tolerance": REGRET_TOL,
        "tied_optima_zero_regret": True,
        "hit_precision_unchanged": True,
        "ranking_unchanged": True,
        "n_wide_rows": int(len(wide)),
        "n_fold_rows": int(len(fold_tbl)),
        "aggregation": "pair_equal_then_fold_mean_plus_sd",
    }
    json_path = dest / "pair_acquisition_regret.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"csv": csv_path, "fold": fold_path, "json": json_path}


def expected_random_metrics(n_event: int, n_samples: int) -> dict:
    p = n_event / n_samples if n_samples else _nan()
    out = {}
    for k in PAIR_ACQ_TOP_K:
        if n_samples < k:
            out[f"random_expected_hit_rate_at_{k}"] = _nan()
            out[f"random_expected_recall_at_{k}"] = _nan()
        else:
            out[f"random_expected_hit_rate_at_{k}"] = float(p)
            out[f"random_expected_recall_at_{k}"] = float(k / n_samples) if n_event else _nan()
    return out


def evaluate_task_strategies(
    part: pd.DataFrame,
    task: dict,
    rng: np.random.Generator,
    n_random: int,
    *,
    high_threshold: float | None = None,
    low_threshold: float | None = None,
) -> list[dict]:
    n_event = int(task["n_event"])
    n_samples = int(len(part))
    high_tau, low_tau = resolved_event_thresholds(high_threshold, low_threshold)
    base = {
        "task": task["task"],
        "metal_species_pair": task["metal_species_pair"],
        "orientation": task["orientation"],
        "n_samples": n_samples,
        "n_event": n_event,
        "n_unique_ligands": int(part["ligand_smiles"].nunique()),
        "event_rate": n_event / n_samples if n_samples else _nan(),
        "is_ree_pair": bool(task.get("is_ree_pair", False)),
        "high_threshold": high_tau,
        "low_threshold": low_tau,
    }
    base.update(expected_random_metrics(n_event, n_samples))
    rows = []
    for strategy in PAIR_ACQ_STRATEGIES:
        ranked = rank_rows(
            part,
            strategy,
            task["task"],
            task["orientation"],
            high_threshold=high_threshold,
            low_threshold=low_threshold,
        )
        rec = dict(base)
        rec["strategy"] = strategy
        rec.update(
            evaluate_topk(
                ranked,
                task["task"],
                task["orientation"],
                n_event,
                high_threshold=high_threshold,
                low_threshold=low_threshold,
            )
        )
        rows.append(rec)
    acc = []
    rec_rand = dict(base)
    rec_rand["strategy"] = PAIR_ACQ_RANDOM_STRATEGY
    rec_rand["n_repeats"] = int(n_random)
    for _ in range(int(n_random)):
        acc.append(
            evaluate_topk(
                random_rank_rows(part, rng),
                task["task"],
                task["orientation"],
                n_event,
                high_threshold=high_threshold,
                low_threshold=low_threshold,
            )
        )
    metric_keys = [k for k in acc[0] if k != "n_rank_corr_all"] if acc else []
    for key in metric_keys:
        vals = np.asarray([d[key] for d in acc], dtype=float) if acc else np.array([])
        rec_rand[key] = float(np.nanmean(vals)) if vals.size and np.isfinite(vals).any() else _nan()
    rec_rand["n_rank_corr_all"] = n_samples
    rows.append(rec_rand)
    return rows


def _bin_mask(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if np.isposinf(hi):
        return values >= lo
    return (values >= lo) & (values < hi)


def fold_calibration(df: pd.DataFrame, fold: str, model_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
    yhat = pd.to_numeric(df["y_pred"], errors="coerce").to_numpy(dtype=float)
    sigma = pd.to_numeric(df["sigma_total"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(sigma)
    y, yhat, sigma = y[mask], yhat[mask], sigma[mask]
    abs_y = np.abs(y)
    abs_yhat = np.abs(yhat)
    abs_err = np.abs(y - yhat)
    fields = {
        "sigma_total": sigma,
        "abs_y": abs_y,
        "abs_yhat": abs_yhat,
        "abs_error": abs_err,
    }
    rho_rows = []
    for name, a, b in RHO_PAIRS:
        rho_rows.append(
            {
                "model_id": model_id,
                "outer_fold": fold,
                "metric_name": name,
                "value": rank_corr(fields[a], fields[b]),
                "n_rows": int(len(y)),
                "scope": "all_heldout_rows",
            }
        )
    bin_rows = []
    regimes = (
        ("abs_yhat_lt_1", abs_yhat < 1.0),
        ("abs_yhat_ge_1", abs_yhat >= 1.0),
        ("abs_y_lt_1", abs_y < 1.0),
        ("abs_y_ge_1", abs_y >= 1.0),
    )
    slices = [(f"abs_yhat_{lab}", _bin_mask(abs_yhat, lo, hi)) for lab, lo, hi in ABS_YHAT_BINS]
    slices.extend(regimes)
    for label, keep in slices:
        n = int(keep.sum())
        rec = {
            "model_id": model_id,
            "outer_fold": fold,
            "bin_id": label,
            "n_rows": n,
            "scope": "all_heldout_rows",
        }
        if n < CALIBRATION_MIN_N:
            rec.update(
                {
                    "rmse": _nan(),
                    "mean_sigma": _nan(),
                    "mean_abs_yhat": _nan(),
                    "mean_abs_y": _nan(),
                    "picp_90": _nan(),
                    "ace": _nan(),
                    "z90": Z90,
                }
            )
        else:
            rec["rmse"] = float(np.sqrt(np.mean((y[keep] - yhat[keep]) ** 2)))
            rec["mean_sigma"] = float(np.mean(sigma[keep]))
            rec["mean_abs_yhat"] = float(np.mean(abs_yhat[keep]))
            rec["mean_abs_y"] = float(np.mean(abs_y[keep]))
            rec["picp_90"] = picp_at_90(y[keep], yhat[keep], sigma[keep])
            rec["ace"] = adaptive_calibration_error(y[keep], yhat[keep], sigma[keep])
            rec["z90"] = Z90
        bin_rows.append(rec)
    return pd.DataFrame(bin_rows), pd.DataFrame(rho_rows)
