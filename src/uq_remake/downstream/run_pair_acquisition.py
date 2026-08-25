"""Run sigma-only pair acquisition from stored workhorse fold scores."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from uq_remake import REMAKE_ROOT  # noqa: E402
from uq_remake.downstream.pair_acquisition import (  # noqa: E402
    TASK_HIGH,
    TASK_LOW,
    eligible_tasks,
    evaluate_task_strategies,
    fold_calibration,
    n_tasks_defined_columns,
    pair_event_counts,
    register_hit_and_precision,
    register_regret,
)
from uq_remake.downstream.protocol import (  # noqa: E402
    AD_RETENTION_DIR,
    DS2_WORKHORSE,
    FOLDS,
    PAIR_ACQ_DIR,
    PAIR_ACQ_EVENT_MIN_EXCLUSIVE,
    PAIR_ACQ_RANDOM_REPEATS,
    PAIR_ACQ_RANDOM_SEED,
    PAIR_ACQ_STRATEGIES,
    PAIR_ACQ_THRESHOLD,
    PAIR_ACQ_TOP_K,
    rel,
)
from uq_remake.metrics.cv_protocol import EVALUATION_PARTITION, FOLD_SD_DDOF  # noqa: E402

SPEC_PATH = REMAKE_ROOT / "configs" / "experiments" / "pair_acquisition_spec.yaml"
SCORE_COLS = [
    "record_id",
    "y_true",
    "y_pred",
    "sigma_total",
    "metal_species_pair",
    "ligand_smiles",
    "is_ree_pair",
    "evaluation_partition",
]
LONG_METRICS = (
    "hit_rate",
    "recall",
    "any_hit",
    "mean_target_observed",
    "mean_abs_selectivity",
    "n_hits",
    "rank_corr_score_target",
    "regret",
    "batch_best",
    "selected_contains_optimum",
)


def _write_csv(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
    else:
        pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path


def _mean_sd(values) -> tuple[float, float, int]:
    x = np.asarray(list(values), dtype=float)
    finite = x[np.isfinite(x)]
    n = int(len(finite))
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(finite))
    sd = float(np.std(finite, ddof=FOLD_SD_DDOF)) if n >= 2 else float("nan")
    return mean, sd, n


def load_fold(fold: str, model_id: str) -> pd.DataFrame:
    path = AD_RETENTION_DIR / "fold_scores" / model_id / f"{fold}.csv"
    df = pd.read_csv(path, usecols=SCORE_COLS)
    if "evaluation_partition" in df.columns:
        part = df["evaluation_partition"].astype(str)
        if not part.eq(EVALUATION_PARTITION).all():
            raise RuntimeError(f"{fold}: expected {EVALUATION_PARTITION}")
    df["outer_fold"] = fold
    df["model_id"] = model_id
    return df


def run_fold(
    df: pd.DataFrame,
    fold: str,
    model_id: str,
    *,
    high_threshold: float | None = None,
    low_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    high_tau = float(PAIR_ACQ_THRESHOLD if high_threshold is None else high_threshold)
    low_tau = float(PAIR_ACQ_THRESHOLD if low_threshold is None else low_threshold)
    tasks = eligible_tasks(df, high_threshold=high_tau, low_threshold=low_tau)
    counts = pair_event_counts(df, high_threshold=high_tau, low_threshold=low_tau)
    counts["outer_fold"] = fold
    counts["model_id"] = model_id
    counts["high_threshold"] = high_tau
    counts["low_threshold"] = low_tau
    fold_index = FOLDS.index(fold)
    rng = np.random.default_rng(PAIR_ACQ_RANDOM_SEED + fold_index * 1009)
    rows = []
    for task in tasks:
        part = df.loc[df["metal_species_pair"] == task["metal_species_pair"]].copy()
        for rec in evaluate_task_strategies(
            part,
            task,
            rng,
            PAIR_ACQ_RANDOM_REPEATS,
            high_threshold=high_tau,
            low_threshold=low_tau,
        ):
            rec["model_id"] = model_id
            rec["outer_fold"] = fold
            rec["event_min_exclusive"] = PAIR_ACQ_EVENT_MIN_EXCLUSIVE
            rec["high_threshold"] = high_tau
            rec["low_threshold"] = low_tau
            rec["threshold"] = high_tau if rec["task"] == TASK_HIGH else low_tau
            rows.append(rec)
    return pd.DataFrame(rows), counts


def summarize_acquisition(metrics: pd.DataFrame) -> pd.DataFrame:
    strategies = list(PAIR_ACQ_STRATEGIES) + ["random"]
    rows = []
    views = [
        (TASK_HIGH, "canonical"),
        (TASK_HIGH, "reversed"),
        (TASK_HIGH, "both"),
        (TASK_LOW, "near_zero"),
    ]
    for task, orientation in views:
        part = metrics[metrics.task == task]
        if orientation != "both":
            part = part[part.orientation == orientation]
        if part.empty:
            continue
        for strategy in strategies:
            sub = part[part.strategy == strategy]
            if sub.empty:
                continue
            n_tasks = sub.groupby("outer_fold").size()
            n_mean, n_sd, n_folds = _mean_sd(n_tasks)
            base = {
                "model_id": str(sub.model_id.iloc[0]),
                "task": task,
                "orientation": orientation,
                "strategy": strategy,
                "n_tasks_fold_mean": n_mean,
                "n_tasks_fold_sd": n_sd,
                "n_valid_folds": n_folds,
                "aggregation": "pair_equal_then_fold_mean_plus_sd",
            }
            for col in ("n_samples", "n_event", "event_rate", "n_unique_ligands", "pool_optimum", "pool_optimum_tied", "n_pool_optima"):
                if col not in sub.columns:
                    continue
                mean, sd, n = _mean_sd(sub.groupby("outer_fold")[col].mean())
                rows.append({**base, "k": "", "metric_name": col, "fold_mean": mean, "fold_sd": sd, "n_valid_folds": n})
            for k in PAIR_ACQ_TOP_K:
                defined = sub.assign(_ok=np.isfinite(pd.to_numeric(sub[f"hit_rate_at_{k}"], errors="coerce")))
                n_def, n_def_sd, _ = _mean_sd(defined.groupby("outer_fold")["_ok"].sum())
                rows.append(
                    {
                        **base,
                        "k": k,
                        "metric_name": "n_tasks_defined",
                        "fold_mean": n_def,
                        "fold_sd": n_def_sd,
                    }
                )
                for stem in LONG_METRICS:
                    col = f"{stem}_at_{k}"
                    if col not in sub.columns:
                        continue
                    mean, sd, n = _mean_sd(sub.groupby("outer_fold")[col].mean())
                    rows.append({**base, "k": k, "metric_name": stem, "fold_mean": mean, "fold_sd": sd, "n_valid_folds": n})
            mean, sd, n = _mean_sd(sub.groupby("outer_fold")["rank_corr_score_target_all"].mean())
            rows.append(
                {
                    **base,
                    "k": "",
                    "metric_name": "rank_corr_score_target_all",
                    "fold_mean": mean,
                    "fold_sd": sd,
                    "n_valid_folds": n,
                }
            )
    return pd.DataFrame(rows)


def summarize_eligibility(
    counts: pd.DataFrame,
    *,
    high_threshold: float | None = None,
    low_threshold: float | None = None,
) -> pd.DataFrame:
    high_tau = float(PAIR_ACQ_THRESHOLD if high_threshold is None else high_threshold)
    low_tau = float(PAIR_ACQ_THRESHOLD if low_threshold is None else low_threshold)
    rows = []
    for name, col in (
        ("high_canonical", "high_canonical_eligible"),
        ("high_reversed", "high_reversed_eligible"),
        ("low_selectivity", "low_eligible"),
    ):
        per_fold = counts.groupby("outer_fold")[col].sum()
        mean, sd, n = _mean_sd(per_fold)
        rows.append(
            {
                "cohort": name,
                "metric_name": "n_eligible_pairs",
                "fold_mean": mean,
                "fold_sd": sd,
                "n_valid_folds": n,
                "event_min_exclusive": PAIR_ACQ_EVENT_MIN_EXCLUSIVE,
                "threshold": high_tau if name != "low_selectivity" else low_tau,
                "high_threshold": high_tau,
                "low_threshold": low_tau,
            }
        )
    high_both = counts.groupby("outer_fold")[["high_canonical_eligible", "high_reversed_eligible"]].sum().sum(axis=1)
    mean, sd, n = _mean_sd(high_both)
    rows.append(
        {
            "cohort": "high_both_orientations",
            "metric_name": "n_eligible_pair_tasks",
            "fold_mean": mean,
            "fold_sd": sd,
            "n_valid_folds": n,
            "event_min_exclusive": PAIR_ACQ_EVENT_MIN_EXCLUSIVE,
            "threshold": high_tau,
            "high_threshold": high_tau,
            "low_threshold": low_tau,
        }
    )
    return pd.DataFrame(rows)


def summarize_bins(bins: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (bin_id,), part in bins.groupby(["bin_id"]):
        rec = {
            "model_id": str(part.model_id.iloc[0]),
            "bin_id": bin_id,
            "scope": "all_heldout_rows",
            "aggregation": "fold_mean_plus_sd",
        }
        for col in ("n_rows", "rmse", "mean_sigma", "mean_abs_yhat", "mean_abs_y", "picp_90", "ace"):
            mean, sd, n = _mean_sd(part[col])
            rows.append({**rec, "metric_name": col, "fold_mean": mean, "fold_sd": sd, "n_valid_folds": n})
    return pd.DataFrame(rows)


def summarize_rho(rho: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, part in rho.groupby("metric_name"):
        mean, sd, n = _mean_sd(part["value"])
        rows.append(
            {
                "model_id": str(part.model_id.iloc[0]),
                "metric_name": name,
                "fold_mean": mean,
                "fold_sd": sd,
                "n_valid_folds": n,
                "scope": "all_heldout_rows",
                "aggregation": "fold_mean_plus_sd",
            }
        )
    return pd.DataFrame(rows)


def wide_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    hit = summary[summary.metric_name == "hit_rate"].copy()
    if hit.empty:
        return pd.DataFrame()
    rows = []
    for (task, orientation, strategy), part in hit.groupby(["task", "orientation", "strategy"]):
        rec = {
            "task": task,
            "orientation": orientation,
            "strategy": strategy,
            "n_tasks_fold_mean": float(part.n_tasks_fold_mean.iloc[0]),
            "n_tasks_fold_sd": float(part.n_tasks_fold_sd.iloc[0]),
        }
        rec.update(
            n_tasks_defined_columns(
                summary[
                    (summary.task == task)
                    & (summary.orientation == orientation)
                    & (summary.strategy == strategy)
                ]
            )
        )
        for k in PAIR_ACQ_TOP_K:
            row = part[pd.to_numeric(part.k, errors="coerce") == k]
            if row.empty:
                continue
            rec[f"hit_rate_at_{k}"] = float(row.fold_mean.iloc[0])
            rec[f"hit_rate_at_{k}_sd"] = float(row.fold_sd.iloc[0])
        mean_col = "mean_abs_selectivity" if task == TASK_LOW else "mean_target_observed"
        mean_src = summary[
            (summary.task == task)
            & (summary.orientation == orientation)
            & (summary.strategy == strategy)
            & (summary.metric_name == mean_col)
        ]
        for k in PAIR_ACQ_TOP_K:
            row = mean_src[pd.to_numeric(mean_src.k, errors="coerce") == k]
            if row.empty:
                continue
            rec[f"{mean_col}_at_{k}"] = float(row.fold_mean.iloc[0])
            rec[f"{mean_col}_at_{k}_sd"] = float(row.fold_sd.iloc[0])
        rows.append(rec)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Sigma-only pair acquisition from stored fold scores.")
    p.add_argument("--model-id", default=DS2_WORKHORSE)
    p.add_argument("--output-dir", default=None, help="Write artifacts here instead of the primary pair-acquisition directory.")
    p.add_argument("--high-threshold", type=float, default=None)
    p.add_argument("--low-threshold", type=float, default=None)
    p.add_argument("--spec", default=None)
    p.add_argument("--skip-calibration", action="store_true")
    args = p.parse_args(argv)
    model_id = str(args.model_id)
    high_tau = float(PAIR_ACQ_THRESHOLD if args.high_threshold is None else args.high_threshold)
    low_tau = float(PAIR_ACQ_THRESHOLD if args.low_threshold is None else args.low_threshold)
    out_dir = Path(args.output_dir) if args.output_dir else PAIR_ACQ_DIR
    spec_path = Path(args.spec) if args.spec else SPEC_PATH
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    metric_rows = []
    count_rows = []
    bin_rows = []
    rho_rows = []
    for fold in FOLDS:
        df = load_fold(fold, model_id)
        metrics, counts = run_fold(df, fold, model_id, high_threshold=high_tau, low_threshold=low_tau)
        metric_rows.append(metrics)
        count_rows.append(counts)
        if not args.skip_calibration:
            bins, rho = fold_calibration(df, fold, model_id)
            bin_rows.append(bins)
            rho_rows.append(rho)
    fold_metrics = pd.concat(metric_rows, ignore_index=True)
    eligibility = pd.concat(count_rows, ignore_index=True)
    summary = summarize_acquisition(fold_metrics)
    elig_sum = summarize_eligibility(eligibility, high_threshold=high_tau, low_threshold=low_tau)
    comparison = wide_comparison(summary)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "pair_acquisition_fold_metrics.csv": fold_metrics,
        "pair_acquisition_eligibility.csv": eligibility,
        "pair_acquisition_eligibility_summary.csv": elig_sum,
        "pair_acquisition_summary.csv": summary,
        "pair_acquisition_comparison.csv": comparison,
    }
    if bin_rows:
        bins = pd.concat(bin_rows, ignore_index=True)
        rho = pd.concat(rho_rows, ignore_index=True)
        names["pair_acquisition_calibration_bins.csv"] = bins
        names["pair_acquisition_calibration_bin_summary.csv"] = summarize_bins(bins)
        names["pair_acquisition_calibration_rho.csv"] = rho
        names["pair_acquisition_calibration_rho_summary.csv"] = summarize_rho(rho)
    for name, frame in names.items():
        _write_csv(out_dir / name, frame)
    payload = {
        "experiment_id": spec.get("experiment_id", "sigma_pair_acquisition"),
        "pass": True,
        "computation_status": "complete",
        "pooled_oof_metrics": False,
        "global_selection": False,
        "distance_scores_used": False,
        "model_id": model_id,
        "n_folds": len(FOLDS),
        "k_grid": list(PAIR_ACQ_TOP_K),
        "strategies": list(PAIR_ACQ_STRATEGIES) + ["random"],
        "event_min_exclusive": PAIR_ACQ_EVENT_MIN_EXCLUSIVE,
        "threshold": high_tau if high_tau == low_tau else None,
        "high_threshold": high_tau,
        "low_threshold": low_tau,
        "spec_path": rel(spec_path),
        "output_dir": rel(out_dir),
        "n_pair_task_rows": int(len(fold_metrics)),
        "primary_pair_acquisition_unchanged": str(out_dir.resolve()) != str(PAIR_ACQ_DIR.resolve()),
    }
    _write_json(out_dir / "pair_acquisition_validation.json", payload)
    _write_json(out_dir / "pair_acquisition_spec_snapshot.json", spec)
    register_hit_and_precision(summary, fold_metrics, out_dir)
    register_regret(summary, fold_metrics, out_dir)
    print(
        json.dumps(
            {k: payload[k] for k in ("experiment_id", "pass", "n_pair_task_rows", "model_id", "high_threshold", "low_threshold")},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
