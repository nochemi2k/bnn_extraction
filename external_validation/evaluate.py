"""Recompute every reported external-validation analysis from fixed predictions.

No training, model selection, raw-data access, plotting, or manuscript generation
is performed. BigSolDB inference uses row RMSE with solute-cluster bootstrap;
lanthanide inference uses MoP RMSE with metal-pair-cluster bootstrap. Intervals
condition on the fixed selected sets and do not include retraining variability.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    BIGSOLDB_BOOTSTRAP_SEED,
    BOOTSTRAP_REPLICATES,
    FOLDS,
    LANTHANIDE_MOP_BOOTSTRAP_SEED,
    PACKAGE_ROOT,
    RANDOM_REPEATS,
    RANDOM_SEED_BASE,
    REPOSITORY_ROOT,
    RETENTIONS,
    Z90,
    load_fixed_predictions,
    validate_prediction_indices,
    write_json,
)

STRATEGIES = ("uncertainty", "tanimoto", "mahalanobis", "random")
COMPARATORS = ("random", "tanimoto", "mahalanobis")
MINIMUM_PAIR_ROWS = 3


def errors(frame: pd.DataFrame, prediction: str = "y_pred") -> np.ndarray:
    return frame["y_true"].to_numpy(dtype=float) - frame[prediction].to_numpy(dtype=float)


def rmse(frame: pd.DataFrame, prediction: str = "y_pred") -> float:
    return float(np.sqrt(np.mean(np.square(errors(frame, prediction)))))


def mae(frame: pd.DataFrame, prediction: str = "y_pred") -> float:
    return float(np.mean(np.abs(errors(frame, prediction))))


def balanced_rmse(frame: pd.DataFrame, group: str, minimum: int = 3) -> tuple[float, int]:
    values = [rmse(selected) for _, selected in frame.groupby(group, sort=True) if len(selected) >= minimum]
    return (float(np.mean(values)) if values else float("nan"), len(values))


def calibration(frame: pd.DataFrame) -> dict:
    error = errors(frame)
    sigma = frame["sigma_total"].to_numpy(dtype=float)
    width = 2.0 * Z90 * sigma
    lower = frame["y_pred"].to_numpy(dtype=float) - Z90 * sigma
    upper = frame["y_pred"].to_numpy(dtype=float) + Z90 * sigma
    y = frame["y_true"].to_numpy(dtype=float)
    interval_score = width + 20.0 * (np.maximum(lower - y, 0.0) + np.maximum(y - upper, 0.0))
    return {
        "picp_90": float(np.mean(np.abs(error) <= Z90 * sigma)),
        "mean_interval_width_90": float(np.mean(width)),
        "median_interval_width_90": float(np.median(width)),
        "interval_score_90": float(np.mean(interval_score)),
        "median_sigma_total": float(np.median(sigma)),
    }


def full_metrics(frame: pd.DataFrame, balanced_groups: list[tuple[str, int]]) -> dict:
    result = {"n_rows": len(frame), "row_rmse": rmse(frame), "row_mae": mae(frame), **calibration(frame)}
    for group, minimum in balanced_groups:
        value, count = balanced_rmse(frame, group, minimum)
        result[f"{group}_balanced_rmse"] = value
        result[f"n_{group}_groups_used"] = count
    return result


def summarize_folds(frame: pd.DataFrame, metric_columns: list[str]) -> dict:
    per_fold = frame.groupby("fold_id")[metric_columns].mean(numeric_only=True)
    return {
        "n_folds": len(per_fold),
        **{f"{column}_fold_mean": float(per_fold[column].mean()) for column in metric_columns},
        **{f"{column}_fold_sd": float(per_fold[column].std(ddof=1)) for column in metric_columns},
    }


def order_frame(frame: pd.DataFrame, strategy: str, dataset: str) -> pd.DataFrame:
    identifier = "record_id" if dataset == "bigsoldb" else "pair_id"
    columns = {
        "uncertainty": "sigma_total",
        "tanimoto": "tanimoto_nn_distance",
        "mahalanobis": "mahalanobis_feature_centroid",
    }
    if strategy not in columns:
        raise KeyError(strategy)
    return frame.sort_values([columns[strategy], identifier], ascending=[True, True], kind="stable")


def selected_at(frame: pd.DataFrame, retention: float) -> pd.DataFrame:
    return frame.iloc[:int(np.floor(retention * len(frame)))].copy()


def support_bin(series: pd.Series) -> pd.Categorical:
    return pd.cut(
        series.astype(float), bins=[-0.5, 0.5, 4.5, 19.5, 99.5, np.inf],
        labels=["0", "1-4", "5-19", "20-99", ">=100"], ordered=True,
    )


def diagnostic_metrics(frame: pd.DataFrame, category: str) -> dict:
    result = {"n_rows": len(frame), "row_rmse": rmse(frame), "row_mae": mae(frame), **calibration(frame)}
    for column in ("fold_id", "solute_smiles", "ligand_id", "metal_pair", "condition_id"):
        if column in frame:
            result[f"n_{column}"] = int(frame[column].nunique())
    result["category"] = category
    return result


def grouped_diagnostics(frames: list[pd.DataFrame], assignments: list[str], output: Path) -> None:
    rows = []
    combined = pd.concat(frames, ignore_index=True)
    for assignment in assignments:
        for fold_id, group in combined.groupby("fold_id", sort=True, observed=True):
            for category, selected in group.groupby(assignment, sort=True, observed=True):
                rows.append({"diagnostic": assignment, "fold_id": fold_id, **diagnostic_metrics(selected, str(category))})
        for category, selected in combined.groupby(assignment, sort=True, observed=True):
            rows.append({"diagnostic": assignment, "fold_id": "pooled_oof", **diagnostic_metrics(selected, str(category))})
    pd.DataFrame(rows).to_csv(output, index=False)


def cluster_arrays(selected: pd.DataFrame, cluster: str, clusters: list[str]) -> tuple[np.ndarray, np.ndarray]:
    work = selected[["y_true", "y_pred", cluster]].copy()
    work["sse"] = np.square(errors(work))
    aggregated = work.groupby(cluster).agg(sse=("sse", "sum"), n=("sse", "size")).reindex(clusters, fill_value=0.0)
    return aggregated["sse"].to_numpy(dtype=float), aggregated["n"].to_numpy(dtype=float)


def cluster_bootstrap(
    selections: list[dict[str, list[pd.DataFrame]]], cluster: str, comparator: str,
    seed: int, replicates: int = BOOTSTRAP_REPLICATES,
) -> dict:
    """Paired fold-stratified cluster bootstrap of fold-mean RMSE differences."""
    rng = np.random.default_rng(seed)
    fold_clusters = [
        sorted(set().union(*(set(frame[cluster]) for strategy in fold.values() for frame in strategy)))
        for fold in selections
    ]
    u_arrays = [cluster_arrays(fold["uncertainty"][0], cluster, clusters) for fold, clusters in zip(selections, fold_clusters)]
    if comparator == "random":
        c_arrays = [
            tuple(np.stack(values, axis=1) for values in zip(*(cluster_arrays(frame, cluster, clusters) for frame in fold["random"])))
            for fold, clusters in zip(selections, fold_clusters)
        ]
    else:
        c_arrays = [cluster_arrays(fold[comparator][0], cluster, clusters) for fold, clusters in zip(selections, fold_clusters)]

    def contrast(weights: list[np.ndarray] | None = None) -> float:
        differences = []
        for fold_number, (u, c) in enumerate(zip(u_arrays, c_arrays)):
            if weights is None:
                u_sse, u_n = u[0].sum(), u[1].sum()
                c_sse, c_n = c[0].sum(axis=0), c[1].sum(axis=0)
            else:
                w = weights[fold_number]
                u_sse, u_n = np.dot(w, u[0]), np.dot(w, u[1])
                c_sse, c_n = np.tensordot(w, c[0], axes=1), np.tensordot(w, c[1], axes=1)
            if u_n <= 0 or np.any(c_n <= 0):
                return float("nan")
            differences.append(np.sqrt(u_sse / u_n) - float(np.mean(np.sqrt(c_sse / c_n))))
        return float(np.mean(differences))

    point = contrast()
    if not np.isfinite(point):
        raise ValueError("Undefined full-sample retention contrast")
    boot = np.empty(replicates, dtype=float)
    rejected = 0
    for replicate in range(replicates):
        for _ in range(10000):
            weights = [np.bincount(rng.integers(0, len(clusters), len(clusters)), minlength=len(clusters)) for clusters in fold_clusters]
            value = contrast(weights)
            if np.isfinite(value):
                boot[replicate] = value
                break
            rejected += 1
        else:
            raise RuntimeError("Could not draw a nonempty paired cluster-bootstrap sample")
    low, high = np.percentile(boot, [2.5, 97.5])
    return {
        "cluster": cluster, "comparator": comparator, "n_bootstrap": replicates,
        "n_resamples_rejected_empty": rejected,
        "point_uncertainty_minus_comparator_rmse": point,
        "ci95_low": float(low), "ci95_high": float(high),
        "interpretation": "negative favors uncertainty; conditional on fixed selected sets; empty paired resamples redrawn",
    }


def make_random_orders(frames: list[pd.DataFrame]) -> list[list[pd.DataFrame]]:
    orders = []
    for fold_number, frame in enumerate(frames):
        rng = np.random.default_rng(RANDOM_SEED_BASE + fold_number)
        orders.append([frame.iloc[rng.permutation(len(frame))] for _ in range(RANDOM_REPEATS)])
    return orders


def retention_analysis(
    frames: list[pd.DataFrame], dataset: str, cluster: str,
    balanced_groups: list[tuple[str, int]], comparators: list[str], out: Path,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    detailed_rows, bootstrap_rows = [], []
    half_summary = {}
    random_orders = make_random_orders(frames)
    for retention in RETENTIONS:
        fold_selections = []
        for fold_number, frame in enumerate(frames):
            strategies = ["uncertainty"] + [c for c in comparators if c != "random"]
            selections = {strategy: [selected_at(order_frame(frame, strategy, dataset), retention)] for strategy in strategies}
            selections["random"] = [selected_at(order, retention) for order in random_orders[fold_number]]
            fold_selections.append(selections)
            for strategy, replicates in selections.items():
                for replicate, selected in enumerate(replicates, start=1):
                    row = {
                        "fold_id": f"fold{fold_number}", "retention": retention, "strategy": strategy,
                        "random_replicate": 0 if strategy != "random" else replicate,
                        "n_retained": len(selected), "row_rmse": rmse(selected), "row_mae": mae(selected),
                        **calibration(selected),
                    }
                    for group, minimum in balanced_groups:
                        value, count = balanced_rmse(selected, group, minimum)
                        row[f"{group}_balanced_rmse"] = value
                        row[f"n_{group}_groups_used"] = count
                    detailed_rows.append(row)
        for comparator in comparators:
            cluster_names = [cluster, "ligand_id"] if dataset == "lanthanide" else [cluster]
            for cluster_name in cluster_names:
                if cluster_name == "ligand_id" and comparator != "random":
                    continue
                result = cluster_bootstrap(
                    fold_selections, cluster_name, comparator,
                    seed=BIGSOLDB_BOOTSTRAP_SEED + int(retention * 1000) + 100 * comparators.index(comparator) + (1 if cluster_name == "ligand_id" else 0),
                )
                bootstrap_rows.append({"retention": retention, **result})
        if retention == 0.5:
            half = pd.DataFrame(detailed_rows)
            half = half[half["retention"] == 0.5]
            for strategy, group in half.groupby("strategy", sort=True):
                metric_columns = ["row_rmse", "row_mae"] + [f"{group}_balanced_rmse" for group, _ in balanced_groups]
                half_summary[strategy] = summarize_folds(group, metric_columns)
    detailed = pd.DataFrame(detailed_rows)
    detailed.to_csv(out / "retention_fold_metrics.csv", index=False)
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(out / "retention_cluster_bootstrap.csv", index=False)
    return {"half_retention": half_summary, "bootstrap": bootstrap.to_dict(orient="records")}


def evaluate_bigsoldb(frames: list[pd.DataFrame], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    validate_prediction_indices("bigsoldb", frames)
    full_rows, baseline_rows = [], []
    for fold, frame in zip(FOLDS, frames):
        if not np.isfinite(frame[["y_pred", "sigma_total", "dummy_pred"]]).all().all() or not (frame["sigma_total"] > 0).all():
            raise ValueError(f"BigSolDB {fold}: invalid prediction or uncertainty")
        full_rows.append({"fold_id": fold, **full_metrics(frame, [("solute_smiles", 3), ("solvent_smiles", 3)])})
        baseline_rows.append({
            "fold_id": fold, "n_rows": len(frame), "row_rmse": rmse(frame, "dummy_pred"), "row_mae": mae(frame, "dummy_pred"),
            "solute_smiles_balanced_rmse": balanced_rmse(frame.assign(y_pred=frame["dummy_pred"]), "solute_smiles", 3)[0],
            "solvent_smiles_balanced_rmse": balanced_rmse(frame.assign(y_pred=frame["dummy_pred"]), "solvent_smiles", 3)[0],
            "n_unseen_solvents": int((~frame["solvent_seen_in_training"]).sum()),
        })
    full, baseline = pd.DataFrame(full_rows), pd.DataFrame(baseline_rows)
    full.to_csv(out / "full_metrics_by_fold.csv", index=False)
    baseline.to_csv(out / "dummy_baseline_by_fold.csv", index=False)
    retention = retention_analysis(
        frames, "bigsoldb", "solute_smiles", [("solute_smiles", 3), ("solvent_smiles", 3)],
        list(COMPARATORS), out,
    )
    grouped_diagnostics(frames, ["solvent_smiles", "training_solute_support_bin"], out / "conditional_diagnostics.csv")
    metrics = ["row_rmse", "row_mae", "solute_smiles_balanced_rmse", "solvent_smiles_balanced_rmse", "picp_90", "mean_interval_width_90", "median_interval_width_90", "interval_score_90"]
    write_json(out / "main_text_summary.json", {
        "status": "complete", "dataset": "BigSolDB v2.2", "n_rows": sum(len(frame) for frame in frames),
        "full_model": summarize_folds(full, metrics),
        "training_solvent_mean_dummy": summarize_folds(baseline, ["row_rmse", "row_mae", "solute_smiles_balanced_rmse", "solvent_smiles_balanced_rmse"]),
        **retention,
        "bootstrap_scope": "solute-cluster, stratified by outer fold, 2000 replicates; fixed selected sets; no retraining variability",
        "balanced_metric_rule": "per-group RMSE then unweighted mean; minimum three retained rows per solute or solvent",
        "canonical_outputs_modified": False,
    })
    print("BigSolDB fixed-prediction evaluation complete", flush=True)


def pair_metrics(selected: pd.DataFrame) -> pd.DataFrame:
    work = selected[["metal_pair", "y_true", "y_pred"]].copy()
    work["sse"] = np.square(work["y_true"].to_numpy(dtype=float) - work["y_pred"].to_numpy(dtype=float))
    grouped = work.groupby("metal_pair", sort=True).agg(sse=("sse", "sum"), n=("sse", "size"))
    grouped = grouped[grouped["n"] >= MINIMUM_PAIR_ROWS].copy()
    grouped["rmse"] = np.sqrt(grouped["sse"] / grouped["n"])
    return grouped


def make_selections(frames: list[pd.DataFrame]) -> dict[float, list[dict[str, list[pd.DataFrame]]]]:
    random_orders = make_random_orders(frames)
    selections = {}
    for retention in RETENTIONS:
        fold_selections = []
        for fold_number, frame in enumerate(frames):
            current = {strategy: [selected_at(order_frame(frame, strategy, "lanthanide"), retention)] for strategy in STRATEGIES if strategy != "random"}
            current["random"] = [selected_at(order, retention) for order in random_orders[fold_number]]
            fold_selections.append(current)
        selections[retention] = fold_selections
    return selections


def build_pair_tables(selections: dict, frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows, eligibility_rows = [], []
    for retention, fold_selections in selections.items():
        for fold_number, (fold, frame) in enumerate(zip(fold_selections, frames)):
            full_pair_count = frame["metal_pair"].nunique()
            for strategy, replicates in fold.items():
                for replicate_number, selected in enumerate(replicates, start=1):
                    replicate = 0 if strategy != "random" else replicate_number
                    metrics = pair_metrics(selected)
                    eligibility_rows.append({
                        "retention": retention, "fold_id": f"fold{fold_number}", "strategy": strategy,
                        "random_replicate": replicate, "n_retained_rows": len(selected),
                        "n_eligible_metal_pairs": len(metrics), "n_full_metal_pairs": full_pair_count,
                        "mop_rmse": float(metrics["rmse"].mean()),
                    })
                    for metal_pair, row in metrics.iterrows():
                        pair_rows.append({
                            "retention": retention, "fold_id": f"fold{fold_number}", "strategy": strategy,
                            "random_replicate": replicate, "metal_pair": metal_pair,
                            "sse": row["sse"], "n": int(row["n"]), "rmse": row["rmse"],
                        })
    return pd.DataFrame(pair_rows), pd.DataFrame(eligibility_rows)


def pair_array(selected: pd.DataFrame, clusters: list[str]) -> np.ndarray:
    return pair_metrics(selected)["rmse"].reindex(clusters).to_numpy(dtype=float)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & (weights > 0)
    if not np.any(valid):
        return float("nan")
    return float(np.dot(weights[valid], values[valid]) / weights[valid].sum())


def weighted_contrast(u_arrays: list[np.ndarray], c_arrays: list[np.ndarray], weights: list[np.ndarray]) -> float:
    differences = []
    for u, c, w in zip(u_arrays, c_arrays, weights):
        u_value = weighted_mean(u, w)
        if c.ndim == 1:
            c_value = weighted_mean(c, w)
        else:
            c_value = float(np.mean([weighted_mean(c[:, j], w) for j in range(c.shape[1])]))
        if not np.isfinite(u_value) or not np.isfinite(c_value):
            return float("nan")
        differences.append(u_value - c_value)
    return float(np.mean(differences))


def pair_win_fraction(fold_selections: list[dict], fold_clusters: list[list[str]], comparator: str) -> tuple[float, int]:
    wins, total = 0, 0
    for fold, clusters in zip(fold_selections, fold_clusters):
        u = pair_array(fold["uncertainty"][0], clusters)
        if comparator == "random":
            c = np.nanmean(np.stack([pair_array(frame, clusters) for frame in fold["random"]], axis=1), axis=1)
        else:
            c = pair_array(fold[comparator][0], clusters)
        both = np.isfinite(u) & np.isfinite(c)
        wins += int(np.sum(u[both] < c[both]))
        total += int(np.sum(both))
    return (float(wins / total) if total else float("nan"), total)


def mop_bootstrap_contrast(fold_selections: list[dict], frames: list[pd.DataFrame], comparator: str, retention: float) -> dict:
    fold_clusters = [sorted(frame["metal_pair"].unique()) for frame in frames]
    u_arrays = [pair_array(fold["uncertainty"][0], clusters) for fold, clusters in zip(fold_selections, fold_clusters)]
    if comparator == "random":
        c_arrays = [np.stack([pair_array(frame, clusters) for frame in fold["random"]], axis=1) for fold, clusters in zip(fold_selections, fold_clusters)]
    else:
        c_arrays = [pair_array(fold[comparator][0], clusters) for fold, clusters in zip(fold_selections, fold_clusters)]
    weights = [np.ones(len(u), dtype=float) for u in u_arrays]
    point = weighted_contrast(u_arrays, c_arrays, weights)
    if not np.isfinite(point):
        raise ValueError("Undefined full-sample MoP contrast")
    rng = np.random.default_rng(LANTHANIDE_MOP_BOOTSTRAP_SEED + int(retention * 1000) + 100 * COMPARATORS.index(comparator))
    boot = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    rejected = 0
    for replicate in range(BOOTSTRAP_REPLICATES):
        for _ in range(10000):
            resample_weights = [np.bincount(rng.integers(0, len(clusters), len(clusters)), minlength=len(clusters)).astype(float) for clusters in fold_clusters]
            value = weighted_contrast(u_arrays, c_arrays, resample_weights)
            if np.isfinite(value):
                boot[replicate] = value
                break
            rejected += 1
        else:
            raise RuntimeError("Could not draw a valid MoP cluster-bootstrap sample")
    low, high = np.percentile(boot, [2.5, 97.5])
    wins, paired_units = pair_win_fraction(fold_selections, fold_clusters, comparator)
    return {
        "retention": retention, "metric": "MoP RMSE", "cluster": "metal_pair", "comparator": comparator,
        "n_bootstrap": BOOTSTRAP_REPLICATES, "n_resamples_rejected_empty": rejected,
        "point_uncertainty_minus_comparator_mop_rmse": point,
        "ci95_low": float(low), "ci95_high": float(high),
        "pair_win_fraction": wins, "n_paired_fold_pair_units": paired_units,
        "interpretation": "negative favors uncertainty; own eligible-pair populations; fixed selected sets; no retraining variability",
    }


def common_eligibility_half(fold_selections: list[dict], frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for fold_number, (fold, frame) in enumerate(zip(fold_selections, frames)):
        eligible_sets = [set(pair_metrics(selected).index) for strategy in STRATEGIES for selected in fold[strategy]]
        common = set(frame["metal_pair"].unique()).intersection(*eligible_sets)
        for strategy in STRATEGIES:
            values = [float(pair_metrics(selected).loc[sorted(common), "rmse"].mean()) for selected in fold[strategy]]
            rows.append({
                "fold_id": f"fold{fold_number}", "strategy": strategy,
                "n_common_metal_pairs": len(common), "n_full_metal_pairs": frame["metal_pair"].nunique(),
                "common_mop_rmse": float(np.mean(values)), "n_random_replicates": len(values) if strategy == "random" else 0,
            })
    return pd.DataFrame(rows)


def evaluate_lanthanide(frames: list[pd.DataFrame], out: Path) -> None:
    validate_prediction_indices("lanthanide", frames)
    main_out = out / "main_text_evaluation"
    distance_out = out / "distance_retention"
    mop_out = out / "mop_inference"
    for directory in (main_out, distance_out, mop_out):
        directory.mkdir(parents=True, exist_ok=True)
    full_rows, baseline_rows = [], []
    for fold, frame in zip(FOLDS, frames):
        if not np.isfinite(frame[["y_pred", "sigma_total", "dummy_pred", "tanimoto_nn_distance", "mahalanobis_feature_centroid"]]).all().all() or not (frame["sigma_total"] > 0).all():
            raise ValueError(f"Lanthanide {fold}: invalid prediction, uncertainty, or distance")
        full_rows.append({"fold_id": fold, **full_metrics(frame, [("metal_pair", 3), ("condition_id", 1), ("ligand_id", 3)])})
        baseline_rows.append({
            "fold_id": fold, "n_rows": len(frame), "row_rmse": rmse(frame, "dummy_pred"), "row_mae": mae(frame, "dummy_pred"),
            "metal_pair_balanced_rmse": balanced_rmse(frame.assign(y_pred=frame["dummy_pred"]), "metal_pair", 3)[0],
            "condition_id_balanced_rmse": balanced_rmse(frame.assign(y_pred=frame["dummy_pred"]), "condition_id", 1)[0],
            "n_unseen_metal_pair_rows": int((~frame["metal_pair_seen_in_training"]).sum()),
        })
    full, baseline = pd.DataFrame(full_rows), pd.DataFrame(baseline_rows)
    full.to_csv(main_out / "full_metrics_by_fold.csv", index=False)
    baseline.to_csv(main_out / "dummy_baseline_by_fold.csv", index=False)
    retention = retention_analysis(
        frames, "lanthanide", "condition_id", [("metal_pair", 3), ("condition_id", 1), ("ligand_id", 3)],
        ["random"], main_out,
    )
    grouped_diagnostics(
        frames, ["metal_pair_training_support_bin", "condition_familiarity", "diluent_familiarity", "diluent_training_support_bin", "metal_pair"],
        main_out / "conditional_diagnostics.csv",
    )
    metrics = ["row_rmse", "row_mae", "metal_pair_balanced_rmse", "condition_id_balanced_rmse", "ligand_id_balanced_rmse", "picp_90", "mean_interval_width_90", "median_interval_width_90", "interval_score_90"]
    write_json(main_out / "main_text_summary.json", {
        "status": "complete", "dataset": "lanthanide log separation factor", "n_rows": sum(len(frame) for frame in frames),
        "full_model": summarize_folds(full, metrics),
        "training_metal_pair_mean_dummy": summarize_folds(baseline, ["row_rmse", "row_mae", "metal_pair_balanced_rmse", "condition_id_balanced_rmse"]),
        **retention,
        "bootstrap_scope": "condition-cell primary and ligand sensitivity, stratified by outer fold, 2000 replicates; fixed selected sets; no retraining variability",
        "condition_signature": "all 21 condition fields; ligand and reference excluded; exact matching only",
        "diluent_signature": "two solvent volume ratios and all 16 solvent-property fields; exact matching only",
        "balanced_metric_rule": "per-group RMSE then unweighted mean; minimum three rows for metal pairs and ligands; all condition cells",
        "canonical_outputs_modified": False,
    })

    distance_result = retention_analysis(
        frames, "lanthanide", "condition_id", [("metal_pair", 3), ("condition_id", 1), ("ligand_id", 3)],
        list(COMPARATORS), distance_out,
    )
    score_rows = [{
        "fold_id": fold,
        "tanimoto_min": float(frame["tanimoto_nn_distance"].min()), "tanimoto_median": float(frame["tanimoto_nn_distance"].median()),
        "tanimoto_max": float(frame["tanimoto_nn_distance"].max()), "mahalanobis_min": float(frame["mahalanobis_feature_centroid"].min()),
        "mahalanobis_median": float(frame["mahalanobis_feature_centroid"].median()), "mahalanobis_max": float(frame["mahalanobis_feature_centroid"].max()),
    } for fold, frame in zip(FOLDS, frames)]
    pd.DataFrame(score_rows).to_csv(distance_out / "distance_score_ranges.csv", index=False)
    write_json(distance_out / "distance_retention_summary.json", {
        "status": "complete", "dataset": "lanthanide log separation factor", "model_training_performed": False,
        "tanimoto": "1 minus maximum ligand-ECFP Tanimoto similarity to any outer-training ligand",
        "mahalanobis": "training-centroid Ledoit-Wolf distance in standardized joint ligand-RDKit + 21 condition + both-metal descriptor block; ECFP excluded",
        **distance_result,
    })

    selections = make_selections(frames)
    pair_table, eligibility = build_pair_tables(selections, frames)
    pair_table.to_csv(mop_out / "retention_pair_metrics.csv.gz", index=False, compression="gzip")
    eligibility.to_csv(mop_out / "retention_strategy_fold_metrics.csv", index=False)
    eligibility_summary = eligibility.groupby(["retention", "strategy"]).agg(
        mean_retained_rows=("n_retained_rows", "mean"), mean_eligible_metal_pairs=("n_eligible_metal_pairs", "mean"),
        sd_eligible_metal_pairs=("n_eligible_metal_pairs", "std"), min_eligible_metal_pairs=("n_eligible_metal_pairs", "min"),
        max_eligible_metal_pairs=("n_eligible_metal_pairs", "max"), mean_mop_rmse=("mop_rmse", "mean"), sd_mop_rmse=("mop_rmse", "std"),
    ).reset_index()
    eligibility_summary.to_csv(mop_out / "eligibility_summary.csv", index=False)
    bootstrap_rows = [mop_bootstrap_contrast(selections[retention], frames, comparator, retention) for retention in RETENTIONS for comparator in COMPARATORS]
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(mop_out / "mop_cluster_bootstrap.csv", index=False)
    common = common_eligibility_half(selections[0.5], frames)
    common.to_csv(mop_out / "common_eligibility_by_fold.csv", index=False)
    common.groupby("strategy")["common_mop_rmse"].agg(["mean", "std"]).reset_index().to_csv(mop_out / "common_eligibility_summary.csv", index=False)
    write_json(mop_out / "mop_inference_summary.json", {
        "status": "complete", "dataset": "lanthanide log separation factor",
        "metric": "per-metal-pair RMSE then unweighted mean; minimum three retained rows per pair",
        "bootstrap": bootstrap_rows, "common_eligibility_half_retention": common.to_dict(orient="records"),
        "bootstrap_seed": LANTHANIDE_MOP_BOOTSTRAP_SEED, "random_repeats": RANDOM_REPEATS,
        "model_training_performed": False,
    })
    print("Lanthanide fixed-prediction evaluation complete", flush=True)


def evaluate_all(output: Path, dataset: str = "all") -> None:
    if dataset in ("bigsoldb", "all"):
        evaluate_bigsoldb(load_fixed_predictions("bigsoldb"), output / "bigsoldb")
    if dataset in ("lanthanide", "all"):
        evaluate_lanthanide(load_fixed_predictions("lanthanide"), output / "lanthanide")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["bigsoldb", "lanthanide", "all"], default="all", nargs="?")
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "results/external_validation_reproduction")
    args = parser.parse_args()
    evaluate_all(args.output, args.dataset)
    print(f"Outputs: {args.output}")


if __name__ == "__main__":
    main()
