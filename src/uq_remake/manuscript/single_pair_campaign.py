"""Analysis task 10: the single-pair prospective demonstration.

For the three most frequent held-out species pairs -- Cu:2.0--Ni:2.0 (1,878
rows / 1,230 ligands, the primary demo pair: a real hydrometallurgical
separation at the Irving-Williams maximum, ties Sec. 2.7 back to Sec. 2.5),
Cu:2.0--Zn:2.0 (2nd), and Ni:2.0--Zn:2.0 (3rd, included so the manuscript can
state whether Cu/Ni was cherry-picked) -- reports, per fold:

  10.1 Campaign setup: candidate-ligand count, the distribution of true
       Delta log K, and how many candidates clear the event threshold. The
       threshold is read from `downstream.pair_acquisition` /
       `downstream.protocol.PAIR_ACQ_THRESHOLD` (not invented here), via the
       same `eligible_tasks` / `pair_event_counts` functions the production
       acquisition pipeline uses.
  10.2 event precision@k and true hit@k, k in {1, 5, 10, 25}, for five
       ranking strategies: predicted selectivity (y_pred), threshold margin
       (the repo's "expected_improvement" strategy -- (predicted - tau)/sigma),
       LCB at 1/2/3 sigma (the same z values `pair_acquisition.py` uses),
       fingerprint nearest-neighbour distance (ascending `tanimoto_nn_distance`
       -- a similarity-gated baseline that ignores the model's own prediction),
       and random (200 repeats, seed 0). Every ranking function is imported
       from `downstream.pair_acquisition`, not reimplemented.
  10.3 The within-pair retention curve for this pair alone (all
       `CANDIDATE_SCORES`, canonical grid), reusing
       `within_pair_ordering`'s per-(pair, fold)-cell retention exactly --
       here there is exactly one pair, so each fold is its own cell.
  10.4 A cross-pair comparison flagging whether Cu/Ni is unusually favourable
       relative to Cu/Zn and Ni/Zn (it must not be silently cherry-picked).

Emits `manuscripts/derived/tables/single_pair_campaign.csv`.

    python -m uq_remake.manuscript.single_pair_campaign
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from uq_remake.downstream.pair_acquisition import (
    eligible_tasks,
    observed_event,
    pair_event_counts,
    random_rank_rows,
    rank_rows,
    resolved_event_thresholds,
)
from uq_remake.downstream.protocol import FOLDS, PAIR_ACQ_RANDOM_SEED, PAIR_COLUMN, RETENTION_FRACTIONS
from uq_remake.manuscript.manuscript_common import CANDIDATE_SCORES, FOLD_SCORES_DIR, DERIVED_TABLES, MIN_CELL_N, fold_mean_sd_row, load_all_folds, write_provenance
from uq_remake.manuscript.within_pair_ordering import _cell_retained_rmse

OUT = DERIVED_TABLES / "single_pair_campaign.csv"

PAIRS_IN_FREQUENCY_ORDER = ("Cu:2.0--Ni:2.0", "Cu:2.0--Zn:2.0", "Ni:2.0--Zn:2.0")
PRIMARY_PAIR = PAIRS_IN_FREQUENCY_ORDER[0]
TOPK = (1, 5, 10, 25)
RANDOM_REPEATS = 200
STRATEGIES_RANKED = ("y_pred", "expected_improvement", "lcb_1sigma", "lcb_2sigma", "lcb_3sigma")
TANIMOTO_STRATEGY = "tanimoto_nn_similarity_gated"
RANDOM_STRATEGY = "random"


def _nan() -> float:
    return float("nan")


def _topk_metrics(ranked: pd.DataFrame, task: str, orientation: str, ks: tuple[int, ...] = TOPK) -> dict:
    """event_precision@k and true hit@k only -- Task 10.2 does not ask for regret/rank-corr."""
    y = pd.to_numeric(ranked["y_true"], errors="coerce").to_numpy(dtype=float)
    event = observed_event(y, task, orientation)
    n = int(len(ranked))
    out = {}
    for k in ks:
        if n < k:
            out[f"event_precision_at_{k}"] = _nan()
            out[f"hit_at_{k}"] = _nan()
            continue
        hits = int(event[:k].sum())
        out[f"event_precision_at_{k}"] = hits / k
        out[f"hit_at_{k}"] = float(hits > 0)
    return out


def _tanimoto_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Similarity-gated baseline: nearest-fingerprint-first, ignores y_pred/sigma entirely."""
    work = df.copy()
    dist = pd.to_numeric(work["tanimoto_nn_distance"], errors="coerce")
    work["_score"] = -dist  # higher score (least negative distance) ranked first, matching rank_rows' descending convention
    keep = work["_score"].notna()
    work = work.loc[keep].copy()
    work["_tie"] = work["record_id"].astype(str)
    work = work.sort_values(["_score", "_tie"], ascending=[False, True], kind="mergesort")
    work["_rank"] = np.arange(1, len(work) + 1)
    return work


def _random_topk_averaged(df: pd.DataFrame, task: str, orientation: str, rng: np.random.Generator, n_repeats: int = RANDOM_REPEATS) -> dict:
    acc = [_topk_metrics(random_rank_rows(df, rng), task, orientation) for _ in range(int(n_repeats))]
    out = {}
    for key in acc[0]:
        vals = np.asarray([d[key] for d in acc], dtype=float)
        out[key] = float(np.nanmean(vals)) if np.isfinite(vals).any() else _nan()
    return out


# ------------------------------------------------------------------ 10.1 ---
def campaign_setup_rows(pair: str, fold_frames: dict[str, pd.DataFrame]) -> tuple[list[dict], list[dict]]:
    fold_level: list[dict] = []
    for fold, df in fold_frames.items():
        part = df[df[PAIR_COLUMN] == pair]
        y = pd.to_numeric(part["y_true"], errors="coerce")
        counts = pair_event_counts(part)
        high_tau, low_tau = resolved_event_thresholds()
        rec = {
            "outer_fold_id": fold,
            "pair": pair,
            "n_rows": int(len(part)),
            "n_candidate_ligands": int(part["ligand_smiles"].nunique()),
            "mean_true_dlogK": float(y.mean()) if len(y) else _nan(),
            "median_true_dlogK": float(y.median()) if len(y) else _nan(),
            "sd_true_dlogK": float(y.std(ddof=1)) if y.notna().sum() > 1 else _nan(),
            "min_true_dlogK": float(y.min()) if len(y) else _nan(),
            "max_true_dlogK": float(y.max()) if len(y) else _nan(),
            "p10_true_dlogK": float(y.quantile(0.10)) if len(y) else _nan(),
            "p90_true_dlogK": float(y.quantile(0.90)) if len(y) else _nan(),
            "event_threshold_high": high_tau,
            "event_threshold_low": low_tau,
        }
        if len(counts):
            c = counts.iloc[0]
            rec.update(
                {
                    "n_event_high_canonical": int(c.n_high_canonical),
                    "n_event_high_reversed": int(c.n_high_reversed),
                    "n_event_low": int(c.n_low),
                }
            )
        else:
            rec.update({"n_event_high_canonical": 0, "n_event_high_reversed": 0, "n_event_low": 0})
        fold_level.append(rec)

    metric_names = [k for k in fold_level[0] if k not in ("outer_fold_id", "pair")]
    summary_rows = []
    for metric in metric_names:
        vals = [rec[metric] for rec in fold_level]
        if metric in ("event_threshold_high", "event_threshold_low"):
            summary_rows.append({"pair": pair, "metric_name": metric, "fold_mean": vals[0], "fold_sd": 0.0, "n_valid_folds": len(vals)})
            continue
        fm = fold_mean_sd_row(vals)
        summary_rows.append({"pair": pair, "metric_name": metric, "fold_mean": fm["fold_mean"], "fold_sd": fm["fold_sd"], "n_valid_folds": fm["n_valid_folds"]})
    return fold_level, summary_rows


# ------------------------------------------------------------------ 10.2 ---
def topk_rows(pair: str, fold_frames: dict[str, pd.DataFrame]) -> list[dict]:
    all_metrics: list[dict] = []
    for i, fold in enumerate(FOLDS):
        df = fold_frames[fold]
        part = df[df[PAIR_COLUMN] == pair]
        if len(part) == 0:
            continue
        for task_def in eligible_tasks(part):
            task, orientation = task_def["task"], task_def["orientation"]
            per_strategy: dict[str, dict] = {}
            for strategy in STRATEGIES_RANKED:
                ranked = rank_rows(part, strategy, task, orientation)
                per_strategy[strategy] = _topk_metrics(ranked, task, orientation)
            per_strategy[TANIMOTO_STRATEGY] = _topk_metrics(_tanimoto_rank(part), task, orientation)
            rng = np.random.default_rng(int(PAIR_ACQ_RANDOM_SEED) + i * 1009)
            per_strategy[RANDOM_STRATEGY] = _random_topk_averaged(part, task, orientation, rng)
            for strategy, metrics in per_strategy.items():
                for metric_name, value in metrics.items():
                    all_metrics.append(
                        {
                            "outer_fold_id": fold,
                            "pair": pair,
                            "task": task,
                            "orientation": orientation,
                            "strategy": strategy,
                            "metric_name": metric_name,
                            "value": value,
                            "n_event": int(task_def["n_event"]),
                            "n_samples": int(len(part)),
                        }
                    )
    fold_df = pd.DataFrame(all_metrics)
    if fold_df.empty:
        return []
    summary = []
    keys = ["pair", "task", "orientation", "strategy", "metric_name"]
    for key, g in fold_df.groupby(keys, dropna=False):
        rec = dict(zip(keys, key))
        fm = fold_mean_sd_row(g["value"].tolist())
        rec.update({"fold_mean": fm["fold_mean"], "fold_sd": fm["fold_sd"], "n_valid_folds": fm["n_valid_folds"]})
        summary.append(rec)
    return summary


# ------------------------------------------------------------------ 10.3 ---
def retention_rows(pair: str, fold_frames: dict[str, pd.DataFrame]) -> list[dict]:
    per_score: dict[str, dict[str, dict[float, float]]] = {}
    for fold, df in fold_frames.items():
        part = df[df[PAIR_COLUMN] == pair]
        if len(part) < MIN_CELL_N:
            continue
        for score in CANDIDATE_SCORES:
            curve = _cell_retained_rmse(part, score, higher_is_risk=True)
            per_score.setdefault(score, {})[fold] = curve

    rows = []
    for score, by_fold in per_score.items():
        for ret in RETENTION_FRACTIONS:
            fold_vals = [by_fold[fold].get(ret, _nan()) for fold in FOLDS if fold in by_fold]
            fm = fold_mean_sd_row(fold_vals)
            rows.append(
                {
                    "pair": pair,
                    "score_name": score,
                    "retention_fraction": ret,
                    "metric_name": "mop_rmse_within_pair",
                    "fold_mean": fm["fold_mean"],
                    "fold_sd": fm["fold_sd"],
                    "n_valid_folds": fm["n_valid_folds"],
                }
            )
    return rows


def main() -> dict:
    fold_frames = load_all_folds()

    campaign_fold_level: list[dict] = []
    campaign_summary: list[dict] = []
    topk_summary: list[dict] = []
    retention_summary: list[dict] = []

    for pair in PAIRS_IN_FREQUENCY_ORDER:
        fl, summ = campaign_setup_rows(pair, fold_frames)
        campaign_fold_level.extend(fl)
        campaign_summary.extend(summ)
        topk_summary.extend(topk_rows(pair, fold_frames))
        retention_summary.extend(retention_rows(pair, fold_frames))

    campaign_df = pd.DataFrame(campaign_summary).assign(table_section="campaign_setup")
    campaign_fold_df = pd.DataFrame(campaign_fold_level).assign(table_section="campaign_setup_fold_level")
    topk_df = pd.DataFrame(topk_summary).assign(table_section="topk_metrics")
    retention_df = pd.DataFrame(retention_summary).assign(table_section="retention_curve")

    # 10.4 cross-pair comparison: is the primary demo pair unusually favourable?
    comparison_rows = []
    prec5 = topk_df[(topk_df.metric_name == "event_precision_at_5") & (topk_df.strategy == "lcb_2sigma") & (topk_df.task == "high_selectivity") & (topk_df.orientation == "canonical")]
    ret50_sigma = retention_df[(retention_df.score_name == "sigma_total") & np.isclose(retention_df.retention_fraction, 0.5)]
    for pair in PAIRS_IN_FREQUENCY_ORDER:
        p5 = prec5[prec5.pair == pair]["fold_mean"]
        r50 = ret50_sigma[ret50_sigma.pair == pair]["fold_mean"]
        comparison_rows.append(
            {
                "pair": pair,
                "event_precision_at_5_lcb2sigma_high_canonical": float(p5.iloc[0]) if len(p5) else _nan(),
                "mop_rmse_at_50pct_sigma_total": float(r50.iloc[0]) if len(r50) else _nan(),
                "is_primary_demo_pair": pair == PRIMARY_PAIR,
            }
        )
    comp_df = pd.DataFrame(comparison_rows)
    if comp_df["event_precision_at_5_lcb2sigma_high_canonical"].notna().all():
        primary_prec = float(comp_df.loc[comp_df.pair == PRIMARY_PAIR, "event_precision_at_5_lcb2sigma_high_canonical"].iloc[0])
        other_prec = comp_df.loc[comp_df.pair != PRIMARY_PAIR, "event_precision_at_5_lcb2sigma_high_canonical"]
        cherry_picked_precision = bool(primary_prec > other_prec.max() + 1e-9)
    else:
        cherry_picked_precision = None
    if comp_df["mop_rmse_at_50pct_sigma_total"].notna().all():
        primary_rmse = float(comp_df.loc[comp_df.pair == PRIMARY_PAIR, "mop_rmse_at_50pct_sigma_total"].iloc[0])
        other_rmse = comp_df.loc[comp_df.pair != PRIMARY_PAIR, "mop_rmse_at_50pct_sigma_total"]
        cherry_picked_rmse = bool(primary_rmse < other_rmse.min() - 1e-9)
    else:
        cherry_picked_rmse = None
    comp_df["table_section"] = "cross_pair_comparison"
    comp_df["cu_ni_strictly_best_precision_at_5"] = cherry_picked_precision
    comp_df["cu_ni_strictly_lowest_rmse_at_50pct"] = cherry_picked_rmse
    print(
        f"single_pair_campaign: Cu/Ni strictly best on precision@5={cherry_picked_precision}, "
        f"strictly lowest RMSE@50%={cherry_picked_rmse} (both True would mean the demo pair was "
        "the most favourable of the three -- report this, do not hide it)",
        flush=True,
    )

    out = pd.concat([campaign_df, campaign_fold_df, topk_df, retention_df, comp_df], ignore_index=True)
    out["source"] = "results/downstream_cv/ad_retention/fold_scores/DS2_RDKIT_MLPENS5_HETERO/fold{0..4}.csv"
    DERIVED_TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    write_provenance(
        OUT,
        module="single_pair_campaign.main",
        inputs=[FOLD_SCORES_DIR / f"{f}.csv" for f in FOLDS],
        notes={
            "pairs": list(PAIRS_IN_FREQUENCY_ORDER),
            "primary_demo_pair": PRIMARY_PAIR,
            "topk": list(TOPK),
            "random_repeats": RANDOM_REPEATS,
            "event_threshold_source": "downstream.pair_acquisition.resolved_event_thresholds (PAIR_ACQ_THRESHOLD)",
        },
    )
    print(f"wrote {OUT} ({len(out)} rows)", flush=True)
    return {"table": out}


if __name__ == "__main__":
    main()
