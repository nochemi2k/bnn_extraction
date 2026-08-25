"""Workhorse selection from fold-level metrics only. No pooled OOF values.

The manuscript operational rule is ``select_workhorse_accuracy_calibration``:
all-pair interval-usability screen, then the RMSE–PICP–ACE Pareto member
closest to the ideal point after min-max normalization.

``select_workhorse_lexicographic`` is a generic fold-mean ranking helper used by
unit tests of retired selection keys. The active pipeline does not call it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from uq_remake import REMAKE_ROOT
from uq_remake.metrics.cv_protocol import (
    OOFTerminologyError,
    archived_model_blocked,
    assert_no_oof_terminology,
    contains_oof_terminology,
)

RULE_PATH = REMAKE_ROOT / "results" / "model_audit" / "workhorse_selection_rule.json"
MSRC = REMAKE_ROOT / "results" / "manuscript_final"
CHEMPROP_DS2 = "DS2_GRAPH_DMPNN_EVID"
POINT_MODEL_TOKENS = ("_MLP_POINT", "_XGB_POINT", "_DMPNN_POINT")

# Operational calibration-usability screen (not a hypothesis test).
PICP_LOWER = 0.85
PICP_UPPER = 0.95
ACE_UPPER = 0.10
N_VALID_REQUIRED = 5
REQUIRE_COMPLETE_NATIVE_UQ = True
PICP_TARGET = 0.90

DISPLAY_NAMES = {
    "DS2_RDKIT_MLP_POINT": "RDKit MLP",
    "DS2_RDKIT_XGB_POINT": "RDKit XGBoost",
    "DS2_GRAPH_DMPNN_POINT": "Chemprop point",
    "DS2_RDKIT_MLP_EVID": "Evidential MLP",
    "DS2_RDKIT_MLPENS5_EVID": "Evidential MLP ensemble",
    "DS2_RDKIT_MLP_HETERO": "Heteroscedastic MLP",
    "DS2_RDKIT_MLPENS5_HETERO": "Heteroscedastic MLP ensemble",
    "DS2_RDKIT_MLP_MCDROPOUT": "MC-dropout MLP",
    "DS2_RDKIT_MLPENS5_MCDROPOUT": "MC-dropout MLP ensemble",
    "DS2_RDKIT_MLP_BAYES_FULL": "Full variational BNN",
    "DS2_RDKIT_MLPENS5_BAYES_FULL": "Full variational BNN ensemble",
    "DS2_RDKIT_SNGP": "SNGP",
    "DS2_RDKIT_SNGPENS5": "SNGP ensemble",
    "DS2_RDKIT_GPR": "Exact GPR",
    "DS2_RDKIT_GPR_SPARSE": "Sparse variational GPR",
    "DS2_GRAPH_DMPNN_EVID": "Chemprop evidential",
}

# Lower is simpler; used only after Pareto-distance ties.
OPERATIONAL_SIMPLICITY = {
    "DS2_RDKIT_MLP_EVID": 0,
    "DS2_RDKIT_MLP_HETERO": 1,
    "DS2_RDKIT_MLP_MCDROPOUT": 2,
    "DS2_RDKIT_SNGP": 3,
    "DS2_RDKIT_GPR_SPARSE": 4,
    "DS2_RDKIT_MLPENS5_EVID": 5,
    "DS2_RDKIT_MLPENS5_HETERO": 6,
    "DS2_RDKIT_MLPENS5_MCDROPOUT": 7,
    "DS2_RDKIT_MLP_BAYES_FULL": 8,
    "DS2_RDKIT_SNGPENS5": 9,
    "DS2_RDKIT_MLPENS5_BAYES_FULL": 10,
    "DS2_GRAPH_DMPNN_EVID": 11,
}

ROW_LEVEL_METRIC_NAMES = {
    "rmse",
    "r2",
    "row_rmse",
    "row_r2",
}


def load_rule() -> dict:
    obj = json.loads(RULE_PATH.read_text(encoding="utf-8"))
    assert_no_oof_terminology(json.dumps(obj), context="workhorse_rule")
    if obj.get("pair_balanced_global_primary"):
        raise ValueError("pair-balanced global metrics must not be primary")
    if obj.get("composite_calibration_score_used"):
        raise ValueError("composite calibration score is forbidden")
    if obj.get("row_level_metrics_used_for_selection"):
        raise ValueError("row-level metrics must not enter workhorse selection")
    return obj


def _lookup(summary: pd.DataFrame, model_id: str, scope: str, metric: str) -> float:
    hit = summary[
        (summary["model_id"] == model_id)
        & (summary["scope"] == scope)
        & (summary["metric_name"] == metric)
    ]
    if hit.empty:
        return float("nan")
    if "pooled_oof_value" in hit.columns or "oof_value" in hit.columns or "concatenated_cv_value" in hit.columns:
        raise OOFTerminologyError("workhorse input contains pooled OOF columns")
    return float(hit.iloc[0]["fold_mean"])


def _lookup_full(summary: pd.DataFrame, model_id: str, scope: str, metric: str) -> dict:
    hit = summary[
        (summary["model_id"] == model_id)
        & (summary["scope"] == scope)
        & (summary["metric_name"] == metric)
    ]
    if hit.empty:
        return {"fold_mean": float("nan"), "fold_sd": float("nan"), "n_valid_folds": 0, "n_folds": 0}
    if any(c in hit.columns for c in ("pooled_oof_value", "oof_value", "concatenated_cv_value")):
        raise OOFTerminologyError("workhorse input contains pooled OOF columns")
    r = hit.iloc[0]
    mean = float(r["fold_mean"]) if pd.notna(r["fold_mean"]) else float("nan")
    sd = float(r["fold_sd"]) if "fold_sd" in hit.columns and pd.notna(r["fold_sd"]) else float("nan")
    n_valid = int(r["n_valid_folds"]) if "n_valid_folds" in hit.columns and pd.notna(r["n_valid_folds"]) else 0
    n_folds = int(r["n_folds"]) if "n_folds" in hit.columns and pd.notna(r["n_folds"]) else 0
    return {"fold_mean": mean, "fold_sd": sd, "n_valid_folds": n_valid, "n_folds": n_folds}


def _is_point_model(model_id: str) -> bool:
    return any(tok in str(model_id) for tok in POINT_MODEL_TOKENS)


def _assert_no_oof_columns(summary: pd.DataFrame) -> None:
    for col in summary.columns:
        if contains_oof_terminology(col):
            raise OOFTerminologyError(f"forbidden column {col}")


def select_workhorse_lexicographic(
    summary: pd.DataFrame,
    ranking_keys: list[tuple[str, str, bool]],
    eligibility: pd.DataFrame | None = None,
    selection_type: str = "lexicographic",
) -> dict:
    """Generic fold-mean lexicographic ranking. Not used by the active manuscript pipeline."""
    if not ranking_keys:
        raise ValueError("ranking_keys must be non-empty")
    _assert_no_oof_columns(summary)
    models = sorted(summary["model_id"].astype(str).unique())
    eligible = []
    skipped = []
    for mid in models:
        if archived_model_blocked(mid):
            skipped.append({"model_id": mid, "reason": "archived_not_part_of_current_grid"})
            continue
        if eligibility is not None and len(eligibility):
            erow = eligibility[eligibility["model_id"] == mid]
            if len(erow) and not bool(erow.iloc[0].get("eligible", True)):
                skipped.append({"model_id": mid, "reason": "eligibility_false"})
                continue
        sub = summary[summary["model_id"] == mid]
        n_valid = int(sub["n_valid_folds"].min()) if "n_valid_folds" in sub.columns else 0
        if n_valid < 5:
            skipped.append({"model_id": mid, "reason": f"incomplete_folds n_valid_folds={n_valid}"})
            continue
        eligible.append(mid)
    if not eligible:
        return {
            "selected": None,
            "reason": "no_eligible_models",
            "skipped": skipped,
            "winner_computed": False,
            "selection_type": selection_type,
        }

    def key(mid: str):
        vals = []
        for scope, metric, higher in ranking_keys:
            v = _lookup(summary, mid, scope, metric)
            if not higher:
                v = -v
            vals.append(v)
        return tuple(vals)

    ranked = sorted(eligible, key=key, reverse=True)
    winner = ranked[0]
    primary = ranking_keys[0]
    return {
        "selected": winner,
        "ranked": ranked,
        "primary": {"scope": primary[0], "metric_name": primary[1], "aggregation": "fold_mean"},
        "skipped": skipped,
        "winner_computed": True,
        "input": "fold_metric_summary_only",
        "selection_type": selection_type,
    }


def _uq_status(mid: str, n_picp: int, n_ace: int, chemprop_invalid_nig_rows: int) -> str:
    if _is_point_model(mid):
        return "no_predictive_uq"
    if mid == CHEMPROP_DS2:
        return "partial_native_invalid_rows" if chemprop_invalid_nig_rows else "complete"
    if n_picp == N_VALID_REQUIRED and n_ace == N_VALID_REQUIRED:
        return "complete"
    if n_picp > 0 or n_ace > 0:
        return "partial"
    return "not_computed"


def _calibration_screen(picp: dict, ace: dict, n_acc: int, uq_status: str, mid: str) -> tuple[bool, str]:
    reasons = []
    if _is_point_model(mid):
        return False, "no_predictive_uq; not operationally eligible"
    if uq_status == "partial_native_invalid_rows" and REQUIRE_COMPLETE_NATIVE_UQ:
        reasons.append(
            "incomplete_native_uq; operational eligibility requires complete row-level native UQ "
            "(PICP/ACE are conditional on valid native-UQ rows and are not used as a fully valid screen)"
        )
    if n_acc < N_VALID_REQUIRED:
        reasons.append(f"incomplete_accuracy_folds n_valid_folds_accuracy={n_acc}")
    if int(picp["n_valid_folds"]) < N_VALID_REQUIRED:
        reasons.append(f"incomplete_picp_folds n_valid_folds_picp={int(picp['n_valid_folds'])}")
    if int(ace["n_valid_folds"]) < N_VALID_REQUIRED:
        reasons.append(f"incomplete_ace_folds n_valid_folds_ace={int(ace['n_valid_folds'])}")
    picp_m = picp["fold_mean"]
    ace_m = ace["fold_mean"]
    if not math.isfinite(picp_m):
        reasons.append("picp_not_finite")
    elif not (PICP_LOWER <= picp_m <= PICP_UPPER):
        reasons.append(
            f"picp_outside_operational_band {picp_m:.6f} not in [{PICP_LOWER}, {PICP_UPPER}]"
        )
    if not math.isfinite(ace_m):
        reasons.append("ace_not_finite")
    elif ace_m > ACE_UPPER:
        reasons.append(f"ace_above_operational_band {ace_m:.6f} > {ACE_UPPER}")
    if reasons:
        return False, "; ".join(reasons)
    return True, ""


def _pareto_objectives(row: dict) -> tuple[float, float, float]:
    rmse = row["all_pair_mop_rmse_mean"]
    picp = row["all_pair_picp_90_mean"]
    ace = row["all_pair_ace_mean"]
    return (
        float(rmse) if math.isfinite(rmse) else 9e9,
        abs(float(picp) - PICP_TARGET) if math.isfinite(picp) else 9e9,
        float(ace) if math.isfinite(ace) else 9e9,
    )


def _dominates(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def _pareto_front(rows: list[dict]) -> list[dict]:
    objs = [_pareto_objectives(r) for r in rows]
    front = []
    for i, oi in enumerate(objs):
        if any(_dominates(oj, oi) for j, oj in enumerate(objs) if j != i):
            continue
        front.append(rows[i])
    return front


def _minmax_distance(row: dict, front: list[dict]) -> float:
    if not front:
        return float("nan")
    objs = [_pareto_objectives(r) for r in front]
    mins = [min(o[k] for o in objs) for k in range(3)]
    maxs = [max(o[k] for o in objs) for k in range(3)]
    o = _pareto_objectives(row)
    norm = []
    for k in range(3):
        span = maxs[k] - mins[k]
        norm.append(0.0 if span <= 0 else (o[k] - mins[k]) / span)
    return math.sqrt(sum(x * x for x in norm))


def _sort_key_usable(row: dict, front: list[dict]) -> tuple:
    dist = _minmax_distance(row, front)
    sign = row["all_pair_sign_accuracy_mean"]
    r2 = row["all_pair_mop_r2_mean"]
    simple = OPERATIONAL_SIMPLICITY.get(row["model_id"], 50)
    dist_key = float(dist) if math.isfinite(dist) else 9e9
    sign_key = -float(sign) if math.isfinite(sign) else 9e9
    r2_key = -float(r2) if math.isfinite(r2) else 9e9
    return (dist_key, sign_key, r2_key, simple)


def select_workhorse_accuracy_calibration(
    summary: pd.DataFrame,
    eligibility: pd.DataFrame | None = None,
    chemprop_invalid_nig_rows: int = 13,
) -> dict:
    """Operational accuracy–calibration compromise. Not a statistical test.

    Stage 1: operational interval-usability screen on all-pair PICP@90 and ACE,
    with complete five-fold all-pair MoP accuracy and complete native UQ for
    operational eligibility.     Stage 2: Pareto front on all-pair MoP RMSE (lower better),
    |row-level PICP@90-0.90| (lower better), and row-level ACE (lower better),
    then the front member closest to the ideal point after min-max
    normalization (Euclidean distance). Restricted rare-earth-element pair
    MoP RMSE is a usage-example diagnostic and is not a selection key.
    Row-level RMSE and R2 are ignored.
    Combined |PICP-0.90|+ACE is not a selection key.
    """
    _assert_no_oof_columns(summary)
    if any(c in summary.columns for c in ROW_LEVEL_METRIC_NAMES if c in {"row_rmse", "row_r2"}):
        # Wide tables may carry display columns; they must not be read as keys.
        pass

    ds2 = summary[summary["model_id"].astype(str).str.startswith("DS2_")].copy()
    models = sorted(ds2["model_id"].astype(str).unique())
    elig_map = {}
    if eligibility is not None and len(eligibility):
        elig_map = {str(r.model_id): bool(r.eligible) for r in eligibility.itertuples(index=False)}

    rows = []
    skipped = []
    for mid in models:
        if archived_model_blocked(mid):
            skipped.append({"model_id": mid, "reason": "archived_not_part_of_current_grid"})
            continue
        all_rmse = _lookup_full(ds2, mid, "all_pairs", "mop_rmse")
        all_r2 = _lookup_full(ds2, mid, "all_pairs", "mop_r2")
        all_sign = _lookup_full(ds2, mid, "all_pairs", "sign_accuracy_all")
        ree_rmse = _lookup_full(ds2, mid, "ree_pairs", "mop_rmse")
        ree_r2 = _lookup_full(ds2, mid, "ree_pairs", "mop_r2")
        ree_sign = _lookup_full(ds2, mid, "ree_pairs", "sign_accuracy_all")
        picp = _lookup_full(ds2, mid, "all_pairs", "marginal_picp_90")
        ace = _lookup_full(ds2, mid, "all_pairs", "marginal_ace")
        mop_picp = _lookup_full(ds2, mid, "all_pairs", "mop_picp_90")
        mop_ace = _lookup_full(ds2, mid, "all_pairs", "mop_ace")
        n_acc = int(all_rmse["n_valid_folds"])
        uq_status = _uq_status(mid, int(picp["n_valid_folds"]), int(ace["n_valid_folds"]), chemprop_invalid_nig_rows)
        ranking_eligible = bool(elig_map.get(mid, not _is_point_model(mid))) and n_acc >= N_VALID_REQUIRED
        if _is_point_model(mid):
            ranking_eligible = False
        usable, reason = _calibration_screen(picp, ace, n_acc, uq_status, mid)
        rows.append(
            {
                "model_id": mid,
                "display_name": DISPLAY_NAMES.get(mid, mid),
                "ranking_eligibility": ranking_eligible,
                "uq_status": uq_status,
                "n_valid_folds_accuracy": n_acc,
                "n_valid_folds_picp": int(picp["n_valid_folds"]),
                "n_valid_folds_ace": int(ace["n_valid_folds"]),
                "all_pair_mop_rmse_mean": all_rmse["fold_mean"],
                "all_pair_mop_rmse_sd": all_rmse["fold_sd"],
                "all_pair_mop_r2_mean": all_r2["fold_mean"],
                "all_pair_mop_r2_sd": all_r2["fold_sd"],
                "all_pair_sign_accuracy_mean": all_sign["fold_mean"],
                "all_pair_sign_accuracy_sd": all_sign["fold_sd"],
                "ree_mop_rmse_mean": ree_rmse["fold_mean"],
                "ree_mop_rmse_sd": ree_rmse["fold_sd"],
                "ree_mop_r2_mean": ree_r2["fold_mean"],
                "ree_mop_r2_sd": ree_r2["fold_sd"],
                "ree_sign_accuracy_mean": ree_sign["fold_mean"],
                "ree_sign_accuracy_sd": ree_sign["fold_sd"],
                "all_pair_picp_90_mean": picp["fold_mean"],
                "all_pair_picp_90_sd": picp["fold_sd"],
                "all_pair_ace_mean": ace["fold_mean"],
                "all_pair_ace_sd": ace["fold_sd"],
                "all_pair_mop_picp_90_mean": mop_picp["fold_mean"],
                "all_pair_mop_picp_90_sd": mop_picp["fold_sd"],
                "all_pair_mop_ace_mean": mop_ace["fold_mean"],
                "all_pair_mop_ace_sd": mop_ace["fold_sd"],
                "n_valid_folds_mop_picp": int(mop_picp["n_valid_folds"]),
                "n_valid_folds_mop_ace": int(mop_ace["n_valid_folds"]),
                "calibration_usable": usable,
                "calibration_exclusion_reason": reason,
                "row_rmse_mean": _lookup_full(ds2, mid, "all_pairs", "rmse")["fold_mean"],
                "row_r2_mean": _lookup_full(ds2, mid, "all_pairs", "r2")["fold_mean"],
            }
        )

    usable_rows = [r for r in rows if r["calibration_usable"]]
    front = _pareto_front(usable_rows)
    front_ids = {r["model_id"] for r in front}
    usable_sorted = sorted(front, key=lambda r: _sort_key_usable(r, front))
    winner = usable_sorted[0]["model_id"] if usable_sorted else None
    rank_map = {r["model_id"]: i + 1 for i, r in enumerate(usable_sorted)}
    for r in rows:
        mid = r["model_id"]
        r["pareto_member"] = bool(r["calibration_usable"] and mid in front_ids)
        r["pareto_distance"] = _minmax_distance(r, front) if r["pareto_member"] else np.nan
        if r["calibration_usable"] and mid in front_ids:
            r["operational_rank"] = rank_map[mid]
            r["selection_status"] = "selected" if mid == winner else "candidate"
        elif r["calibration_usable"]:
            r["operational_rank"] = np.nan
            r["selection_status"] = "dominated_screen_passer"
        elif _is_point_model(mid):
            r["operational_rank"] = np.nan
            r["selection_status"] = "point_baseline_not_operationally_eligible"
        elif mid == CHEMPROP_DS2:
            r["operational_rank"] = np.nan
            r["selection_status"] = "scientific_comparator_partial_uq_not_operational_candidate"
        else:
            r["operational_rank"] = np.nan
            r["selection_status"] = "screened_out"
        r.pop("row_rmse_mean", None)
        r.pop("row_r2_mean", None)

    out = {
        "selection_type": "operational_accuracy_calibration_compromise",
        "primary_accuracy_metric": "all_pair_mop_rmse",
        "secondary_usage_metric": "ree_mop_rmse",
        "calibration_metrics": ["all_pair_picp_90", "all_pair_ace"],
        "calibration_screen": {
            "name": "operational_calibration_usability_screen",
            "picp_lower": PICP_LOWER,
            "picp_upper": PICP_UPPER,
            "ace_upper": ACE_UPPER,
            "n_valid_folds_required": N_VALID_REQUIRED,
            "formal_hypothesis_test": False,
        },
        "pareto_objectives": ["all_pair_mop_rmse", "abs_picp_90_minus_0.90", "all_pair_ace"],
        "pareto_normalization": "minmax_among_pareto_set",
        "pareto_selection": "closest_euclidean_to_ideal",
        "r2_role": "auxiliary_diagnostic",
        "sign_accuracy_role": "directional_diagnostic",
        "row_level_metrics_used_for_selection": False,
        "retraining_performed": False,
        "selection_nested": False,
        "composite_calibration_score_used": False,
        "calibration_deviation_plot_is_visualization_aid_only": True,
        "tie_breakers": [
            "higher_all_pair_sign_accuracy",
            "higher_all_pair_mop_r2",
            "operational_simplicity",
        ],
        "risk_ordering_used": False,
        "risk_ordering_reason": (
            "Fold-local retention-curve AUC is not a grid-wide selection metric "
            "and is not used as a hidden selection key."
        ),
        "require_complete_native_uq_for_operational_eligibility": REQUIRE_COMPLETE_NATIVE_UQ,
        "chemprop_native_evidential_scientific_comparator": True,
        "chemprop_native_evidential_operational_candidate": False,
        "chemprop_invalid_nig_rows": chemprop_invalid_nig_rows,
        "partial_uq_picp_ace_conditional_on_valid_rows": True,
        "selected": winner,
        "ranked": [r["model_id"] for r in usable_sorted],
        "pareto_front": [r["model_id"] for r in front],
        "skipped": skipped,
        "winner_computed": winner is not None,
        "input": "fold_metric_summary_only",
        "rows": rows,
        "description": (
            "the best operational compromise among the evaluated models under the "
            "declared all-pair interval-usability screen and the RMSE–PICP–ACE Pareto rule"
        ),
        "not_a_statistical_test": True,
        "not_universal_superiority": True,
    }
    return out


def write_rule() -> Path:
    obj = {
        "selection_type": "operational_accuracy_calibration_compromise",
        "primary_accuracy_metric": "all_pair_mop_rmse",
        "primary_metric": "all_pairs_mean_of_pairs_RMSE",
        "secondary_usage_metric": "ree_mop_rmse",
        "primary_aggregation": "fold_mean",
        "evaluation_units": "heldout_fold",
        "ranking_aggregation": "mean_of_pairs",
        "calibration_metrics": ["all_pair_picp_90", "all_pair_ace"],
        "calibration_screen": {
            "name": "operational_calibration_usability_screen",
            "picp_lower": PICP_LOWER,
            "picp_upper": PICP_UPPER,
            "ace_upper": ACE_UPPER,
            "n_valid_folds_required": N_VALID_REQUIRED,
            "formal_hypothesis_test": False,
        },
        "r2_role": "auxiliary_diagnostic",
        "sign_accuracy_role": "directional_diagnostic",
        "row_level_metrics_used_for_selection": False,
        "retraining_performed": False,
        "selection_nested": False,
        "pooled_concatenation_forbidden": True,
        "pareto_objectives": ["all_pair_mop_rmse", "abs_picp_90_minus_0.90", "all_pair_ace"],
        "pareto_normalization": "minmax_among_pareto_set",
        "pareto_selection": "closest_euclidean_to_ideal",
        "tie_breakers": [
            "higher_all_pair_sign_accuracy",
            "higher_all_pair_MoP_R2",
            "operational_simplicity",
        ],
        "chemprop_native_evidential_eligible": True,
        "chemprop_native_evidential_operational_candidate": False,
        "chemprop_mc_dropout_eligible": False,
        "variational_bnn_eligible": True,
        "sngp_eligible": True,
        "gpr_sparse_eligible": True,
        "exact_gpr_eligible": False,
        "require_complete_native_uq_for_operational_eligibility": True,
        "pair_balanced_global_primary": False,
        "composite_calibration_score_used": False,
        "picp_gate_070_obsolete": True,
        "selection_input": "fold_metrics_only",
        "forbidden_fields": ["concatenated_cv_value", "pooled_heldout_value"],
        "note": (
            "The DS2 workhorse is selected as an operational accuracy–interval-usability compromise. "
            "Candidate models must have complete five-fold all-pair MoP accuracy metrics and "
            "usable all-pair predictive-interval calibration. Among candidates satisfying the "
            "operational interval-usability screen (0.85 ≤ row-level PICP@90 ≤ 0.95 and row-level ACE ≤ 0.10, "
            "n_valid_folds=5), the Pareto front is formed on all-pair MoP RMSE (lower better), "
            "|row-level PICP@90-0.90| (lower better), and row-level ACE (lower better). "
            "A model dominates another if it is no worse on all three objectives and strictly better on at least one. "
            "The selected model is the front member closest to the ideal point after "
            "min-max normalization among the Pareto set, using Euclidean distance. Restricted rare-earth-element pair MoP RMSE is a chemically focused "
            "usage example and does not determine selection. MoP R2 is an auxiliary diagnostic; "
            "sign accuracy is a directional diagnostic used only for remaining distance ties. "
            "This is not a formal statistical test. Row-level RMSE and R2 are not used for selection. "
            "PICP and ACE used by the screen and Pareto rule are row-level (marginal) coverage metrics. "
            "Chemprop partial native-UQ status is preserved; "
            "complete native UQ is required for operational workhorse eligibility. Selection is not nested."
        ),
    }
    RULE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RULE_PATH.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return RULE_PATH


def write_accuracy_calibration_artifacts(
    choice: dict,
    dest: Path | None = None,
) -> dict[str, Path]:
    dest = dest or MSRC
    dest.mkdir(parents=True, exist_ok=True)
    rows = list(choice.get("rows") or [])
    table = pd.DataFrame(rows)
    keep = [
        "model_id",
        "display_name",
        "ranking_eligibility",
        "uq_status",
        "n_valid_folds_accuracy",
        "n_valid_folds_picp",
        "n_valid_folds_ace",
        "all_pair_mop_rmse_mean",
        "all_pair_mop_rmse_sd",
        "all_pair_mop_r2_mean",
        "all_pair_mop_r2_sd",
        "all_pair_sign_accuracy_mean",
        "all_pair_sign_accuracy_sd",
        "ree_mop_rmse_mean",
        "ree_mop_rmse_sd",
        "ree_mop_r2_mean",
        "ree_mop_r2_sd",
        "ree_sign_accuracy_mean",
        "ree_sign_accuracy_sd",
        "all_pair_picp_90_mean",
        "all_pair_picp_90_sd",
        "all_pair_ace_mean",
        "all_pair_ace_sd",
        "all_pair_mop_picp_90_mean",
        "all_pair_mop_picp_90_sd",
        "all_pair_mop_ace_mean",
        "all_pair_mop_ace_sd",
        "pareto_member",
        "pareto_distance",
        "calibration_usable",
        "calibration_exclusion_reason",
        "operational_rank",
        "selection_status",
    ]
    table = table[[c for c in keep if c in table.columns]]
    json_obj = {k: v for k, v in choice.items() if k != "rows"}
    json_obj["selection_type"] = "operational_accuracy_calibration_compromise"
    json_obj["primary_accuracy_metric"] = "all_pair_mop_rmse"
    json_obj["secondary_usage_metric"] = "ree_mop_rmse"
    json_obj["calibration_metrics"] = ["all_pair_picp_90", "all_pair_ace"]
    json_obj["calibration_screen"] = choice["calibration_screen"]
    json_obj["r2_role"] = "auxiliary_diagnostic"
    json_obj["sign_accuracy_role"] = "directional_diagnostic"
    json_obj["row_level_metrics_used_for_selection"] = False
    json_obj["retraining_performed"] = False
    json_obj["selection_nested"] = False
    json_path = dest / "workhorse_selection_accuracy_calibration.json"
    csv_path = dest / "workhorse_selection_accuracy_calibration.csv"
    cand_path = dest / "workhorse_selection_candidates.csv"
    json_path.write_text(json.dumps(json_obj, indent=2, default=str), encoding="utf-8")
    table.to_csv(csv_path, index=False)
    candidates = table[table["calibration_usable"] == True].copy()  # noqa: E712
    candidates.to_csv(cand_path, index=False)
    return {
        "json": json_path,
        "csv": csv_path,
        "candidates": cand_path,
    }
