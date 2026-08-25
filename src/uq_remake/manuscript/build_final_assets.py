"""Build manuscript registries, tables, and a numeric snapshot from canonical artifacts.

Does not retrain models. Reruns workhorse selection from fold-level summaries only.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uq_remake import REMAKE_ROOT
from uq_remake.downstream.ad_hoc_selectivity import SELECTIVITY_AD_HOC
from uq_remake.downstream.protocol import PAIR_ACQ_EVENT_MIN_EXCLUSIVE, family_display_name, score_display_name
from uq_remake.metrics.workhorse import (
    select_workhorse_accuracy_calibration,
    write_accuracy_calibration_artifacts,
    write_rule,
)
from uq_remake.training.protocol import canonical_model_id

from uq_remake.manuscript.manuscript_paths import MAN, MSRC, TABLES

GRID = "shared_ds1_ds2_uq_v3"
WORKHORSE = "DS2_RDKIT_MLP_MCDROPOUT"  # overwritten by operational selection in main()
ARCHIVE = REMAKE_ROOT / "manuscripts" / "archive" / "pre_corrected_rewrite"

DISPLAY = {
    "MLP_POINT": "RDKit MLP",
    "XGB_POINT": "RDKit XGBoost",
    "GRAPH_DMPNN_POINT": "Chemprop point",
    "MLP_EVID": "Evidential MLP",
    "MLPENS5_EVID": "Evidential MLP ensemble",
    "MLP_HETERO": "Heteroscedastic MLP",
    "MLPENS5_HETERO": "Heteroscedastic MLP ensemble",
    "MLP_MCDROPOUT": "MC-dropout MLP",
    "MLPENS5_MCDROPOUT": "MC-dropout MLP ensemble",
    "MLP_BAYES_FULL": "Full variational BNN",
    "MLPENS5_BAYES_FULL": "Full variational BNN ensemble",
    "SNGP": "SNGP",
    "SNGPENS5": "SNGP ensemble",
    "GPR": "Exact GPR",
    "GPR_SPARSE": "Sparse variational GPR",
    "GRAPH_DMPNN_EVID": "Chemprop evidential",
}

UQ_TYPE = {
    "POINT_MLP": "none",
    "POINT_XGB": "none",
    "CHEMPROP_POINT_NATIVE": "none",
    "EVID_MLP": "evidential_NIG",
    "EVID_MLPENS5": "evidential_NIG_ensemble_LoTV",
    "HETERO_MLP": "heteroscedastic",
    "HETERO_MLPENS5": "heteroscedastic_ensemble_LoTV",
    "MCDROPOUT_MLP": "mc_dropout",
    "MCDROPOUT_MLPENS5": "mc_dropout_ensemble_LoTV",
    "BAYES_FULL": "full_variational_bnn",
    "BAYES_FULLENS5": "full_variational_bnn_ensemble_LoTV",
    "SNGP": "spectral_normalized_gp_readout",
    "SNGPENS5": "sngp_ensemble_LoTV",
    "GPR": "exact_gp",
    "GPR_SPARSE": "sparse_variational_gp",
    "CHEMPROP_EVIDENTIAL_NATIVE": "native_chemprop_evidential_NIG",
}

FEATURE = {
    "POINT_MLP": "RDKit tabular",
    "POINT_XGB": "RDKit tabular",
    "CHEMPROP_POINT_NATIVE": "Chemprop graph",
    "EVID_MLP": "RDKit tabular",
    "EVID_MLPENS5": "RDKit tabular",
    "HETERO_MLP": "RDKit tabular",
    "HETERO_MLPENS5": "RDKit tabular",
    "MCDROPOUT_MLP": "RDKit tabular",
    "MCDROPOUT_MLPENS5": "RDKit tabular",
    "BAYES_FULL": "RDKit tabular",
    "BAYES_FULLENS5": "RDKit tabular",
    "SNGP": "RDKit tabular",
    "SNGPENS5": "RDKit tabular",
    "GPR": "RDKit tabular",
    "GPR_SPARSE": "RDKit tabular",
    "CHEMPROP_EVIDENTIAL_NATIVE": "Chemprop graph",
}

INTERVAL = {
    "POINT_MLP": "unavailable",
    "POINT_XGB": "unavailable",
    "CHEMPROP_POINT_NATIVE": "unavailable",
    "EVID_MLP": "student_t_nig_predictive",
    "EVID_MLPENS5": "student_t_nig_predictive",
    "HETERO_MLP": "normal_moment",
    "HETERO_MLPENS5": "normal_moment",
    "MCDROPOUT_MLP": "normal_moment",
    "MCDROPOUT_MLPENS5": "normal_moment",
    "BAYES_FULL": "empirical_posterior_predictive",
    "BAYES_FULLENS5": "empirical_posterior_predictive",
    "SNGP": "gaussian_predictive_moment",
    "SNGPENS5": "gaussian_predictive_moment",
    "GPR": "gaussian_predictive_moment",
    "GPR_SPARSE": "gaussian_predictive_moment",
    "CHEMPROP_EVIDENTIAL_NATIVE": "student_t_nig_predictive",
}

SAMPLES = {
    # Heteroscedastic families use 30 Monte Carlo dropout passes at prediction
    # (`mc_dropout_passes` in the active grid and `predict_hetero_mc`).
    "POINT_MLP": 0,
    "POINT_XGB": 0,
    "CHEMPROP_POINT_NATIVE": 0,
    "EVID_MLP": 0,
    "EVID_MLPENS5": 0,
    "HETERO_MLP": 30,
    "HETERO_MLPENS5": 30,
    "MCDROPOUT_MLP": 30,
    "MCDROPOUT_MLPENS5": 30,
    "BAYES_FULL": 30,
    "BAYES_FULLENS5": 30,
    "SNGP": 0,
    "SNGPENS5": 0,
    "GPR": 0,
    "GPR_SPARSE": 0,
    "CHEMPROP_EVIDENTIAL_NATIVE": 0,
}

DS2_SCOPES = [
    ("all_pairs", "all-pair"),
    ("ree_pairs", "REE–REE"),
    ("high_selectivity_row", "high-selectivity (row)"),
    ("high_selectivity_all_pair_membership", "high-selectivity (pair membership)"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def fmt(mean, sd=None, n_valid=None, digits=3) -> str:
    if mean is None or (isinstance(mean, float) and not np.isfinite(mean)):
        return "NA"
    if sd is None or (isinstance(sd, float) and not np.isfinite(sd)):
        s = f"{float(mean):.{digits}f}"
    else:
        s = f"{float(mean):.{digits}f} ± {float(sd):.{digits}f}"
    if n_valid is not None and int(n_valid) != 5:
        s += f" (n_valid={int(n_valid)})"
    return s


def lookup(summary: pd.DataFrame, model_id: str, scope: str, metric: str) -> dict:
    hit = summary[
        (summary["model_id"] == model_id)
        & (summary["scope"] == scope)
        & (summary["metric_name"] == metric)
    ]
    if hit.empty:
        return {"fold_mean": np.nan, "fold_sd": np.nan, "n_folds": 0, "n_valid_folds": 0}
    r = hit.iloc[0]
    return {
        "fold_mean": float(r["fold_mean"]) if pd.notna(r["fold_mean"]) else np.nan,
        "fold_sd": float(r["fold_sd"]) if pd.notna(r["fold_sd"]) else np.nan,
        "n_folds": int(r["n_folds"]) if pd.notna(r.get("n_folds", np.nan)) else 0,
        "n_valid_folds": int(r["n_valid_folds"]) if pd.notna(r.get("n_valid_folds", np.nan)) else 0,
    }


def short_of(model_id: str) -> str:
    if model_id.endswith("_GRAPH_DMPNN_EVID"):
        return "GRAPH_DMPNN_EVID"
    if model_id.endswith("_GRAPH_DMPNN_POINT"):
        return "GRAPH_DMPNN_POINT"
    rest = model_id.split("_", 2)[-1] if model_id.startswith(("DS1_", "DS2_")) else model_id
    # DS1_RDKIT_MLP_POINT -> RDKIT_MLP_POINT; drop RDKIT_
    if rest.startswith("RDKIT_"):
        rest = rest[len("RDKIT_") :]
    return rest


def display_of(model_id: str) -> str:
    return DISPLAY.get(short_of(model_id), short_of(model_id))


def archive_old_manuscript() -> None:
    src = REMAKE_ROOT / "manuscripts" / "full_manuscript"
    if not src.exists():
        return
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists():
        return
    shutil.copytree(src, ARCHIVE)


def fold_size_table() -> pd.DataFrame:
    rows = []
    for ds, schema in (("DS1", "rdkit_ds1"), ("DS2", "rdkit_ds2")):
        for fold in ("fold0", "fold1", "fold2", "fold3", "fold4"):
            man = json.loads(
                (REMAKE_ROOT / "data" / "processed" / "features" / schema / fold / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            rows.append(
                {
                    "dataset": ds,
                    "fold_id": fold,
                    "n_train": int(man["n_train"]),
                    "n_val": int(man["n_val"]),
                    "n_test": int(man["n_test"]),
                    "feature_dimension": int(man["feature_dimension"]),
                    "artifact_path": f"data/processed/features/{schema}/{fold}/manifest.json",
                }
            )
    return pd.DataFrame(rows)


def ds2_scope_counts() -> dict:
    df = pd.read_csv(REMAKE_ROOT / "data" / "processed" / "ds2.csv", usecols=["ligand_smiles", "is_ree_pair", "metal_species_pair", "selectivity_logK"])
    ree = df["is_ree_pair"]
    if ree.dtype == object:
        ree = ree.astype(str).str.lower().isin({"true", "1", "yes"})
    ree = ree.astype(bool)
    tau = 1.0
    hs = df["selectivity_logK"].abs() > tau
    pair = df["metal_species_pair"]
    ree_pairs = pair[ree]
    out = {
        "n_rows": int(len(df)),
        "n_ligands": int(df["ligand_smiles"].nunique()),
        "n_species_pairs": int(pair.nunique()),
        "n_ree_rows": int(ree.sum()),
        "n_ree_species_pairs": int(ree_pairs.nunique()),
        "n_highsel_rows": int(hs.sum()),
        "n_ree_highsel_rows": int((ree & hs).sum()),
        "high_selectivity_tau": tau,
        "high_selectivity_rule": "abs(selectivity_logK) > 1.0",
        "min_pair_n": 3,
        "pair_id_example": "Fe:2.0--Co:2.0",
        "status": "canonical_cv_result",
        "artifact_path": "data/processed/ds2.csv",
        "note": "REE∩high-selectivity row count is a composition diagnostic only; no combined CV scope exists in fold_metric_summary.csv.",
    }
    return out


def build_model_tables(reg: dict, eligibility: pd.DataFrame, choice: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = {m: i + 1 for i, m in enumerate(choice.get("ranked") or [])}
    skipped = {s["model_id"]: s["reason"] for s in choice.get("skipped") or []}
    op_status = {r["model_id"]: r.get("selection_status") for r in choice.get("rows") or []}
    selected_id = choice.get("selected")
    rows = []
    status_rows = []
    for m in reg["models"]:
        fam = m["model_family"]
        short = m["short_name"]
        mid = m["model_id"]
        role = "comparator"
        if mid == selected_id or mid == WORKHORSE:
            role = "workhorse"
        elif fam == "GPR":
            role = "excluded_infeasible"
        elif mid.startswith("DS2_") and short in {"MLP_POINT", "XGB_POINT", "GRAPH_DMPNN_POINT"}:
            role = "point_baseline_not_workhorse_eligible"
        elif mid == "DS2_GRAPH_DMPNN_EVID":
            role = "scientific_comparator_partial_uq"
        sel_status = "not_applicable"
        if m["dataset"] == "DS2" and fam != "GPR":
            if mid == selected_id:
                sel_status = "selected"
            elif mid in op_status:
                sel_status = str(op_status[mid])
            elif mid in ranked:
                sel_status = f"ranked_{ranked[mid]}"
            elif mid in skipped:
                sel_status = f"skipped_{skipped[mid]}"
        rows.append(
            {
                "display_name": DISPLAY[short],
                "model_id": mid,
                "dataset": m["dataset"],
                "model_family": fam,
                "short_name": short,
                "feature_representation": FEATURE[fam],
                "point_prediction": "yes",
                "uncertainty_type": UQ_TYPE[fam],
                "ensemble_size": m["n_ensemble_members"],
                "stochastic_sample_count": SAMPLES[fam],
                "active_status": m["overall_status"],
                "ranking_status": m["ranking_status"],
                "pair_metric_status": m["ranking_status"],
                "uq_status": m["uq_status"],
                "interval_status": m["interval_status"],
                "interval_method": INTERVAL[fam],
                "manuscript_role": role,
                "selection_status": sel_status,
                "exclusion_reason": m.get("exclusion_reason") or "",
            }
        )
        status_rows.append(
            {
                "model_id": mid,
                "display_name": DISPLAY[short],
                "dataset": m["dataset"],
                "overall_status": m["overall_status"],
                "training_status": m["training_status"],
                "prediction_status": m["prediction_status"],
                "point_metric_status": m["point_metric_status"],
                "uq_status": m["uq_status"],
                "interval_status": m["interval_status"],
                "ranking_status": m["ranking_status"],
                "pair_metric_status": m["ranking_status"],
                "native_uq_status": m["native_uq_status"],
                "invalid_native_nig_rows": m["invalid_native_nig_rows"],
                "active_for_manuscript": m["active_for_manuscript"],
                "canonical_artifact_prefix": m["canonical_artifact_prefix"],
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(status_rows)


def _sort_formatted_by_mean(df: pd.DataFrame, col: str, ascending: bool = True) -> pd.DataFrame:
    means = []
    for v in df[col]:
        s = str(v).split("±")[0].strip()
        try:
            means.append(float(s))
        except ValueError:
            means.append(np.inf if ascending else -np.inf)
    out = df.copy()
    out["_sort"] = means
    out = out.sort_values("_sort", ascending=ascending, kind="mergesort").drop(columns="_sort")
    return out.reset_index(drop=True)


def pair_level_frame(df: pd.DataFrame, diagnostic_class: str) -> pd.DataFrame:
    cols = [
        "display_name",
        "model_id",
        "scope",
        "scope_key",
        "mop_rmse",
        "mop_r2",
        "sign_accuracy",
        "mop_picp_90",
        "mop_ace",
        "n_valid_folds_point",
        "n_valid_folds_uq",
        "uq_status",
        "aggregation",
    ]
    out = df[[c for c in cols if c in df.columns]].copy()
    out["diagnostic_class"] = diagnostic_class
    return out


def row_level_frame(df: pd.DataFrame, diagnostic_class: str) -> pd.DataFrame:
    cols = [
        "display_name",
        "model_id",
        "scope",
        "scope_key",
        "row_rmse",
        "row_r2",
        "sign_accuracy",
        "picp_90",
        "ace",
        "n_valid_folds_point",
        "n_valid_folds_uq",
        "aggregation",
    ]
    out = df[[c for c in cols if c in df.columns]].copy()
    out["diagnostic_class"] = diagnostic_class
    return out


TABLE3_ADHOC_COMPARATORS = (
    {"model_id": "DS1_RDKIT_MLP_POINT", "display_name": "RDKit MLP", "has_uq": False},
    {"model_id": "DS1_GRAPH_DMPNN_EVID", "display_name": "Chemprop evidential", "has_uq": True},
)


def _adhoc_scope_row(summary: pd.DataFrame, model_id: str, scope: str) -> pd.Series:
    hit = summary[
        (summary["source_model_id"].astype(str) == model_id)
        & (summary["scope"].astype(str) == scope)
        & (summary["selectivity_method"].astype(str) == SELECTIVITY_AD_HOC)
    ]
    if hit.empty:
        raise RuntimeError(f"Table 3 ad hoc comparator missing from summary: {model_id} {scope}")
    return hit.iloc[0]


def _fmt_adhoc_metric(row: pd.Series, col: str, *, has_uq: bool) -> str:
    interval_cols = {"picp_90", "ace", "mop_picp_90", "mop_ace"}
    if (not has_uq) and col in interval_cols:
        return "NA (no UQ)"
    if col not in row.index:
        return "NA (no UQ)" if col in interval_cols else "NA"
    return fmt(row[col], row.get(f"{col}_sd"), row.get("n_valid", 5))


def table3_adhoc_comparator_rows() -> list[dict]:
    """Named all-pair ad hoc comparators for Table 3 (not workhorse candidates)."""
    path = REMAKE_ROOT / "results" / "downstream_cv" / "ad_hoc_selectivity" / "ad_hoc_selectivity_summary.csv"
    if not path.exists():
        raise RuntimeError(f"missing ad hoc summary required for Table 3: {path}")
    summary = pd.read_csv(path)
    rows = []
    for spec in TABLE3_ADHOC_COMPARATORS:
        allp = _adhoc_scope_row(summary, spec["model_id"], "all_pairs")
        ree = _adhoc_scope_row(summary, spec["model_id"], "ree_pairs")
        has_uq = bool(spec["has_uq"])
        if has_uq and ("mop_picp_90" not in allp.index or pd.isna(allp.get("mop_picp_90", np.nan))):
            raise RuntimeError(
                f"{spec['model_id']} is missing all-pair MoP PICP/ACE; re-run ad hoc evaluation"
            )
        if "row_r2" not in allp.index or pd.isna(allp.get("row_r2", np.nan)):
            raise RuntimeError(
                f"{spec['model_id']} is missing all-pair row-level R2; re-run ad hoc evaluation"
            )
        rows.append(
            {
                "display_name": spec["display_name"],
                "model_id": spec["model_id"],
                "regime": "ad hoc",
                "all_pair_mop_rmse": _fmt_adhoc_metric(allp, "mop_rmse", has_uq=True),
                "all_pair_mop_r2": _fmt_adhoc_metric(allp, "mop_r2", has_uq=True),
                "all_pair_row_r2": _fmt_adhoc_metric(allp, "row_r2", has_uq=True),
                "ree_mop_rmse": _fmt_adhoc_metric(ree, "mop_rmse", has_uq=True),
                "ree_mop_r2": _fmt_adhoc_metric(ree, "mop_r2", has_uq=True),
                "ree_sign_accuracy": _fmt_adhoc_metric(ree, "sign_accuracy", has_uq=True),
                "all_pair_picp_90": _fmt_adhoc_metric(allp, "mop_picp_90", has_uq=has_uq),
                "all_pair_ace": _fmt_adhoc_metric(allp, "mop_ace", has_uq=has_uq),
                "all_pair_row_picp_90": _fmt_adhoc_metric(allp, "picp_90", has_uq=has_uq),
                "all_pair_row_ace": _fmt_adhoc_metric(allp, "ace", has_uq=has_uq),
                "calibration_usable": False,
                "operational_role": "ad hoc DS1-difference comparator; not a workhorse candidate",
                "native_uq_status": "propagated_independent_quadrature" if has_uq else "no_predictive_uq",
                "selection_status": "",
                "operational_rank": "",
                "calibration_exclusion_reason": "ad_hoc_comparator_not_in_operational_screen",
                "aggregation": "fold_mean_plus_sd_ddof1",
                "primary_accuracy_metric": "all_pair_mop_rmse",
                "r2_role": "auxiliary_diagnostic",
            }
        )
    return rows


def build_ds_tables(summary: pd.DataFrame, choice: dict | None = None) -> dict[str, pd.DataFrame]:
    out = {}
    # DS1 point
    point_ids = [
        canonical_model_id("DS1", "POINT_MLP"),
        canonical_model_id("DS1", "POINT_XGB"),
        canonical_model_id("DS1", "CHEMPROP_POINT_NATIVE"),
    ]
    uq_fams = [
        "EVID_MLP",
        "EVID_MLPENS5",
        "HETERO_MLP",
        "HETERO_MLPENS5",
        "MCDROPOUT_MLP",
        "MCDROPOUT_MLPENS5",
        "BAYES_FULL",
        "BAYES_FULLENS5",
        "SNGP",
        "SNGPENS5",
        "GPR_SPARSE",
        "CHEMPROP_EVIDENTIAL_NATIVE",
    ]
    rows = []
    for mid in point_ids:
        rmse = lookup(summary, mid, "metal_complex", "rmse")
        r2 = lookup(summary, mid, "metal_complex", "r2")
        rows.append(
            {
                "display_name": display_of(mid),
                "model_id": mid,
                "scope": "metal_complex",
                "rmse": fmt(rmse["fold_mean"], rmse["fold_sd"], rmse["n_valid_folds"]),
                "r2": fmt(r2["fold_mean"], r2["fold_sd"], r2["n_valid_folds"]),
                "picp_90": "NA (no UQ)",
                "ace": "NA (no UQ)",
                "interval_status": "unavailable",
                "n_valid_folds": rmse["n_valid_folds"],
                "metric_status": "complete" if rmse["n_valid_folds"] == 5 else "partial",
                "aggregation": "fold_mean_plus_sd_ddof1",
            }
        )
    out["main_ds1_point_results"] = _sort_formatted_by_mean(pd.DataFrame(rows), "rmse")

    rows = []
    for fam in uq_fams:
        mid = canonical_model_id("DS1", fam)
        rmse = lookup(summary, mid, "metal_complex", "rmse")
        r2 = lookup(summary, mid, "metal_complex", "r2")
        picp = lookup(summary, mid, "metal_complex", "marginal_picp_90")
        ace = lookup(summary, mid, "metal_complex", "marginal_ace")
        rows.append(
            {
                "display_name": display_of(mid),
                "model_id": mid,
                "scope": "metal_complex",
                "rmse": fmt(rmse["fold_mean"], rmse["fold_sd"], rmse["n_valid_folds"]),
                "r2": fmt(r2["fold_mean"], r2["fold_sd"], r2["n_valid_folds"]),
                "picp_90": fmt(picp["fold_mean"], picp["fold_sd"], picp["n_valid_folds"]),
                "ace": fmt(ace["fold_mean"], ace["fold_sd"], ace["n_valid_folds"]),
                "interval_method": INTERVAL[fam],
                "n_valid_folds": min(rmse["n_valid_folds"], picp["n_valid_folds"] or 0),
                "uq_status": "complete" if picp["n_valid_folds"] == 5 else "partial",
                "interval_status": "ok" if picp["n_valid_folds"] == 5 else "partial",
                "aggregation": "fold_mean_plus_sd_ddof1",
            }
        )
    out["main_ds1_uq_results"] = _sort_formatted_by_mean(pd.DataFrame(rows), "rmse")

    ds2_ids = []
    for fam in [
        "POINT_MLP",
        "POINT_XGB",
        "CHEMPROP_POINT_NATIVE",
        *uq_fams,
    ]:
        ds2_ids.append((fam, canonical_model_id("DS2", fam)))

    def ds2_block(scope_key, scope_label, name):
        rows = []
        for fam, mid in ds2_ids:
            row_rmse = lookup(summary, mid, scope_key, "rmse")
            r2 = lookup(summary, mid, scope_key, "r2")
            mop_rmse = lookup(summary, mid, scope_key, "mop_rmse")
            mop_r2 = lookup(summary, mid, scope_key, "mop_r2")
            sign = lookup(summary, mid, scope_key, "sign_accuracy_all")
            picp = lookup(summary, mid, scope_key, "marginal_picp_90")
            ace = lookup(summary, mid, scope_key, "marginal_ace")
            mop_picp = lookup(summary, mid, scope_key, "mop_picp_90")
            mop_ace = lookup(summary, mid, scope_key, "mop_ace")
            uq_ok = picp["n_valid_folds"] == 5
            rows.append(
                {
                    "display_name": display_of(mid),
                    "model_id": mid,
                    "scope": scope_label,
                    "scope_key": scope_key,
                    "row_rmse": fmt(row_rmse["fold_mean"], row_rmse["fold_sd"], row_rmse["n_valid_folds"]),
                    "row_r2": fmt(r2["fold_mean"], r2["fold_sd"], r2["n_valid_folds"]),
                    "mop_rmse": fmt(mop_rmse["fold_mean"], mop_rmse["fold_sd"], mop_rmse["n_valid_folds"]),
                    "mop_r2": fmt(mop_r2["fold_mean"], mop_r2["fold_sd"], mop_r2["n_valid_folds"]),
                    "sign_accuracy": fmt(sign["fold_mean"], sign["fold_sd"], sign["n_valid_folds"]),
                    "picp_90": fmt(picp["fold_mean"], picp["fold_sd"], picp["n_valid_folds"]) if picp["n_valid_folds"] else "NA (no UQ)",
                    "ace": fmt(ace["fold_mean"], ace["fold_sd"], ace["n_valid_folds"]) if ace["n_valid_folds"] else "NA (no UQ)",
                    "mop_picp_90": fmt(mop_picp["fold_mean"], mop_picp["fold_sd"], mop_picp["n_valid_folds"]) if mop_picp["n_valid_folds"] else "NA (no UQ)",
                    "mop_ace": fmt(mop_ace["fold_mean"], mop_ace["fold_sd"], mop_ace["n_valid_folds"]) if mop_ace["n_valid_folds"] else "NA (no UQ)",
                    "picp_scope": scope_label,
                    "n_valid_folds_point": row_rmse["n_valid_folds"],
                    "n_valid_folds_uq": picp["n_valid_folds"],
                    "uq_status": "complete" if uq_ok else ("partial" if picp["n_valid_folds"] else "not_computed"),
                    "aggregation": "fold_mean_plus_sd_ddof1",
                }
            )
        out[name] = pd.DataFrame(rows)

    ds2_block("all_pairs", "all-pair", "ds2_all_pairs")
    ds2_block("ree_pairs", "REE–REE", "ds2_ree_pairs")
    ds2_block("high_selectivity_row", "high-selectivity (row)", "ds2_highsel_row")
    ds2_block("high_selectivity_all_pair_membership", "high-selectivity (pair membership)", "ds2_highsel_pair")
    out["ds2_ree_pairs"]["ree_picp_90"] = out["ds2_ree_pairs"]["picp_90"]
    out["ds2_ree_pairs"]["ree_ace"] = out["ds2_ree_pairs"]["ace"]
    out["ds2_ree_pairs"]["ree_mop_picp_90"] = out["ds2_ree_pairs"]["mop_picp_90"]
    out["ds2_ree_pairs"]["ree_mop_ace"] = out["ds2_ree_pairs"]["mop_ace"]
    out["ds2_all_pairs"]["all_pair_picp_90"] = out["ds2_all_pairs"]["mop_picp_90"]
    out["ds2_all_pairs"]["all_pair_ace"] = out["ds2_all_pairs"]["mop_ace"]
    out["ds2_all_pairs"]["all_pair_row_picp_90"] = out["ds2_all_pairs"]["picp_90"]
    out["ds2_all_pairs"]["all_pair_row_ace"] = out["ds2_all_pairs"]["ace"]

    # Operational DS2 comparison (MoP PICP/ACE displayed; row-level PICP/ACE kept for the screen)
    if choice is None:
        elig = pd.read_csv(REMAKE_ROOT / "results" / "model_audit" / "workhorse_eligibility.csv")
        choice = select_workhorse_accuracy_calibration(summary, elig)
    selected = choice.get("selected")
    by_id = {r["model_id"]: r for r in choice.get("rows") or []}
    rows = []
    for fam, mid in ds2_ids:
        rec = by_id.get(mid, {})
        all_rmse = lookup(summary, mid, "all_pairs", "mop_rmse")
        all_r2 = lookup(summary, mid, "all_pairs", "mop_r2")
        row_r2 = lookup(summary, mid, "all_pairs", "r2")
        mop_rmse = lookup(summary, mid, "ree_pairs", "mop_rmse")
        mop_r2 = lookup(summary, mid, "ree_pairs", "mop_r2")
        sign = lookup(summary, mid, "ree_pairs", "sign_accuracy_all")
        picp = lookup(summary, mid, "all_pairs", "marginal_picp_90")
        ace = lookup(summary, mid, "all_pairs", "marginal_ace")
        mop_picp = lookup(summary, mid, "all_pairs", "mop_picp_90")
        mop_ace = lookup(summary, mid, "all_pairs", "mop_ace")
        usable = bool(rec.get("calibration_usable", False))
        uq_status = rec.get("uq_status", "not_computed")
        if mid == selected:
            role = "selected operational accuracy-calibration compromise"
        elif usable:
            role = "calibration-usable candidate"
        elif fam in {"POINT_MLP", "POINT_XGB", "CHEMPROP_POINT_NATIVE"}:
            role = "point baseline; no predictive UQ; not operationally eligible"
        elif fam == "CHEMPROP_EVIDENTIAL_NATIVE":
            role = "scientific comparator; partial native UQ; not operational candidate"
        else:
            role = "screened out by operational calibration-usability screen"
        rows.append(
            {
                "display_name": display_of(mid),
                "model_id": mid,
                "regime": "direct",
                "all_pair_mop_rmse": fmt(all_rmse["fold_mean"], all_rmse["fold_sd"], all_rmse["n_valid_folds"]),
                "all_pair_mop_r2": fmt(all_r2["fold_mean"], all_r2["fold_sd"], all_r2["n_valid_folds"]),
                "all_pair_row_r2": fmt(row_r2["fold_mean"], row_r2["fold_sd"], row_r2["n_valid_folds"]),
                "ree_mop_rmse": fmt(mop_rmse["fold_mean"], mop_rmse["fold_sd"], mop_rmse["n_valid_folds"]),
                "ree_mop_r2": fmt(mop_r2["fold_mean"], mop_r2["fold_sd"], mop_r2["n_valid_folds"]),
                "ree_sign_accuracy": fmt(sign["fold_mean"], sign["fold_sd"], sign["n_valid_folds"]),
                "all_pair_picp_90": fmt(mop_picp["fold_mean"], mop_picp["fold_sd"], mop_picp["n_valid_folds"]) if mop_picp["n_valid_folds"] else "NA (no UQ)",
                "all_pair_ace": fmt(mop_ace["fold_mean"], mop_ace["fold_sd"], mop_ace["n_valid_folds"]) if mop_ace["n_valid_folds"] else "NA (no UQ)",
                "all_pair_row_picp_90": fmt(picp["fold_mean"], picp["fold_sd"], picp["n_valid_folds"]) if picp["n_valid_folds"] else "NA (no UQ)",
                "all_pair_row_ace": fmt(ace["fold_mean"], ace["fold_sd"], ace["n_valid_folds"]) if ace["n_valid_folds"] else "NA (no UQ)",
                "calibration_usable": usable,
                "operational_role": role,
                "native_uq_status": uq_status,
                "selection_status": rec.get("selection_status", ""),
                "operational_rank": rec.get("operational_rank", ""),
                "calibration_exclusion_reason": rec.get("calibration_exclusion_reason", ""),
                "aggregation": "fold_mean_plus_sd_ddof1",
                "primary_accuracy_metric": "all_pair_mop_rmse",
                "r2_role": "auxiliary_diagnostic",
                "_usable_sort": 0 if usable else (2 if fam in {"POINT_MLP", "POINT_XGB", "CHEMPROP_POINT_NATIVE"} else 1),
                "_rank_sort": rec.get("operational_rank") if pd.notna(rec.get("operational_rank", np.nan)) else 99,
                "_rmse_sort": all_rmse["fold_mean"] if np.isfinite(all_rmse["fold_mean"]) else 9e9,
            }
        )
    op_direct = pd.DataFrame(rows).sort_values(
        ["_usable_sort", "_rank_sort", "_rmse_sort"], kind="mergesort"
    ).drop(columns=["_usable_sort", "_rank_sort", "_rmse_sort"]).reset_index(drop=True)
    op = pd.concat([op_direct, pd.DataFrame(table3_adhoc_comparator_rows())], ignore_index=True)
    out["main_ds2_operational_comparison"] = op
    out["main_ds2_workhorse_selection"] = op_direct.copy()
    out["main_ds2_model_comparison"] = out["ds2_ree_pairs"][
        [c for c in out["ds2_ree_pairs"].columns if not str(c).startswith("row_")]
    ].copy()

    pair_frames = [
        pair_level_frame(out["ds2_ree_pairs"], "S4.1 primary_ree_pair_level"),
        pair_level_frame(out["ds2_all_pairs"], "S4.1 all_pair_pair_level"),
        pair_level_frame(out["ds2_highsel_pair"], "S4.5 high_selectivity_pair_membership"),
        pair_level_frame(out["ds2_highsel_row"], "S4.5_note_highsel_row_also_has_pair_metrics"),
    ]
    row_frames = [
        row_level_frame(out["ds2_all_pairs"], "S4.3 secondary_row_weighted_all_pairs"),
        row_level_frame(out["ds2_ree_pairs"], "S4.3 secondary_row_weighted_ree"),
        row_level_frame(out["ds2_highsel_row"], "S4.4 high_selectivity_row_estimand"),
        row_level_frame(out["ds2_highsel_pair"], "S4.4_note_highsel_pair_membership_row_diagnostics"),
    ]
    out["supplementary_pair_level_results"] = pd.concat(pair_frames, ignore_index=True)
    out["supplementary_row_level_diagnostics"] = pd.concat(row_frames, ignore_index=True)
    sel_keep = [
        "display_name",
        "model_id",
        "calibration_usable",
        "all_pair_mop_rmse",
        "all_pair_picp_90",
        "all_pair_ace",
        "all_pair_row_picp_90",
        "all_pair_row_ace",
        "all_pair_mop_r2",
        "ree_mop_rmse",
        "ree_mop_r2",
        "ree_sign_accuracy",
        "selection_status",
        "operational_role",
        "native_uq_status",
        "operational_rank",
        "calibration_exclusion_reason",
    ]
    out["supplementary_workhorse_selection"] = op_direct[sel_keep].copy()
    return out


FULL_RETENTION_GRID_LABEL = "{0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00}"
FULL_RETENTION_AUC_DEFINITION = "trapz(metric, retention) on the prespecified 10-100% grid; not a continuous-retention estimate"


def write_retention_auc_summary() -> pd.DataFrame:
    src = REMAKE_ROOT / "results" / "downstream_cv" / "ad_retention" / "retention_aurc.csv"
    if not src.exists():
        return pd.DataFrame()
    out = pd.read_csv(src).copy()
    out["retention_grid"] = FULL_RETENTION_GRID_LABEL
    out["auc_definition"] = FULL_RETENTION_AUC_DEFINITION
    out["auc_mean"] = out["fold_mean"]
    out["auc_sd"] = out["fold_sd"]
    dest = REMAKE_ROOT / "results" / "downstream_cv" / "ad_retention" / "retention_auc_summary.csv"
    write_csv(dest, out)
    write_csv(TABLES / "retention_auc_summary.csv", out)
    return out


def ad_table() -> pd.DataFrame:
    aurc_path = REMAKE_ROOT / "results" / "downstream_cv" / "ad_retention" / "retention_auc_summary.csv"
    if not aurc_path.exists():
        aurc_path = REMAKE_ROOT / "results" / "downstream_cv" / "ad_retention" / "retention_aurc.csv"
    aurc = pd.read_csv(aurc_path)
    defs_path = REMAKE_ROOT / "results" / "downstream_cv" / "ad_retention" / "score_definitions.csv"
    if not defs_path.exists():
        defs_path = REMAKE_ROOT / "results" / "downstream_cv" / "ad" / "ad_score_definitions.csv"
    defs = pd.read_csv(defs_path) if defs_path.exists() else pd.DataFrame()
    dmap = {r.score_name: r for r in defs.itertuples(index=False)} if len(defs) and "score_name" in defs.columns else {}
    ret50_path = REMAKE_ROOT / "results" / "downstream_cv" / "ad_retention" / "retention_50pct.csv"
    ret50 = pd.read_csv(ret50_path) if ret50_path.exists() else pd.DataFrame()
    work = aurc[
        (aurc.dataset.astype(str) == "DS2")
        & (aurc.model_id.astype(str) == WORKHORSE)
        & (aurc.scope.astype(str).isin(["all_pairs", "ree_pairs"]))
    ].copy()
    rows = []

    def _fmt_hit(hit: pd.DataFrame) -> tuple[str, int]:
        if hit is None or hit.empty:
            return "NA", 0
        r = hit.iloc[0]
        n = int(r.n_valid_folds) if "n_valid_folds" in hit.columns and pd.notna(r.n_valid_folds) else 5
        return fmt(r.fold_mean, r.fold_sd, n), n

    def _mae50(scope: str, score: str, metric: str) -> str:
        if ret50 is None or ret50.empty:
            return "NA"
        hit = ret50[
            (ret50.dataset.astype(str) == "DS2")
            & (ret50.model_id.astype(str) == WORKHORSE)
            & (ret50.scope.astype(str) == scope)
            & (ret50.score_name.astype(str) == score)
            & (ret50.metric_name.astype(str) == metric)
        ]
        if hit.empty:
            return "NA"
        r = hit.iloc[0]
        n = int(r.n_valid_folds) if pd.notna(r.n_valid_folds) else 5
        return fmt(r.retained_mean, r.retained_sd, n)

    for scope in ("all_pairs", "ree_pairs"):
        scoped = work[work.scope.astype(str) == scope]
        for score in sorted(scoped.score_name.astype(str).unique()):
            mop = scoped[(scoped.score_name.astype(str) == score) & (scoped.metric_name.astype(str) == "retention_curve_auc_mop_rmse")]
            rowm = scoped[(scoped.score_name.astype(str) == score) & (scoped.metric_name.astype(str) == "retention_curve_auc_rmse")]
            if mop.empty and rowm.empty:
                continue
            meta = dmap.get(score)
            pair_aurc, n_pair = _fmt_hit(mop)
            row_aurc, n_row = _fmt_hit(rowm)
            pair_mae50 = _mae50(scope, score, "mop_rmse")
            row_mae50 = _mae50(scope, score, "rmse")
            rows.append(
                {
                    "scope": scope,
                    "score_name": score,
                    "score_display": score_display_name(score),
                    "definition": getattr(meta, "definition", "") if meta is not None else "",
                    "fit_partition": getattr(meta, "fit_partition", "") if meta is not None else "",
                    "orientation": getattr(meta, "orientation", "higher_is_greater_risk") if meta is not None else "higher_is_greater_risk",
                    "pair_aurc": pair_aurc,
                    "mop_rmse_at_50": pair_mae50,
                    "aurc": pair_aurc,
                    "rmse_at_50pct_retention": pair_mae50,
                    "row_aurc": row_aurc,
                    "row_rmse_at_50": row_mae50,
                    "row_rmse_at_50pct_retention": row_mae50,
                    "n_valid_folds": n_pair or n_row or 5,
                    "aggregation": "fold_mean_plus_sd_ddof1",
                    "comparison_type": "predeclared_descriptive_no_winner",
                    "aurc_estimand": "pair_level_mop_rmse_vs_retention_after_row_ranking",
                    "row_aurc_estimand": "row_rmse_vs_retention",
                    "retention_grid": FULL_RETENTION_GRID_LABEL,
                    "auc_definition": FULL_RETENTION_AUC_DEFINITION,
                }
            )
    return pd.DataFrame(rows)


def ad_retention_tables() -> dict[str, pd.DataFrame]:
    root = REMAKE_ROOT / "results" / "downstream_cv" / "ad_retention"
    out = {}
    mapping = {
        "ad_retention_50pct": "retention_50pct.csv",
        "ad_retention_summary": "retention_summary.csv",
        "ad_random_retention_summary": "random_retention_summary.csv",
        "ad_enrichment_summary": "enrichment_summary.csv",
        "ad_retention_aurc": "retention_aurc.csv",
        "retention_auc_summary": "retention_auc_summary.csv",
        "ad_utility": "utility_fold.csv",
        "ad_score_definitions": "score_definitions.csv",
        "ad_leakage_audit": "leakage_audit.csv",
    }
    for name, fname in mapping.items():
        path = root / fname
        out[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return out


def metal_medium_table() -> pd.DataFrame:
    path = REMAKE_ROOT / "results" / "downstream_cv" / "ad" / "ad_metal_medium_strata_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    s = pd.read_csv(path)
    s = s[s.metric_name.isin(["n_rows", "n_pairs", "rmse", "mop_rmse", "picp_90", "sign_accuracy"])].copy()
    displays = []
    for a, b, n, met in zip(s.fold_mean, s.fold_sd, s.n_valid_folds, s.metric_name):
        digits = 1 if met in {"n_rows", "n_pairs"} else 3
        displays.append(fmt(a, b, n, digits=digits))
    s["display"] = displays
    return s


def ad_distance_error_corr_display_table() -> pd.DataFrame:
    path = REMAKE_ROOT / "results" / "downstream_cv" / "ad_distance_error_corr" / "ad_distance_error_corr_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(path)
    if raw.empty:
        return pd.DataFrame()
    score_order = {
        "tanimoto_nn_distance": 0,
        "tanimoto_knn_mean_5": 1,
        "tanimoto_knn_mean_10": 2,
        "tanimoto_median_distance": 3,
        "mahalanobis_feature_centroid": 4,
        "mahalanobis_feature_nn": 5,
        "mahalanobis_feature_knn_mean_5": 6,
        "mahalanobis_feature_knn_mean_10": 7,
        "euclidean_feature_nn": 8,
        "euclidean_feature_median": 9,
        "chemprop_std_euclidean_nn": 10,
        "chemprop_std_euclidean_knn_5": 11,
        "chemprop_std_euclidean_knn_10": 12,
        "chemprop_euclidean_nn": 13,
        "chemprop_cosine_nn": 14,
        "chemprop_mahalanobis_centroid": 15,
        "chemprop_mahalanobis_nn": 16,
    }
    scope_order = {"all_pairs": 0, "ree_pairs": 1}
    model_order = {"DS2_RDKIT_MLPENS5_HETERO": 0, "DS2_GRAPH_DMPNN_EVID": 1}
    rows = []
    keys = [c for c in ("model_id", "scope", "score_name") if c in raw.columns]
    for key, g in raw.groupby(keys, dropna=False):
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        pear = g[g.metric_name.astype(str) == "pearson_abs_residual"]
        spear = g[g.metric_name.astype(str) == "spearman_abs_residual"]
        if pear.empty or spear.empty:
            continue
        pr, sr = pear.iloc[0], spear.iloc[0]
        n_valid = int(min(pr.n_valid_folds, sr.n_valid_folds))
        n_rows = pr.n_rows_used_fold_mean if "n_rows_used_fold_mean" in pear.columns else np.nan
        scope = rec.get("scope", "")
        score = rec.get("score_name", "")
        model_id = rec.get("model_id", pr.model_id if "model_id" in pr.index else "")
        rows.append(
            {
                "model_id": model_id,
                "scope": scope,
                "score_name": score,
                "score_display": score_display_name(score),
                "pearson_abs_residual": fmt(pr.fold_mean, pr.fold_sd, n_valid),
                "spearman_abs_residual": fmt(sr.fold_mean, sr.fold_sd, n_valid),
                "pearson_mean": pr.fold_mean,
                "pearson_sd": pr.fold_sd,
                "spearman_mean": sr.fold_mean,
                "spearman_sd": sr.fold_sd,
                "n_rows_used_fold_mean": n_rows,
                "n_valid_folds": n_valid,
                "error_definition": "row_level_absolute_residual",
                "role": "si_diagnostic_only",
                "used_for_selection": False,
                "_model_ord": model_order.get(str(model_id), 99),
                "_scope_ord": scope_order.get(str(scope), 99),
                "_score_ord": score_order.get(str(score), 99),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["_model_ord", "_scope_ord", "_score_ord"]).drop(
            columns=["_model_ord", "_scope_ord", "_score_ord"]
        )
    return out


def pair_density_display_table() -> pd.DataFrame:
    dens = pd.read_csv(REMAKE_ROOT / "results" / "downstream_cv" / "pair_density" / "pair_density_summary.csv")
    keep = dens[(dens.scope == "all_pairs") & (dens.metric_name.isin(["rmse", "mop_rmse", "picp_90", "mean_sigma_total", "n_rows", "n_pairs"]))].copy()
    keep["source_metric_name"] = keep["metric_name"]
    keep["metric_name"] = keep["metric_name"].replace({"n_rows": "n_test_rows", "n_pairs": "n_test_pairs"})
    keep["count_definition"] = keep["metric_name"].map(
        lambda m: (
            "fold mean of test rows or test pairs assigned to each training-density bin; "
            "not unique-pair counts pooled across folds; not retained MoP-pair counts"
            if m in {"n_test_rows", "n_test_pairs"}
            else ""
        )
    )
    order = {"0": 0, "1-5": 1, "6-10": 2, "11-25": 3, "26-50": 4, ">50": 5}
    keep["_ord"] = keep["bin"].map(order)
    keep = keep.sort_values(["metric_name", "_ord"])
    count_metrics = {"n_test_rows", "n_test_pairs", "n_rows", "n_pairs"}
    keep["display"] = [
        fmt(a, b, n, digits=1 if met in count_metrics else 3)
        for a, b, n, met in zip(keep.fold_mean, keep.fold_sd, keep.n_valid_folds, keep.metric_name)
    ]
    return keep.drop(columns=["_ord"])


def paired_difference_table() -> pd.DataFrame:
    path = MSRC / "workhorse_fold_paired_difference_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    s = pd.read_csv(path)
    s["display_name"] = [display_of(m) for m in s.comparator_model_id]
    if {"delta_all_pair_mop_rmse_mean", "delta_all_pair_mop_rmse_sd", "delta_all_pair_mop_rmse_n_valid"}.issubset(s.columns):
        s["delta_all_pair_mop_rmse"] = [fmt(a, b, n) for a, b, n in zip(s.delta_all_pair_mop_rmse_mean, s.delta_all_pair_mop_rmse_sd, s.delta_all_pair_mop_rmse_n_valid)]
    s["delta_ree_mop_rmse"] = [fmt(a, b, n) for a, b, n in zip(s.delta_ree_mop_rmse_mean, s.delta_ree_mop_rmse_sd, s.delta_ree_mop_rmse_n_valid)]
    s["delta_all_pair_picp_90"] = [fmt(a, b, n) for a, b, n in zip(s.delta_all_pair_picp_90_mean, s.delta_all_pair_picp_90_sd, s.delta_all_pair_picp_90_n_valid)]
    s["delta_all_pair_ace"] = [fmt(a, b, n) for a, b, n in zip(s.delta_all_pair_ace_mean, s.delta_all_pair_ace_sd, s.delta_all_pair_ace_n_valid)]
    return s


def mop_pair_validity_table() -> pd.DataFrame:
    path = MSRC / "workhorse_mop_pair_validity.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def grid_sigma_risk_table() -> pd.DataFrame:
    path = MSRC / "grid_sigma_total_risk_ordering.csv"
    if not path.exists():
        return pd.DataFrame()
    s = pd.read_csv(path)
    s["display_name"] = [display_of(m) for m in s.model_id]
    s["mop_aurc"] = [fmt(a, b, n) for a, b, n in zip(s.mop_aurc_mean, s.mop_aurc_sd, s.mop_aurc_n_valid)]
    s["row_aurc"] = [fmt(a, b, n) for a, b, n in zip(s.row_aurc_mean, s.row_aurc_sd, s.row_aurc_n_valid)]
    rmse_mean = "mop_rmse_at_50pct_retention_mean" if "mop_rmse_at_50pct_retention_mean" in s.columns else "mop_mae_at_50pct_retention_mean"
    rmse_sd = "mop_rmse_at_50pct_retention_sd" if "mop_rmse_at_50pct_retention_sd" in s.columns else "mop_mae_at_50pct_retention_sd"
    rmse_n = "mop_rmse_at_50pct_retention_n_valid" if "mop_rmse_at_50pct_retention_n_valid" in s.columns else "mop_mae_at_50pct_retention_n_valid"
    s["mop_rmse_at_50pct"] = [fmt(a, b, n) for a, b, n in zip(s[rmse_mean], s[rmse_sd], s[rmse_n])]
    return s.sort_values(["scope", "mop_aurc_mean"], kind="mergesort").reset_index(drop=True)


def write_cmi_ligand_overlap() -> dict:
    """Count shared ligands on the official CMI assignment versus project folds."""
    df = pd.read_csv(REMAKE_ROOT / "data" / "processed" / "m4_processed.csv", usecols=["ligand_smiles"])
    ext = pd.read_csv(REMAKE_ROOT / "data" / "processed" / "m4_external.csv", usecols=["ligand_smiles"])
    cmi = json.loads((REMAKE_ROOT / "data" / "splits" / "cmi_official_split.json").read_text(encoding="utf-8"))
    folds = json.loads((REMAKE_ROOT / "data" / "splits" / "project_folds.json").read_text(encoding="utf-8"))
    train = df.iloc[cmi["train_idx"]].copy()
    test = df.iloc[cmi["test_idx"]].copy()
    train["ligand"] = train["ligand_smiles"].astype(str)
    test["ligand"] = test["ligand_smiles"].astype(str)
    ext_l = set(ext["ligand_smiles"].astype(str))
    tr_l = set(train["ligand"])
    te_l = set(test["ligand"])
    fold_overlap = {}
    for fid, spec in folds.items():
        tr = set(df.iloc[spec["train_idx"]]["ligand_smiles"].astype(str))
        va = set(df.iloc[spec["val_idx"]]["ligand_smiles"].astype(str))
        te = set(df.iloc[spec["test_idx"]]["ligand_smiles"].astype(str))
        fold_overlap[fid] = {
            "train_test": len(tr & te),
            "train_val": len(tr & va),
            "val_test": len(va & te),
            "n_test_ligands": len(te),
            "n_test_rows": len(spec["test_idx"]),
        }
    out = {
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "n_ext_rows": int(len(ext)),
        "n_train_ligands": int(len(tr_l)),
        "n_test_ligands": int(len(te_l)),
        "n_ext_ligands": int(len(ext_l)),
        "n_ligands_shared_train_test": int(len(tr_l & te_l)),
        "frac_test_ligands_seen_in_train": float(len(tr_l & te_l) / len(te_l)) if te_l else None,
        "n_test_rows_with_train_ligand": int(test["ligand"].isin(tr_l).sum()),
        "frac_test_rows_with_train_ligand": float(test["ligand"].isin(tr_l).mean()),
        "n_train_rows_with_test_ligand": int(train["ligand"].isin(te_l).sum()),
        "n_ligands_shared_train_ext": int(len(tr_l & ext_l)),
        "n_ligands_shared_test_ext": int(len(te_l & ext_l)),
        "ligand_disjoint_project_folds": fold_overlap,
        "note": (
            "Official CMI train/test assignment is a random row split after a small ligand-held-out "
            "external set. Shared ligands mean the same ligand appears with other metals or conditions "
            "on both sides of the split."
        ),
    }
    dest = REMAKE_ROOT / "results" / "canonical" / "cmi_split_ligand_overlap.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def figure_registry() -> pd.DataFrame:
    from uq_remake.manuscript.figure_specs import FIGURE_SPECS

    return pd.DataFrame(
        [
            {
                "figure_id": s["figure_id"],
                "purpose": s["purpose"],
                "source_artifact": s["source_artifact"],
                "dataset": s["dataset"],
                "model_ids": s["model_ids"],
                "scope": s["scope"],
                "metric_name": s["metric_name"],
                "aggregation": s["aggregation"],
                "status": s["status"],
            }
            for s in FIGURE_SPECS
        ]
    )


def main() -> None:
    MAN.mkdir(parents=True, exist_ok=True)
    MSRC.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    log = []
    log.append(f"started {utc_now()}")
    archive_old_manuscript()
    log.append(f"archived old manuscript to {ARCHIVE.as_posix() if ARCHIVE.exists() else 'already present'}")
    write_cmi_ligand_overlap()
    log.append("wrote results/canonical/cmi_split_ligand_overlap.json")

    reg = json.loads((REMAKE_ROOT / "configs" / "experiments" / "active_registry.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(REMAKE_ROOT / "results" / "cv_metrics" / "fold_metric_summary.csv")
    fold_metrics = pd.read_csv(REMAKE_ROOT / "results" / "cv_metrics" / "fold_metrics.csv")
    if "metric_name" in summary.columns:
        summary = summary[~summary.metric_name.astype(str).str.contains("spearman", case=False, regex=False)].copy()
    if "metric_name" in fold_metrics.columns:
        fold_metrics = fold_metrics[~fold_metrics.metric_name.astype(str).str.contains("spearman", case=False, regex=False)].copy()
    eligibility = pd.read_csv(REMAKE_ROOT / "results" / "model_audit" / "workhorse_eligibility.csv")
    write_rule()
    ds2_sum = summary[summary["dataset"].astype(str).str.upper() == "DS2"]
    old_choice_path = REMAKE_ROOT / "results" / "model_audit" / "workhorse_selection.json"
    if old_choice_path.exists() and not (MSRC / "workhorse_selection_pre_rerun.json").exists():
        shutil.copy2(old_choice_path, MSRC / "workhorse_selection_pre_rerun.json")
    choice = select_workhorse_accuracy_calibration(ds2_sum, eligibility)
    global WORKHORSE
    if not choice.get("winner_computed") or not choice.get("selected"):
        raise RuntimeError(f"operational workhorse selection failed: {choice}")
    WORKHORSE = choice["selected"]
    choice["rerun_after_metric_corrections"] = True
    choice["rerun_utc"] = utc_now()
    paths = write_accuracy_calibration_artifacts(choice, MSRC)
    write_json(MSRC / "workhorse_selection_rerun.json", {k: v for k, v in choice.items() if k != "rows"})
    write_json(old_choice_path, {k: v for k, v in choice.items() if k != "rows"})
    log.append(
        f"operational workhorse selected={choice['selected']} n_usable={len(choice.get('ranked') or [])}"
    )
    from uq_remake.downstream.si_diagnostics import run_after_workhorse_selection

    run_after_workhorse_selection()
    log.append("wrote fold-wise paired differences among interval-usable DS2 models")
    write_retention_auc_summary()
    log.append("wrote full-grid retention_auc_summary.csv")

    grid_df, status_df = build_model_tables(reg, eligibility, choice)
    write_csv(TABLES / "model_grid_final.csv", grid_df)
    write_csv(TABLES / "model_status_final.csv", status_df)

    tables = build_ds_tables(summary, choice)
    for name, df in tables.items():
        write_csv(TABLES / f"{name}.csv", df)
    for name in (
        "workhorse_selection_accuracy_calibration.csv",
        "workhorse_selection_candidates.csv",
    ):
        src = MSRC / name
        if src.exists():
            shutil.copy2(src, TABLES / name)

    ad = ad_table()
    write_csv(TABLES / "main_ds2_ad_results.csv", ad)
    pair_keep = [
        "scope",
        "score_name",
        "score_display",
        "definition",
        "fit_partition",
        "orientation",
        "pair_aurc",
        "mop_rmse_at_50",
        "n_valid_folds",
        "aggregation",
        "aurc_estimand",
        "retention_grid",
        "auc_definition",
    ]
    row_keep = [
        "scope",
        "score_name",
        "score_display",
        "definition",
        "fit_partition",
        "orientation",
        "row_aurc",
        "row_rmse_at_50",
        "n_valid_folds",
        "aggregation",
        "row_aurc_estimand",
        "retention_grid",
        "auc_definition",
    ]
    pair_ad = ad[[c for c in pair_keep if c in ad.columns]].copy()
    row_ad = ad[[c for c in row_keep if c in ad.columns]].copy()
    write_csv(TABLES / "main_ds2_ad_pair_results.csv", pair_ad)
    write_csv(TABLES / "supplementary_ds2_ad_row_results.csv", row_ad)
    # supplementary_ds2_ad_row_level.csv used to be written here as a byte-identical
    # second copy of the line above under an older name. Nothing reads it: write_manuscript
    # and validate_manuscript both use the "_results" name. Dropped rather than maintained.
    support_src = REMAKE_ROOT / "results" / "downstream_cv" / "ad" / "ad_pair_level_retention_support.csv"
    if support_src.exists():
        support = pd.read_csv(support_src)
        if "n_pairs_with_at_least_3_rows" in support.columns and "n_pairs_with_at_least_3_retained" not in support.columns:
            support["n_pairs_with_at_least_3_retained"] = support["n_pairs_with_at_least_3_rows"]
        support = support.drop(columns=[c for c in support.columns if c in {"mop_mae", "mae", "row_mae"}], errors="ignore")
        write_csv(TABLES / "ad_pair_level_retention_support.csv", support)
    ad_ret = ad_retention_tables()
    for name, df in ad_ret.items():
        if len(df):
            out = df.copy()
            if "score_name" in out.columns:
                out["score_display"] = out["score_name"].map(score_display_name)
            if "family" in out.columns and "family_display" not in out.columns:
                out["family_display"] = out["family"].map(family_display_name)
            write_csv(TABLES / f"{name}.csv", out)
    mm = metal_medium_table()
    if len(mm):
        write_csv(TABLES / "supplementary_metal_medium_strata.csv", mm)
    dens_disp = pair_density_display_table()
    write_csv(TABLES / "supplementary_pair_density_bins.csv", dens_disp)
    corr_disp = ad_distance_error_corr_display_table()
    if len(corr_disp):
        write_csv(TABLES / "ad_distance_error_corr.csv", corr_disp)
    paired = paired_difference_table()
    if len(paired):
        write_csv(TABLES / "supplementary_fold_paired_differences.csv", paired)
    mop_valid = mop_pair_validity_table()
    if len(mop_valid):
        write_csv(TABLES / "supplementary_mop_pair_validity.csv", mop_valid)
    grid_risk = grid_sigma_risk_table()
    if len(grid_risk):
        write_csv(TABLES / "supplementary_grid_sigma_total_risk_ordering.csv", grid_risk)

    folds = fold_size_table()
    write_csv(TABLES / "fold_sizes.csv", folds)
    write_csv(TABLES / "supplementary_complete_model_results.csv", summary)
    write_csv(TABLES / "supplementary_fold_breakdowns.csv", fold_metrics)
    # model_status_final.csv is the name write_manuscript reads; the former
    # supplementary_model_status.csv was a byte-identical duplicate and is no longer written.
    exp_audit = REMAKE_ROOT / "results" / "audit" / "final" / "experiment_completeness.csv"
    exp_release = MSRC / "experiment_completeness.csv"
    exp_src = exp_audit if exp_audit.exists() else exp_release
    if not exp_src.exists():
        raise FileNotFoundError(
            "experiment completeness inventory is missing; expected "
            f"{exp_audit.as_posix()} or {exp_release.as_posix()}"
        )
    exp = pd.read_csv(exp_src)
    stale = {"nested_workhorse_crc", "top_k_screening", "topk_utility_screening", "topk_directional_screening"}
    exp = exp[~exp.experiment_id.astype(str).isin(stale)].copy()
    exp["experiment_id"] = exp["experiment_id"].replace({"fold_local_risk_coverage": "fold_local_retention_error"})
    if "known_limitations" in exp.columns:
        exp["known_limitations"] = (
            exp["known_limitations"]
            .astype(str)
            .str.replace("computed in S8.3 as a small-slice diagnostic", "computed in S8.2 as a small-slice diagnostic", regex=False)
        )
    chem_lim = exp["experiment_id"].astype(str) == "chemprop_ds2_five_fold_reevaluation"
    if "known_limitations" in exp.columns and chem_lim.any():
        exp.loc[chem_lim, "known_limitations"] = (
            exp.loc[chem_lim, "known_limitations"]
            .astype(str)
            .str.replace("ranking complete_5_of_5", "pair_metric_status complete_5_of_5", regex=False)
        )
    exp = exp[~exp.experiment_id.astype(str).isin({"fold_local_retention_error", "fold_local_risk_coverage"})].copy()
    def _interp(status) -> str:
        s = str(status)
        return "canonical" if s in {"complete", "partial"} else s

    exp["computation_status"] = exp["status"]
    exp["interpretation_status"] = exp["status"].map(_interp)
    ad_row = {
        "experiment_id": "ad_retention_study",
        "scientific_question": "Empirical AD error-ordering at fixed retention",
        "dataset": "DS1/DS2",
        "model": "workhorse_and_chemprop",
        "status": "complete",
        "computation_status": "complete",
        "interpretation_status": "canonical",
        "canonical_artifact": "results/downstream_cv/ad_retention/fold_retention.csv",
        "aggregation_method": "fold_mean_plus_sd",
        "known_limitations": "canonical 10-100% fold-local study; plotted means in retention_summary.csv; AUC in retention_auc_summary.csv; 50% table in retention_50pct.csv",
    }
    adhoc_val_path = REMAKE_ROOT / "results" / "downstream_cv" / "ad_hoc_selectivity" / "ad_hoc_validation.json"
    adhoc_complete = False
    if adhoc_val_path.exists():
        adhoc_payload = json.loads(adhoc_val_path.read_text(encoding="utf-8"))
        adhoc_complete = bool(adhoc_payload.get("pass")) and adhoc_payload.get("computation_status") == "complete"
    corr_path = REMAKE_ROOT / "results" / "downstream_cv" / "ad_distance_error_corr" / "ad_distance_error_corr_summary.csv"
    corr_row = {
        "experiment_id": "ad_distance_error_corr",
        "scientific_question": "Are ligand-level molecular-distance scores associated with row-level absolute residual on held-out DS2?",
        "dataset": "DS2",
        "model": "workhorse",
        "status": "complete" if corr_path.exists() else "not_computed",
        "computation_status": "complete" if corr_path.exists() else "not_computed",
        "interpretation_status": "canonical" if corr_path.exists() else "not_computed",
        "canonical_artifact": "results/downstream_cv/ad_distance_error_corr/ad_distance_error_corr_summary.csv",
        "aggregation_method": "fold_mean_plus_sd",
        "known_limitations": "descriptive association between molecular-distance scores and row-level absolute residual $|y-\\widehat y|$; not a selection metric; not a substitute for fixed-retention error ordering",
    }
    adhoc_row = {
        "experiment_id": "adhoc_differencing",
        "scientific_question": "Is ad hoc differencing of DS1 absolute-binding predictions an adequate substitute for direct DS2 selectivity modeling?",
        "dataset": "DS1/DS2",
        "model": "ds1_uq_and_direct_ds2",
        "status": "complete" if adhoc_complete else "not_computed",
        "computation_status": "complete" if adhoc_complete else "not_computed",
        "interpretation_status": "canonical" if adhoc_complete else "not_computed",
        "canonical_artifact": "results/downstream_cv/ad_hoc_selectivity/ad_hoc_selectivity_summary.csv",
        "aggregation_method": "fold_mean_plus_sd",
        "known_limitations": "aligned DS2-fold comparison; independent-error propagation of two absolute-binding predictive standard deviations; restricted formulation diagnostic; not a pair-acquisition study",
    }
    pair_acq_val = REMAKE_ROOT / "results" / "downstream_cv" / "pair_acquisition" / "pair_acquisition_validation.json"
    pair_acq_complete = False
    if pair_acq_val.exists():
        pair_acq_payload = json.loads(pair_acq_val.read_text(encoding="utf-8"))
        pair_acq_complete = bool(pair_acq_payload.get("pass")) and pair_acq_payload.get("computation_status") == "complete"
    pair_acq_row = {
        "experiment_id": "sigma_pair_acquisition",
        "scientific_question": "Does sigma-only top-k ranking recover high-selectivity and near-zero selectivity ligands within held-out metal-species pairs?",
        "dataset": "DS2",
        "model": "workhorse",
        "status": "complete" if pair_acq_complete else "not_computed",
        "computation_status": "complete" if pair_acq_complete else "not_computed",
        "interpretation_status": "canonical" if pair_acq_complete else "not_computed",
        "canonical_artifact": "results/downstream_cv/pair_acquisition/pair_acquisition_summary.csv",
        "aggregation_method": "pair_equal_then_fold_mean_plus_sd",
        "known_limitations": "n_event>" + str(int(PAIR_ACQ_EVENT_MIN_EXCLUSIVE)) + "; sigma-only ranking; not a distance AD; not a prospective campaign",
    }
    pair_acq_sens_val = REMAKE_ROOT / "results" / "downstream_cv" / "pair_acquisition_tau_2_0p5" / "pair_acquisition_validation.json"
    pair_acq_sens_complete = False
    if pair_acq_sens_val.exists():
        pair_acq_sens_payload = json.loads(pair_acq_sens_val.read_text(encoding="utf-8"))
        pair_acq_sens_complete = bool(pair_acq_sens_payload.get("pass")) and pair_acq_sens_payload.get("computation_status") == "complete"
    pair_acq_sens_row = {
        "experiment_id": "pair_acquisition_threshold_2_half",
        "scientific_question": "Using the same stored fold-local scores, does sigma-only top-k ranking recover ligands with observed high selectivity (target-direction Delta log K > 2) and near-zero selectivity (|Delta log K| < 0.5)?",
        "dataset": "DS2",
        "model": "workhorse",
        "status": "complete" if pair_acq_sens_complete else "not_computed",
        "computation_status": "complete" if pair_acq_sens_complete else "not_computed",
        "interpretation_status": "supporting" if pair_acq_sens_complete else "not_computed",
        "canonical_artifact": "results/downstream_cv/pair_acquisition_tau_2_0p5/pair_acquisition_summary.csv",
        "aggregation_method": "pair_equal_then_fold_mean_plus_sd",
        "known_limitations": "supporting sensitivity; high event y>2; near-zero $|y|<0.5$; does not replace Figure 5 or Table 4; n_event>" + str(int(PAIR_ACQ_EVENT_MIN_EXCLUSIVE)) + "; sigma-only ranking; not prospective",
    }
    planned_rows = [
        {
            "experiment_id": "tanimoto_stratified_vs_ligand_disjoint",
            "scientific_question": "Does Tanimoto-stratified GroupKFold change ligand-disjoint DS1 log K error relative to the current folds?",
            "dataset": "DS1",
            "model": "DS1_GRAPH_DMPNN_POINT;DS1_RDKIT_MLP_POINT",
            "status": "not_computed",
            "computation_status": "not_computed",
            "interpretation_status": "not_computed",
            "canonical_artifact": "",
            "aggregation_method": "",
            "known_limitations": "not_computed; current Tanimoto–residual association is already near zero",
        },
        {
            "experiment_id": "cmi_competitor_ligand_disjoint_grid",
            "scientific_question": "Do CMI Random Forest, ChemBERTa, and MolBERT recover official-split log K accuracy under ligand-disjoint CV?",
            "dataset": "DS1",
            "model": "unspecified",
            "status": "not_computed",
            "computation_status": "not_computed",
            "interpretation_status": "not_computed",
            "canonical_artifact": "",
            "aggregation_method": "",
            "known_limitations": "not_computed; only Chemprop point has an official-split retrain",
        },
        {
            "experiment_id": "pair_acquisition_on_pareto_neighbors",
            "scientific_question": "Do other RMSE–PICP–ACE Pareto models match workhorse pair-acquisition event precision@$k$ and true hit@$k$?",
            "dataset": "DS2",
            "model": "pareto_neighbors",
            "status": "not_computed",
            "computation_status": "not_computed",
            "interpretation_status": "not_computed",
            "canonical_artifact": "",
            "aggregation_method": "",
            "known_limitations": "not_computed; workhorse was not selected for the ranking task",
        },
        {
            "experiment_id": "prospective_pair_acquisition_campaign",
            "scientific_question": "Do within-pair shortlists recover high- or near-zero-selectivity ligands in new experiments?",
            "dataset": "DS2",
            "model": "workhorse",
            "status": "not_computed",
            "computation_status": "not_computed",
            "interpretation_status": "not_computed",
            "canonical_artifact": "",
            "aggregation_method": "",
            "known_limitations": "not_computed; current metrics are retrospective held-out ranking",
        },
        {
            "experiment_id": "gaussian_expected_improvement_integral",
            "scientific_question": "Does the Gaussian expected-improvement integral change pair-acquisition event precision@$k$ or true hit@$k$ relative to the specified standardized margin?",
            "dataset": "DS2",
            "model": "workhorse",
            "status": "not_computed",
            "computation_status": "not_computed",
            "interpretation_status": "not_computed",
            "canonical_artifact": "",
            "aggregation_method": "",
            "known_limitations": "not_computed; current score is the standardized threshold-margin (yhat-threshold)/sigma",
        },
    ]
    for row in (ad_row, adhoc_row, corr_row, pair_acq_row, pair_acq_sens_row, *planned_rows):
        for k in row:
            if k not in exp.columns:
                exp[k] = ""
        exp = exp[exp.experiment_id.astype(str) != row["experiment_id"]]
        exp = pd.concat([exp, pd.DataFrame([row])], ignore_index=True, sort=False)
    exp = exp.fillna("")
    exp = exp.drop_duplicates(subset=["experiment_id"], keep="last")
    write_csv(TABLES / "supplementary_experiment_status.csv", exp)
    write_csv(MSRC / "experiment_completeness.csv", exp)

    figreg = figure_registry()
    write_csv(MAN / "figure_data_registry.csv", figreg)
    write_csv(TABLES / "figure_data_registry.csv", figreg)

    def _read_json(*rels: str) -> dict:
        for rel in rels:
            path = REMAKE_ROOT / rel
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        raise FileNotFoundError(rels)

    ds1c = _read_json("results/canonical/ds1_composition.json")
    ds2c = _read_json("results/canonical/ds2_composition.json")
    hplus = _read_json("results/canonical/hplus_decision.json")
    ds2_counts = ds2_scope_counts()
    schema1 = json.loads((REMAKE_ROOT / "data" / "processed" / "features" / "ds1_rdkit_schema.json").read_text(encoding="utf-8"))
    schema2 = json.loads((REMAKE_ROOT / "data" / "processed" / "features" / "ds2_rdkit_schema.json").read_text(encoding="utf-8"))

    snapshot_rows = []
    claims = []
    sources = []

    def add_claim(**kwargs):
        claims.append(kwargs)
        snapshot_rows.append(
            {
                "claim_id": kwargs["claim_id"],
                "dataset": kwargs.get("dataset", ""),
                "model_id": kwargs.get("model_id", ""),
                "experiment_id": kwargs.get("experiment_id", ""),
                "scope": kwargs.get("scope", ""),
                "metric_name": kwargs.get("metric_name", ""),
                "value_display": kwargs.get("value_display", ""),
                "fold_mean": kwargs.get("fold_mean", ""),
                "fold_sd": kwargs.get("fold_sd", ""),
                "n_valid_folds": kwargs.get("n_valid_folds", ""),
                "status": kwargs.get("status", ""),
            }
        )
        sources.append(
            {
                "claim_id": kwargs["claim_id"],
                "dataset": kwargs.get("dataset", ""),
                "model_id": kwargs.get("model_id", ""),
                "experiment_id": kwargs.get("experiment_id", ""),
                "scope": kwargs.get("scope", ""),
                "metric_name": kwargs.get("metric_name", ""),
                "artifact_path": kwargs.get("artifact_path", ""),
                "fold_level_source": kwargs.get("fold_level_source", ""),
                "aggregation_method": kwargs.get("aggregation_method", ""),
                "status": kwargs.get("status", ""),
            }
        )

    add_claim(
        claim_id="ds1_n_records",
        dataset="DS1",
        experiment_id="data",
        metric_name="n_records",
        value_display=str(ds1c["n_records"]),
        artifact_path="results/canonical/ds1_composition.json",
        aggregation_method="dataset_count",
        status="canonical_cv_result",
    )
    add_claim(
        claim_id="ds1_n_ligands",
        dataset="DS1",
        experiment_id="data",
        metric_name="n_unique_ligands",
        value_display=str(ds1c["n_unique_ligands"]),
        artifact_path="results/canonical/ds1_composition.json",
        aggregation_method="dataset_count",
        status="canonical_cv_result",
    )
    add_claim(
        claim_id="ds1_n_metals",
        dataset="DS1",
        experiment_id="data",
        metric_name="n_metals",
        value_display=str(ds1c["n_metals"]),
        artifact_path="results/canonical/ds1_composition.json",
        aggregation_method="dataset_count",
        status="canonical_cv_result",
    )
    add_claim(
        claim_id="ds1_n_metal_complex",
        dataset="DS1",
        experiment_id="data",
        metric_name="n_metal_complex",
        value_display=str(ds1c["family_breakdown"]["metal_complex"]),
        artifact_path="results/canonical/ds1_composition.json",
        aggregation_method="dataset_count",
        status="canonical_cv_result",
    )
    add_claim(
        claim_id="ds1_n_protonation",
        dataset="DS1",
        experiment_id="data",
        metric_name="n_protonation",
        value_display=str(ds1c["family_breakdown"]["protonation"]),
        artifact_path="results/canonical/ds1_composition.json",
        aggregation_method="dataset_count",
        status="canonical_cv_result",
    )
    add_claim(
        claim_id="ds2_n_rows",
        dataset="DS2",
        experiment_id="data",
        metric_name="n_pairs_rows",
        value_display=str(ds2c["n_pairs_rows"]),
        artifact_path="results/canonical/ds2_composition.json",
        aggregation_method="dataset_count",
        status="canonical_cv_result",
    )
    add_claim(
        claim_id="ds2_n_ligands",
        dataset="DS2",
        experiment_id="data",
        metric_name="n_unique_ligands",
        value_display=str(ds2c["n_unique_ligands"]),
        artifact_path="results/canonical/ds2_composition.json",
        aggregation_method="dataset_count",
        status="canonical_cv_result",
    )
    add_claim(
        claim_id="ds2_n_species_pairs",
        dataset="DS2",
        experiment_id="data",
        metric_name="n_metal_species_pairs",
        value_display=str(ds2c["n_metal_species_pairs"]),
        artifact_path="results/canonical/ds2_composition.json",
        aggregation_method="dataset_count",
        status="canonical_cv_result",
    )
    add_claim(
        claim_id="chemprop_ds2_invalid_nig",
        dataset="DS2",
        model_id="DS2_GRAPH_DMPNN_EVID",
        experiment_id="chemprop_ds2_five_fold_reevaluation",
        metric_name="n_uq_invalid_rows_total",
        value_display="13",
        artifact_path="results/cv_metrics/chemprop_evidential_status_v2.csv",
        fold_level_source="results/cv_metrics/chemprop_evidential_uq_diagnostics_v2.csv",
        aggregation_method="sum_over_folds",
        status="partial_result",
    )
    add_claim(
        claim_id="workhorse_id",
        dataset="DS2",
        model_id=WORKHORSE,
        experiment_id="workhorse_reproduction",
        metric_name="selected_model",
        value_display=WORKHORSE,
        artifact_path="results/manuscript_final/workhorse_selection_accuracy_calibration.json",
        aggregation_method="operational_interval_usability_screen_then_rmse_picp_ace_pareto",
        status="canonical_cv_result",
    )

    headline_specs = [
        (WORKHORSE, "all_pairs", "mop_rmse", "ds2_point_uq_grid"),
        (WORKHORSE, "all_pairs", "mop_r2", "ds2_point_uq_grid"),
        (WORKHORSE, "all_pairs", "sign_accuracy_all", "ds2_point_uq_grid"),
        (WORKHORSE, "all_pairs", "mop_picp_90", "ds2_point_uq_grid"),
        (WORKHORSE, "all_pairs", "mop_ace", "ds2_point_uq_grid"),
        (WORKHORSE, "all_pairs", "marginal_picp_90", "ds2_point_uq_grid"),
        (WORKHORSE, "all_pairs", "marginal_ace", "ds2_point_uq_grid"),
        (WORKHORSE, "ree_pairs", "mop_rmse", "ds2_point_uq_grid"),
        (WORKHORSE, "ree_pairs", "rmse", "ds2_point_uq_grid"),
        (WORKHORSE, "ree_pairs", "r2", "ds2_point_uq_grid"),
        (WORKHORSE, "ree_pairs", "mop_r2", "ds2_point_uq_grid"),
        (WORKHORSE, "ree_pairs", "sign_accuracy_all", "ds2_point_uq_grid"),
        (WORKHORSE, "all_pairs", "rmse", "ds2_point_uq_grid"),
        ("DS2_GRAPH_DMPNN_EVID", "ree_pairs", "mop_rmse", "chemprop_ds2_five_fold_reevaluation"),
        ("DS2_GRAPH_DMPNN_EVID", "ree_pairs", "sign_accuracy_all", "chemprop_ds2_five_fold_reevaluation"),
        ("DS2_GRAPH_DMPNN_EVID", "all_pairs", "marginal_picp_90", "chemprop_ds2_five_fold_reevaluation"),
        ("DS1_GRAPH_DMPNN_POINT", "metal_complex", "rmse", "ds1_point_model_comparison"),
        ("DS1_RDKIT_MLP_POINT", "metal_complex", "rmse", "ds1_point_model_comparison"),
        ("DS1_RDKIT_XGB_POINT", "metal_complex", "rmse", "ds1_point_model_comparison"),
        ("DS1_RDKIT_MLPENS5_EVID", "metal_complex", "rmse", "ds1_uq_model_comparison"),
        ("DS1_RDKIT_MLPENS5_EVID", "metal_complex", "marginal_picp_90", "ds1_uq_model_comparison"),
    ]
    seen_ids = set()
    for mid, scope, metric, exp in headline_specs:
        d = lookup(summary, mid, scope, metric)
        cid = f"{mid}|{scope}|{metric}"
        seen_ids.add(cid)
        add_claim(
            claim_id=cid,
            dataset=mid.split("_")[0],
            model_id=mid,
            experiment_id=exp,
            scope=scope,
            metric_name=metric,
            value_display=fmt(d["fold_mean"], d["fold_sd"], d["n_valid_folds"]),
            fold_mean=d["fold_mean"],
            fold_sd=d["fold_sd"],
            n_valid_folds=d["n_valid_folds"],
            artifact_path="results/cv_metrics/fold_metric_summary.csv",
            fold_level_source="results/cv_metrics/fold_metrics.csv",
            aggregation_method="fold_mean_plus_sd_ddof1",
            status="canonical_cv_result" if d["n_valid_folds"] == 5 else "partial_result",
        )
    for r in summary.itertuples(index=False):
        if "spearman" in str(r.metric_name).lower():
            continue
        cid = f"{r.model_id}|{r.scope}|{r.metric_name}"
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        n_valid = int(r.n_valid_folds) if pd.notna(r.n_valid_folds) else 0
        add_claim(
            claim_id=cid,
            dataset=str(r.dataset),
            model_id=str(r.model_id),
            experiment_id="canonical_fold_metrics",
            scope=str(r.scope),
            metric_name=str(r.metric_name),
            value_display=fmt(r.fold_mean, r.fold_sd, n_valid),
            fold_mean=r.fold_mean,
            fold_sd=r.fold_sd,
            n_valid_folds=n_valid,
            artifact_path="results/cv_metrics/fold_metric_summary.csv",
            fold_level_source="results/cv_metrics/fold_metrics.csv",
            aggregation_method="fold_mean_plus_sd_ddof1",
            status="canonical_cv_result" if n_valid == 5 else ("partial_result" if n_valid > 0 else "not_computed"),
        )
    ret50 = ad_ret.get("ad_retention_50pct", pd.DataFrame())
    if len(ret50):
        for r in ret50.itertuples(index=False):
                if "spearman" in str(getattr(r, "metric_name", "")).lower():
                    continue
                if int(getattr(r, "n_valid_folds", 0) or 0) == 0:
                    continue
                n_valid = int(r.n_valid_folds)
                mean = getattr(r, "retained_mean", None)
                sd = getattr(r, "retained_sd", None)
                add_claim(
                    claim_id=f"ad50|{r.dataset}|{r.model_id}|{r.score_name}|{getattr(r, 'scope', '')}|{r.metric_name}",
                    dataset=str(r.dataset),
                    model_id=str(r.model_id),
                    experiment_id="ad_retention_study",
                    scope=str(getattr(r, "scope", "retention_50pct")),
                    metric_name=str(r.metric_name),
                    value_display=fmt(mean, sd, n_valid),
                    fold_mean=mean,
                    fold_sd=sd,
                    n_valid_folds=n_valid,
                artifact_path="results/downstream_cv/ad_retention/retention_50pct.csv",
                fold_level_source="results/downstream_cv/ad_retention/fold_retention.csv",
                aggregation_method="fold_mean_plus_sd_ddof1",
                status="canonical_fold_result" if n_valid == 5 else "partial_result",
            )

    # AD AURC claims (canonical 10-100% grid)
    for score in ("sigma_total", "tanimoto_nn_distance", "mahalanobis_feature_centroid", "pair_density_risk"):
        hit = ad[(ad.scope == "all_pairs") & (ad.score_name == score)]
        if len(hit):
            add_claim(
                claim_id=f"ad_aurc_all_pairs|{score}",
                dataset="DS2",
                model_id=WORKHORSE,
                experiment_id="fold_local_retention_error",
                scope="all_pairs",
                metric_name=f"pair_aurc[{score}]",
                value_display=hit.iloc[0]["pair_aurc"] if "pair_aurc" in hit.columns else hit.iloc[0]["aurc"],
                artifact_path="results/downstream_cv/ad_retention/retention_auc_summary.csv",
                fold_level_source="results/downstream_cv/ad_retention/fold_retention.csv",
                aggregation_method="fold_mean_plus_sd_ddof1",
                status="canonical_fold_result",
            )
            add_claim(
                claim_id=f"ad_row_aurc_all_pairs|{score}",
                dataset="DS2",
                model_id=WORKHORSE,
                experiment_id="fold_local_retention_error_row",
                scope="all_pairs",
                metric_name=f"row_aurc[{score}]",
                value_display=hit.iloc[0]["row_aurc"] if "row_aurc" in hit.columns else hit.iloc[0]["aurc"],
                artifact_path="results/downstream_cv/ad_retention/retention_auc_summary.csv",
                fold_level_source="results/downstream_cv/ad_retention/fold_retention.csv",
                aggregation_method="fold_mean_plus_sd_ddof1",
                status="canonical_fold_result",
            )

    write_csv(MSRC / "manuscript_claim_registry.csv", pd.DataFrame(claims))
    write_csv(MAN / "claim_registry.csv", pd.DataFrame(claims))
    write_csv(MSRC / "manuscript_metric_snapshot.csv", pd.DataFrame(snapshot_rows))
    write_csv(MSRC / "manuscript_source_registry.csv", pd.DataFrame(sources))

    art_status = []
    for rel, desc, status in [
        ("results/cv_metrics/fold_metric_summary.csv", "Fold mean±SD metrics", "canonical_cv_result"),
        ("results/cv_metrics/fold_metrics.csv", "Fold-level metrics", "canonical_fold_result"),
        ("results/cv_metrics/chemprop_evidential_status_v2.csv", "Chemprop DS2 evidential v2 status", "partial_result"),
        ("results/downstream_cv/ad_retention/retention_summary.csv", "AD quantile retention summary", "canonical_fold_result"),
        ("results/downstream_cv/ad_hoc_selectivity/ad_hoc_selectivity_summary.csv", "Aligned ad hoc vs direct selectivity summary", "canonical_fold_result"),
        ("results/downstream_cv/ad_hoc_selectivity/ad_hoc_selectivity_fold_metrics.csv", "Aligned ad hoc vs direct fold metrics", "canonical_fold_result"),
        ("results/downstream_cv/ad_retention/retention_50pct.csv", "AD 50% retention comparison", "canonical_fold_result"),
        ("results/downstream_cv/ad_retention/retention_auc_summary.csv", "Full-grid retention-curve AUC summary", "canonical_fold_result"),
        ("results/downstream_cv/ad/ad_metal_medium_strata_summary.csv", "Metal-medium seen/unseen", "canonical_fold_result"),
        ("results/downstream_cv/ad_distance_error_corr/ad_distance_error_corr_summary.csv", "Molecular-distance score vs row-level absolute-residual association", "canonical_fold_result"),
        ("results/canonical/ds1_composition.json", "DS1 composition", "canonical_cv_result"),
        ("results/canonical/ds2_composition.json", "DS2 composition", "canonical_cv_result"),
        ("checkpoints/active/DS1_RDKIT_GPR/fold0/member_0/infeasible.json", "Exact GPR infeasible", "excluded"),
        ("results/downstream_cv/replicate_spread", "Replicate-spread vs UQ", "not_identifiable"),
        ("results/downstream_cv/ad/ad_medium_identifiability.csv", "Medium strata", "not_identifiable"),
    ]:
        art_status.append({"artifact_path": rel, "description": desc, "status": status, "exists": (REMAKE_ROOT / rel).exists()})

    write_csv(MSRC / "manuscript_artifact_status.csv", pd.DataFrame(art_status))
    shutil.copy2(MSRC / "manuscript_artifact_status.csv", MAN / "manuscript_artifact_status.csv")
    shutil.copy2(MSRC / "manuscript_source_registry.csv", MAN / "manuscript_source_registry.csv")
    shutil.copy2(MSRC / "manuscript_metric_snapshot.csv", MAN / "manuscript_metric_snapshot.csv")

    numbers = {
        "grid_id": GRID,
        "workhorse_model_id": WORKHORSE,
        "workhorse_display": display_of(WORKHORSE),
        "manuscript_accuracy_metrics": "all_pair_mop_rmse primary; displayed all_pair_picp_90/all_pair_ace are MoP; operational screen uses row-level PICP/ACE; all_pair_mop_r2 and all_pair_row_r2 auxiliary; two named ad hoc comparators; ree_mop_rmse usage example",
        "coded_selection_metric": "operational_interval_usability_screen then RMSE-PICP-ACE Pareto",
        "primary_metric": "all_pairs mop_rmse after operational interval-usability screen and RMSE-PICP-ACE Pareto selection",
        "selection_type": "operational_accuracy_calibration_compromise",
        "r2_role": "auxiliary_diagnostic",
        "sign_accuracy_role": "directional_diagnostic",
        "rank_correlation_role": "not_used",
        "ad_distance_error_corr_role": "si_diagnostic_only",
        "pair_acquisition_status": "sigma_only_n_event_gt_" + str(int(PAIR_ACQ_EVENT_MIN_EXCLUSIVE)),
        "ad_retention_status": "fold_local empirical error-ordering diagnostic; not ligand-selection evidence",
        "row_level_metrics_used_for_selection": False,
        "selection_nested": False,
        "retraining_performed": False,
        "selection_rerun": True,
        "chemprop_rank": "not_operational_candidate_partial_native_uq",
        "ad_retention_status": "fold_local empirical error-ordering diagnostic; not ligand-selection evidence",
        "ds1": ds1c,
        "ds2": ds2c,
        "ds2_scope_counts": ds2_counts,
        "hplus_decision": hplus["decision"],
        "hplus_r2_with": fmt(hplus["summary"][0]["fold_mean"], hplus["summary"][0]["fold_sd"]),
        "hplus_r2_without": fmt(hplus["summary"][3]["fold_mean"], hplus["summary"][3]["fold_sd"]),
        "rdkit_ds1_dim": schema1["feature_dimension"],
        "rdkit_ds2_dim": schema2["feature_dimension"],
        "rdkit_n_selected": schema1.get("rdkit_n_descriptors", 187),
        "chemprop_ds1_extra": 16,
        "chemprop_ds2_extra": 27,
        "mc_samples": 30,
        "bayes_samples": 30,
        "ensemble_size": 5,
        "gpr_inducing": 512,
        "sngp_rff": 512,
        "min_pair_n": 3,
        "high_sel_tau": 1.0,
        "fold_sd_ddof": 1,
        "imputation": False,
        "pca": False,
        "target_transform": "none_native_units",
        "ad_primary_scores_workhorse": ["sigma_total", "tanimoto_nn_distance", "mahalanobis_feature_centroid"],
        "ad_primary_scores_chemprop": [
            "sigma_total",
            "tanimoto_nn_distance",
            "chemprop_std_euclidean_nn",
            "chemprop_mahalanobis_centroid",
        ],
        "ad_retention_grid": [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
        "ad_primary_retention": 0.50,
        "ad_status": (
            "fold-local retention analysis (empirical error ordering); 50% retention is the prespecified RMSE comparison; "
            "sigma-only pair acquisition is the primary screening analysis; not a formal coverage or accept/reject guarantee"
        ),
        "not_computed": [
            "feature_ablations",
            "source_tag_holdout",
            "metal_pair_extrapolation",
            "random_vs_ligand_disjoint",
            "posthoc_variance_calibration",
            "restricted_mape_pooled",
            "ree_highselectivity_combined_scope",
            "repeated_ligand_disjoint_cv",
            "tanimoto_stratified_vs_ligand_disjoint",
            "cmi_competitor_ligand_disjoint_grid",
            "pair_acquisition_on_pareto_neighbors",
            "prospective_pair_acquisition_campaign",
            "gaussian_expected_improvement_integral",
        ],
        "not_identifiable": ["replicate_spread", "medium_ligand_medium_strata"],
        "excluded": ["exact_GPR", "chemprop_mc_dropout", "BAYES_LASTLAYER", "ranking_aux"],
        "workhorse_all_pair_mop_rmse": lookup(summary, WORKHORSE, "all_pairs", "mop_rmse"),
        "workhorse_all_pair_mop_r2": lookup(summary, WORKHORSE, "all_pairs", "mop_r2"),
        "workhorse_all_pair_sign": lookup(summary, WORKHORSE, "all_pairs", "sign_accuracy_all"),
        "workhorse_ree_mop_rmse": lookup(summary, WORKHORSE, "ree_pairs", "mop_rmse"),
        "workhorse_ree_mop_r2": lookup(summary, WORKHORSE, "ree_pairs", "mop_r2"),
        "workhorse_ree_row_rmse": lookup(summary, WORKHORSE, "ree_pairs", "rmse"),
        "workhorse_ree_row_r2": lookup(summary, WORKHORSE, "ree_pairs", "r2"),
        "workhorse_ree_sign": lookup(summary, WORKHORSE, "ree_pairs", "sign_accuracy_all"),
        "workhorse_all_picp": lookup(summary, WORKHORSE, "all_pairs", "marginal_picp_90"),
        "workhorse_all_ace": lookup(summary, WORKHORSE, "all_pairs", "marginal_ace"),
        "workhorse_all_mop_picp": lookup(summary, WORKHORSE, "all_pairs", "mop_picp_90"),
        "workhorse_all_mop_ace": lookup(summary, WORKHORSE, "all_pairs", "mop_ace"),
        "workhorse_ree_picp": lookup(summary, WORKHORSE, "ree_pairs", "marginal_picp_90"),
        "workhorse_ree_ace": lookup(summary, WORKHORSE, "ree_pairs", "marginal_ace"),
        "ranked": choice["ranked"],
    }
    r2_inv = fold_metrics[
        (fold_metrics.model_id == WORKHORSE)
        & (fold_metrics.scope == "ree_pairs")
        & (fold_metrics.metric_name == "mop_r2")
    ]
    numbers["workhorse_ree_mop_r2_n_pairs_invalid"] = [int(x) for x in r2_inv["n_pairs_invalid"].tolist()]
    numbers["workhorse_ree_mop_r2_n_pairs_valid"] = [int(x) for x in r2_inv["n_pairs_valid"].tolist()]
    # formatted convenience
    numbers["fmt"] = {
        "wh_mop_rmse": fmt(numbers["workhorse_all_pair_mop_rmse"]["fold_mean"], numbers["workhorse_all_pair_mop_rmse"]["fold_sd"], numbers["workhorse_all_pair_mop_rmse"]["n_valid_folds"]),
        "wh_mop_r2": fmt(numbers["workhorse_all_pair_mop_r2"]["fold_mean"], numbers["workhorse_all_pair_mop_r2"]["fold_sd"], numbers["workhorse_all_pair_mop_r2"]["n_valid_folds"]),
        "wh_ree_mop_rmse": fmt(numbers["workhorse_ree_mop_rmse"]["fold_mean"], numbers["workhorse_ree_mop_rmse"]["fold_sd"], numbers["workhorse_ree_mop_rmse"]["n_valid_folds"]),
        "wh_ree_mop_r2": fmt(numbers["workhorse_ree_mop_r2"]["fold_mean"], numbers["workhorse_ree_mop_r2"]["fold_sd"], numbers["workhorse_ree_mop_r2"]["n_valid_folds"]),
        "wh_row_rmse": fmt(numbers["workhorse_ree_row_rmse"]["fold_mean"], numbers["workhorse_ree_row_rmse"]["fold_sd"], numbers["workhorse_ree_row_rmse"]["n_valid_folds"]),
        "wh_row_r2": fmt(numbers["workhorse_ree_row_r2"]["fold_mean"], numbers["workhorse_ree_row_r2"]["fold_sd"], numbers["workhorse_ree_row_r2"]["n_valid_folds"]),
        "wh_sign": fmt(numbers["workhorse_ree_sign"]["fold_mean"], numbers["workhorse_ree_sign"]["fold_sd"], numbers["workhorse_ree_sign"]["n_valid_folds"]),
        "wh_picp": fmt(numbers["workhorse_all_picp"]["fold_mean"], numbers["workhorse_all_picp"]["fold_sd"], numbers["workhorse_all_picp"]["n_valid_folds"]),
        "wh_ace": fmt(numbers["workhorse_all_ace"]["fold_mean"], numbers["workhorse_all_ace"]["fold_sd"], numbers["workhorse_all_ace"]["n_valid_folds"]),
        "wh_mop_picp": fmt(numbers["workhorse_all_mop_picp"]["fold_mean"], numbers["workhorse_all_mop_picp"]["fold_sd"], numbers["workhorse_all_mop_picp"]["n_valid_folds"]),
        "wh_mop_ace": fmt(numbers["workhorse_all_mop_ace"]["fold_mean"], numbers["workhorse_all_mop_ace"]["fold_sd"], numbers["workhorse_all_mop_ace"]["n_valid_folds"]),
        "wh_ree_picp": fmt(numbers["workhorse_ree_picp"]["fold_mean"], numbers["workhorse_ree_picp"]["fold_sd"], numbers["workhorse_ree_picp"]["n_valid_folds"]),
        "wh_ree_ace": fmt(numbers["workhorse_ree_ace"]["fold_mean"], numbers["workhorse_ree_ace"]["fold_sd"], numbers["workhorse_ree_ace"]["n_valid_folds"]),
    }
    # ranked operational candidates by Pareto rule
    ranked_rows = []
    for i, mid in enumerate(choice.get("ranked") or [], 1):
        d = lookup(summary, mid, "all_pairs", "mop_rmse")
        picp = lookup(summary, mid, "all_pairs", "marginal_picp_90")
        ace = lookup(summary, mid, "all_pairs", "marginal_ace")
        r2 = lookup(summary, mid, "all_pairs", "mop_r2")
        ranked_rows.append(
            {
                "rank": i,
                "model_id": mid,
                "display": display_of(mid),
                **d,
                "display_value": fmt(d["fold_mean"], d["fold_sd"], d["n_valid_folds"]),
                "picp_90": fmt(picp["fold_mean"], picp["fold_sd"], picp["n_valid_folds"]),
                "ace": fmt(ace["fold_mean"], ace["fold_sd"], ace["n_valid_folds"]),
                "mop_r2": fmt(r2["fold_mean"], r2["fold_sd"], r2["n_valid_folds"]),
            }
        )
    numbers["ranked_detail"] = ranked_rows
    write_json(MSRC / "numbers.json", numbers)

    table_reg = []
    for p in sorted(TABLES.glob("*.csv")):
        table_reg.append(
            {
                "table_id": p.stem,
                "path": str(p.relative_to(REMAKE_ROOT)).replace("\\", "/"),
                "n_rows": len(pd.read_csv(p)),
                "aggregation": "fold_mean_plus_sd_ddof1" if "fold" in p.stem or p.stem.startswith(("main_", "ds2_")) else "mixed",
            }
        )
    write_csv(MAN / "table_registry.csv", pd.DataFrame(table_reg))

    refs = pd.DataFrame(
        [
            {"ref_id": "karunaratne2025", "text": "Karunaratne et al., JCIM 2025, CMI M4 stability-constant models (Table 1 published MAE/R2)."},
            {"ref_id": "chemprop16", "text": "Chemprop 1.6.1 directed message-passing neural network; native evidential regression head."},
            {"ref_id": "auldfold", "text": "Amini et al., evidential deep regression (NIG)."},
            {"ref_id": "gal2016", "text": "Gal & Ghahramani, MC dropout as approximate Bayesian inference (used here as MC dropout, not as a full BNN)."},
            {"ref_id": "sngp", "text": "Liu et al., spectral-normalized neural Gaussian process."},
            {"ref_id": "svgp", "text": "Hensman et al., stochastic variational Gaussian processes."},
            {"ref_id": "ad_retention", "text": "Applicability-domain scores as empirical error-ordering tools at fixed retention."},
        ]
    )
    write_csv(MAN / "manuscript_references.csv", refs)

    log.append(f"finished {utc_now()}")
    print("\n".join(log))
    print("wrote", MAN)
    print("selected", choice["selected"], "ranked", len(choice["ranked"]))


def rebuild_display_tables() -> None:
    """Rebuild manuscript-facing metric tables from fold summaries. No retraining."""
    global WORKHORSE
    TABLES.mkdir(parents=True, exist_ok=True)
    MSRC.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(REMAKE_ROOT / "results" / "cv_metrics" / "fold_metric_summary.csv")
    if "metric_name" in summary.columns:
        summary = summary[~summary.metric_name.astype(str).str.contains("spearman", case=False, regex=False)].copy()
    eligibility = pd.read_csv(REMAKE_ROOT / "results" / "model_audit" / "workhorse_eligibility.csv")
    ds2_sum = summary[summary["dataset"].astype(str).str.upper() == "DS2"]
    choice = select_workhorse_accuracy_calibration(ds2_sum, eligibility)
    WORKHORSE = choice["selected"]
    write_rule()
    write_accuracy_calibration_artifacts(choice, MSRC)
    tables = build_ds_tables(summary, choice)
    for name, df in tables.items():
        write_csv(TABLES / f"{name}.csv", df)
    numbers_path = MSRC / "numbers.json"
    numbers = json.loads(numbers_path.read_text(encoding="utf-8")) if numbers_path.exists() else {}
    numbers["workhorse_model_id"] = WORKHORSE
    numbers["workhorse_display"] = display_of(WORKHORSE)
    numbers["manuscript_accuracy_metrics"] = (
        "all_pair_mop_rmse primary; displayed all_pair_picp_90/all_pair_ace are MoP; "
        "operational screen uses row-level PICP/ACE; all_pair_mop_r2 and all_pair_row_r2 auxiliary; "
        "two named ad hoc comparators; ree_mop_rmse usage example"
    )
    numbers["coded_selection_metric"] = "operational_interval_usability_screen then RMSE-PICP-ACE Pareto"
    numbers["primary_metric"] = (
        "all_pairs mop_rmse after operational interval-usability screen and RMSE-PICP-ACE Pareto selection"
    )
    numbers["selection_type"] = "operational_accuracy_calibration_compromise"
    numbers["r2_role"] = "auxiliary_diagnostic"
    numbers["sign_accuracy_role"] = "directional_diagnostic"
    numbers["rank_correlation_role"] = "not_used"
    numbers["ad_distance_error_corr_role"] = "si_diagnostic_only"
    numbers["row_level_metrics_used_for_selection"] = False
    numbers["selection_nested"] = False
    numbers["retraining_performed"] = False
    numbers["ad_retention_status"] = "fold_local empirical error-ordering diagnostic; not ligand-selection evidence"
    numbers["workhorse_all_pair_mop_rmse"] = lookup(summary, WORKHORSE, "all_pairs", "mop_rmse")
    numbers["workhorse_all_pair_mop_r2"] = lookup(summary, WORKHORSE, "all_pairs", "mop_r2")
    numbers["workhorse_ree_mop_r2"] = lookup(summary, WORKHORSE, "ree_pairs", "mop_r2")
    numbers["workhorse_ree_mop_rmse"] = lookup(summary, WORKHORSE, "ree_pairs", "mop_rmse")
    numbers["workhorse_ree_sign"] = lookup(summary, WORKHORSE, "ree_pairs", "sign_accuracy_all")
    numbers["workhorse_all_picp"] = lookup(summary, WORKHORSE, "all_pairs", "marginal_picp_90")
    numbers["workhorse_all_ace"] = lookup(summary, WORKHORSE, "all_pairs", "marginal_ace")
    numbers["workhorse_all_mop_picp"] = lookup(summary, WORKHORSE, "all_pairs", "mop_picp_90")
    numbers["workhorse_all_mop_ace"] = lookup(summary, WORKHORSE, "all_pairs", "mop_ace")
    numbers["workhorse_ree_picp"] = lookup(summary, WORKHORSE, "ree_pairs", "marginal_picp_90")
    numbers["workhorse_ree_ace"] = lookup(summary, WORKHORSE, "ree_pairs", "marginal_ace")
    fmt_block = numbers.setdefault("fmt", {})
    fmt_block["wh_mop_rmse"] = fmt(
        numbers["workhorse_all_pair_mop_rmse"]["fold_mean"],
        numbers["workhorse_all_pair_mop_rmse"]["fold_sd"],
        numbers["workhorse_all_pair_mop_rmse"]["n_valid_folds"],
    )
    fmt_block["wh_mop_r2"] = fmt(
        numbers["workhorse_all_pair_mop_r2"]["fold_mean"],
        numbers["workhorse_all_pair_mop_r2"]["fold_sd"],
        numbers["workhorse_all_pair_mop_r2"]["n_valid_folds"],
    )
    fmt_block["wh_ree_mop_rmse"] = fmt(
        numbers["workhorse_ree_mop_rmse"]["fold_mean"],
        numbers["workhorse_ree_mop_rmse"]["fold_sd"],
        numbers["workhorse_ree_mop_rmse"]["n_valid_folds"],
    )
    fmt_block["wh_ree_mop_r2"] = fmt(
        numbers["workhorse_ree_mop_r2"]["fold_mean"],
        numbers["workhorse_ree_mop_r2"]["fold_sd"],
        numbers["workhorse_ree_mop_r2"]["n_valid_folds"],
    )
    fmt_block["wh_sign"] = fmt(
        numbers["workhorse_ree_sign"]["fold_mean"],
        numbers["workhorse_ree_sign"]["fold_sd"],
        numbers["workhorse_ree_sign"]["n_valid_folds"],
    )
    fmt_block["wh_picp"] = fmt(
        numbers["workhorse_all_picp"]["fold_mean"],
        numbers["workhorse_all_picp"]["fold_sd"],
        numbers["workhorse_all_picp"]["n_valid_folds"],
    )
    fmt_block["wh_ace"] = fmt(
        numbers["workhorse_all_ace"]["fold_mean"],
        numbers["workhorse_all_ace"]["fold_sd"],
        numbers["workhorse_all_ace"]["n_valid_folds"],
    )
    fmt_block["wh_mop_picp"] = fmt(
        numbers["workhorse_all_mop_picp"]["fold_mean"],
        numbers["workhorse_all_mop_picp"]["fold_sd"],
        numbers["workhorse_all_mop_picp"]["n_valid_folds"],
    )
    fmt_block["wh_mop_ace"] = fmt(
        numbers["workhorse_all_mop_ace"]["fold_mean"],
        numbers["workhorse_all_mop_ace"]["fold_sd"],
        numbers["workhorse_all_mop_ace"]["n_valid_folds"],
    )
    fmt_block["wh_ree_picp"] = fmt(
        numbers["workhorse_ree_picp"]["fold_mean"],
        numbers["workhorse_ree_picp"]["fold_sd"],
        numbers["workhorse_ree_picp"]["n_valid_folds"],
    )
    fmt_block["wh_ree_ace"] = fmt(
        numbers["workhorse_ree_ace"]["fold_mean"],
        numbers["workhorse_ree_ace"]["fold_sd"],
        numbers["workhorse_ree_ace"]["n_valid_folds"],
    )
    write_json(numbers_path, numbers)
    print("rebuilt display tables", sorted(tables), "selected", WORKHORSE)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--tables-only":
        rebuild_display_tables()
    else:
        main()
