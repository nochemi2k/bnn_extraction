"""Run aligned ad hoc DS1-difference versus direct DS2 selectivity."""
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

from uq_remake import REMAKE_ROOT  # noqa: E402
from uq_remake.downstream.ad_hoc_infer import (  # noqa: E402
    POINT_MODELS,
    UQ_MODELS,
    cache_path,
    infer_ds1_model,
)
from uq_remake.downstream.ad_hoc_selectivity import (  # noqa: E402
    SCOPES,
    SELECTIVITY_AD_HOC,
    SELECTIVITY_DIRECT,
    AdHocAlignmentError,
    ad_hoc_comparison_phrase,
    assemble_ad_hoc_pairs,
    assert_no_ligand_overlap,
    evaluate_ad_hoc_fold,
    method_class,
    summarize_fold_metrics,
    _display_model,
)
from uq_remake.downstream.protocol import AD_HOC_DIR, FOLDS, rel  # noqa: E402
from uq_remake.metrics.cv_protocol import EVALUATION_PARTITION, MIN_PAIR_N  # noqa: E402
from uq_remake.training.protocol import load_folds, load_table  # noqa: E402


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


def ds2_fold_pairs() -> dict[str, pd.DataFrame]:
    table = load_table("DS2")
    folds = load_folds("DS2")
    out = {}
    for fold in FOLDS:
        idx = np.asarray(folds[fold]["test_idx"], dtype=int)
        out[fold] = table.iloc[idx].copy()
    return out


def ds1_fold_ligands() -> dict[str, dict[str, set[str]]]:
    table = load_table("DS1")
    folds = load_folds("DS1")
    out = {}
    for fold in FOLDS:
        split = folds[fold]
        out[fold] = {
            "train": set(table.iloc[np.asarray(split["train_idx"], dtype=int)]["ligand_smiles"].astype(str)),
            "val": set(table.iloc[np.asarray(split["val_idx"], dtype=int)]["ligand_smiles"].astype(str)),
            "test": set(table.iloc[np.asarray(split["test_idx"], dtype=int)]["ligand_smiles"].astype(str)),
        }
    return out


def load_or_infer_ds1(model_id: str) -> pd.DataFrame:
    cached = cache_path(model_id)
    if cached.exists():
        return pd.read_csv(cached)
    print(f"infer {model_id}", flush=True)
    pred = infer_ds1_model(model_id)
    _write_csv(cached, pred)
    return pred


def _source_fold_for_record(pred: pd.DataFrame) -> pd.Series:
    if "source_fold" in pred.columns:
        return pred["source_fold"].astype(str)
    if "outer_fold_id" in pred.columns:
        return pred["outer_fold_id"].astype(str)
    return pred["fold_id"].astype(str)


def check_alignment(ds2_pairs: pd.DataFrame, ds1_pred: pd.DataFrame, ds1_ligands: dict) -> None:
    indexed = ds1_pred.copy()
    indexed["record_id"] = indexed["record_id"].astype(str)
    indexed["_src_fold"] = _source_fold_for_record(indexed)
    by_id = indexed.drop_duplicates("record_id").set_index("record_id")
    ligands = ds2_pairs["ligand_smiles"].astype(str)
    for rec_col in ("record_id_x", "record_id_y"):
        src = by_id.loc[ds2_pairs[rec_col].astype(str), "_src_fold"].astype(str).to_numpy()
        for fold in FOLDS:
            mask = src == fold
            if not bool(mask.any()):
                continue
            held = set(ligands.to_numpy()[mask])
            train = ds1_ligands[fold]["train"] | ds1_ligands[fold]["val"]
            assert_no_ligand_overlap(train, held)
            missing = held - ds1_ligands[fold]["test"]
            if missing:
                raise AdHocAlignmentError(f"{len(missing)} ligands are not in DS1 {fold} test ligands")


def evaluate_model(model_id: str, ds2_folds: dict[str, pd.DataFrame], ds1_ligands: dict) -> list[dict]:
    pred = load_or_infer_ds1(model_id)
    rows = []
    for fold, pairs in ds2_folds.items():
        check_alignment(pairs, pred, ds1_ligands)
        assembled = assemble_ad_hoc_pairs(pairs, pred, model_id, fold)
        for scope in SCOPES:
            rows.append(evaluate_ad_hoc_fold(assembled, model_id, scope))
    return rows


def direct_ds2_fold_rows() -> list[dict]:
    path = REMAKE_ROOT / "results" / "cv_metrics" / "fold_metrics.csv"
    iv = REMAKE_ROOT / "results" / "cv_metrics" / "fold_interval_metrics.csv"
    metrics = pd.read_csv(path)
    intervals = pd.read_csv(iv) if iv.exists() else pd.DataFrame()
    keep = metrics[(metrics.dataset.astype(str) == "DS2") & (metrics.scope.astype(str).isin(SCOPES))].copy()
    wanted = {
        "mop_rmse": "mop_rmse",
        "mop_r2": "mop_r2",
        "mop_picp_90": "mop_picp_90",
        "mop_ace": "mop_ace",
        "sign_accuracy_all": "sign_accuracy",
        "rmse": "row_rmse",
        "r2": "row_r2",
    }
    rows = []
    keys = ["model_id", "fold_id", "scope"]
    for key, g in keep.groupby(keys, dropna=False):
        rec = {k: v for k, v in zip(keys, key if isinstance(key, tuple) else (key,))}
        wide = {str(r.metric_name): r for _, r in g.iterrows()}
        mop = wide.get("mop_rmse")
        rec.update(
            {
                "dataset": "DS2",
                "outer_fold": rec.pop("fold_id"),
                "source_model": _display_model(rec["model_id"]),
                "source_model_id": rec.pop("model_id"),
                "selectivity_method": SELECTIVITY_DIRECT,
                "n_rows": int(mop.n_rows_used) if mop is not None and pd.notna(mop.n_rows_used) else 0,
                "n_pairs": int(mop.n_pairs_retained) if mop is not None and pd.notna(getattr(mop, "n_pairs_retained", np.nan)) else 0,
                "uncertainty_status": "direct_ds2_native",
                "propagation_method": "none",
                "n_valid": 1,
                "evaluation_partition": EVALUATION_PARTITION,
                "min_pair_size": MIN_PAIR_N,
            }
        )
        for src, dest in wanted.items():
            hit = wide.get(src)
            rec[dest] = float(hit.value) if hit is not None and pd.notna(hit.value) else np.nan
        if len(intervals):
            ivg = intervals[
                (intervals.dataset.astype(str) == "DS2")
                & (intervals.model_id.astype(str) == rec["source_model_id"])
                & (intervals.fold_id.astype(str) == rec["outer_fold"])
                & (intervals.scope.astype(str) == rec["scope"])
            ]
            picp = ivg[ivg.metric_name.astype(str) == "marginal_picp_90"]
            ace = ivg[ivg.metric_name.astype(str) == "marginal_ace"]
            rec["picp_90"] = float(picp.iloc[0].value) if len(picp) and pd.notna(picp.iloc[0].value) else np.nan
            rec["ace"] = float(ace.iloc[0].value) if len(ace) and pd.notna(ace.iloc[0].value) else np.nan
        else:
            rec["picp_90"] = rec["ace"] = np.nan
        rows.append(rec)
    return rows


def comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary is None or summary.empty:
        return pd.DataFrame()
    work = summary[summary.scope.astype(str) == "ree_pairs"].copy()
    work["method_class"] = [method_class(m, i) for m, i in zip(work.selectivity_method, work.source_model_id)]
    class_order = {"direct_ds2": 0, "ad_hoc_ds1_uncertainty": 1, "ad_hoc_ds1_point": 2}
    work["_ord"] = work["method_class"].map(class_order).fillna(9)
    work = work.sort_values(["_ord", "mop_rmse"]).drop(columns=["_ord"])
    return work


def manuscript_table(cmp: pd.DataFrame) -> pd.DataFrame:
    if cmp is None or cmp.empty:
        return pd.DataFrame()
    work = cmp.copy()
    # Keep method classes separate. Direct point models are not the ad hoc baseline.
    if "method_class" in work.columns:
        work = work[work.method_class.astype(str).isin(["direct_ds2", "ad_hoc_ds1_uncertainty", "ad_hoc_ds1_point"])]
    if "source_model_id" in work.columns:
        direct_point = (work.method_class.astype(str) == "direct_ds2") & work.source_model_id.astype(str).str.contains("POINT|XGB", regex=True)
        work = work.loc[~direct_point].copy()
    keep = [
        "method_class",
        "selectivity_method",
        "source_model",
        "source_model_id",
        "scope",
        "mop_rmse",
        "mop_rmse_sd",
        "mop_r2",
        "mop_r2_sd",
        "sign_accuracy",
        "sign_accuracy_sd",
        "picp_90",
        "picp_90_sd",
        "ace",
        "ace_sd",
        "n_valid",
        "uncertainty_status",
        "propagation_method",
    ]
    cols = [c for c in keep if c in work.columns]
    return work[cols].copy()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Aligned ad hoc DS1-difference vs direct DS2 selectivity")
    p.add_argument("--uq-only", action="store_true", help="Skip optional point-model ad hoc baselines")
    p.add_argument("--models", default="", help="Comma-separated DS1 model ids; default is the UQ grid")
    args = p.parse_args(argv)
    AD_HOC_DIR.mkdir(parents=True, exist_ok=True)
    ds2_folds = ds2_fold_pairs()
    ds1_ligands = ds1_fold_ligands()
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = list(UQ_MODELS)
        if not args.uq_only:
            models.extend(POINT_MODELS)
    fold_rows = []
    failures = []
    for model_id in models:
        try:
            print(model_id, flush=True)
            fold_rows.extend(evaluate_model(model_id, ds2_folds, ds1_ligands))
        except Exception as exc:
            failures.append({"source_model_id": model_id, "error": str(exc)})
            print(f"  FAILED {model_id}: {exc}", flush=True)
    fold_rows.extend(direct_ds2_fold_rows())
    fold_df = pd.DataFrame(fold_rows)
    if len(fold_df):
        fold_df["method_class"] = [
            method_class(m, i) for m, i in zip(fold_df.selectivity_method, fold_df.source_model_id)
        ]
    summary = summarize_fold_metrics(fold_df)
    cmp = comparison_table(summary)
    ms_tbl = manuscript_table(cmp)
    _write_csv(AD_HOC_DIR / "ad_hoc_selectivity_fold_metrics.csv", fold_df)
    _write_csv(AD_HOC_DIR / "ad_hoc_selectivity_summary.csv", summary)
    _write_csv(AD_HOC_DIR / "ad_hoc_selectivity_comparison.csv", cmp)
    man_tables = REMAKE_ROOT / "manuscripts" / "final" / "tables" / "ad_hoc_selectivity_comparison.csv"
    man_data = REMAKE_ROOT / "manuscripts" / "final" / "data" / "ad_hoc_selectivity_comparison.csv"
    _write_csv(man_tables, ms_tbl)
    _write_csv(man_data, ms_tbl)
    phrase = ad_hoc_comparison_phrase(summary)
    snapshot = {
        "experiment_id": "adhoc_differencing",
        "computation_status": "complete" if len(fold_df) else "not_computed",
        "selectivity_methods": [SELECTIVITY_DIRECT, SELECTIVITY_AD_HOC],
        "propagation_method": "independent_quadrature_sigma_total",
        "min_pair_size": MIN_PAIR_N,
        "alignment": "DS2 outer folds; DS1 checkpoint is the ligand-held-out fold",
        "pooled_oof_forbidden": True,
        "comparison_phrase": phrase,
        "failures": failures,
        "n_fold_rows": int(len(fold_df)),
        "n_summary_rows": int(len(summary)),
    }
    _write_json(AD_HOC_DIR / "ad_hoc_spec_snapshot.json", snapshot)
    report = {
        "pass": bool(len(fold_df) and fold_df.loc[fold_df.selectivity_method == SELECTIVITY_AD_HOC, "outer_fold"].nunique() == 5),
        "computation_status": snapshot["computation_status"],
        "n_ad_hoc_models": int(fold_df.loc[fold_df.selectivity_method == SELECTIVITY_AD_HOC, "source_model_id"].nunique()) if len(fold_df) else 0,
        "comparison_phrase": phrase,
        "failures": failures,
        "spec_path": rel(REPO_ROOT / "configs" / "experiments" / "ad_hoc_selectivity_spec.yaml"),
    }
    _write_json(AD_HOC_DIR / "ad_hoc_validation.json", report)
    print(json.dumps({k: report[k] for k in ("pass", "n_ad_hoc_models")}, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
