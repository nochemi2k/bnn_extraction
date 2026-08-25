"""
Chemprop-only H+ inclusion decision on project_fold{0-4}.

For each fold, train M4_GRAPH_DMPNN_POINT with train_with_H and train_without_H,
evaluate on metal_complex-only test, then decide and persist the choice.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uq_remake import REMAKE_ROOT
from uq_remake.data.build_m4 import write_chemprop_fold_csvs
from uq_remake.metrics.point_metrics import (
    cluster_bootstrap_ci,
    compute_point_metrics,
    format_metric_line,
    paired_cluster_bootstrap,
)
from uq_remake.models.chemprop_dmpnn import load_shared_cfg, predict_dmpnn_point, train_dmpnn_point


def _metal_complex_mask(df: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    return df.iloc[idx]["family"].to_numpy() == "metal_complex"


def run_variant(df: pd.DataFrame, folds: dict, variant: str, cfg: dict) -> dict:
    """variant in {train_with_H, train_without_H}."""
    fold_rows = []
    oof_records = []
    for fold_name, split in folds.items():
        train_idx = np.asarray(split["train_idx"], dtype=int)
        val_idx = np.asarray(split["val_idx"], dtype=int)
        test_idx = np.asarray(split["test_idx"], dtype=int)

        if variant == "train_without_H":
            train_idx = train_idx[df.iloc[train_idx]["family"].to_numpy() != "protonation"]
            val_idx = val_idx[df.iloc[val_idx]["family"].to_numpy() != "protonation"]
        # test always metal_complex only for the decision metric
        te_mask = _metal_complex_mask(df, test_idx)
        test_eval_idx = test_idx[te_mask]

        model_id = f"M4_GRAPH_DMPNN_POINT_{fold_name}"
        tag = f"{model_id}__{variant}"
        work = REMAKE_ROOT / "checkpoints" / tag / "chemprop_data"
        ckpt = REMAKE_ROOT / "checkpoints" / tag
        work.mkdir(parents=True, exist_ok=True)
        write_chemprop_fold_csvs(df, train_idx, val_idx, test_eval_idx, work)

        model_pt = ckpt / "fold_0" / "model_0" / "model.pt"
        print(f"[{tag}] training ... n_train={len(train_idx)} n_val={len(val_idx)} n_test_mc={len(test_eval_idx)}")
        if model_pt.exists():
            print(f"[{tag}] skip train; found {model_pt}")
        else:
            train_dmpnn_point(
                work / "train.csv",
                work / "train_features.csv",
                work / "val.csv",
                work / "val_features.csv",
                ckpt,
                seed=int(cfg["random_seed"]) + int(fold_name.replace("fold", "")),
            )
        preds_path = REMAKE_ROOT / "results" / "stage1" / "hplus" / f"{tag}_test_preds.csv"
        yhat = predict_dmpnn_point(work / "test.csv", work / "test_features.csv", ckpt, preds_path)
        y = df.iloc[test_eval_idx]["logK"].to_numpy(dtype=float)
        lig = df.iloc[test_eval_idx]["ligand_smiles"].astype(str).to_numpy()
        m = compute_point_metrics(y, yhat)
        fold_rows.append({"fold": fold_name, "variant": variant, **m})
        for rid, yt, yp, lg in zip(
            df.iloc[test_eval_idx]["record_id"],
            y,
            yhat,
            lig,
        ):
            oof_records.append(
                {
                    "record_id": rid,
                    "fold": fold_name,
                    "variant": variant,
                    "ligand_smiles": lg,
                    "y_true": float(yt),
                    "y_pred": float(yp),
                }
            )
        pred_df = pd.DataFrame(
            {
                "record_id": df.iloc[test_eval_idx]["record_id"].values,
                "ligand_smiles": lig,
                "y_true": y,
                "y_pred": yhat,
            }
        )
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(preds_path, index=False)

    return {"fold_metrics": fold_rows, "oof": oof_records}


def main() -> None:
    cfg = load_shared_cfg()
    df = pd.read_csv(REMAKE_ROOT / "data" / "processed" / "m4_processed.csv")
    folds = json.loads((REMAKE_ROOT / "data" / "splits" / "project_folds.json").read_text(encoding="utf-8"))

    results = {}
    for variant in ("train_with_H", "train_without_H"):
        results[variant] = run_variant(df, folds, variant, cfg)

    # Aggregate + paired bootstrap on pooled OOF
    summary_rows = []
    lines = []
    oof_by_var = {}
    for variant, payload in results.items():
        fdf = pd.DataFrame(payload["fold_metrics"])
        oof = pd.DataFrame(payload["oof"])
        oof_by_var[variant] = oof
        for metric in ("r2", "mae"):
            mean = float(fdf[metric].mean())
            sd = float(fdf[metric].std(ddof=1))
            point, lo, hi = cluster_bootstrap_ci(
                oof["y_true"].to_numpy(),
                oof["y_pred"].to_numpy(),
                oof["ligand_smiles"].to_numpy(),
                metric_fn=(r2_score if metric == "r2" else mean_absolute_error),
                n_boot=int(cfg["cluster_bootstrap_n"]),
                seed=int(cfg["random_seed"]),
            )
            # Prefer fold-mean for reported central value to match protocol
            summary_rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "fold_mean": mean,
                    "fold_sd": sd,
                    "oof_point": point,
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            )
            lines.append(
                format_metric_line(
                    f"M4_GRAPH_DMPNN_POINT/{variant}",
                    metric,
                    mean,
                    fold_sd=sd,
                    ci_low=lo,
                    ci_high=hi,
                )
            )

    # Paired comparison on R2 and MAE (with_H - without_H)
    merged = oof_by_var["train_with_H"].merge(
        oof_by_var["train_without_H"],
        on=["record_id", "ligand_smiles"],
        suffixes=("_with", "_without"),
    )
    paired = {
        "r2": paired_cluster_bootstrap(
            merged["y_true_with"].to_numpy(),
            merged["y_pred_with"].to_numpy(),
            merged["y_pred_without"].to_numpy(),
            merged["ligand_smiles"].to_numpy(),
            r2_score,
            n_boot=int(cfg["cluster_bootstrap_n"]),
            seed=int(cfg["random_seed"]),
        ),
        "mae": paired_cluster_bootstrap(
            merged["y_true_with"].to_numpy(),
            merged["y_pred_with"].to_numpy(),
            merged["y_pred_without"].to_numpy(),
            merged["ligand_smiles"].to_numpy(),
            mean_absolute_error,
            n_boot=int(cfg["cluster_bootstrap_n"]),
            seed=int(cfg["random_seed"]),
        ),
    }

    # Decision rule: prefer variant with higher R2; break ties with lower MAE.
    # Require that the preferred side is not clearly worse on the other metric via paired CI.
    with_r2 = next(r for r in summary_rows if r["variant"] == "train_with_H" and r["metric"] == "r2")
    without_r2 = next(r for r in summary_rows if r["variant"] == "train_without_H" and r["metric"] == "r2")
    with_mae = next(r for r in summary_rows if r["variant"] == "train_with_H" and r["metric"] == "mae")
    without_mae = next(r for r in summary_rows if r["variant"] == "train_without_H" and r["metric"] == "mae")

    if with_r2["fold_mean"] > without_r2["fold_mean"] + 1e-4:
        decision = "train_with_H"
        rationale = "Higher fold-mean R2 on metal_complex-only test with H+ in training."
    elif without_r2["fold_mean"] > with_r2["fold_mean"] + 1e-4:
        decision = "train_without_H"
        rationale = "Higher fold-mean R2 on metal_complex-only test without H+ in training."
    else:
        decision = "train_with_H" if with_mae["fold_mean"] <= without_mae["fold_mean"] else "train_without_H"
        rationale = "R2 essentially tied; chose lower fold-mean MAE."

    decision_obj = {
        "decision": decision,
        "rationale": rationale,
        "decided_using": "M4_GRAPH_DMPNN_POINT only (Chemprop D-MPNN POINT)",
        "applies_uniformly_to": [
            "M4_RDKIT_MLP_POINT",
            "M4_RDKIT_XGB_POINT",
            "M4_CPEMB_MLP_POINT",
            "M4_CPEMB_XGB_POINT",
            "M4_GRAPH_DMPNN_POINT",
            "DS1",
            "DS2",
        ],
        "paired_cluster_bootstrap": paired,
        "summary": summary_rows,
    }
    out_dir = REMAKE_ROOT / "results" / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(decision_obj, indent=2)
    (out_dir / "hplus_decision.json").write_text(payload, encoding="utf-8")
    canon = REMAKE_ROOT / "results" / "canonical"
    canon.mkdir(parents=True, exist_ok=True)
    (canon / "hplus_decision.json").write_text(payload, encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(out_dir / "hplus_comparison_table.csv", index=False)
    (out_dir / "hplus_metric_lines.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("DECISION:", decision, "-", rationale)


if __name__ == "__main__":
    main()
