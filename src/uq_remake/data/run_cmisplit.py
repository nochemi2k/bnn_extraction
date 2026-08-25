"""
CMI official-split verification (cmisplit).

Trains a blank Chemprop D-MPNN POINT model on CMI train, early-stops on a
ligand-disjoint holdout carved from train (so official test/ext stay clean),
evaluates on CMI test and CMI ext (validation), compares to published Table 1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uq_remake import REMAKE_ROOT
from uq_remake.data.build_m4 import write_chemprop_fold_csvs
from uq_remake.metrics.point_metrics import compute_point_metrics, format_metric_line
from uq_remake.models.chemprop_dmpnn import (
    extract_fingerprints,
    load_shared_cfg,
    predict_dmpnn_point,
    train_dmpnn_point,
)

MODEL_ID = "M4_GRAPH_DMPNN_POINT_cmisplit"


def main() -> None:
    cfg = load_shared_cfg()
    df = pd.read_csv(REMAKE_ROOT / "data" / "processed" / "m4_processed.csv")
    external = pd.read_csv(REMAKE_ROOT / "data" / "processed" / "m4_external.csv")
    cmi = json.loads((REMAKE_ROOT / "data" / "splits" / "cmi_official_split.json").read_text(encoding="utf-8"))

    train_idx = np.asarray(cmi["train_idx"], dtype=int)
    test_idx = np.asarray(cmi["test_idx"], dtype=int)
    ext_idx = np.arange(len(external), dtype=int)

    # Early-stopping val: ligand-disjoint 15% of CMI train
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=int(cfg["random_seed"]))
    tr_rel, va_rel = next(
        gss.split(train_idx, groups=df.iloc[train_idx]["ligand_smiles"].astype(str).values)
    )
    tr_idx = train_idx[tr_rel]
    va_idx = train_idx[va_rel]

    work = REMAKE_ROOT / "checkpoints" / MODEL_ID / "chemprop_data"
    ckpt = REMAKE_ROOT / "checkpoints" / MODEL_ID
    work.mkdir(parents=True, exist_ok=True)

    write_chemprop_fold_csvs(
        df,
        tr_idx,
        va_idx,
        test_idx,
        work,
        extra=None,
    )
    from uq_remake.data.build_m4 import chemprop_train_medians, write_chemprop_csvs

    med = chemprop_train_medians(df, tr_idx)
    write_chemprop_csvs(
        external,
        ext_idx,
        work / "ext.csv",
        work / "ext_features.csv",
        train_feature_medians=med,
    )

    model_pt = ckpt / "fold_0" / "model_0" / "model.pt"
    skip_train = "--skip-train" in sys.argv or model_pt.exists()
    if skip_train and model_pt.exists():
        print(f"[{MODEL_ID}] skipping train; using existing checkpoint {model_pt}")
    else:
        print(f"[{MODEL_ID}] training blank Chemprop D-MPNN on cmi_official_split ...")
        train_dmpnn_point(
            train_data=work / "train.csv",
            train_features=work / "train_features.csv",
            val_data=work / "val.csv",
            val_features=work / "val_features.csv",
            save_dir=ckpt,
            seed=int(cfg["random_seed"]),
        )

    results = {"model_id": MODEL_ID, "split": "cmi_official_split", "evaluations": {}}
    lines = []
    for split_name, table, idx, data_csv, feat_csv in [
        ("cmi_test", df, test_idx, work / "test.csv", work / "test_features.csv"),
        ("cmi_ext", external, ext_idx, work / "ext.csv", work / "ext_features.csv"),
    ]:
        preds_path = REMAKE_ROOT / "results" / "stage1" / f"{MODEL_ID}_{split_name}_preds.csv"
        yhat = predict_dmpnn_point(data_csv, feat_csv, ckpt, preds_path)
        y = table.iloc[idx]["logK"].to_numpy(dtype=float)
        fam = table.iloc[idx]["family"].to_numpy()
        metrics_all = compute_point_metrics(y, yhat)
        mask_mc = fam == "metal_complex"
        metrics_mc = compute_point_metrics(y[mask_mc], yhat[mask_mc])
        results["evaluations"][split_name] = {
            "all_records": metrics_all,
            "metal_complex_only": metrics_mc,
        }
        for scope, m in [("all", metrics_all), ("metal_complex", metrics_mc)]:
            for metric in ("r2", "mae"):
                lines.append(
                    format_metric_line(
                        f"{MODEL_ID}/{split_name}/{scope}",
                        metric,
                        m[metric],
                        dispersion_applies=False,
                    )
                )
        pred_out = table.iloc[idx][["record_id", "complex_smiles", "ligand_smiles", "family", "logK"]].copy()
        pred_out["y_pred"] = yhat
        pred_out.to_csv(preds_path, index=False)

    # Store cmisplit embeddings separately (never reuse as CPEMB)
    emb_dir = REMAKE_ROOT / "embeddings" / "cmisplit"
    for split_name, table, idx, data_csv, feat_csv in [
        ("train", df, tr_idx, work / "train.csv", work / "train_features.csv"),
        ("val", df, va_idx, work / "val.csv", work / "val_features.csv"),
        ("test", df, test_idx, work / "test.csv", work / "test_features.csv"),
        ("ext", external, ext_idx, work / "ext.csv", work / "ext_features.csv"),
    ]:
        ids = table.iloc[idx]["record_id"].tolist()
        extract_fingerprints(
            data_csv,
            feat_csv,
            ckpt,
            emb_dir / f"M4_cmisplit_{split_name}.npz",
            ids,
        )

    published = {
        "cmi_published_m4_test_mae": cfg["cmi_published_m4_test_mae"],
        "cmi_published_m4_test_r2": cfg["cmi_published_m4_test_r2"],
        "cmi_published_m4_ext_mae": cfg["cmi_published_m4_ext_mae"],
        "cmi_published_m4_ext_r2": cfg["cmi_published_m4_ext_r2"],
        "note": (
            "Karunaratne et al. JCIM 2025 Table 1: 'test' metrics are 5-fold CV averages (±); "
            "'ext' metrics are external-set ensemble predictions. Our cmisplit run is a single "
            "from-scratch retrain on the published train assignment; compare especially to ext "
            "(validation file, 398 rows) and to test file (4338 rows) as open observations."
        ),
    }
    results["cmi_published"] = published

    # Comparison verdicts (open observations)
    ext_mc = results["evaluations"]["cmi_ext"]["all_records"]
    test_mc = results["evaluations"]["cmi_test"]["all_records"]
    results["comparison"] = {
        "vs_published_ext_r2": {
            "ours": ext_mc["r2"],
            "published": cfg["cmi_published_m4_ext_r2"],
            "diff_ours_minus_published": ext_mc["r2"] - cfg["cmi_published_m4_ext_r2"],
        },
        "vs_published_ext_mae": {
            "ours": ext_mc["mae"],
            "published": cfg["cmi_published_m4_ext_mae"],
            "diff_ours_minus_published": ext_mc["mae"] - cfg["cmi_published_m4_ext_mae"],
        },
        "vs_published_test_r2_note": (
            "Published test R2 is a 5-fold CV mean; our single-split test-file R2 is "
            f"{test_mc['r2']:.4f} vs published {cfg['cmi_published_m4_test_r2']}"
        ),
    }

    out_json = REMAKE_ROOT / "results" / "stage1" / "stage1a_cmisplit_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    (REMAKE_ROOT / "results" / "stage1" / "stage1a_metric_lines.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    # Compact comparison table
    rows = [
        {
            "quantity": "ext_r2",
            "reproduced": ext_mc["r2"],
            "cmi_published": cfg["cmi_published_m4_ext_r2"],
            "reproduced_minus_published": ext_mc["r2"] - cfg["cmi_published_m4_ext_r2"],
        },
        {
            "quantity": "ext_mae",
            "reproduced": ext_mc["mae"],
            "cmi_published": cfg["cmi_published_m4_ext_mae"],
            "reproduced_minus_published": ext_mc["mae"] - cfg["cmi_published_m4_ext_mae"],
        },
        {
            "quantity": "test_file_r2",
            "reproduced": test_mc["r2"],
            "cmi_published": cfg["cmi_published_m4_test_r2"],
            "reproduced_minus_published": test_mc["r2"] - cfg["cmi_published_m4_test_r2"],
        },
        {
            "quantity": "test_file_mae",
            "reproduced": test_mc["mae"],
            "cmi_published": cfg["cmi_published_m4_test_mae"],
            "reproduced_minus_published": test_mc["mae"] - cfg["cmi_published_m4_test_mae"],
        },
    ]
    pd.DataFrame(rows).to_csv(REMAKE_ROOT / "results" / "stage1" / "stage1a_comparison_table.csv", index=False)
    print("\n".join(lines))
    print("Wrote", out_json)


if __name__ == "__main__":
    main()
