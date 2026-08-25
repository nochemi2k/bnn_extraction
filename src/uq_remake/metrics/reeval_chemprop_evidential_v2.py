"""Re-evaluate existing DS2 native Chemprop evidential checkpoints.

Does not retrain, does not modify checkpoints, fold assignments, features,
targets, or lambda_reg. Inference-time NIG validation only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import torch  # noqa: F401  — import before rdkit on Windows

from uq_remake import REMAKE_ROOT
from uq_remake.data.feature_schema import feature_schema_hash
from uq_remake.metrics.cv_protocol import (
    ACE_LEVELS,
    EVALUATION_PARTITION,
    FOLD_SD_DDOF,
    MIN_PAIR_N,
    PAIR_COLUMN,
    fold_mean_sd,
)
from uq_remake.metrics.fold_metrics import (
    apply_scope,
    coverage_at,
    ds2_scopes,
    evaluate_one_fold,
    nig_student_t_interval,
)
from uq_remake.models.chemprop_dmpnn import predict_dmpnn_evidential_uq, resolve_chemprop_model_pt
from uq_remake.models.chemprop_evidential_validate import (
    fold_uq_status,
    validate_native_evidential_output,
)
from uq_remake.training.protocol import (
    PREPROCESSING_SCHEMA_HASH,
    checkpoint_dir,
    extra_meta_frame,
    load_folds,
    load_table,
    record_ids_for_split,
    target_name,
)

DATASET = "DS2"
MODEL_ID = "DS2_GRAPH_DMPNN_EVID"
MODEL_DIR_V2 = "DS2_GRAPH_DMPNN_EVID_v2"
FOLDS = ["fold0", "fold1", "fold2", "fold3", "fold4"]
INTERVAL_METHOD = "student_t_nig_predictive"

PRED_ROOT = REMAKE_ROOT / "results" / "cv_predictions" / "fold_test_predictions" / MODEL_DIR_V2
METRICS = REMAKE_ROOT / "results" / "cv_metrics"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_bundle(fold: str) -> dict:
    table = load_table(DATASET)
    split = load_folds(DATASET)[fold]
    te = np.asarray(split["test_idx"], dtype=int)
    y_true = table.iloc[te][target_name(DATASET)].to_numpy(dtype=float)
    chem_dir = REMAKE_ROOT / "data" / "processed" / "features" / "chemprop_ds2" / fold
    chem_test = pd.read_csv(chem_dir / "test.csv")
    if len(chem_test) != len(te):
        raise RuntimeError(f"{fold}: chemprop test.csv {len(chem_test)} != split {len(te)}")
    if not np.allclose(chem_test["logK"].to_numpy(dtype=float), y_true, equal_nan=True, rtol=0, atol=1e-6):
        raise RuntimeError(f"{fold}: chemprop test targets not aligned to table")
    return {
        "fold": fold,
        "table": table,
        "te": te,
        "y_true": y_true,
        "chem_dir": chem_dir,
        "record_ids": record_ids_for_split(table, DATASET, te),
        "extra": extra_meta_frame(table, DATASET, te),
    }


def _align_report(bundle: dict, pred: pd.DataFrame) -> dict:
    expected_ids = np.asarray(bundle["record_ids"]).astype(str)
    got_ids = pred["record_id"].astype(str).to_numpy()
    expected_set = set(expected_ids)
    got_set = set(got_ids)
    missing = sorted(expected_set - got_set)
    unexpected = sorted(got_set - expected_set)
    n_dup = int(len(got_ids) - len(set(got_ids)))
    y_mismatches = 0
    if len(got_ids) == len(expected_ids) and n_dup == 0 and not missing and not unexpected:
        order_ok = np.array_equal(got_ids, expected_ids)
        if order_ok:
            yt = pred["y_true"].to_numpy(dtype=float)
            y_mismatches = int(
                (~np.isclose(yt, bundle["y_true"], equal_nan=True, rtol=0, atol=1e-8)).sum()
            )
        else:
            y_mismatches = int(len(expected_ids))
    else:
        y_mismatches = int(len(expected_ids))
    ok = (
        len(got_ids) == len(expected_ids)
        and n_dup == 0
        and not missing
        and not unexpected
        and y_mismatches == 0
        and np.array_equal(got_ids, expected_ids)
    )
    return {
        "dataset": DATASET,
        "model_id": MODEL_ID,
        "fold_id": bundle["fold"],
        "expected_rows": int(len(expected_ids)),
        "predicted_rows": int(len(got_ids)),
        "missing_rows": int(len(missing)),
        "unexpected_rows": int(len(unexpected)),
        "duplicate_rows": n_dup,
        "y_true_mismatches": int(y_mismatches),
        "alignment_status": "ok" if ok else "failed",
    }


def _uq_on_scope(df: pd.DataFrame, scope: str) -> dict:
    scoped = apply_scope(df, DATASET, scope)
    n_total = int(len(scoped))
    uq_ok = scoped["uq_valid"].astype(bool).to_numpy()
    yt = scoped["y_true"].to_numpy(dtype=float)
    yp = scoped["y_pred"].to_numpy(dtype=float)
    row_ok = uq_ok & np.isfinite(yt) & np.isfinite(yp) & np.isfinite(scoped["var_total"].to_numpy(dtype=float))
    row_ok = row_ok & (scoped["var_total"].to_numpy(dtype=float) >= 0)
    n_valid = int(row_ok.sum())
    n_invalid = n_total - n_valid
    work = scoped.loc[row_ok].copy()
    if n_valid == 0:
        picp = ace = float("nan")
        coverages = {float(q): float("nan") for q in ACE_LEVELS}
        interval_status = "invalid_fold"
    else:
        lo90, hi90, valid90 = nig_student_t_interval(
            work["mu_native"], work["nu_native"], work["alpha_native"], work["beta_native"], 0.9
        )
        picp = coverage_at(work["y_true"], lo90, hi90, valid90)
        coverages = {}
        ace_vals = []
        for q in ACE_LEVELS:
            lo, hi, vq = nig_student_t_interval(
                work["mu_native"], work["nu_native"], work["alpha_native"], work["beta_native"], float(q)
            )
            cov = coverage_at(work["y_true"], lo, hi, vq)
            coverages[float(q)] = cov
            ace_vals.append(abs(cov - float(q)) if np.isfinite(cov) else float("nan"))
        ace = float(np.mean(ace_vals)) if all(np.isfinite(ace_vals)) else float("nan")
        interval_status = "complete" if n_invalid == 0 else "partial_native_invalid_rows"
        if len(ACE_LEVELS) != 9:
            raise RuntimeError("ACE must use the nine-level grid 0.1..0.9")

    n_pairs_total = n_pairs_valid = n_pairs_invalid = ""
    if PAIR_COLUMN in scoped.columns:
        col = PAIR_COLUMN
        n_pairs_total = int(scoped[col].nunique())
        pair_ok = scoped.assign(_ok=row_ok).groupby(col, sort=False)["_ok"].all()
        n_pairs_valid = int(pair_ok.sum())
        n_pairs_invalid = int((~pair_ok).sum())

    return {
        "dataset": DATASET,
        "model_id": MODEL_ID,
        "fold_id": str(df["fold_id"].iloc[0]),
        "scope": scope,
        "interval_method": INTERVAL_METHOD,
        "n_rows_total": n_total,
        "n_rows_with_valid_uq": n_valid,
        "n_rows_invalid_uq": n_invalid,
        "fraction_invalid_uq": float(n_invalid / n_total) if n_total else float("nan"),
        "n_pairs_total": n_pairs_total,
        "n_pairs_with_valid_intervals": n_pairs_valid,
        "n_pairs_invalid_intervals": n_pairs_invalid,
        "marginal_picp_90": picp,
        "marginal_ace": ace,
        "interval_status": interval_status,
        "evaluation_partition": EVALUATION_PARTITION,
    }


def _summarize(fold_df: pd.DataFrame, metric_col: str, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in fold_df.groupby(keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        stats = fold_mean_sd(g[metric_col].to_numpy())
        rec = {k: v for k, v in zip(keys, key)}
        rec.update(
            {
                "fold_mean": stats["fold_mean"],
                "fold_sd": stats["fold_sd"],
                "n_folds_attempted": stats["n_folds"],
                "n_valid_folds": stats["n_valid_folds"],
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def _metric_status(value, aligned: bool, n_used: int) -> str:
    if not aligned:
        return "alignment_failed"
    if n_used <= 0:
        return "no_valid_rows"
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "undefined_metric"
    return "ok"


def run_fold(fold: str) -> dict:
    bundle = _load_bundle(fold)
    ckpt = checkpoint_dir(MODEL_ID, fold, 0)
    ident = json.loads((ckpt / "checkpoint_identity.json").read_text(encoding="utf-8"))
    for key, want in {"dataset": DATASET, "fold": fold, "member_id": 0, "model_id": MODEL_ID}.items():
        got = ident.get(key)
        if key == "member_id":
            got = int(got)
            want = int(want)
        if str(got) != str(want):
            raise RuntimeError(f"{fold}: checkpoint identity {key}={got!r} != {want!r}")
    meta = json.loads((ckpt / "meta.json").read_text(encoding="utf-8"))
    lam = meta.get("selected_hps", {}).get("evidential_regularization")
    if float(lam) != 0.2:
        raise RuntimeError(f"{fold}: refusing to proceed; evidential_regularization={lam!r} (expected 0.2)")
    if str(meta.get("selected_hps", {}).get("loss_function", "")).lower() != "evidential":
        raise RuntimeError(f"{fold}: checkpoint is not evidential")

    model_pt = resolve_chemprop_model_pt(
        ckpt, dataset=DATASET, fold=fold, member_id=0, model_id=MODEL_ID
    )
    hash_before = _sha256_file(model_pt)
    feat_hash = str(meta.get("feature_hash") or feature_schema_hash(DATASET))
    live_hash = feature_schema_hash(DATASET)
    if feat_hash != live_hash:
        raise RuntimeError(f"{fold}: feature_schema_hash mismatch checkpoint={feat_hash} live={live_hash}")

    native_pred = predict_dmpnn_evidential_uq(
        bundle["chem_dir"] / "test.csv",
        bundle["chem_dir"] / "test_features.csv",
        ckpt,
        n_expected=len(bundle["te"]),
        dataset=DATASET,
        fold=fold,
        member_id=0,
        model_id=MODEL_ID,
    )
    hash_after = _sha256_file(model_pt)
    if hash_before != hash_after:
        raise RuntimeError(f"{fold}: checkpoint hash changed during inference")

    validated = validate_native_evidential_output(
        {
            "mu": native_pred["mu"],
            "nu": native_pred["nu"],
            "alpha": native_pred["alpha"],
            "beta": native_pred["beta"],
        }
    )
    n = validated["n_rows_total"]
    extra = bundle["extra"]
    pred = pd.DataFrame(
        {
            "dataset": DATASET,
            "model_id": MODEL_ID,
            "fold_id": fold,
            "record_id": np.asarray(bundle["record_ids"]).astype(str),
            "row_id": np.arange(n, dtype=int),
            "table_idx": np.asarray(bundle["te"], dtype=int),
            "evaluation_partition": EVALUATION_PARTITION,
            "y_true": np.asarray(bundle["y_true"], dtype=float),
            "y_pred": validated["mu"],
            "mu_native": validated["mu"],
            "nu_native": validated["nu"],
            "alpha_native": validated["alpha"],
            "beta_native": validated["beta"],
            "gamma": validated["mu"],
            "nu": validated["nu"],
            "alpha": validated["alpha"],
            "beta": validated["beta"],
            "var_aleatoric": validated["var_aleatoric"],
            "var_epistemic": validated["var_epistemic"],
            "var_total": validated["var_total"],
            "point_valid": validated["point_valid"],
            "uq_valid": validated["uq_valid"],
            "uncertainty_status": validated["uncertainty_status"],
            "checkpoint_hash": hash_before,
            "feature_schema_hash": feat_hash,
            "preprocessing_schema_hash": PREPROCESSING_SCHEMA_HASH,
            "interval_method": INTERVAL_METHOD,
            "ligand_smiles": extra["ligand_smiles"].to_numpy(),
            "family": extra["family"].to_numpy(),
            "metal_species_pair": extra["metal_species_pair"].to_numpy(),
            "is_ree_pair": extra["is_ree_pair"].to_numpy(),
            "pair_id": extra["pair_id"].to_numpy(),
            "member_id": 0,
        }
    )
    out_path = PRED_ROOT / fold / "fold_test_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_path, index=False)

    align = _align_report(bundle, pred)
    aligned = align["alignment_status"] == "ok"
    uq_status = fold_uq_status(
        n_point_invalid=validated["n_point_invalid"],
        n_uq_invalid=validated["n_uq_invalid"],
        n_rows=n,
        aligned=aligned,
    )
    diag = {
        "dataset": DATASET,
        "model_id": MODEL_ID,
        "fold_id": fold,
        "n_rows_total": validated["n_rows_total"],
        "n_point_valid": validated["n_point_valid"],
        "n_point_invalid": validated["n_point_invalid"],
        "n_uq_valid": validated["n_uq_valid"],
        "n_uq_invalid": validated["n_uq_invalid"],
        "fraction_uq_invalid": float(validated["n_uq_invalid"] / n) if n else float("nan"),
        "n_invalid_nu": validated["n_invalid_nu"],
        "n_invalid_alpha": validated["n_invalid_alpha"],
        "n_invalid_beta": validated["n_invalid_beta"],
        "n_nonfinite_native_values": validated["n_nonfinite_native_values"],
        "min_nu": validated["min_nu"],
        "min_alpha": validated["min_alpha"],
        "min_beta": validated["min_beta"],
        "uncertainty_status": uq_status,
        "checkpoint_hash": hash_before,
        "checkpoint_hash_after_infer": hash_after,
        "checkpoint_unchanged": hash_before == hash_after,
    }

    point_src = pred.copy()
    point_src["evaluation_partition"] = EVALUATION_PARTITION
    eval_out = evaluate_one_fold(point_src, DATASET, MODEL_ID) if aligned else {
        "fold_metrics": [],
        "interval_metrics": [],
    }
    point_rows = []
    wanted = {
        "mop_rmse",
        "sign_accuracy_all",
        "rmse",
        "r2",
    }
    for row in eval_out["fold_metrics"]:
        if row["metric_name"] not in wanted:
            continue
        rec = {
            "dataset": DATASET,
            "model_id": MODEL_ID,
            "fold_id": fold,
            "scope": row["scope"],
            "metric_name": row["metric_name"],
            "value": row["value"],
            "n_rows_total": row.get("n_rows_total", n),
            "n_rows_used": row.get("n_rows_used", ""),
            "n_pairs_total": row.get("n_pairs_total", ""),
            "n_pairs_valid": row.get("n_pairs_valid", ""),
            "evaluation_partition": EVALUATION_PARTITION,
        }
        rec["metric_status"] = _metric_status(
            rec["value"], aligned, int(rec["n_rows_used"] or 0) if rec["n_rows_used"] != "" else 0
        )
        if not aligned:
            rec["metric_status"] = "alignment_failed"
        point_rows.append(rec)

    uq_rows = []
    if aligned:
        for scope in ds2_scopes():
            uq_rows.append(_uq_on_scope(pred, scope))
    else:
        for scope in ds2_scopes():
            uq_rows.append(
                {
                    "dataset": DATASET,
                    "model_id": MODEL_ID,
                    "fold_id": fold,
                    "scope": scope,
                    "interval_method": INTERVAL_METHOD,
                    "n_rows_total": n,
                    "n_rows_with_valid_uq": 0,
                    "n_rows_invalid_uq": n,
                    "fraction_invalid_uq": 1.0,
                    "n_pairs_total": "",
                    "n_pairs_with_valid_intervals": "",
                    "n_pairs_invalid_intervals": "",
                    "marginal_picp_90": float("nan"),
                    "marginal_ace": float("nan"),
                    "interval_status": "invalid_fold",
                    "evaluation_partition": EVALUATION_PARTITION,
                }
            )

    print(
        f"  {fold}: n={n} point_valid={validated['n_point_valid']} "
        f"uq_invalid={validated['n_uq_invalid']} align={align['alignment_status']} "
        f"uq_status={uq_status} hash={hash_before[:12]}",
        flush=True,
    )
    return {
        "align": align,
        "diag": diag,
        "point_rows": point_rows,
        "uq_rows": uq_rows,
        "n_uq_invalid": validated["n_uq_invalid"],
        "aligned": aligned,
        "n_point_valid": validated["n_point_valid"],
        "n_uq_valid": validated["n_uq_valid"],
        "fold_uq_status": uq_status,
    }


def main(argv=None):
    argparse.ArgumentParser(description="DS2 Chemprop evidential v2 re-evaluation (no retrain)").parse_args(argv)
    METRICS.mkdir(parents=True, exist_ok=True)
    PRED_ROOT.mkdir(parents=True, exist_ok=True)

    aligns, diags, point_rows, uq_rows = [], [], [], []
    n_rank_valid = 0
    n_uq_complete = 0
    n_uq_invalid_total = 0
    n_attempted = 0
    for fold in FOLDS:
        n_attempted += 1
        out = run_fold(fold)
        aligns.append(out["align"])
        diags.append(out["diag"])
        point_rows.extend(out["point_rows"])
        uq_rows.extend(out["uq_rows"])
        n_uq_invalid_total += int(out["n_uq_invalid"])
        if out["aligned"] and out["n_point_valid"] > 0:
            n_rank_valid += 1
        if out["fold_uq_status"] == "complete":
            n_uq_complete += 1

    align_df = pd.DataFrame(aligns)
    diag_df = pd.DataFrame(diags)
    point_df = pd.DataFrame(point_rows)
    uq_df = pd.DataFrame(uq_rows)
    align_df.to_csv(METRICS / "chemprop_evidential_alignment_v2.csv", index=False)
    diag_df.to_csv(METRICS / "chemprop_evidential_uq_diagnostics_v2.csv", index=False)
    point_df.to_csv(METRICS / "chemprop_evidential_fold_point_metrics_v2.csv", index=False)

    uq_long = []
    for _, r in uq_df.iterrows():
        for metric in ("marginal_picp_90", "marginal_ace"):
            uq_long.append(
                {
                    "dataset": r["dataset"],
                    "model_id": r["model_id"],
                    "fold_id": r["fold_id"],
                    "scope": r["scope"],
                    "metric_name": metric,
                    "value": r[metric],
                    "n_rows_total": r["n_rows_total"],
                    "n_rows_with_valid_uq": r["n_rows_with_valid_uq"],
                    "n_rows_invalid_uq": r["n_rows_invalid_uq"],
                    "fraction_invalid_uq": r["fraction_invalid_uq"],
                    "n_pairs_total": r["n_pairs_total"],
                    "n_pairs_with_valid_intervals": r["n_pairs_with_valid_intervals"],
                    "n_pairs_invalid_intervals": r["n_pairs_invalid_intervals"],
                    "interval_status": r["interval_status"],
                    "evaluation_partition": EVALUATION_PARTITION,
                }
            )
    uq_long_df = pd.DataFrame(uq_long)
    uq_long_df.to_csv(METRICS / "chemprop_evidential_fold_uq_metrics_v2.csv", index=False)
    point_sum = _summarize(point_df, "value", ["dataset", "model_id", "scope", "metric_name"]) if len(point_df) else pd.DataFrame()
    if len(point_sum):
        point_sum.to_csv(METRICS / "chemprop_evidential_point_summary_v2.csv", index=False)
    uq_sum = _summarize(uq_long_df, "value", ["dataset", "model_id", "scope", "metric_name"]) if len(uq_long_df) else pd.DataFrame()
    if len(uq_sum):
        uq_sum.to_csv(METRICS / "chemprop_evidential_uq_summary_v2.csv", index=False)

    ranking_status = "complete_5_of_5" if n_rank_valid == 5 and n_attempted == 5 else f"incomplete_{n_rank_valid}_of_{n_attempted}"
    uq_status = "partial_native_invalid_rows" if n_uq_invalid_total > 0 else "complete_native_valid_rows"
    notes = (
        "No Chemprop retraining. Native NIG parameters preserved. "
        f"Invalid native-UQ rows total={n_uq_invalid_total}. "
        f"Folds with zero invalid NIG rows={n_uq_complete}. "
        "PICP/ACE use student_t_nig_predictive on uq_valid rows only; "
        "invalid rows retained in prediction files with NaN variances."
    )
    status = pd.DataFrame(
        [
            {
                "dataset": DATASET,
                "model_id": MODEL_ID,
                "ranking_status": ranking_status,
                "uq_status": uq_status,
                "n_folds_attempted": n_attempted,
                "n_ranking_valid_folds": n_rank_valid,
                "n_uq_valid_folds": n_uq_complete,
                "n_uq_invalid_rows_total": n_uq_invalid_total,
                "notes": notes,
            }
        ]
    )
    status.to_csv(METRICS / "chemprop_evidential_status_v2.csv", index=False)
    print("ranking_status", ranking_status, "uq_status", uq_status, flush=True)
    print("Wrote", PRED_ROOT, "and v2 CSVs under", METRICS, flush=True)


if __name__ == "__main__":
    main()
