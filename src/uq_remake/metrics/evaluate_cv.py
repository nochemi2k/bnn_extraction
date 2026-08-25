"""Evaluate held-out fold predictions independently. Never concatenate folds."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uq_remake import REMAKE_ROOT  # noqa: E402
from uq_remake.metrics.cv_protocol import (  # noqa: E402
    EVALUATION_PARTITION,
    LONG_METRIC_FIELDS,
    OOFTerminologyError,
    PUBLISHED_RISK_METRIC_COLUMNS,
    assert_active_outputs_have_no_oof,
    assert_published_metric_names,
    fold_mean_sd,
    metric_row,
)
from uq_remake.metrics.fold_metrics import evaluate_one_fold  # noqa: E402
from uq_remake.metrics.workhorse import load_rule, write_rule  # noqa: E402

PRED_ROOT = REMAKE_ROOT / "results" / "cv_predictions" / "fold_test_predictions"
ACTIVE_PRED = REMAKE_ROOT / "results" / "active_grid"
METRICS_ROOT = REMAKE_ROOT / "results" / "cv_metrics"
AUDIT = REMAKE_ROOT / "results" / "model_audit"


def _standardize_pred(df: pd.DataFrame, model_id: str, fold: str) -> pd.DataFrame:
    out = df.copy()
    if "fold_id" not in out.columns:
        out["fold_id"] = fold
    if "model_id" not in out.columns:
        out["model_id"] = model_id
    if "evaluation_partition" not in out.columns:
        out["evaluation_partition"] = EVALUATION_PARTITION
    if out["evaluation_partition"].astype(str).str.contains("oof", case=False, na=False).any():
        raise OOFTerminologyError(f"{model_id} {fold} has OOF evaluation_partition")
    out["evaluation_partition"] = EVALUATION_PARTITION
    folds = out["fold_id"].astype(str).unique()
    if len(folds) != 1:
        raise OOFTerminologyError(f"{model_id} file mixes folds {list(folds)}")
    return out


def discover_prediction_files() -> list[tuple[str, str, Path]]:
    found: list[tuple[str, str, Path]] = []
    if ACTIVE_PRED.exists():
        for path in ACTIVE_PRED.glob("*/*/test_preds.csv"):
            model_id = path.parent.parent.name
            fold = path.parent.name
            if fold.startswith("fold"):
                found.append((model_id, fold, path))
    if PRED_ROOT.exists():
        for path in PRED_ROOT.glob("*/*/fold_test_predictions.csv"):
            model_id = path.parent.parent.name
            fold = path.parent.name
            found.append((model_id, fold, path))
    # unique by model/fold, prefer cv_predictions copy
    uniq: dict[tuple[str, str], Path] = {}
    for mid, fold, path in found:
        key = (mid, fold)
        if key not in uniq or path.name == "fold_test_predictions.csv":
            uniq[key] = path
    # Accepted Chemprop DS2 evidential artifact is the versioned v2 re-evaluation.
    # Do not treat DS2_GRAPH_DMPNN_EVID_v2 as a second model family.
    v2_root = PRED_ROOT / "DS2_GRAPH_DMPNN_EVID_v2"
    if v2_root.exists():
        for fold_dir in sorted(v2_root.glob("fold*")):
            pred = fold_dir / "fold_test_predictions.csv"
            if pred.exists():
                uniq[("DS2_GRAPH_DMPNN_EVID", fold_dir.name)] = pred
        for key in list(uniq):
            if key[0].endswith("_v2"):
                uniq.pop(key, None)
    return [(m, f, p) for (m, f), p in sorted(uniq.items())]


def copy_fold_prediction(df: pd.DataFrame, model_id: str, fold: str) -> Path:
    dest = PRED_ROOT / model_id / fold / "fold_test_predictions.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest


def summarize_fold_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "model_id", "scope", "metric_name"]
    rows = []
    for key, g in fold_df.groupby(keys, dropna=False):
        stats = fold_mean_sd(g["value"].to_numpy())
        rows.append(
            {
                "dataset": key[0],
                "model_id": key[1],
                "scope": key[2],
                "metric_name": key[3],
                **stats,
            }
        )
    return pd.DataFrame(rows)


def _metric_key(row) -> tuple:
    return (
        str(row.get("dataset", "")),
        str(row.get("model_id", "")),
        str(row.get("fold_id", "")),
        str(row.get("scope", "")),
        str(row.get("metric_name", "")),
    )


def compare_to_stored(metric_rows: list[dict], stored_path: Path, *, atol: float = 1e-8) -> list[str]:
    issues: list[str] = []
    if not stored_path.exists():
        return [f"missing_stored:{stored_path}"]
    stored = pd.read_csv(stored_path)
    live = pd.DataFrame(metric_rows)
    if live.empty:
        return ["no_live_metrics"]
    stored_map = {}
    for rec in stored.to_dict(orient="records"):
        stored_map[_metric_key(rec)] = rec.get("value")
    n_checked = 0
    for rec in live.to_dict(orient="records"):
        key = _metric_key(rec)
        if key not in stored_map:
            issues.append(f"missing_stored_row:{key}")
            continue
        n_checked += 1
        a = rec.get("value")
        b = stored_map[key]
        a_num = pd.to_numeric(pd.Series([a]), errors="coerce").iloc[0]
        b_num = pd.to_numeric(pd.Series([b]), errors="coerce").iloc[0]
        if pd.isna(a_num) and pd.isna(b_num):
            continue
        if pd.isna(a_num) or pd.isna(b_num) or abs(float(a_num) - float(b_num)) > atol:
                    issues.append(f"value_mismatch:{key} live={a} stored={b}")
    if n_checked == 0:
        issues.append("no_overlapping_metric_rows")
    return issues


def validate_stored_metrics() -> dict:
    """Schema/completeness check for retained fold metrics (no prediction dumps)."""
    summary_path = METRICS_ROOT / "fold_metric_summary.csv"
    fold_path = METRICS_ROOT / "fold_metrics.csv"
    issues: list[str] = []
    if not summary_path.exists():
        issues.append(f"missing:{summary_path}")
    if not fold_path.exists():
        issues.append(f"missing:{fold_path}")
    if issues:
        raise SystemExit("\n".join(issues))
    summary = pd.read_csv(summary_path)
    fold = pd.read_csv(fold_path)
    needed = {"dataset", "model_id", "scope", "metric_name", "fold_mean", "fold_sd", "n_valid_folds"}
    missing = sorted(needed - set(summary.columns))
    if missing:
        issues.append(f"summary_missing_columns:{missing}")
    for mid in ("DS1_RDKIT_MLPENS5_EVID", "DS2_RDKIT_MLPENS5_EVID"):
        if not (summary.model_id.astype(str) == mid).any():
            issues.append(f"missing_workhorse_summary:{mid}")
        if not (fold.model_id.astype(str) == mid).any():
            issues.append(f"missing_workhorse_fold:{mid}")
    forbidden = {"pooled_oof_value", "oof_value", "concatenated_cv_value"}
    if forbidden & set(summary.columns):
        issues.append("oof_columns_in_summary")
    if issues:
        print("STORED METRICS FAIL")
        print("\n".join(issues))
        raise SystemExit(1)
    report = {
        "status": "pass",
        "mode": "stored-only",
        "n_summary_rows": int(len(summary)),
        "n_fold_rows": int(len(fold)),
        "n_models": int(summary.model_id.nunique()),
    }
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description="Fold-local CV metrics (no pooled OOF)")
    p.add_argument("--select-workhorse", action="store_true", help="Select workhorse only from complete fold summaries")
    p.add_argument(
        "--compare-stored",
        action="store_true",
        help="Recompute fold metrics and compare to results/cv_metrics; do not write",
    )
    p.add_argument(
        "--stored-only",
        action="store_true",
        help="Validate retained fold_metric_summary.csv without reading prediction dumps",
    )
    args = p.parse_args(argv)

    if args.stored_only:
        validate_stored_metrics()
        return

    files = discover_prediction_files()
    if not files:
        raise SystemExit(
            "No fold prediction files found. This release ships stored metrics only; "
            "use --stored-only or python scripts/evaluate.py"
        )
    metric_rows: list[dict] = []
    interval_rows: list[dict] = []
    coverage_rows: list[dict] = []
    risk_rows: list[dict] = []

    for model_id, fold, path in files:
        raw = pd.read_csv(path)
        df = _standardize_pred(raw, model_id, fold)
        if not args.compare_stored:
            copy_fold_prediction(df, model_id, fold)
        dataset = str(df["dataset"].iloc[0]) if "dataset" in df.columns else ("DS2" if "DS2_" in model_id else "DS1")
        out = evaluate_one_fold(df, dataset, model_id)
        metric_rows.extend(out["fold_metrics"])
        interval_rows.extend(out["interval_metrics"])
        coverage_rows.extend(out["coverage_diagnostics"])
        risk_rows.extend(out["risk_metrics"])
        print(f"  metrics {model_id} {fold} n={len(df)}", flush=True)

    if args.compare_stored:
        issues = compare_to_stored(metric_rows, METRICS_ROOT / "fold_metrics.csv")
        issues.extend(compare_to_stored(interval_rows, METRICS_ROOT / "fold_interval_metrics.csv"))
        if issues:
            print("COMPARE FAIL")
            print("\n".join(issues[:50]))
            if len(issues) > 50:
                print(f"... {len(issues) - 50} more")
            raise SystemExit(1)
        print(json.dumps({"status": "pass", "n_prediction_files": len(files), "n_metric_rows": len(metric_rows)}, indent=2))
        return

    write_rule()
    METRICS_ROOT.mkdir(parents=True, exist_ok=True)
    PRED_ROOT.mkdir(parents=True, exist_ok=True)

    fold_metrics = pd.DataFrame(metric_rows)
    if len(fold_metrics):
        cols = [c for c in LONG_METRIC_FIELDS if c in fold_metrics.columns] + [
            c for c in fold_metrics.columns if c not in LONG_METRIC_FIELDS
        ]
        fold_metrics = fold_metrics[cols]
    if len(fold_metrics):
        assert_published_metric_names(fold_metrics["metric_name"].astype(str).unique())
    fold_metrics.to_csv(METRICS_ROOT / "fold_metrics.csv", index=False)
    interval_df = pd.DataFrame(interval_rows)
    if len(interval_df):
        assert_published_metric_names(interval_df["metric_name"].astype(str).unique())
    interval_df.to_csv(METRICS_ROOT / "fold_interval_metrics.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(METRICS_ROOT / "fold_coverage_diagnostics.csv", index=False)
    risk_df = pd.DataFrame(risk_rows)
    if len(risk_df):
        risk_metric_cols = [
            c for c in risk_df.columns if c.endswith("_retained") and not c.startswith("n_")
        ]
        assert_published_metric_names(risk_metric_cols, extra_allowed=PUBLISHED_RISK_METRIC_COLUMNS)
    risk_df.to_csv(METRICS_ROOT / "fold_risk_metrics.csv", index=False)

    all_long = pd.concat(
        [fold_metrics, interval_df] if len(interval_df) and len(fold_metrics) else [fold_metrics],
        ignore_index=True,
    ) if len(fold_metrics) or len(interval_df) else pd.DataFrame()
    if len(all_long):
        summary = summarize_fold_metrics(all_long)
        forbidden = {"pooled_oof_value", "oof_value", "concatenated_cv_value"}
        if forbidden & set(summary.columns):
            raise OOFTerminologyError("summary gained OOF columns")
        summary.to_csv(METRICS_ROOT / "fold_metric_summary.csv", index=False)
    else:
        summary = pd.DataFrame(
            columns=["dataset", "model_id", "scope", "metric_name", "fold_mean", "fold_sd", "n_folds", "n_valid_folds"]
        )
        summary.to_csv(METRICS_ROOT / "fold_metric_summary.csv", index=False)

    if args.select_workhorse and len(summary):
        from uq_remake.metrics.workhorse import select_workhorse_accuracy_calibration

        elig_path = AUDIT / "workhorse_eligibility.csv"
        elig = pd.read_csv(elig_path) if elig_path.exists() else None
        ds2 = summary[summary["dataset"].astype(str).str.upper() == "DS2"]
        choice = select_workhorse_accuracy_calibration(ds2, elig)
        payload = {k: v for k, v in choice.items() if k != "rows"}
        (AUDIT / "workhorse_selection.json").write_text(
            __import__("json").dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        print("workhorse:", choice.get("selected"), "computed=", choice.get("winner_computed"), flush=True)
    else:
        print("Workhorse not selected (need --select-workhorse and complete folds).", flush=True)

    load_rule()
    assert_active_outputs_have_no_oof()
    print(f"Wrote fold-local metrics for {len(files)} prediction files to {METRICS_ROOT}", flush=True)


if __name__ == "__main__":
    main()
