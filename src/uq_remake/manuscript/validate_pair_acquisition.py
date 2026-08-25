"""Validate sigma-only pair-acquisition artifacts."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from uq_remake import REMAKE_ROOT
from uq_remake.downstream.pair_acquisition import assert_fold_hit_monotone
from uq_remake.downstream.protocol import (
    DS2_WORKHORSE,
    FOLDS,
    PAIR_ACQ_DIR,
    PAIR_ACQ_STRATEGIES,
    PAIR_ACQ_TOP_K,
)

REQUIRED = (
    "pair_acquisition_fold_metrics.csv",
    "pair_acquisition_summary.csv",
    "pair_acquisition_comparison.csv",
    "pair_acquisition_eligibility_summary.csv",
    "pair_acquisition_calibration_bin_summary.csv",
    "pair_acquisition_calibration_rho_summary.csv",
    "pair_acquisition_hit_and_precision.csv",
    "pair_acquisition_regret.csv",
    "pair_acquisition_regret_fold.csv",
    "magnitude_adjusted_risk_summary.csv",
    "magnitude_adjusted_risk_fold.csv",
    "pair_acquisition_metric_migration.json",
    "pair_acquisition_validation.json",
)


def add(rows, name, ok, detail=""):
    rows.append({"check": name, "pass": bool(ok), "detail": detail})


def main() -> dict:
    rows = []
    for fname in REQUIRED:
        path = PAIR_ACQ_DIR / fname
        add(rows, f"artifact_{fname}", path.exists(), str(path))

    val_path = PAIR_ACQ_DIR / "pair_acquisition_validation.json"
    if val_path.exists():
        payload = json.loads(val_path.read_text(encoding="utf-8"))
        add(rows, "validation_pass_flag", bool(payload.get("pass")))
        add(rows, "computation_complete", payload.get("computation_status") == "complete")
        add(rows, "no_pooled_oof", payload.get("pooled_oof_metrics") is False)
        add(rows, "no_global_selection", payload.get("global_selection") is False)
        add(rows, "no_distance_scores", payload.get("distance_scores_used") is False)
        add(rows, "workhorse_model", payload.get("model_id") == DS2_WORKHORSE, str(payload.get("model_id")))

    fold_path = PAIR_ACQ_DIR / "pair_acquisition_fold_metrics.csv"
    if fold_path.exists():
        fold = pd.read_csv(fold_path)
        add(rows, "fold_rows_nonempty", len(fold) > 0, f"n={len(fold)}")
        add(rows, "five_outer_folds", int(fold.outer_fold.nunique()) == 5, str(sorted(fold.outer_fold.astype(str).unique())))
        add(rows, "heldout_folds_only", set(fold.outer_fold.astype(str)) == set(FOLDS))
        add(rows, "has_high_and_low", {"high_selectivity", "low_selectivity"} <= set(fold.task.astype(str)))
        add(rows, "has_lcb3", "lcb_3sigma" in set(fold.strategy.astype(str)))
        add(rows, "has_random", "random" in set(fold.strategy.astype(str)))
        add(rows, "strategies_complete", set(PAIR_ACQ_STRATEGIES).issubset(set(fold.strategy.astype(str))))
        add(rows, "no_spearman_token", "spearman" not in " ".join(fold.columns).lower())
        text = " ".join(fold.astype(str).head(3).fillna("").to_string().split()).lower()
        add(rows, "no_oof_token", "oof" not in text)
        for k in PAIR_ACQ_TOP_K:
            add(rows, f"has_hit_rate_at_{k}", f"hit_rate_at_{k}" in fold.columns)
            add(rows, f"has_any_hit_at_{k}", f"any_hit_at_{k}" in fold.columns)
            add(rows, f"has_regret_at_{k}", f"regret_at_{k}" in fold.columns)
        if "regret_at_1" in fold.columns:
            det = fold[fold.strategy.astype(str) != "random"]
            for k in PAIR_ACQ_TOP_K:
                vals = pd.to_numeric(det[f"regret_at_{k}"], errors="coerce")
                add(rows, f"regret_at_{k}_nonnegative", bool((vals.dropna() >= -1e-12).all()), f"min={float(vals.min()) if vals.notna().any() else 'NA'}")
                if f"selected_contains_optimum_at_{k}" in det.columns:
                    hit = det[pd.to_numeric(det[f"selected_contains_optimum_at_{k}"], errors="coerce") >= 1.0 - 1e-12]
                    rvals = pd.to_numeric(hit[f"regret_at_{k}"], errors="coerce")
                    add(
                        rows,
                        f"regret_at_{k}_zero_if_contains_optimum",
                        bool(rvals.dropna().empty or (rvals.dropna().abs() <= 1e-12).all()),
                        f"n={int(rvals.notna().sum())}",
                    )
        try:
            assert_fold_hit_monotone(fold)
            add(rows, "task_hit_monotone", True)
        except RuntimeError as exc:
            add(rows, "task_hit_monotone", False, str(exc))

    summary_path = PAIR_ACQ_DIR / "pair_acquisition_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        add(rows, "summary_uses_fold_aggregation", "fold" in " ".join(summary.aggregation.astype(str).unique()).lower())
        add(rows, "summary_has_five_folds", bool((pd.to_numeric(summary.n_valid_folds, errors="coerce") == 5).any()))

    rho_path = PAIR_ACQ_DIR / "pair_acquisition_calibration_rho_summary.csv"
    if rho_path.exists():
        rho = pd.read_csv(rho_path)
        needed = {"rho_sigma_abs_error", "rho_sigma_abs_y", "rho_sigma_abs_yhat", "rho_abs_yhat_abs_error"}
        add(rows, "calibration_rho_complete", needed.issubset(set(rho.metric_name.astype(str))))
        add(rows, "rho_five_folds", bool((pd.to_numeric(rho.n_valid_folds, errors="coerce") == 5).all()))

    mag_path = PAIR_ACQ_DIR / "magnitude_adjusted_risk_summary.csv"
    mag_fold = PAIR_ACQ_DIR / "magnitude_adjusted_risk_fold.csv"
    if mag_path.exists() and mag_fold.exists():
        mag = pd.read_csv(mag_path)
        magf = pd.read_csv(mag_fold)
        needed_mag = {
            "rho_partial_sigma_error_given_abs_yhat",
            "rho_sigma_abs_error",
            "rho_sigma_abs_yhat",
            "rho_abs_yhat_abs_error",
        }
        add(rows, "magnitude_adjusted_metrics_present", needed_mag.issubset(set(mag.metric_name.astype(str))))
        add(rows, "magnitude_adjusted_five_folds", bool((pd.to_numeric(mag.n_valid_folds, errors="coerce") == 5).all()))
        add(rows, "magnitude_adjusted_fold_local", int((magf.metric_name.astype(str) == "rho_partial_sigma_error_given_abs_yhat").sum()) == 5)
        add(rows, "magnitude_adjusted_not_pooled", "pooled" not in " ".join(mag.metric_name.astype(str)).lower())
        man_mag = REMAKE_ROOT / "manuscripts" / "final" / "tables" / "magnitude_adjusted_risk_summary.csv"
        add(rows, "manuscript_magnitude_table", man_mag.exists(), str(man_mag))

    man = REMAKE_ROOT / "manuscripts" / "final" / "tables" / "pair_acquisition_comparison.csv"
    add(rows, "manuscript_comparison_table", man.exists(), str(man))

    hit_path = PAIR_ACQ_DIR / "pair_acquisition_hit_and_precision.csv"
    if hit_path.exists():
        wide = pd.read_csv(hit_path)
        add(rows, "hit_precision_has_hit_at_1", "hit_at_1" in wide.columns)
        add(rows, "hit_precision_has_event_precision_at_1", "event_precision_at_1" in wide.columns)
        if {"hit_rate_at_1", "event_precision_at_1"}.issubset(wide.columns):
            add(
                rows,
                "legacy_hit_rate_equals_event_precision",
                np.allclose(wide.hit_rate_at_1.to_numpy(dtype=float), wide.event_precision_at_1.to_numpy(dtype=float), equal_nan=True),
            )
        if {"any_hit_at_1", "hit_at_1"}.issubset(wide.columns):
            add(
                rows,
                "any_hit_equals_true_hit",
                np.allclose(wide.any_hit_at_1.to_numpy(dtype=float), wide.hit_at_1.to_numpy(dtype=float), equal_nan=True),
            )
        if {"event_precision_at_5", "hit_at_5"}.issubset(wide.columns):
            add(
                rows,
                "precision_not_conflated_with_hit",
                not np.allclose(
                    wide.event_precision_at_5.to_numpy(dtype=float),
                    wide.hit_at_5.to_numpy(dtype=float),
                    equal_nan=True,
                ),
            )
    man_hit = REMAKE_ROOT / "manuscripts" / "final" / "tables" / "pair_acquisition_hit_and_precision.csv"
    add(rows, "manuscript_hit_precision_table", man_hit.exists(), str(man_hit))
    man_regret = REMAKE_ROOT / "manuscripts" / "final" / "tables" / "pair_acquisition_regret.csv"
    add(rows, "manuscript_regret_table", man_regret.exists(), str(man_regret))
    regret_path = PAIR_ACQ_DIR / "pair_acquisition_regret.csv"
    if regret_path.exists():
        rg = pd.read_csv(regret_path)
        add(rows, "regret_has_k1", "regret_at_1" in rg.columns)
        add(rows, "regret_has_pool_optimum", "pool_optimum" in rg.columns)
        add(rows, "regret_has_contains_optimum", "selected_contains_optimum_at_5" in rg.columns)
        add(rows, "regret_lower_is_better_flag", "lower_is_better" not in rg.columns or bool(rg.lower_is_better.astype(str).str.lower().isin(["true", "1"]).all()))
    mig = PAIR_ACQ_DIR / "pair_acquisition_metric_migration.json"
    if mig.exists():
        payload = json.loads(mig.read_text(encoding="utf-8"))
        add(rows, "migration_legacy_is_precision", payload.get("correct_name") == "event_precision_at_k")
        add(rows, "migration_true_hit_is_any_hit", payload.get("true_hit_column") == "any_hit_at_k")
        add(rows, "migration_no_rerank", payload.get("ranking_unchanged") is True)

    report = {
        "pass": all(r["pass"] for r in rows),
        "n_checks": len(rows),
        "n_pass": int(sum(r["pass"] for r in rows)),
        "checks": rows,
    }
    out = PAIR_ACQ_DIR / "pair_acquisition_validator.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    rep = main()
    print(json.dumps({k: rep[k] for k in ("pass", "n_checks", "n_pass")}, indent=2))
    if not rep["pass"]:
        for row in rep["checks"]:
            if not row["pass"]:
                print(row)
        raise SystemExit(1)
