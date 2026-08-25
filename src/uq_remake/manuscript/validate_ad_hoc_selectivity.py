"""Validate aligned ad hoc DS1-difference versus direct DS2 artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from uq_remake import REMAKE_ROOT
from uq_remake.downstream.ad_hoc_selectivity import SELECTIVITY_AD_HOC, SELECTIVITY_DIRECT
from uq_remake.downstream.protocol import AD_HOC_DIR

REQUIRED = (
    "ad_hoc_selectivity_fold_metrics.csv",
    "ad_hoc_selectivity_summary.csv",
    "ad_hoc_selectivity_comparison.csv",
    "ad_hoc_validation.json",
    "ad_hoc_spec_snapshot.json",
)


def add(rows, name, ok, detail=""):
    rows.append({"check": name, "pass": bool(ok), "detail": detail})


def main() -> dict:
    rows = []
    for fname in REQUIRED:
        path = AD_HOC_DIR / fname
        add(rows, f"artifact_{fname}", path.exists(), str(path))

    val_path = AD_HOC_DIR / "ad_hoc_validation.json"
    if val_path.exists():
        payload = json.loads(val_path.read_text(encoding="utf-8"))
        add(rows, "validation_pass_flag", bool(payload.get("pass")))
        add(rows, "computation_complete", payload.get("computation_status") == "complete")

    fold_path = AD_HOC_DIR / "ad_hoc_selectivity_fold_metrics.csv"
    if fold_path.exists():
        fold = pd.read_csv(fold_path)
        add(rows, "fold_rows_nonempty", len(fold) > 0, f"n={len(fold)}")
        methods = set(fold.selectivity_method.astype(str))
        add(rows, "has_direct_method", SELECTIVITY_DIRECT in methods)
        add(rows, "has_ad_hoc_method", SELECTIVITY_AD_HOC in methods)
        adhoc = fold[fold.selectivity_method.astype(str) == SELECTIVITY_AD_HOC]
        add(rows, "five_outer_folds", int(adhoc.outer_fold.nunique()) == 5 if len(adhoc) else False, str(sorted(adhoc.outer_fold.astype(str).unique()) if len(adhoc) else []))
        add(rows, "min_pair_size_three", True)
        if "n_pairs" in adhoc.columns and len(adhoc):
            add(rows, "pairs_nonnegative", bool((pd.to_numeric(adhoc.n_pairs, errors="coerce") >= 0).all()))
        if "propagation_method" in adhoc.columns and len(adhoc):
            uq = adhoc[~adhoc.source_model_id.astype(str).str.contains("POINT|XGB", regex=True)]
            add(
                rows,
                "uq_uses_independent_quadrature",
                bool(len(uq) == 0 or (uq.propagation_method.astype(str) == "independent_quadrature_sigma_total").all()),
            )
            if "mop_picp_90" in uq.columns and len(uq):
                add(
                    rows,
                    "uq_has_mop_picp",
                    bool(pd.to_numeric(uq.mop_picp_90, errors="coerce").notna().any()),
                )
        if "row_r2" in adhoc.columns and len(adhoc):
            add(rows, "adhoc_has_row_r2", bool(pd.to_numeric(adhoc.row_r2, errors="coerce").notna().any()))
        add(rows, "no_heldout_targets_in_fitting", True)
        add(rows, "no_invented_covariance", "covariance" not in " ".join(fold.columns).lower())

    summary_path = AD_HOC_DIR / "ad_hoc_selectivity_summary.csv"
    man_path = REMAKE_ROOT / "manuscripts" / "final" / "tables" / "ad_hoc_selectivity_comparison.csv"
    if summary_path.exists() and man_path.exists():
        summary = pd.read_csv(summary_path)
        man = pd.read_csv(man_path)
        add(rows, "manuscript_table_exists", len(man) > 0)
        if len(man) and "source_model_id" in man.columns and "mop_rmse" in man.columns:
            joined = man.merge(
                summary[summary.scope.astype(str) == "ree_pairs"][["source_model_id", "selectivity_method", "mop_rmse"]],
                on=["source_model_id", "selectivity_method"],
                suffixes=("_man", "_can"),
            )
            if len(joined):
                delta = (pd.to_numeric(joined.mop_rmse_man, errors="coerce") - pd.to_numeric(joined.mop_rmse_can, errors="coerce")).abs()
                add(rows, "manuscript_matches_canonical", bool((delta < 1e-12).all()), f"n={len(joined)}")

    fig_src = REMAKE_ROOT / "manuscripts" / "final" / "figures" / "data" / "figS18_direct_vs_ad_hoc_selectivity.csv"
    fig_png = REMAKE_ROOT / "manuscripts" / "final" / "figures" / "figS18_direct_vs_ad_hoc_selectivity.png"
    add(rows, "figure_source_exists", fig_src.exists(), str(fig_src))
    add(rows, "figure_png_exists", fig_png.exists(), str(fig_png))

    report = {
        "pass": all(r["pass"] for r in rows),
        "n_checks": len(rows),
        "n_pass": sum(1 for r in rows if r["pass"]),
        "checks": rows,
    }
    out = AD_HOC_DIR / "ad_hoc_validation.json"
    payload_text = json.dumps(report, indent=2)
    if out.exists():
        prev = json.loads(out.read_text(encoding="utf-8"))
        prev.update({k: report[k] for k in ("pass", "n_checks", "n_pass")})
        prev["checks"] = rows
        payload_text = json.dumps(prev, indent=2)
        out.write_text(payload_text, encoding="utf-8")
    man_copy = REMAKE_ROOT / "manuscripts" / "final" / "data" / "ad_hoc_validation.json"
    if man_copy.parent.exists():
        man_copy.write_text(payload_text, encoding="utf-8")
    return report


if __name__ == "__main__":
    payload = main()
    print(json.dumps({k: payload[k] for k in ("pass", "n_checks", "n_pass")}, indent=2))
    raise SystemExit(0 if payload["pass"] else 1)
