"""Validation of the fold-local AD retention study and manuscript text."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from uq_remake import REMAKE_ROOT
from uq_remake.downstream.protocol import CHEMPROP_ONLY_SCORES, FOLDS, RETENTION_FRACTIONS

MAN = REMAKE_ROOT / "manuscripts" / "final"
MSRC = REMAKE_ROOT / "results" / "manuscript_final"
AD = REMAKE_ROOT / "results" / "downstream_cv" / "ad_retention"

FORBIDDEN = [
    r"\bCRC\b",
    r"\bcrc\b",
    r"conformal",
    r"risk budget",
    r"risk-budget",
    r"bounded[- ]risk",
    r"lambda threshold",
    r"nested threshold",
    r"validated abstention",
    r"R_CRC",
    r"risk_budget_alpha",
]
ACTIVE_TEXT = [
    MAN / "main_manuscript.md",
    MAN / "supplementary_information.md",
    MAN / "figures" / "captions.md",
]
ARCHIVE_OK = ()


def add(rows: list, name: str, ok: bool, detail: str = "") -> None:
    rows.append({"check": name, "pass": bool(ok), "detail": detail})


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def scan_forbidden(text: str, label: str) -> list[str]:
    hits = []
    for pat in FORBIDDEN:
        for m in re.finditer(pat, text, flags=re.I):
            line = text.count("\n", 0, m.start()) + 1
            snippet = text[max(0, m.start() - 40) : m.end() + 40].replace("\n", " ")
            hits.append(f"{label}:{line}:{pat}:{snippet}")
    return hits


def main() -> dict:
    rows: list[dict] = []
    main_md = _read(MAN / "main_manuscript.md")
    si_md = _read(MAN / "supplementary_information.md")
    captions = _read(MAN / "figures" / "captions.md")
    combined = "\n".join([main_md, si_md, captions])

    for path in ACTIVE_TEXT:
        hits = scan_forbidden(_read(path), path.name)
        add(rows, f"no_crc_language_{path.name}", not hits, "; ".join(hits[:8]))

    add(rows, "empirical_risk_ordering_framing", "error-ordering" in main_md.lower() or "error ordering" in main_md.lower() or "empirical risk-ordering" in main_md.lower() or "empirical AD risk" in main_md)
    add(rows, "no_coverage_guarantee_claim", "do not provide coverage" in main_md.lower() or "not a coverage" in combined.lower() or "deployment-risk guarantees" in main_md)
    add(rows, "certainty_vs_pair_acquisition", "lowest-risk" in main_md.lower() and "sigma-only" in si_md.lower() and "pair-acquisition" in si_md.lower())
    add(rows, "primary_retention_50", "50%" in main_md and "retention" in main_md.lower())
    add(rows, "table3_is_ad_retention", "50% retention" in main_md and "**Table 3." in main_md)
    add(rows, "chemprop_embedding_paragraph", "learned molecular representation" in combined)
    add(rows, "chemprop_fold_local_embeddings", ("training fold" in combined.lower() or "training-fold" in combined.lower()) and "held-out ligands were scored" in combined)
    add(rows, "random_retention_mentioned", "random" in main_md.lower() and "retention" in main_md.lower())
    add(rows, "enrichment_defined", "Enrichment" in combined or "enrichment" in combined.lower())
    add(rows, "rmse_enrichment_defined", "enrichment" in combined.lower())
    add(rows, "error_ordering_diagnostic", "error-ordering" in main_md.lower() or "error ordering" in main_md.lower() or "fixed-retention" in main_md.lower())
    add(rows, "hypotheses_predeclared", all(h in si_md for h in ("H1", "H2", "H3", "H4", "H5")))
    add(rows, "registered_spec_exists", (REMAKE_ROOT / "configs" / "experiments" / "ad_retention_spec.yaml").exists())

    required_artifacts = [
        AD / "retention_summary.csv",
        AD / "fold_retention.csv",
        AD / "retention_50pct.csv",
        AD / "random_retention_summary.csv",
        AD / "random_fold_retention.csv",
        AD / "enrichment_summary.csv",
        AD / "enrichment_fold.csv",
        AD / "retention_aurc.csv",
        AD / "leakage_audit.csv",
        AD / "score_definitions.csv",
        AD / "ad_retention_spec_snapshot.json",
    ]
    for path in required_artifacts:
        add(rows, f"artifact_{path.name}", path.exists(), str(path))

    leak_path = AD / "leakage_audit.csv"
    if leak_path.exists():
        leak = pd.read_csv(leak_path)
        add(rows, "leakage_audit_nonempty", len(leak) > 0, f"n={len(leak)}")
        add(rows, "all_distances_fold_local", (leak.status.astype(str) == "pass").all(), str(leak.status.value_counts().to_dict()))
        add(rows, "no_heldout_tanimoto_reference", not leak.tanimoto_heldout_in_reference.astype(bool).any())
        add(rows, "no_heldout_embedding_reference", not leak.embedding_heldout_in_reference.astype(bool).any())
        add(rows, "no_chemprop_scores_on_other_models", not leak.chemprop_scores_on_non_chemprop.astype(bool).any())
        add(rows, "five_folds_per_job", int(leak.groupby(["dataset", "model_id"]).outer_fold_id.nunique().min()) == 5)

    fold_path = AD / "fold_retention.csv"
    if fold_path.exists():
        fold = pd.read_csv(fold_path)
        add(rows, "no_pooled_column", not any("pool" in c.lower() for c in fold.columns))
        add(rows, "evaluation_partition_heldout", "heldout" in str(fold.get("evaluation_partition", pd.Series(dtype=str)).astype(str).str.lower().unique()))
        curve = fold[fold.metric_name.isin(["rmse", "mop_rmse"])]
        grouped = curve.groupby(["dataset", "model_id", "scope", "score_name", "metric_name"]).outer_fold_id.nunique()
        add(rows, "every_curve_has_five_folds", bool(len(grouped) and (grouped == 5).all()), f"min={int(grouped.min()) if len(grouped) else 0} n_curves={len(grouped)}")
        grid = set(float(x) for x in RETENTION_FRACTIONS)
        observed = set(pd.to_numeric(fold.retention_fraction, errors="coerce").dropna().unique())
        add(rows, "retention_grid_complete", grid.issubset(observed), f"missing={sorted(grid - observed)}")

        ds1_pairs = fold[(fold.dataset == "DS1") & (fold.metric_name.isin(["n_pairs_used", "n_pairs_total"]))]
        # DS1 pair counts may be stored as 0 in fold files; the 50% table must not display them as if they were computed.
        t50 = AD / "retention_50pct.csv"
        if t50.exists():
            tab = pd.read_csv(t50)
            ds1_tab = tab[(tab.dataset == "DS1") & (tab.metric_name == "rmse")]
            pair_cols = [c for c in ds1_tab.columns if "pairs" in c.lower()]
            if pair_cols and len(ds1_tab):
                bad = ds1_tab[pair_cols].apply(pd.to_numeric, errors="coerce")
                add(rows, "ds1_50pct_pairs_are_na", bool(bad.isna().all().all()), "DS1 pair columns must be NA, not 0")
            else:
                add(rows, "ds1_50pct_pairs_are_na", True, "no DS1 pair columns")
        enrich = AD / "enrichment_summary.csv"
        if enrich.exists():
            enr = pd.read_csv(enrich)
            add(rows, "rmse_enrichment_in_table", any(str(x).startswith("enrichment_10_") for x in enr.metric_name.astype(str)), str(enr.metric_name.unique()[:8]))
            add(rows, "no_rank_correlation_in_enrichment", "spearman" not in " ".join(enr.metric_name.astype(str)).lower())

        ds2_mop = fold[
            (fold.dataset == "DS2")
            & (fold.metric_name.isin(["mop_rmse"]))
            & (fold.scope == "ree_pairs")
        ]
        if len(ds2_mop):
            finite = ds2_mop[pd.to_numeric(ds2_mop.value, errors="coerce").notna()]
            ok_support = (
                (pd.to_numeric(finite.n_pairs_used, errors="coerce") >= 3)
                | (finite.metric_status.astype(str).isin(["insufficient_pair_coverage", "empty_retention", "not_applicable_ds1"]))
            )
            add(rows, "ds2_mop_respects_pair_support", bool(ok_support.all()), f"violations={int((~ok_support).sum())}")
        else:
            add(rows, "ds2_mop_respects_pair_support", False, "no DS2 MoP rows")

        workhorse = fold[fold.family.astype(str) == "workhorse"]
        chem_on_wh = [c for c in CHEMPROP_ONLY_SCORES if c in set(workhorse.score_name.astype(str))]
        add(rows, "chemprop_only_scores_not_on_workhorse", not chem_on_wh, str(chem_on_wh))

    scores_root = AD / "fold_scores"
    if scores_root.exists():
        bad = []
        for model_dir in scores_root.iterdir():
            if not model_dir.is_dir():
                continue
            if "GRAPH_DMPNN" in model_dir.name:
                continue
            for csv in model_dir.glob("fold*.csv"):
                cols = set(pd.read_csv(csv, nrows=0).columns)
                hit = [c for c in CHEMPROP_ONLY_SCORES if c in cols]
                if hit:
                    bad.append(f"{model_dir.name}/{csv.name}:{hit}")
        add(rows, "chemprop_columns_absent_from_non_chemprop_score_tables", not bad, "; ".join(bad[:6]))

    tables = MAN / "tables"
    add(rows, "table_50pct_copied", (tables / "ad_retention_50pct.csv").exists() or (AD / "retention_50pct.csv").exists())
    add(rows, "no_main_crc_table", not (tables / "main_crc_results.csv").exists())

    n_fail = sum(1 for r in rows if not r["pass"])
    report = {
        "n_checks": len(rows),
        "n_pass": len(rows) - n_fail,
        "n_fail": n_fail,
        "pass": n_fail == 0,
        "checks": rows,
        "experiment_id": "ad_retention_study",
        "formal_risk_control": False,
        "pooled_oof_metrics": False,
    }
    out = MSRC / "ad_retention_validation.json"
    MSRC.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (MAN / "ad_retention_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(MSRC / "ad_retention_validation.csv", index=False)
    print(json.dumps({k: report[k] for k in ("n_checks", "n_pass", "n_fail", "pass")}, indent=2))
    if n_fail:
        for r in rows:
            if not r["pass"]:
                print("FAIL", r["check"], r["detail"])
    return report


if __name__ == "__main__":
    report = main()
    raise SystemExit(0 if report["pass"] else 1)
