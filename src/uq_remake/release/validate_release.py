"""Validate that this checkout has the artifacts required for the release workflow.

Fails loudly if expected data, splits, checkpoints, canonical results, or
manuscript figures are missing or malformed. Does not retrain models.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch  # noqa: F401  — before rdkit on Windows
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from uq_remake import REMAKE_ROOT  # noqa: E402
from uq_remake.manuscript.figure_specs import FIGURE_SPECS  # noqa: E402
from uq_remake.training.protocol import load_folds, load_table  # noqa: E402

FOLDS = ("fold0", "fold1", "fold2", "fold3", "fold4")
WORKHORSE = {
    "DS1": "DS1_RDKIT_MLPENS5_EVID",
    "DS2": "DS2_RDKIT_MLPENS5_HETERO",
}
EXPECTED_MODELS = [
    "DS1_GRAPH_DMPNN_EVID",
    "DS1_GRAPH_DMPNN_POINT",
    "DS1_RDKIT_GPR",
    "DS1_RDKIT_GPR_SPARSE",
    "DS1_RDKIT_MLPENS5_BAYES_FULL",
    "DS1_RDKIT_MLPENS5_EVID",
    "DS1_RDKIT_MLPENS5_HETERO",
    "DS1_RDKIT_MLPENS5_MCDROPOUT",
    "DS1_RDKIT_MLP_BAYES_FULL",
    "DS1_RDKIT_MLP_EVID",
    "DS1_RDKIT_MLP_HETERO",
    "DS1_RDKIT_MLP_MCDROPOUT",
    "DS1_RDKIT_MLP_POINT",
    "DS1_RDKIT_SNGP",
    "DS1_RDKIT_SNGPENS5",
    "DS1_RDKIT_XGB_POINT",
    "DS2_GRAPH_DMPNN_EVID",
    "DS2_GRAPH_DMPNN_POINT",
    "DS2_RDKIT_GPR",
    "DS2_RDKIT_GPR_SPARSE",
    "DS2_RDKIT_MLPENS5_BAYES_FULL",
    "DS2_RDKIT_MLPENS5_EVID",
    "DS2_RDKIT_MLPENS5_HETERO",
    "DS2_RDKIT_MLPENS5_MCDROPOUT",
    "DS2_RDKIT_MLP_BAYES_FULL",
    "DS2_RDKIT_MLP_EVID",
    "DS2_RDKIT_MLP_HETERO",
    "DS2_RDKIT_MLP_MCDROPOUT",
    "DS2_RDKIT_MLP_POINT",
    "DS2_RDKIT_SNGP",
    "DS2_RDKIT_SNGPENS5",
    "DS2_RDKIT_XGB_POINT",
]
REQUIRED_PATHS = [
    "configs/data.yaml",
    "configs/evaluation.yaml",
    "configs/figures.yaml",
    "configs/default.yaml",
    "configs/active_grid.json",
    "configs/shared_dmpnn.json",
    "configs/experiments/active_registry.yaml",
    "configs/experiments/ad_retention_spec.yaml",
    "configs/experiments/ad_hoc_selectivity_spec.yaml",
    "configs/experiments/pair_acquisition_spec.yaml",
    "configs/experiments/pair_acquisition_threshold_sensitivity_spec.yaml",
    "data/processed/ds1.csv",
    "data/processed/ds2.csv",
    "data/processed/metal_property_table.json",
    "data/processed/features/ds1_rdkit_schema.json",
    "data/processed/features/ds2_rdkit_schema.json",
    "data/processed/audits/prepare_data_findings.json",
    "data/splits/project_folds.json",
    "data/splits/ds2_project_folds.json",
    "results/canonical/ds1_composition.json",
    "results/canonical/ds2_composition.json",
    "results/canonical/hplus_decision.json",
    "results/stage1/stage1a_cmisplit_results.json",
    "results/stage1/cmi_reproduction_results.json",
    "results/model_audit/workhorse_selection_rule.json",
    "results/cv_metrics/fold_metrics.csv",
    "results/cv_metrics/fold_metric_summary.csv",
    "results/cv_metrics/chemprop_evidential_status_v2.csv",
    "results/downstream_cv/ad_retention/fold_retention.csv",
    "results/downstream_cv/ad_retention/retention_summary.csv",
    "results/downstream_cv/ad_retention/retention_50pct.csv",
    "results/downstream_cv/ad_retention/retention_auc_summary.csv",
    "results/downstream_cv/ad/ad_fold_scores.csv",
    "results/downstream_cv/ad/ad_orientation_audit.csv",
    "results/downstream_cv/ad_retention/fold_scores/DS2_RDKIT_MLPENS5_HETERO/fold0.csv",
    "results/downstream_cv/ad_hoc_selectivity/ad_hoc_selectivity_summary.csv",
    "results/downstream_cv/pair_acquisition/pair_acquisition_summary.csv",
    "results/downstream_cv/pair_acquisition/pair_acquisition_comparison.csv",
    "results/downstream_cv/pair_acquisition_tau_2_0p5/pair_acquisition_summary.csv",
    "results/downstream_cv/pair_event_support/pair_event_support_pooled.csv",
    "results/downstream_cv/ad_hoc_selectivity/ad_hoc_selectivity_comparison.csv",
    "results/downstream_cv/pair_density/pair_density_summary.csv",
    "results/downstream_cv/ad_distance_error_corr/ad_distance_error_corr_summary.csv",
    "manuscripts/final/tables/ad_distance_error_corr.csv",
    "results/downstream_cv/sign_stability/sign_stability_summary.csv",
    "results/manuscript_final/numbers.json",
    "results/manuscript_final/workhorse_selection_accuracy_calibration.csv",
    "manuscripts/final/main_manuscript_iz.md",
    "manuscripts/final/supplementary_information.md",
    "manuscripts/final/tables/ad_retention_50pct.csv",
    "manuscripts/final/tables/main_ds2_operational_comparison.csv",
]
FORBIDDEN_TOP = {
    "embeddings",
    "docs",
    "tools",
    "archive",
}
FORBIDDEN_REL_MANUSCRIPT = {
    "manuscripts/archive",
    "manuscripts/full_manuscript",
    "manuscripts/jcim_draft",
    "results/active_grid",
    "results/archive",
    "results/audit",
    "results/cv_predictions",
    "src/uq_remake/audit",
}
FORBIDDEN_REL_FULL = {
    "manuscripts/archive",
    "manuscripts/full_manuscript",
    "manuscripts/jcim_draft",
    "results/active_grid",
    "results/archive",
    "results/audit",
    "src/uq_remake/audit",
}


class Gate:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append({"check": name, "pass": bool(ok), "detail": detail})
        if not ok:
            print(f"FAIL  {name}: {detail}", flush=True)
        else:
            print(f"PASS  {name}" + (f": {detail}" if detail else ""), flush=True)

    @property
    def ok(self) -> bool:
        return all(r["pass"] for r in self.rows)


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def check_paths(g: Gate, profile: str) -> None:
    forbidden_rel = FORBIDDEN_REL_MANUSCRIPT if profile == "manuscript" else FORBIDDEN_REL_FULL
    for rel in REQUIRED_PATHS:
        path = REMAKE_ROOT / rel
        g.check(f"path:{rel}", path.exists(), "" if path.exists() else f"missing {path}")
    for name in FORBIDDEN_TOP:
        g.check(f"absent_top:{name}", not (REMAKE_ROOT / name).exists())
    for rel in forbidden_rel:
        g.check(f"absent:{rel}", not (REMAKE_ROOT / rel).exists())


def check_tables_splits(g: Gate) -> None:
    for ds in ("DS1", "DS2"):
        table = load_table(ds)
        folds = load_folds(ds)
        g.check(f"{ds}_table_rows", len(table) > 0, f"n={len(table)}")
        id_col = "record_id" if ds == "DS1" else "pair_id"
        g.check(f"{ds}_has_{id_col}", id_col in table.columns, ",".join(table.columns[:8]))
        g.check(f"{ds}_five_folds", set(folds) >= set(FOLDS), str(sorted(folds)[:8]))
        for fold in FOLDS:
            spec = folds[fold]
            n_test = len(spec["test_idx"])
            g.check(f"{ds}_{fold}_test_idx", n_test > 0, f"n_test={n_test}")
            max_idx = max(spec["test_idx"])
            g.check(f"{ds}_{fold}_idx_in_range", max_idx < len(table), f"max={max_idx} n={len(table)}")


def check_metrics(g: Gate) -> None:
    summary = pd.read_csv(REMAKE_ROOT / "results" / "cv_metrics" / "fold_metric_summary.csv")
    needed = {"dataset", "model_id", "scope", "metric_name", "fold_mean", "fold_sd", "n_valid_folds"}
    g.check("fold_metric_summary_schema", needed <= set(summary.columns), str(sorted(summary.columns)[:20]))
    for ds, mid in WORKHORSE.items():
        hit = summary[summary.model_id.astype(str) == mid]
        g.check(f"metrics_present:{mid}", len(hit) > 0, f"n_rows={len(hit)}")
    chem = pd.read_csv(REMAKE_ROOT / "results" / "cv_metrics" / "chemprop_evidential_status_v2.csv")
    g.check("chemprop_status_rows", len(chem) > 0)
    ad50 = pd.read_csv(REMAKE_ROOT / "results" / "downstream_cv" / "ad_retention" / "retention_50pct.csv")
    g.check("retention_50pct_rows", len(ad50) > 0)
    pair_acq = pd.read_csv(REMAKE_ROOT / "results" / "downstream_cv" / "pair_acquisition" / "pair_acquisition_summary.csv")
    g.check("pair_acquisition_summary_rows", len(pair_acq) > 0)
    for mid in WORKHORSE.values():
        root = REMAKE_ROOT / "results" / "downstream_cv" / "ad_retention" / "fold_scores" / mid
        present = all((root / f"{fold}.csv").exists() for fold in FOLDS)
        g.check(f"fold_scores:{mid}", present, str(root))


def check_figures(g: Gate) -> None:
    fig_dir = REMAKE_ROOT / "manuscripts" / "final" / "figures"
    for spec in FIGURE_SPECS:
        src = REMAKE_ROOT / spec["source_artifact"]
        g.check(f"figure_source:{spec['figure_id']}", src.exists(), spec["source_artifact"])
        png = fig_dir / f"{spec['figure_id']}.png"
        pdf = fig_dir / f"{spec['figure_id']}.pdf"
        g.check(f"figure_png:{spec['figure_id']}", png.exists(), str(png))
        g.check(f"figure_pdf:{spec['figure_id']}", pdf.exists(), str(pdf))
    g.check("captions.md", (fig_dir / "captions.md").exists())
    g.check("main_manuscript", (REMAKE_ROOT / "manuscripts" / "final" / "main_manuscript_iz.md").exists())
    g.check("si", (REMAKE_ROOT / "manuscripts" / "final" / "supplementary_information.md").exists())


def _load_weight(path: Path) -> str:
    if path.suffix == ".pt":
        try:
            torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            # Chemprop 1.6 checkpoints pickle argparse.Namespace; these files are local training artifacts.
            torch.load(path, map_location="cpu", weights_only=False)
        return "torch"
    if path.suffix == ".json":
        _load_json(path)
        return "json"
    raise ValueError(path.suffix)


def check_checkpoints(g: Gate, load_weights: bool) -> None:
    root = REMAKE_ROOT / "checkpoints" / "active"
    g.check("checkpoints_root", root.exists(), str(root))
    for mid in EXPECTED_MODELS:
        model_dir = root / mid
        g.check(f"checkpoint_dir:{mid}", model_dir.exists(), str(model_dir))
        if not model_dir.exists():
            continue
        if mid.endswith("_GPR") and not mid.endswith("SPARSE"):
            inf = [p for p in model_dir.rglob("infeasible.json")]
            g.check(f"gpr_infeasible:{mid}", len(inf) == 5, f"n={len(inf)}")
            for path in inf:
                _load_json(path)
            continue
        weights = list(model_dir.rglob("model.pt")) + list(model_dir.rglob("model.json"))
        g.check(f"checkpoint_weights:{mid}", len(weights) > 0, f"n={len(weights)}")
        if load_weights:
            n_ok = 0
            for path in weights:
                try:
                    _load_weight(path)
                    n_ok += 1
                except Exception as exc:  # noqa: BLE001
                    g.check(f"load:{path.relative_to(REMAKE_ROOT).as_posix()}", False, str(exc))
            if weights:
                g.check(f"loaded_weights:{mid}", n_ok == len(weights), f"{n_ok}/{len(weights)}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Validate release data, splits, checkpoints, results, and figures.")
    p.add_argument(
        "--profile",
        choices=("manuscript", "full"),
        default=None,
        help="manuscript: forbid cv_predictions. full: allow cv_predictions. "
        "Default is full when results/cv_predictions exists, otherwise manuscript.",
    )
    p.add_argument(
        "--skip-checkpoint-load",
        action="store_true",
        help="Check checkpoint paths only; do not torch.load weights.",
    )
    args = p.parse_args(argv)
    if REMAKE_ROOT != ROOT:
        print(f"warning: package root {REMAKE_ROOT} != scripts parent {ROOT}", flush=True)
    profile = args.profile
    if profile is None:
        profile = "full" if (REMAKE_ROOT / "results" / "cv_predictions").exists() else "manuscript"
        print(f"profile: {profile} (auto)", flush=True)
    g = Gate()
    check_paths(g, profile=profile)
    check_tables_splits(g)
    check_metrics(g)
    check_figures(g)
    check_checkpoints(g, load_weights=not args.skip_checkpoint_load)
    n_fail = sum(1 for r in g.rows if not r["pass"])
    print(json.dumps({"n_checks": len(g.rows), "n_fail": n_fail, "pass": n_fail == 0}, indent=2), flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
