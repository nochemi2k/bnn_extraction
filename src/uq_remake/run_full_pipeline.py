"""Canonical reproduction pipeline.

Default mode rebuilds data, trains the active grid, evaluates, and regenerates
the manuscript. ``--dry`` uses stored artifacts without retraining.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"

# (step_name, script, live_args, dry_args). dry_args=None means skip in --dry.
STEPS = [
    ("prepare_data", "src/uq_remake/data/prepare_data.py", [], ["--verify"]),
    ("cmisplit", "src/uq_remake/data/run_cmisplit.py", [], None),
    ("cmi_repro", "src/uq_remake/data/run_cmi_reproduction.py", ["--skip-if-complete"], None),
    ("hplus", "src/uq_remake/data/run_hplus_decision.py", [], None),
    ("finalize_ds1", "src/uq_remake/data/finalize_ds1.py", [], None),
    ("active_grid", "src/uq_remake/training/run_active_grid.py", [], ["--audit-only"]),
    ("cv_metrics", "src/uq_remake/metrics/evaluate_cv.py", [], ["--stored-only"]),
    ("downstream_cv", "src/uq_remake/downstream/run_downstream_cv.py", ["--rescore"], ["--from-existing-scores"]),
    # MUST run after downstream_cv. run_ad_hoc_selectivity.py (inside downstream_cv)
    # rewrites ad_hoc_selectivity_{summary,comparison,fold_metrics}.csv and the manuscript
    # rendering copy from the independently folded DS1 checkpoints. The fold-matched
    # protocol that main-text Table 2 and SI S13 actually report then patches its rows back
    # in. Without this step a full rebuild silently reverts Table 2 to the superseded
    # alignment whose DS1 training ligands could intersect the DS2 test fold.
    ("ad_hoc_foldmatched", "src/uq_remake/downstream/run_ad_hoc_foldmatched.py", ["--stage", "all"], ["--stage", "eval"]),
    ("manuscript", "src/uq_remake/manuscript/run_manuscript_rebuild.py", [], []),
    ("validate", "src/uq_remake/release/validate_release.py", ["--skip-checkpoint-load"], ["--skip-checkpoint-load"]),
]


def _env() -> dict:
    env = os.environ.copy()
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("MPLBACKEND", "Agg")
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    rdkit_data = REPO_ROOT / ".venv311" / "Lib" / "site-packages" / "rdkit"
    if rdkit_data.exists():
        env["RDBASE"] = str(rdkit_data)
    return env


def main(argv=None):
    names = [s[0] for s in STEPS]
    p = argparse.ArgumentParser(
        description="Canonical DS1/DS2 pipeline. Default retrains from processed rebuild; --dry uses stored artifacts."
    )
    p.add_argument(
        "--from",
        dest="start",
        default=None,
        choices=names,
        help="First step. Default is prepare_data (live and --dry).",
    )
    p.add_argument(
        "--dry",
        action="store_true",
        help="No retraining. Verify data, score stored CV metrics, rebuild downstream from fold scores, regenerate figures/prose.",
    )
    p.add_argument("--skip-manuscript", action="store_true")
    p.add_argument(
        "--reuse-downstream",
        action="store_true",
        help="With --dry, skip downstream recomputation when canonical artifacts already exist.",
    )
    args = p.parse_args(argv)
    start = args.start or "prepare_data"
    i0 = names.index(start)
    env = _env()
    mode = "dry" if args.dry else "live"
    print(f"Canonical pipeline mode={mode} from step={start}", flush=True)
    for name, rel, live_extra, dry_extra in STEPS[i0:]:
        if name == "manuscript" and args.skip_manuscript:
            print("\n######## manuscript: skipped ########\n", flush=True)
            continue
        extra = dry_extra if args.dry else live_extra
        if extra is None:
            print(f"\n######## {name}: skipped in --dry ########\n", flush=True)
            continue
        extra = list(extra)
        if args.dry and args.reuse_downstream and name == "downstream_cv":
            extra = ["--from-existing-scores", "--skip-if-complete"]
        path = REPO_ROOT / rel
        print(f"\n######## {name}: {rel} {' '.join(extra)} ########\n", flush=True)
        proc = subprocess.run([sys.executable, "-u", str(path), *extra], cwd=str(REPO_ROOT), env=env)
        if proc.returncode != 0:
            raise SystemExit(f"{name} failed with {proc.returncode}")
    print("\nCanonical pipeline complete.")


if __name__ == "__main__":
    main()
