"""Canonical downstream rebuild: AD retention, context diagnostics, pair acquisition.

Default uses stored fold-local AD scores (no Tanimoto/Mahalanobis/embedding rescoring).
Pass ``--rescore`` only after a live retrain when those distances must be recomputed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uq_remake.downstream.protocol import (  # noqa: E402
    AD_HOC_DIR,
    AD_RETENTION_DIR,
    CORR_DIR,
    DS1_WORKHORSE,
    DS2_WORKHORSE,
    FOLDS,
    PAIR_ACQ_DIR,
    PAIR_ACQ_SENS_DIR,
    PAIR_ACQ_SENS_HIGH_THRESHOLD,
    PAIR_ACQ_SENS_LOW_THRESHOLD,
    PAIR_ACQ_SENS_SPEC,
    PAIR_DIR,
    PAIR_EVENT_DIR,
    SIGN_DIR,
)


def _fold_scores_complete(model_id: str) -> bool:
    root = AD_RETENTION_DIR / "fold_scores" / model_id
    return all((root / f"{fold}.csv").exists() for fold in FOLDS)


def canonical_artifacts() -> dict[str, bool]:
    return {
        "ad": (AD_RETENTION_DIR / "retention_summary.csv").exists()
        and _fold_scores_complete(DS2_WORKHORSE)
        and _fold_scores_complete(DS1_WORKHORSE),
        "context": (PAIR_DIR / "pair_density_summary.csv").exists()
        and (SIGN_DIR / "sign_stability_summary.csv").exists()
        and (CORR_DIR / "ad_distance_error_corr_summary.csv").exists(),
        "pair_acq": (PAIR_ACQ_DIR / "pair_acquisition_summary.csv").exists()
        and (PAIR_ACQ_DIR / "pair_acquisition_validation.json").exists(),
        "pair_acq_sens": (PAIR_ACQ_SENS_DIR / "pair_acquisition_summary.csv").exists()
        and (PAIR_ACQ_SENS_DIR / "pair_acquisition_validation.json").exists(),
        "si_diagnostics": (PAIR_EVENT_DIR / "pair_event_support_pooled.csv").exists(),
        "ad_hoc": (AD_HOC_DIR / "ad_hoc_selectivity_summary.csv").exists()
        and (AD_HOC_DIR / "ad_hoc_validation.json").exists(),
    }


def _run_ad(argv: list[str]) -> int:
    from uq_remake.downstream.run_ad_study import main as ad_main

    return int(ad_main(argv) or 0)


def _run_pair_acq(argv: list[str]) -> int:
    from uq_remake.downstream.run_pair_acquisition import main as pair_acq_main

    return int(pair_acq_main(argv) or 0)


def _run_ad_hoc(argv: list[str]) -> int:
    from uq_remake.downstream.run_ad_hoc_selectivity import main as adhoc_main

    return int(adhoc_main(argv) or 0)


def _run_pair_acq_sensitivity() -> int:
    from uq_remake.downstream.run_pair_acquisition import main as pair_acq_main

    return int(
        pair_acq_main(
            [
                "--high-threshold",
                str(PAIR_ACQ_SENS_HIGH_THRESHOLD),
                "--low-threshold",
                str(PAIR_ACQ_SENS_LOW_THRESHOLD),
                "--output-dir",
                str(PAIR_ACQ_SENS_DIR),
                "--spec",
                str(PAIR_ACQ_SENS_SPEC),
                "--skip-calibration",
            ]
        )
        or 0
    )


def _run_si_diagnostics() -> int:
    from uq_remake.downstream.si_diagnostics import run_from_fold_scores

    run_from_fold_scores()
    return 0


def _run_context() -> int:
    from uq_remake.downstream.context_diagnostics import rebuild_context_diagnostics

    rebuild_context_diagnostics(DS2_WORKHORSE)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Fold-local AD, pair-density/sign-stability, and sigma-only pair acquisition."
    )
    p.add_argument("--skip-chemprop", action="store_true")
    p.add_argument("--skip-random", action="store_true")
    p.add_argument("--skip-ad", action="store_true")
    p.add_argument("--skip-context", action="store_true")
    p.add_argument("--skip-pair-acq", action="store_true")
    p.add_argument("--skip-pair-acq-sens", action="store_true")
    p.add_argument("--skip-si-diagnostics", action="store_true")
    p.add_argument("--skip-ad-hoc", action="store_true")
    p.add_argument(
        "--from-existing-scores",
        action="store_true",
        help="Recompute AD MAE tables from fold_scores (default unless --rescore).",
    )
    p.add_argument(
        "--rescore",
        action="store_true",
        help="Recompute Tanimoto/Mahalanobis/embeddings from tables. Slow; use after a live retrain.",
    )
    p.add_argument(
        "--skip-if-complete",
        action="store_true",
        help="Skip AD/context/pair-acquisition steps whose canonical artifacts already exist.",
    )
    args = p.parse_args(argv)

    present = canonical_artifacts()
    run_ad = not args.skip_ad
    run_context = not args.skip_context
    run_pair_acq = not args.skip_pair_acq
    run_pair_acq_sens = not args.skip_pair_acq_sens
    run_si = not args.skip_si_diagnostics
    run_ad_hoc = not args.skip_ad_hoc
    if args.skip_if_complete:
        if present["ad"]:
            run_ad = False
            print("AD artifacts present; skipping AD", flush=True)
        if present["context"]:
            run_context = False
            print("Context artifacts present; skipping pair-density/sign-stability", flush=True)
        if present["pair_acq"]:
            run_pair_acq = False
            print("Pair-acquisition artifacts present; skipping pair acquisition", flush=True)
        if present["pair_acq_sens"]:
            run_pair_acq_sens = False
            print("Pair-acquisition sensitivity artifacts present; skipping", flush=True)
        if present["si_diagnostics"]:
            run_si = False
            print("SI diagnostic artifacts present; skipping", flush=True)
        if present["ad_hoc"]:
            run_ad_hoc = False
            print("Ad hoc selectivity artifacts present; skipping ad hoc selectivity", flush=True)

    status = {
        "ad": "skipped",
        "context": "skipped",
        "pair_acq": "skipped",
        "pair_acq_sens": "skipped",
        "si_diagnostics": "skipped",
        "ad_hoc": "skipped",
        "from_existing_scores": not args.rescore,
        "rescore": bool(args.rescore),
    }

    if run_ad:
        ad_argv = []
        if args.skip_chemprop:
            ad_argv.append("--skip-chemprop")
        if args.skip_random:
            ad_argv.append("--skip-random")
        if not args.rescore:
            ad_argv.append("--from-existing-scores")
        print(f"######## downstream: AD {' '.join(ad_argv)} ########", flush=True)
        code = _run_ad(ad_argv)
        if code:
            return code
        status["ad"] = "ran"

    if run_context:
        if not _fold_scores_complete(DS2_WORKHORSE):
            raise FileNotFoundError(AD_RETENTION_DIR / "fold_scores" / DS2_WORKHORSE)
        print("######## downstream: context diagnostics ########", flush=True)
        code = _run_context()
        if code:
            return code
        status["context"] = "ran"

    if run_pair_acq:
        if not _fold_scores_complete(DS2_WORKHORSE):
            raise FileNotFoundError(AD_RETENTION_DIR / "fold_scores" / DS2_WORKHORSE)
        print("######## downstream: pair acquisition ########", flush=True)
        code = _run_pair_acq([])
        if code:
            return code
        status["pair_acq"] = "ran"

    if run_pair_acq_sens:
        if not _fold_scores_complete(DS2_WORKHORSE):
            raise FileNotFoundError(AD_RETENTION_DIR / "fold_scores" / DS2_WORKHORSE)
        print("######## downstream: pair acquisition threshold sensitivity ########", flush=True)
        code = _run_pair_acq_sensitivity()
        if code:
            return code
        status["pair_acq_sens"] = "ran"

    if run_si:
        if not _fold_scores_complete(DS2_WORKHORSE):
            raise FileNotFoundError(AD_RETENTION_DIR / "fold_scores" / DS2_WORKHORSE)
        print("######## downstream: SI diagnostics ########", flush=True)
        code = _run_si_diagnostics()
        if code:
            return code
        status["si_diagnostics"] = "ran"

    if run_ad_hoc:
        print("######## downstream: ad hoc selectivity ########", flush=True)
        code = _run_ad_hoc([])
        if code:
            return code
        status["ad_hoc"] = "ran"

    if args.skip_chemprop:
        status["chemprop"] = "skipped"
    status["artifacts"] = canonical_artifacts()
    AD_RETENTION_DIR.mkdir(parents=True, exist_ok=True)
    out = AD_RETENTION_DIR.parent / "downstream_status.json"
    out.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2), flush=True)
    missing = [k for k, ok in status["artifacts"].items() if not ok]
    if missing:
        print(f"downstream incomplete: {missing}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
