"""Rebuild manuscript tables, figures, and prose from canonical artifacts.

Does not retrain models. Requires existing CV metrics, AD retention, and
sigma-only pair-acquisition artifacts.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _run_jacs() -> int:
    """JACS repositioning analyses: reads frozen fold scores / processed data only,
    writes under manuscripts/jacs/. Does not touch manuscripts/final/ and does not
    run any part of the JCIM rebuild path above."""
    from uq_remake.manuscript.within_pair_ordering import main as within_pair_main
    from uq_remake.manuscript.scale_free_error import main as scale_free_main
    from uq_remake.manuscript.logk_censoring import main as censoring_main
    from uq_remake.manuscript.uncertainty_by_regime import main as regime_main
    from uq_remake.manuscript.sigma_chemical_axes import main as sigma_axes_main
    from uq_remake.manuscript.coverage_override_grid import main as coverage_grid_main
    from uq_remake.manuscript.external_set_emulation import main as external_set_main
    from uq_remake.manuscript.single_pair_campaign import main as single_pair_main
    from uq_remake.manuscript.validate_jacs_analyses import main as validate_jacs_main

    print("######## jacs: within_pair_ordering (Tasks 1-2) ########", flush=True)
    within_pair_main()
    print("######## jacs: scale_free_error (Task 3) ########", flush=True)
    scale_free_main()
    print("######## jacs: logk_censoring (Task 4) ########", flush=True)
    censoring_main()
    print("######## jacs: uncertainty_by_regime (Task 5) ########", flush=True)
    regime_main()
    print("######## jacs: sigma_chemical_axes (Task 6) ########", flush=True)
    sigma_axes_main()
    print("######## jacs: coverage_override_grid (Tasks 7-8) ########", flush=True)
    coverage_grid_main()
    print("######## jacs: external_set_emulation (Task 9) ########", flush=True)
    external_set_main()
    print("######## jacs: single_pair_campaign (Task 10) ########", flush=True)
    single_pair_main()
    print("######## jacs: validate_jacs_analyses ########", flush=True)
    report = validate_jacs_main()
    print({k: report[k] for k in ("pass", "n_checks", "n_pass")}, flush=True)
    if not report["pass"]:
        for row in report["checks"]:
            if not row["pass"]:
                print(row)
    return 0 if report["pass"] else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Rebuild manuscript from canonical fold artifacts (no retraining)")
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--skip-validate", action="store_true")
    p.add_argument(
        "--skip-tables",
        action="store_true",
        help="Do not run build_final_assets.py; regenerate figures/prose from retained tables.",
    )
    p.add_argument(
        "--jacs",
        action="store_true",
        help=(
            "Run the JACS repositioning analyses instead of the JCIM rebuild path: "
            "within-pair risk ordering/retention, scale-free error, logK censoring, "
            "uncertainty-by-regime, sigma-vs-chemistry axes, the coverage-override "
            "grid and its flagged-region composition, and their validator. Writes "
            "under manuscripts/jacs/ only; does not touch manuscripts/final/."
        ),
    )
    args = p.parse_args(argv)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYTHONUTF8", "1")

    if args.jacs:
        return _run_jacs()

    if not args.skip_tables:
        from uq_remake.manuscript.build_final_assets import main as build_main

        print("######## manuscript: build_final_assets ########", flush=True)
        build_main()

    if not args.skip_figures:
        from uq_remake.manuscript.generate_figures import main as fig_main
        from uq_remake.manuscript.finalize_main_figures import main as finalize_figs

        print("######## manuscript: generate_figures ########", flush=True)
        fig_main()

        # generate_figures.py is frozen and emits the historical figure ids. The
        # manuscript references the finalized ids, and figure_specs/validate_release
        # enforce them, so this step is not optional on a figure rebuild.
        print("######## manuscript: finalize_main_figures ########", flush=True)
        if finalize_figs() != 0:
            raise SystemExit("finalize_main_figures failed; committed figure ids are incomplete")

        # Six figures are redrawn from the same stored summaries because the frozen
        # generator's layout obscured the result each one exists to show (see the
        # module docstring). This must run last: it overwrites the ids that
        # finalize_main_figures has just placed. No displayed number changes.
        from uq_remake.manuscript.redraw_figures import main as redraw_figs

        print("######## manuscript: redraw_figures ########", flush=True)
        if redraw_figs([]) != 0:
            raise SystemExit("redraw_figures failed; committed figures are stale")

    from uq_remake.manuscript.write_manuscript import main as write_main

    print("######## manuscript: write_manuscript ########", flush=True)
    write_main()

    if args.skip_validate:
        return 0

    from uq_remake.manuscript.iz_risk_score_table import main as iz_table3_main
    from uq_remake.manuscript.validate_no_spearman import main as no_rank_main
    from uq_remake.manuscript.validate_ad_hoc_selectivity import main as adhoc_val_main
    from uq_remake.manuscript.validate_pair_acquisition import main as pair_acq_val_main
    from uq_remake.manuscript.validate_ad_retention import main as ad_val_main
    from uq_remake.manuscript.validate_manuscript import main as ms_val_main

    # Table 3 ships as a CSV that used to have no generator. Verify, do not overwrite.
    print("######## manuscript: verify Table 3 (iz_risk_score_table) ########", flush=True)
    iz_table3 = iz_table3_main([]) == 0

    print("######## manuscript: validate_no_spearman ########", flush=True)
    no_rank = no_rank_main()
    print({k: no_rank[k] for k in ("pass", "n_files_scanned", "n_hits") if k in no_rank}, flush=True)
    print("######## manuscript: validate_ad_hoc_selectivity ########", flush=True)
    adhoc = adhoc_val_main()
    print({k: adhoc[k] for k in ("pass", "n_checks", "n_pass") if k in adhoc}, flush=True)
    print("######## manuscript: validate_pair_acquisition ########", flush=True)
    pair_acq = pair_acq_val_main()
    print({k: pair_acq[k] for k in ("pass", "n_checks", "n_pass") if k in pair_acq}, flush=True)
    print("######## manuscript: validate_ad_retention ########", flush=True)
    ad = ad_val_main()
    print("######## manuscript: validate_manuscript ########", flush=True)
    ms = ms_val_main()
    ok = iz_table3 and all(bool(r.get("pass")) for r in (no_rank, adhoc, pair_acq, ad, ms))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
