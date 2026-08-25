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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Rebuild manuscript from canonical fold artifacts (no retraining)")
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--skip-validate", action="store_true")
    p.add_argument(
        "--skip-tables",
        action="store_true",
        help="Do not run build_final_assets.py; regenerate figures/prose from retained tables.",
    )
    args = p.parse_args(argv)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYTHONUTF8", "1")

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
