"""Map generated figure outputs onto the manuscript's main-figure numbering.

`generate_figures.py` is a frozen generator: it writes the historical figure ids
(`fig01_study_design`, `fig02_workhorse_selection`, `fig03_risk_ordering`,
`fig04_predicted_magnitude`, `fig05_pair_acquisition`). The submitted manuscript
displays four main figures, drops the study-design schematic, and uses the
five-score risk-ordering panel produced by `iz_risk_ordering_figure.py` rather
than the three-score version produced by `generate_figures.py`.

This module reconciles the two. It regenerates the five-score risk-ordering
figure and copies the remaining generator outputs onto the committed ids in
`figure_specs.FIGURE_SPECS`, so that a full figure rebuild
reproduces exactly the filenames referenced by `main_manuscript_iz.md`:

    fig01_workhorse_selection   <- fig02_workhorse_selection
    fig02_risk_ordering         <- iz_risk_ordering_figure (five scores)
    fig03_predicted_magnitude   <- fig04_predicted_magnitude
    fig04_pair_acquisition      <- fig05_pair_acquisition

The SI shifts too: the high-selectivity pair-membership panel was dropped when its
table was collapsed into the row-scope table (Section S4.4), so figS7..figS19 become
figS6..figS18. Three generator outputs are not displayed by either document:
`fig01_study_design`, the three-score `fig03_risk_ordering`, and
`figS6_ds2_highsel_pair`. Retired copies are listed in `_deprecated/MANIFEST.md`.

Run after `generate_figures.py`:

    python -m uq_remake.manuscript.finalize_main_figures
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uq_remake.manuscript.manuscript_paths import FIGURES  # noqa: E402

# generator id -> manuscript figure id
MAIN_FIGURE_MAP: dict[str, str] = {
    "fig02_workhorse_selection": "fig01_workhorse_selection",
    "fig04_predicted_magnitude": "fig03_predicted_magnitude",
    "fig05_pair_acquisition": "fig04_pair_acquisition",
}

# The SI dropped the high-selectivity pair-membership panel (its table was collapsed into
# the row-scope table in Section S4.4), so every later supplementary figure shifts down one.
SI_FIGURE_MAP: dict[str, str] = {
    "figS7_workhorse_fold_stability": "figS6_workhorse_fold_stability",
    "figS8_chemprop_embedding_retention": "figS7_chemprop_embedding_retention",
    "figS9_uncertainty_decomposition": "figS8_uncertainty_decomposition",
    "figS10_ad_orientation": "figS9_ad_orientation",
    "figS11_retention_all_pairs": "figS10_retention_all_pairs",
    "figS12_pair_density": "figS11_pair_density",
    "figS13_pair_event_support": "figS12_pair_event_support",
    "figS14_sign_stability_detail": "figS13_sign_stability_detail",
    "figS15_pair_acquisition_threshold_sensitivity": "figS14_pair_acquisition_threshold_sensitivity",
    "figS16_ad_hoc_interval_calibration": "figS15_ad_hoc_interval_calibration",
    "figS17_retention_ree": "figS16_retention_ree",
    "figS18_direct_vs_ad_hoc_selectivity": "figS17_direct_vs_ad_hoc_selectivity",
    "figS19_ree_fold_stability": "figS18_ree_fold_stability",
}

# Generator outputs the manuscript does not display.
RETIRED_FIGURE_IDS: tuple[str, ...] = (
    "fig01_study_design",
    "fig03_risk_ordering",
    "figS6_ds2_highsel_pair",
)

EXTENSIONS: tuple[str, ...] = (".png", ".pdf")


def _copy_pair(src_id: str, dst_id: str, missing: list[str]) -> list[Path]:
    written: list[Path] = []
    for ext in EXTENSIONS:
        src = FIGURES / f"{src_id}{ext}"
        if not src.exists():
            missing.append(str(src))
            continue
        dst = FIGURES / f"{dst_id}{ext}"
        shutil.copyfile(src, dst)
        written.append(dst)
    return written


def main(argv=None) -> int:
    # fig02_risk_ordering is produced directly under its manuscript id.
    from uq_remake.manuscript.iz_risk_ordering_figure import main as risk_ordering_main

    print("######## finalize_main_figures: fig02_risk_ordering ########", flush=True)
    risk_ordering_main()

    missing: list[str] = []
    n_written = 0
    # SI first: the shift is a chain (figS7->figS6, figS8->figS7, ...), so copying in
    # ascending source order would let an earlier copy overwrite a later source. Sorting
    # by source index descending makes every copy read an untouched file.
    def _idx(fid: str) -> int:
        digits = "".join(c for c in fid.split("_")[0] if c.isdigit())
        return int(digits) if digits else 0

    ordered_si = sorted(SI_FIGURE_MAP.items(), key=lambda kv: _idx(kv[0]), reverse=True)
    for src_id, dst_id in ordered_si + list(MAIN_FIGURE_MAP.items()):
        written = _copy_pair(src_id, dst_id, missing)
        n_written += len(written)
        for path in written:
            print(f"wrote {path}", flush=True)
    print(f"finalize_main_figures: {n_written} files written", flush=True)

    for fid in RETIRED_FIGURE_IDS:
        for ext in EXTENSIONS:
            stale = FIGURES / f"{fid}{ext}"
            if stale.exists():
                print(f"note: {stale.name} is not displayed by the manuscript", flush=True)

    if missing:
        for path in missing:
            print(f"MISSING generator output: {path}", flush=True)
        print("Run uq_remake.manuscript.generate_figures first.", flush=True)
        return 1

    # The committed inventory is figure_specs.FIGURE_SPECS; confirm every id now exists.
    from uq_remake.manuscript.manuscript_paths import committed_figure_ids

    absent = [
        f"{fid}{ext}"
        for fid in committed_figure_ids()
        for ext in EXTENSIONS
        if not (FIGURES / f"{fid}{ext}").exists()
    ]
    if absent:
        for name in absent:
            print(f"MISSING committed figure: {name}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
