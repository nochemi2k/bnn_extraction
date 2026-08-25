"""Canonical paths for manuscript assembly. Science artifacts live under results/."""
from __future__ import annotations

from uq_remake import REMAKE_ROOT

MAN = REMAKE_ROOT / "manuscripts" / "final"
MAIN_MANUSCRIPT = MAN / "main_manuscript_iz.md"
SUPPLEMENTARY = MAN / "supplementary_information.md"
MSRC = REMAKE_ROOT / "results" / "manuscript_final"
TABLES = MAN / "tables"
FIGURES = MAN / "figures"

RESULTS = REMAKE_ROOT / "results"
CV_METRICS = RESULTS / "cv_metrics"
DOWNSTREAM = RESULTS / "downstream_cv"
CANONICAL = RESULTS / "canonical"
STAGE1 = RESULTS / "stage1"
AUDITS = REMAKE_ROOT / "data" / "processed" / "audits"

SIGN_STABILITY_SUMMARY = DOWNSTREAM / "sign_stability" / "sign_stability_summary.csv"
PAIR_DENSITY_SUMMARY = DOWNSTREAM / "pair_density" / "pair_density_summary.csv"
CHEMPROP_STATUS = CV_METRICS / "chemprop_evidential_status_v2.csv"
CHEMPROP_DIAGNOSTICS = CV_METRICS / "chemprop_evidential_uq_diagnostics_v2.csv"
CMI_OFFICIAL_SPLIT = STAGE1 / "stage1a_cmisplit_results.json"
CMI_REPRODUCTION = STAGE1 / "cmi_reproduction_results.json"
CMI_OVERLAP = CANONICAL / "cmi_split_ligand_overlap.json"
HPLUS_DECISION = CANONICAL / "hplus_decision.json"
PREPARE_DATA_FINDINGS = AUDITS / "prepare_data_findings.json"
PAIR_ACQUISITION = DOWNSTREAM / "pair_acquisition"
PAIR_ACQ_SENS = DOWNSTREAM / "pair_acquisition_tau_2_0p5"
PAIR_EVENT = DOWNSTREAM / "pair_event_support"
AD_HOC_SELECTIVITY = DOWNSTREAM / "ad_hoc_selectivity"
AD_RETENTION = DOWNSTREAM / "ad_retention"
NUMBERS_JSON = MSRC / "numbers.json"

# The figure inventory has ONE definition: `figure_specs.FIGURE_SPECS`, which is what
# release/validate_release.py and tests/test_artifact_paths.py enforce. The duplicate
# COMMITTED_FIGURE_IDS tuple that used to live here was never imported and had already
# drifted from FIGURE_SPECS; it was removed rather than maintained in two places.
def committed_figure_ids() -> tuple[str, ...]:
    """Figure ids the manuscript displays, in document order."""
    from uq_remake.manuscript.figure_specs import FIGURE_SPECS

    return tuple(str(spec["figure_id"]) for spec in FIGURE_SPECS)
