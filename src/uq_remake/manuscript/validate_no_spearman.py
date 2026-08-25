"""Reject Spearman tokens in active manuscript-facing artifacts."""
from __future__ import annotations

import json
import re
from pathlib import Path

from uq_remake import REMAKE_ROOT

TOKENS = (
    "spearman",
    "mop_spearman",
    "row_spearman",
    "spearman_gain",
    "retention_curve_auc_mop_spearman",
    "ree_mop_spearman",
    "pair_balanced_spearman",
    "pair_balanced_global_spearman",
)
ENDPOINT_TOKENS = (
    "mop_spearman",
    "row_spearman",
    "spearman_gain",
    "retention_curve_auc_mop_spearman",
    "ree_mop_spearman",
    "pair_balanced_spearman",
    "pair_balanced_global_spearman",
)

ALLOW_PARTS = (
    "manuscripts/archive",
    "manuscripts\\archive",
    "results/archive",
    "results\\archive",
    "results/manuscript_final/audit",
    "results\\manuscript_final\\audit",
    "ad_framing_revision",
    "historical",
    "ad_distance_error_corr",
)

SCAN_ROOTS = (
    REMAKE_ROOT / "manuscripts" / "final",
    REMAKE_ROOT / "results" / "manuscript_final",
    REMAKE_ROOT / "results" / "cv_metrics",
    REMAKE_ROOT / "results" / "downstream_cv",
    REMAKE_ROOT / "configs" / "experiments",
    REMAKE_ROOT / "configs" / "active_grid.json",
)

SKIP_SUFFIXES = {".png", ".pdf", ".npz", ".pt", ".pth", ".ckpt", ".pyc", ".zip"}
TEXT_SUFFIXES = {".md", ".csv", ".json", ".yaml", ".yml", ".txt", ".tex"}
PAT = re.compile("|".join(re.escape(t) for t in TOKENS), re.I)


def _allowed(path: Path) -> bool:
    s = str(path).replace("\\", "/").lower()
    if any(part.replace("\\", "/").lower() in s for part in ALLOW_PARTS):
        return True
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return False
    if any(tok in text for tok in ENDPOINT_TOKENS):
        return False
    if path.name == "main_manuscript.md":
        return True
    return "ad_distance_error_corr" in text or (
        path.name == "supplementary_information.md" and "table s8.5" in text
    )


def _iter_files():
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        if root.is_file():
            yield root
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() in SKIP_SUFFIXES:
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES and path.suffix:
                continue
            yield path


def main() -> dict:
    hits = []
    scanned = 0
    for path in _iter_files():
        if path.name.endswith("_validation.json") or path.name.endswith("validation_report.json") or path.name.endswith("_validator.json"):
            continue
        if _allowed(path):
            continue
        scanned += 1
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        found = sorted(set(m.group(0).lower() for m in PAT.finditer(text)))
        if found:
            rel = str(path.relative_to(REMAKE_ROOT))
            hits.append({"path": rel, "tokens": found})
    report = {
        "pass": len(hits) == 0,
        "n_files_scanned": scanned,
        "n_hits": len(hits),
        "hits": hits[:80],
    }
    out = REMAKE_ROOT / "manuscripts" / "final" / "no_rank_correlation_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    rep = main()
    print(json.dumps({k: rep[k] for k in ("pass", "n_files_scanned", "n_hits")}, indent=2))
    if not rep["pass"]:
        for hit in rep["hits"][:20]:
            print(hit)
        raise SystemExit(1)
