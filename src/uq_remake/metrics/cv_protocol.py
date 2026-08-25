"""CV evaluation protocol: fold-local metrics only. Pooled OOF is forbidden."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from uq_remake import REMAKE_ROOT

OOF_PATTERN = re.compile(
    r"(?<![a-z])oof(?![a-z])|out[_-]?of[_-]?fold",
    re.IGNORECASE,
)
SPECIES_PAIR_RE = re.compile(r"^.+:-?\d+\.\d+--.+:-?\d+\.\d+$")
ARCHIVED_MODELS = {
    "BAYES_LASTLAYER",
}
FORBIDDEN_MODEL_TOKENS = (
    "BAYES_LASTLAYER",
    "BAYES_LAST",
    "GRAPH_DMPNN_BAYES",
    "GRAPH_DMPNN_MCDROPOUT",
    "CHEMPROP_MC",
    "RANK",
    "SNGP_INCOMPLETE",
    "GPR_BAYES",
    "CPEMB_BAYES",
    "RDKIT_BAYES_FULL",
)
ACTIVE_OUTPUT_DIRS = (
    REMAKE_ROOT / "results" / "cv_metrics",
    REMAKE_ROOT / "results" / "downstream_cv",
)
FORBIDDEN_RESULT_DIRS = (
    REMAKE_ROOT / "results" / "oof",
    REMAKE_ROOT / "results" / "oof_metrics",
    REMAKE_ROOT / "results" / "oof_predictions",
    REMAKE_ROOT / "results" / "pooled_oof",
)
MIN_PAIR_N = 3
HIGH_SEL_TAU = 1.0
PAIR_COLUMN = "metal_species_pair"
ACE_LEVELS = tuple(round(x, 1) for x in np.arange(0.1, 1.0, 0.1))
RETENTION_LEVELS = tuple(round(x, 1) for x in np.arange(0.1, 1.01, 0.1))
FOLD_SD_DDOF = 1
EVALUATION_PARTITION = "heldout_fold"

PUBLISHED_POINT_METRIC_NAMES = frozenset(
    {
        "rmse",
        "r2",
        "sign_accuracy_all",
        "n_exact_zero_observed",
        "mop_rmse",
        "mop_r2",
        "mop_picp_90",
        "mop_ace",
        "marginal_picp_90",
        "marginal_ace",
    }
)
PUBLISHED_RISK_METRIC_COLUMNS = frozenset({"rmse_retained"})
FORBIDDEN_PUBLISHED_METRIC_NAMES = frozenset(
    {
        "mae",
        "mop_mae",
        "mae_retained",
        "pair_balanced_mae",
        "pair_balanced_global_r2",
        "pair_balanced_r2",
        "mape",
        "restricted_mape",
        "restricted_mape_pooled",
    }
)


def assert_published_metric_names(names, *, extra_allowed: frozenset[str] | None = None) -> None:
    names = {str(n) for n in names if str(n)}
    forbidden = sorted(names & FORBIDDEN_PUBLISHED_METRIC_NAMES)
    if forbidden:
        raise ValueError(f"forbidden published metric names: {forbidden}")
    allowed = PUBLISHED_POINT_METRIC_NAMES | (extra_allowed or frozenset())
    extra = sorted(names - allowed)
    if extra:
        raise ValueError(f"unpublished metric names: {extra}")


LONG_METRIC_FIELDS = [
    "dataset",
    "model_id",
    "fold_id",
    "scope",
    "metric_name",
    "aggregation",
    "pair_column",
    "min_pair_size",
    "high_selectivity_threshold",
    "high_selectivity_estimand",
    "n_rows_total",
    "n_rows_used",
    "n_rows_invalid",
    "n_pairs_total",
    "n_pairs_retained",
    "n_pairs_valid",
    "n_pairs_invalid",
    "n_rows_in_retained_pairs",
    "value",
    "interval_method",
    "nominal_level",
    "n_valid_intervals",
    "n_invalid_intervals",
    "evaluation_partition",
]


class PooledFoldError(ValueError):
    """Raised when a metric function is given more than one held-out fold."""


class OOFTerminologyError(ValueError):
    """Raised when OOF / pooled-OOF terminology appears in an active artifact."""


class InvalidPairIdentifierError(ValueError):
    """Raised when DS2 rows lack species-level pair IDs."""


def contains_oof_terminology(text: str) -> bool:
    return bool(OOF_PATTERN.search(str(text)))


def assert_no_oof_terminology(*parts: str, context: str = "") -> None:
    blob = " ".join(str(p) for p in parts)
    if contains_oof_terminology(blob):
        raise OOFTerminologyError(
            f"OOF terminology is forbidden in active metrics ({context}): {blob[:200]}"
        )


def assert_single_heldout_fold(df: pd.DataFrame, context: str = "") -> str:
    """Primary metric functions accept exactly one held-out fold."""
    if df is None or len(df) == 0:
        raise PooledFoldError(f"{context}: empty frame")
    if "fold_id" not in df.columns and "fold" in df.columns:
        df = df.rename(columns={"fold": "fold_id"})
    if "fold_id" not in df.columns:
        raise PooledFoldError(f"{context}: fold_id is required; pooled frames are forbidden")
    folds = pd.Index(df["fold_id"].astype(str).unique())
    if len(folds) != 1:
        raise PooledFoldError(
            f"{context}: got {len(folds)} folds {list(folds)}; "
            "do not concatenate held-out predictions before metrics"
        )
    if "evaluation_partition" in df.columns:
        parts = {str(x) for x in df["evaluation_partition"].dropna().unique()}
        bad = {p for p in parts if contains_oof_terminology(p) or p.lower() in {"oof", "pooled_oof"}}
        if bad:
            raise OOFTerminologyError(f"{context}: evaluation_partition={bad}")
        unexpected = parts - {EVALUATION_PARTITION, ""}
        if unexpected:
            raise PooledFoldError(f"{context}: evaluation_partition must be {EVALUATION_PARTITION}, got {unexpected}")
    assert_no_oof_terminology(context, str(folds[0]))
    return str(folds[0])


def require_species_pair(df: pd.DataFrame, context: str = "") -> str:
    if PAIR_COLUMN not in df.columns:
        raise InvalidPairIdentifierError(
            f"{context}: DS2 metrics require {PAIR_COLUMN} (e.g. Fe:2.0--Co:2.0). "
            "Element-only pairs such as Fe--Co are forbidden."
        )
    vals = df[PAIR_COLUMN].astype(str)
    if vals.eq("").any() or vals.eq("nan").any():
        raise InvalidPairIdentifierError(f"{context}: blank {PAIR_COLUMN} values")
    bad = vals[~vals.str.match(SPECIES_PAIR_RE)]
    if len(bad):
        raise InvalidPairIdentifierError(
            f"{context}: {len(bad)} pair IDs are not species-level "
            f"(example {bad.iloc[0]!r}). Expected Element:ox--Element:ox."
        )
    return PAIR_COLUMN


def archived_model_blocked(model_id: str) -> bool:
    mid = str(model_id).upper()
    return any(tok in mid for tok in FORBIDDEN_MODEL_TOKENS)


def finite_mask(y_true, y_pred) -> np.ndarray:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return np.isfinite(yt) & np.isfinite(yp)


def fold_mean_sd(values) -> dict:
    """Unbiased fold SD (ddof=1). Primary estimand is the mean of fold metrics."""
    x = np.asarray(list(values), dtype=float)
    finite = x[np.isfinite(x)]
    n_valid = int(len(finite))
    n = int(len(x))
    if n_valid == 0:
        return {"fold_mean": float("nan"), "fold_sd": float("nan"), "n_folds": n, "n_valid_folds": 0}
    mean = float(np.mean(finite))
    sd = float(np.std(finite, ddof=FOLD_SD_DDOF)) if n_valid >= 2 else float("nan")
    return {"fold_mean": mean, "fold_sd": sd, "n_folds": n, "n_valid_folds": n_valid}


def scan_active_outputs_for_oof(roots: tuple[Path, ...] | None = None) -> list[str]:
    hits: list[str] = []
    for root in roots or ACTIVE_OUTPUT_DIRS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".csv", ".json", ".md", ".txt"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if contains_oof_terminology(path.name) or contains_oof_terminology(text[:200_000]):
                try:
                    hits.append(str(path.relative_to(REMAKE_ROOT)))
                except ValueError:
                    hits.append(str(path))
    return hits


def assert_active_outputs_have_no_oof() -> None:
    existing = [p for p in FORBIDDEN_RESULT_DIRS if p.exists()]
    if existing:
        raise OOFTerminologyError(f"Forbidden OOF result directories exist: {existing}")
    hits = scan_active_outputs_for_oof()
    if hits:
        raise OOFTerminologyError("OOF terminology in active outputs:\n" + "\n".join(hits[:50]))


def metric_row(**kwargs) -> dict:
    row = {k: kwargs.get(k, np.nan if k.startswith("n_") or k == "value" else "") for k in LONG_METRIC_FIELDS}
    row.update(kwargs)
    row["evaluation_partition"] = kwargs.get("evaluation_partition", EVALUATION_PARTITION)
    assert_no_oof_terminology(*[str(v) for v in row.values()], context="metric_row")
    return row


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
