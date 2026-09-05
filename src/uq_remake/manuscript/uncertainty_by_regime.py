"""Analysis task 5: uncertainty composition by selectivity regime.

Splits held-out rows into high-selectivity (|y_true| > 1, the same
`HIGH_SEL_TAU` threshold used everywhere else in this repo) and near-zero
(|y_true| < 1), and reports fold-level means of sigma_aleatoric,
sigma_epistemic_within, sigma_epistemic_between, sigma_total, |error|, and
the aleatoric variance share, then fold mean +/- SD (ddof=1) across the 5
folds. The aleatoric share is the ratio of the fold's pooled
sigma_aleatoric^2 to pooled sigma_total^2 (law-of-total-variance identity,
verified to hold row-wise to ~1e-3): sigma_total^2 = sigma_aleatoric^2 +
sigma_epistemic_within^2 + sigma_epistemic_between^2.

This supports the reframing of the near-zero region: not a practical
retention target, but evidence that the model's uncertainty is dominated by
irreducible (aleatoric) variance in both regimes.

`manuscripts/derived/analysis/chemical_sanity_checks.json` gives rough (`~`)
figures for this task (aleatoric share ~0.85 high-selectivity, ~0.87
near-zero; sigma_total 2.296 vs 0.939) rather than the multi-decimal
regression targets used elsewhere. sigma_total reproduces to 3 decimals.
The aleatoric-share figures are directional/order-of-magnitude checks only:
this module's fold-pooled-variance-ratio definition lands at ~0.89-0.92
(high-selectivity) and ~0.92 (near-zero) -- same ranking (near-zero
slightly higher, both aleatoric-dominated) and the same qualitative
conclusion, but not an exact match to the "~0.85 / ~0.87" figures under any
ratio definition tried (row-mean-of-ratios lands at ~0.80/~0.85 instead).
Reported here rather than silently forced to match; see the validator and
the task summary for how this is flagged.

Writes `manuscripts/derived/tables/uncertainty_by_regime.csv`.

    python -m uq_remake.manuscript.uncertainty_by_regime
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from uq_remake.downstream.protocol import FOLDS
from uq_remake.manuscript.manuscript_common import (
    FOLD_SCORES_DIR,
    HIGH_SEL_TAU,
    DERIVED_TABLES,
    WORKHORSE,
    fold_mean_sd_row,
    load_all_folds,
    with_abs_residual,
    write_provenance,
)

OUT = DERIVED_TABLES / "uncertainty_by_regime.csv"
COMPONENT_COLS = ("sigma_aleatoric", "sigma_epistemic_within", "sigma_epistemic_between", "sigma_total")


def _regime_mask(y_true: pd.Series, regime: str) -> pd.Series:
    v = pd.to_numeric(y_true, errors="coerce").abs()
    if regime == "high_selectivity":
        return v > HIGH_SEL_TAU
    if regime == "near_zero":
        return v < HIGH_SEL_TAU
    raise ValueError(regime)


def regime_rows(fold_frames: dict[str, pd.DataFrame], regime: str) -> list[dict]:
    per_fold: dict[str, dict] = {}
    for fold, df in fold_frames.items():
        work = with_abs_residual(df)
        sub = work.loc[_regime_mask(work["y_true"], regime)]
        stats = {"n_rows": int(len(sub))}
        for col in COMPONENT_COLS:
            stats[col] = float(pd.to_numeric(sub[col], errors="coerce").mean()) if len(sub) else float("nan")
        stats["abs_error"] = float(sub["_abs_residual"].mean()) if len(sub) else float("nan")
        al2 = pd.to_numeric(sub["sigma_aleatoric"], errors="coerce") ** 2
        tot2 = pd.to_numeric(sub["sigma_total"], errors="coerce") ** 2
        stats["aleatoric_variance_share"] = float(al2.sum() / tot2.sum()) if len(sub) and tot2.sum() > 0 else float("nan")
        per_fold[fold] = stats

    metric_names = list(COMPONENT_COLS) + ["abs_error", "aleatoric_variance_share"]
    rows = []
    for metric in metric_names:
        fold_vals = [per_fold[fold][metric] for fold in FOLDS]
        fm = fold_mean_sd_row(fold_vals)
        rows.append(
            {
                "regime": regime,
                "regime_definition": f"|y_true| {'>' if regime == 'high_selectivity' else '<'} {HIGH_SEL_TAU}",
                "metric_name": metric,
                "fold_mean": fm["fold_mean"],
                "fold_sd": fm["fold_sd"],
                "n_valid_folds": fm["n_valid_folds"],
                "n_rows_per_fold": [per_fold[fold]["n_rows"] for fold in FOLDS],
            }
        )
    return rows


def main() -> dict:
    fold_frames = load_all_folds()
    rows: list[dict] = []
    rows.extend(regime_rows(fold_frames, "high_selectivity"))
    rows.extend(regime_rows(fold_frames, "near_zero"))
    out = pd.DataFrame(rows)
    out["source"] = f"results/downstream_cv/ad_retention/fold_scores/{WORKHORSE}/fold{{0..4}}.csv"

    DERIVED_TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    write_provenance(
        OUT,
        module="uncertainty_by_regime.main",
        inputs=[FOLD_SCORES_DIR / f"{f}.csv" for f in FOLDS],
        notes={
            "aleatoric_share_definition": "ratio of fold-pooled sigma_aleatoric^2 to fold-pooled sigma_total^2",
            "prototype_reproduction": "sigma_total exact; aleatoric share directional only (see module docstring)",
        },
    )
    print(f"wrote {OUT} ({len(out)} rows)", flush=True)
    return {"table": out}


if __name__ == "__main__":
    main()
