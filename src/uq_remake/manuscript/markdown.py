"""Shared Markdown helpers for the main article and SI."""
from __future__ import annotations

import pandas as pd


def md_cell(x) -> str:
    if pd.isna(x):
        return ""
    parts = str(x).split("$")
    return "$".join(part.replace("|", "/") if i % 2 == 0 else part for i, part in enumerate(parts))


def md_table(df: pd.DataFrame, cols: list[str], rename: dict | None = None) -> str:
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].copy()
    if rename:
        sub = sub.rename(columns=rename)
    headers = list(sub.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in sub.itertuples(index=False):
        cells = [md_cell(x) for x in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def note_block(text: str) -> str:
    return f"\n> **Aggregation note.** {text}\n"


def fmt_msd(mean, sd, n_valid=5, digits=3) -> str:
    if mean is None or pd.isna(mean):
        return "NA"
    if sd is None or pd.isna(sd):
        s = f"{float(mean):.{digits}f}"
    else:
        s = f"{float(mean):.{digits}f} ± {float(sd):.{digits}f}"
    if n_valid is not None and int(n_valid) != 5:
        s += f" (n_valid={int(n_valid)})"
    return s


def acq_wide_metric(df, task, orientation, strategy, prefix, k, digits=3) -> str:
    if df is None or getattr(df, "empty", True):
        return "NA"
    hit = df[
        (df.task.astype(str) == task)
        & (df.orientation.astype(str) == orientation)
        & (df.strategy.astype(str) == strategy)
    ]
    if hit.empty:
        return "NA"
    r = hit.iloc[0]
    mean_c = f"{prefix}_at_{int(k)}"
    sd_c = f"{mean_c}_sd"
    if mean_c not in r.index:
        return "NA"
    n = int(r.n_valid_folds) if "n_valid_folds" in r.index and pd.notna(r.n_valid_folds) else 5
    return fmt_msd(r[mean_c], r[sd_c] if sd_c in r.index else None, n, digits=digits)


PAIR_ACQ_COMPACT_ROWS = (
    ("high_selectivity", "both", "high selectivity", "y_pred", "predicted signed selectivity"),
    ("high_selectivity", "both", "high selectivity", "expected_improvement", "threshold-margin score"),
    ("high_selectivity", "both", "high selectivity", "random", "random"),
    ("low_selectivity", "near_zero", "near-zero selectivity", "y_pred", "predicted magnitude"),
    ("low_selectivity", "near_zero", "near-zero selectivity", "lcb_2sigma", "LCB-2"),
    ("low_selectivity", "near_zero", "near-zero selectivity", "random", "random"),
)


def pair_acq_compact_md(hit, regret) -> str:
    if hit is None or getattr(hit, "empty", True):
        return ""
    precision_prefix = "event_precision" if "event_precision_at_1" in hit.columns else "hit_rate"
    rows = []
    for task, orientation, task_label, strategy, strat_label in PAIR_ACQ_COMPACT_ROWS:
        rows.append(
            {
                "task": task_label,
                "strategy": strat_label,
                "true hit@1": acq_wide_metric(hit, task, orientation, strategy, "hit", 1),
                "event precision@1": acq_wide_metric(hit, task, orientation, strategy, precision_prefix, 1),
                "true hit@5": acq_wide_metric(hit, task, orientation, strategy, "hit", 5),
                "event precision@5": acq_wide_metric(hit, task, orientation, strategy, precision_prefix, 5),
                "regret@1": acq_wide_metric(regret, task, orientation, strategy, "regret", 1),
                "regret@5": acq_wide_metric(regret, task, orientation, strategy, "regret", 5),
                "n@1": acq_wide_metric(hit, task, orientation, strategy, "n_tasks_defined", 1, digits=1),
                "n@5": acq_wide_metric(hit, task, orientation, strategy, "n_tasks_defined", 5, digits=1),
            }
        )
    return md_table(pd.DataFrame(rows), list(rows[0].keys()))
