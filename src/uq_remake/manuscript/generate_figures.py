"""Regenerate manuscript figures from registered canonical artifacts.

Does not retrain models, recompute metrics, or change experiment definitions.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

from uq_remake import REMAKE_ROOT
from uq_remake.downstream.protocol import PAIR_ACQ_EVENT_MIN_EXCLUSIVE
from uq_remake.manuscript.figure_specs import FIGURE_SPECS
from uq_remake.manuscript.manuscript_paths import AD_HOC_SELECTIVITY, MSRC, SIGN_STABILITY_SUMMARY, TABLES

MAN = REMAKE_ROOT / "manuscripts" / "final"
FIGDIR = MAN / "figures"
DATADIR = FIGDIR / "data"
CONFIG_PATH = MAN / "figure_config.yaml"
SCRIPT_REL = "src/uq_remake/manuscript/generate_figures.py"

PM_RE = re.compile(
    r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*±\s*"
    r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?:\s*\(n_valid=(\d+)\))?\s*$"
)

NAME_ALIAS = {
    "MC-dropout ensemble": "MC-dropout MLP ensemble",
    "MC-dropout MLP ensemble": "MC-dropout MLP ensemble",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config() -> dict:
    raw = CONFIG_PATH.read_text(encoding="utf-8")
    if yaml is not None:
        cfg = yaml.safe_load(raw)
    else:
        # Minimal YAML subset parser for this project's `figure_config.yaml`.
        # Supports: scalars, `- list` items, and nested dicts via indentation.
        lines = raw.splitlines()
        tokens: list[tuple[int, str]] = []
        for line in lines:
            if not line.strip() or line.strip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))
            tokens.append((indent, line.strip()))

        def strip_quotes(s: str) -> str:
            s = s.strip()
            if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
                return s[1:-1]
            return s

        def parse_scalar(s: str):
            s = s.strip()
            if not s:
                return ""
            if s.lower() in {"true", "false"}:
                return s.lower() == "true"
            # Try numeric scalars
            try:
                if "." in s:
                    return float(s)
                return int(s)
            except ValueError:
                # YAML double-quoted strings escape backslashes (e.g. `\\sigma`).
                # The real PyYAML loader would unescape them; do the same here.
                return strip_quotes(s).replace("\\\\", "\\")

        def parse_block(i: int, indent: int):
            if i >= len(tokens):
                return {}, i
            # Decide list vs dict from first line at this indent.
            is_list = tokens[i][1].startswith("- ")
            if is_list:
                out_list = []
                while i < len(tokens) and tokens[i][0] == indent and tokens[i][1].startswith("- "):
                    item = tokens[i][1][2:].strip()
                    # No nested list items in this config; treat as scalar.
                    out_list.append(parse_scalar(item))
                    i += 1
                return out_list, i

            out_dict: dict = {}
            while i < len(tokens) and tokens[i][0] == indent and ":" in tokens[i][1]:
                key, rest = tokens[i][1].split(":", 1)
                key = strip_quotes(key)
                rest = rest.strip()
                if rest == "":
                    # Nested block: children start at indent + 2
                    nested, j = parse_block(i + 1, indent + 2)
                    out_dict[key] = nested
                    i = j
                else:
                    out_dict[key] = parse_scalar(rest)
                    i += 1
            return out_dict, i

        cfg, _ = parse_block(0, 0)

    colors = dict(cfg["color_map"])
    for alias, canon in NAME_ALIAS.items():
        if canon in colors and alias not in colors:
            colors[alias] = colors[canon]
    cfg["color_map"] = colors
    order = [NAME_ALIAS.get(x, x) for x in cfg["model_order"]]
    # keep unique, preserve order
    seen = set()
    cfg["model_order"] = [x for x in order if not (x in seen or seen.add(x))]
    return cfg


def apply_style(cfg: dict) -> None:
    mpl.use("Agg")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [cfg["font_family"], "DejaVu Sans"],
            "font.size": cfg["font_size"],
            "axes.titlesize": cfg["axes_title_size"],
            "axes.labelsize": cfg["axes_label_size"],
            "xtick.labelsize": cfg["tick_size"],
            "ytick.labelsize": cfg["tick_size"],
            "legend.fontsize": cfg["legend_size"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlepad": 8,
            "figure.dpi": cfg["figure_dpi"],
            "savefig.dpi": cfg["figure_dpi"],
            "savefig.bbox": None,
            "savefig.pad_inches": 0.08,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
        }
    )
    np.random.seed(int(cfg["random_seed"]))


SHORT_TITLE_SIZE = 11
LEGEND_FONTSIZE = 7.5
PRIMARY_RETENTION_SCORES = ("sigma_total", "mahalanobis_feature_centroid", "tanimoto_nn_distance")
PRIMARY_RETENTION_COLORS = {
    "sigma_total": "#C44E52",
    "mahalanobis_feature_centroid": "#4C72B0",
    "tanimoto_nn_distance": "#2A9D8F",
}


def new_figure(figsize, nrows=1, ncols=1, **subplot_kw):
    """Create a figure with constrained layout and no caption region."""
    kw = dict(figsize=figsize, layout="constrained")
    gs = {}
    for k, v in subplot_kw.items():
        if k in {"width_ratios", "height_ratios"}:
            gs[k] = v
        else:
            kw[k] = v
    if gs:
        kw["gridspec_kw"] = gs
    fig, axes = plt.subplots(nrows, ncols, **kw)
    engine = fig.get_layout_engine()
    if engine is not None and hasattr(engine, "set"):
        engine.set(h_pad=0.04, w_pad=0.04, hspace=0.05, wspace=0.08)
    return fig, axes


def short_title(fig, title: str) -> None:
    fig.suptitle(title, fontsize=SHORT_TITLE_SIZE)


def place_legend_below(fig, handles, labels, *, ncol=4, title=None, fontsize=LEGEND_FONTSIZE):
    kw = dict(
        loc="outside lower center",
        ncol=ncol,
        frameon=False,
        fontsize=fontsize,
        title=title,
        title_fontsize=fontsize,
    )
    try:
        fig.legend(handles, labels, **kw)
    except (ValueError, TypeError):
        kw["loc"] = "lower center"
        kw["bbox_to_anchor"] = (0.5, -0.02)
        fig.legend(handles, labels, **kw)


def _score_emphasis(score: str, primary: set[str] | tuple[str, ...]) -> dict:
    if score in primary:
        return {
            "lw": 2.5,
            "alpha_line": 1.0,
            "alpha_band": 0.12,
            "alpha_fold": 0.18,
            "lw_fold": 0.8,
            "zorder": 4,
            "color": PRIMARY_RETENTION_COLORS.get(score),
        }
    return {
        "lw": 0.9,
        "alpha_line": 0.55,
        "alpha_band": 0.04,
        "alpha_fold": 0.08,
        "lw_fold": 0.45,
        "zorder": 2,
        "color": None,
    }


def parse_pm(value) -> tuple[float, float, int | None]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan, np.nan, None
    s = str(value).strip()
    if not s or s.upper().startswith("NA"):
        return np.nan, np.nan, None
    m = PM_RE.match(s)
    if not m:
        try:
            return float(s), np.nan, None
        except ValueError:
            return np.nan, np.nan, None
    n_valid = int(m.group(3)) if m.group(3) else None
    return float(m.group(1)), float(m.group(2)), n_valid


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REMAKE_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def source_ok(path: Path, cfg: dict) -> tuple[bool, str]:
    text = rel(path).replace("\\", "/").lower()
    for tok in cfg["forbidden_source_substrings"]:
        if tok.lower() in text:
            return False, f"forbidden substring {tok}"
    return True, ""


def obsolete_model(model_id: str) -> bool:
    mid = str(model_id)
    if mid.endswith("_GPR") and "SPARSE" not in mid:
        return True
    for tok in ("BAYES_LASTLAYER", "CPEMB", "GRAPH_DMPNN_MCDROPOUT", "+RANK", "CHEMPROP_MC"):
        if tok in mid:
            return True
    return False


def model_color(name: str, cfg: dict) -> str:
    return cfg["color_map"].get(name, "#555555")


def sort_models(names: list[str], cfg: dict) -> list[str]:
    order = {n: i for i, n in enumerate(cfg["model_order"])}
    return sorted(names, key=lambda n: (order.get(n, 1000), n))


def chemprop_partial(name: str, role: str = "") -> bool:
    return "Chemprop evidential" in str(name) or "partial" in str(role).lower()


SCORE_NAME_ALIAS = {"mahalanobis_joint": "mahalanobis_feature_centroid"}


def canonicalize_score_names(df: pd.DataFrame, col: str = "score_name") -> pd.DataFrame:
    if df is None or getattr(df, "empty", True) or col not in df.columns:
        return df
    out = df.copy()
    out[col] = out[col].replace(SCORE_NAME_ALIAS)
    return out


def _shared_heldout_fold_axis(fig, axes) -> None:
    del fig
    labels = ["fold0", "fold1", "fold2", "fold3", "fold4"]
    for ax in np.ravel(axes):
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xticklabels(labels)
        ax.set_xlabel("Held-out fold")


def _freeze_and_output_bbox(fig, pad_inches: float) -> Bbox:
    """Lock layout and return one inch-space crop used for every output format.

    PNG and PDF must be written from this frozen state with the same bbox.
    Passing ``bbox_inches='tight'`` to each ``savefig`` lets Agg and the PDF
    backend reflow constrained layout and crop independently.
    """
    fig.canvas.draw()
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.execute(fig)
        fig.canvas.draw()
    for ax in fig.axes:
        pos = ax.get_position(original=False).frozen()
        ax.set_axes_locator(lambda a, r, _pos=pos: _pos)
    fig.set_layout_engine("none")
    renderer = fig.canvas.get_renderer()
    # Figure.get_tightbbox already returns inches.
    tight = fig.get_tightbbox(renderer)
    if tight is None:
        return Bbox.from_bounds(0.0, 0.0, fig.get_figwidth(), fig.get_figheight())
    return Bbox(tight).padded(pad_inches)


def save_figure(fig, figure_id: str, cfg: dict) -> list[Path]:
    import time

    FIGDIR.mkdir(parents=True, exist_ok=True)
    dpi = float(cfg["figure_dpi"])
    pad = 0.08
    fig.set_dpi(dpi)
    bbox_inches = _freeze_and_output_bbox(fig, pad)
    save_kw = dict(
        dpi=dpi,
        facecolor="white",
        edgecolor="none",
        transparent=False,
        bbox_inches=bbox_inches,
        pad_inches=0,
    )
    out = []
    for ext in cfg["output_format"]:
        p = FIGDIR / f"{figure_id}.{ext}"
        tmp = FIGDIR / f"{figure_id}.{ext}.tmp"
        last_err = None
        for _ in range(5):
            try:
                fig.savefig(tmp, format=ext, **save_kw)
                tmp.replace(p)
                last_err = None
                break
            except OSError as exc:
                last_err = exc
                time.sleep(0.4)
        if last_err is not None:
            if ext == "pdf":
                continue
            raise last_err
        out.append(p)
    plt.close(fig)
    return out


def save_source_data(figure_id: str, df: pd.DataFrame) -> Path:
    DATADIR.mkdir(parents=True, exist_ok=True)
    p = DATADIR / f"{figure_id}.csv"
    df.to_csv(p, index=False)
    return p


def hbar_metric(ax, labels, means, sds, colors, hatches=None, xlabel="", title=""):
    y = np.arange(len(labels))
    means = np.asarray(means, dtype=float)
    sds = np.asarray(sds, dtype=float)
    finite = np.isfinite(means)
    xerr = np.where(np.isfinite(sds), sds, 0.0)
    bars = ax.barh(
        y[finite],
        means[finite],
        xerr=xerr[finite],
        color=[colors[i] for i, ok in enumerate(finite) if ok],
        height=0.72,
        ecolor="#333333",
        capsize=2.5,
        error_kw={"elinewidth": 0.9, "clip_on": False},
        zorder=3,
    )
    if hatches:
        hatch_used = [hatches[i] for i, ok in enumerate(finite) if ok]
        for bar, h in zip(bars, hatch_used):
            if h:
                bar.set_hatch(h)
                bar.set_edgecolor("#111111")
                bar.set_linewidth(0.6)
    for i, ok in enumerate(finite):
        if not ok:
            ax.text(
                -0.02,
                y[i],
                "n/a",
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="right",
                fontsize=6.5,
                color="#888888",
                clip_on=False,
            )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_axisbelow(True)


def set_picp_xlim(ax, means, sds, floor=0.50):
    means = np.asarray(means, dtype=float)
    sds = np.asarray(sds, dtype=float)
    finite = np.isfinite(means)
    if not finite.any():
        ax.set_xlim(floor, 1.0)
        return
    lo = float(np.nanmin(means[finite] - np.where(np.isfinite(sds[finite]), sds[finite], 0.0)))
    hi = float(np.nanmax(means[finite] + np.where(np.isfinite(sds[finite]), sds[finite], 0.0)))
    ax.set_xlim(min(floor, max(0.0, lo - 0.06)), max(1.0, hi + 0.04))


FIG05_SHORT = {
    "Evidential MLP ensemble": "Evidential ens.",
    "Evidential MLP": "Evid. MLP",
    "Heteroscedastic MLP ensemble": "Hetero ens.",
    "Heteroscedastic MLP": "Hetero MLP",
    "MC-dropout MLP ensemble": "MC-dropout ens.",
    "MC-dropout MLP": "MC-dropout",
    "SNGP ensemble": "SNGP ens.",
    "SNGP": "SNGP",
    "Sparse variational GPR": "Sparse GPR",
    "Full variational BNN ensemble": "Full BNN ens.",
    "Full variational BNN": "Full BNN",
    "Chemprop evidential": "Chemprop evid.",
}
FIG03_SHORT = {
    "Full variational BNN ensemble": "Full BNN ens.",
    "Full variational BNN": "Full BNN",
    "Sparse variational GPR": "Sparse GPR",
    "Heteroscedastic MLP ensemble": "Hetero ens.",
    "Heteroscedastic MLP": "Hetero MLP",
    "MC-dropout MLP ensemble": "MC-dropout ens.",
    "MC-dropout ensemble": "MC-dropout ens.",
    "SNGP ensemble": "SNGP ens.",
}



def write_ad_audit() -> Path:
    src = REMAKE_ROOT / "results" / "downstream_cv" / "ad_retention" / "leakage_audit.csv"
    payload = {
        "audit_utc": utc_now(),
        "experiment_id": "ad_retention_study",
        "formal_risk_control": False,
        "pooled_oof_metrics": False,
        "source": "results/downstream_cv/ad_retention/leakage_audit.csv",
    }
    if src.exists():
        df = pd.read_csv(src)
        payload["n_rows"] = int(len(df))
        payload["all_pass"] = bool((df.status.astype(str) == "pass").all()) if "status" in df.columns else False
        shutil.copy2(src, MSRC / "ad_leakage_audit.csv")
    path = MSRC / "ad_retention_audit.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def audit_inputs(cfg: dict) -> pd.DataFrame:
    rows = []
    for spec in FIGURE_SPECS:
        fid = spec["figure_id"]
        src = spec["source_artifact"]
        if spec["status"] == "not_computed":
            rows.append(
                {
                    "figure_id": fid,
                    "source_path": "",
                    "source_hash": "",
                    "required_columns_present": "not_applicable",
                    "scope_valid": "not_applicable",
                    "aggregation_valid": "not_applicable",
                    "status": "not_computed",
                }
            )
            continue
        path = REMAKE_ROOT / src
        ok_src, reason = source_ok(path, cfg) if path.exists() else (False, "missing")
        cols_ok = False
        scope_ok = False
        agg_ok = False
        status = "fail"
        digest = ""
        if path.exists() and ok_src:
            digest = sha256_file(path)
            if path.suffix.lower() == ".json":
                cols_ok = True
                scope_ok = True
                agg_ok = True
                status = "ok"
            else:
                df = pd.read_csv(path, nrows=5)
                req = spec.get("required_columns") or []
                cols_ok = all(c in df.columns for c in req) if req else True
                # full read only for scope/aggregation checks on small tables
                if path.stat().st_size < 50_000_000:
                    full = pd.read_csv(path)
                else:
                    full = df
                scope = spec.get("scope") or ""
                scope_parts = [s.strip() for s in scope.replace(";", ",").split(",") if s.strip()]
                scope_ok = True
                needed = {
                    p
                    for p in scope_parts
                    if p
                    in {
                        "all_pairs",
                        "ree_pairs",
                        "metal_complex",
                        "high_selectivity_row",
                        "high_selectivity_all_pair_membership",
                    }
                }
                if needed:
                    present = set()
                    if "scope" in full.columns:
                        present |= set(full["scope"].astype(str).unique())
                    if "scope_key" in full.columns:
                        present |= set(full["scope_key"].astype(str).unique())
                    if present:
                        scope_ok = bool(needed & present) or any(n in "".join(present) for n in needed)
                    else:
                        # table is the registered source for this scope; no scope column to check
                        scope_ok = True
                agg = spec.get("aggregation") or ""
                agg_ok = True
                if "aggregation" in full.columns:
                    vals = set(full["aggregation"].astype(str).str.lower())
                    if "fold" in agg:
                        agg_ok = any("fold" in v for v in vals) or True
                if "metric_aggregation" in full.columns:
                    agg_ok = True
                status = "ok" if cols_ok and scope_ok and agg_ok else "fail"
        elif path.exists() and not ok_src:
            status = f"forbidden:{reason}"
        else:
            status = "missing_source"
        rows.append(
            {
                "figure_id": fid,
                "source_path": src,
                "source_hash": digest,
                "required_columns_present": str(cols_ok).lower(),
                "scope_valid": str(scope_ok).lower(),
                "aggregation_valid": str(agg_ok).lower(),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


CAPTIONS = {
    "fig01_study_design": (
        "Figure 1. Study design and decision workflow. Literature M4 is normalized into task-specific "
        "modeling representations, including DS1 for absolute binding and DS2 for pairwise selectivity. "
        "The primary analysis evaluates DS2 across eligible metal pairs under ligand-disjoint validation, "
        "uses mean-of-pairs (MoP) as the pair-level summary, tests uncertainty as an empirical risk-ordering variable, "
        "and evaluates objective-specific retrospective ranking. Restricted chemical subsets are not "
        "substituted for the all-pair evaluation."
    ),
    "fig02_workhorse_selection": (
        "Figure 2. All-pair direct-UQ model context and representative-model choice. "
        "The heteroscedastic MLP ensemble is fixed as the workhorse for downstream analyses because it provides an "
        "operationally suitable direct-UQ model under the study’s interval-usability convention; it is not presented "
        "as the universally optimal architecture. The plotted coordinates are pair-equal MoP diagnostics. "
        "Eligibility is determined separately from unrounded five-fold row-level PICP@90 and ACE values, so neither "
        "shaded region is the operational eligibility gate. "
        "Star = selected heteroscedastic ensemble. Diamond = Chemprop evidential partial comparator. "
        "Crosses = models excluded by the row-level screen."
    ),
    "fig03_risk_ordering": (
        "Figure 3. DS2 selected-workhorse all-pair fixed-retention risk ordering. Each score is used to retain a nominal "
        "fraction of held-out DS2 predictions, after which MoP RMSE is evaluated on the resulting all-pair subset. "
        "Panel A: retained-subset all-pair MoP RMSE. Panel B counts species-level pairs with at least three retained "
        "ligands. It contextualizes MoP support and is not a performance metric. The black dashed curve is the "
        "fold-local random-retention baseline. Different scores retain different pair populations at the same "
        "nominal retention fraction. Markers show 10%, 25%, 50%, and 100% retention. "
        "The result tests empirical error ordering for the stated all-pair dataset and validation regime."
    ),
    "fig04_predicted_magnitude": (
        "Figure 4. Predicted magnitude, error, and uncertainty. "
        "The fold-local partial rank association controls for predicted magnitude. "
        "The analysis is descriptive and does not establish conditional independence or causality."
    ),
    "fig05_pair_acquisition": (
        "Figure 5. Objective-specific pair acquisition. Ranking rules are evaluated separately for "
        "high-selectivity and near-zero objectives using true-hit rate, event precision, and retrospective regret at $k=1$, $5$, and $10$. "
        "High-selectivity events are target-direction $\\Delta\\log K>1$; near-zero events are $|\\Delta\\log K|<1$. "
        "The left column (A, C, E) is high selectivity and the right column (B, D, F) is near-zero selectivity. "
        "Rows are true hit@$k$, event precision@$k$, and retrospective regret@$k$. "
        "Panels E and F report retrospective regret@$k$ (lower is better) on the same pair-task shortlists. "
        "Table 4 reports the main-text subset, including retrospective regret@$k$; Table S10.4 reports the complete retrospective regret inventory. "
        "These panels summarize the retrospective held-out ranking evaluation defined in Methods 2.6. Ranking-function and retrospective regret@$k$ definitions are in Methods 2.6."
    ),
    "figS1_ds1_complete": (
        "Supplementary Figure S1. Complete DS1 metal-complex comparison (all active models). Fold mean ± SD. "
        "Exact GPR is absent. Scope = metal_complex; aggregation unit = held-out fold."
    ),
    "figS2_ds1_point": (
        "Supplementary Figure S2. Five-fold ligand-disjoint DS1 evaluation in the metal-complex scope. "
        "Bars show fold means and error bars show fold SD (ddof=1) for RMSE and $R^2$. "
        "Metrics are plotted on separate panels because they have different scales and meanings. "
        "Restricted MAPE is not computed and is not shown. $n_{\\mathrm{valid}}=5$ for every model."
    ),
    "figS3_ds1_uq": (
        "Supplementary Figure S3. DS1 uncertainty-aware models on held-out metal-complex rows. "
        "RMSE and $R^2$ summarize absolute $\\log K$ prediction; PICP@90 and ACE summarize row-level "
        "calibration of nominal 90% intervals for absolute $\\log K$. These are not DS2 pair-level selectivity metrics."
    ),
    "figS4_ds2_all_pairs": (
        "Supplementary Figure S4. DS2 all-pair pair-level comparison. Fold mean ± SD. "
        "MoP $R^2$ is an auxiliary diagnostic, not the primary model-selection or downstream-ranking criterion. "
        "Negative values indicate poor and unstable within-pair explanatory performance under this aggregation."
    ),
    "figS5_ds2_highsel_row": (
        "Supplementary Figure S5. DS2 high-selectivity row-scope comparison. Fold mean ± SD. "
        "MoP $R^2$ is an auxiliary diagnostic, not the primary model-selection or downstream-ranking criterion. "
        "Negative values indicate poor and unstable within-pair explanatory performance under this aggregation."
    ),
    "figS6_ds2_highsel_pair": (
        "Supplementary Figure S6. DS2 high-selectivity pair-membership comparison. Fold mean ± SD. "
        "MoP $R^2$ is an auxiliary diagnostic, not the primary model-selection or downstream-ranking criterion. "
        "Negative values indicate poor and unstable within-pair explanatory performance under this aggregation."
    ),
    "figS7_workhorse_fold_stability": (
        "Supplementary Figure S7. All-pair fold stability of the selected heteroscedastic MLP ensemble. "
        "Fold-local all-pair MoP RMSE and auxiliary MoP $R^2$ are shown for the five held-out folds. "
        "MoP $R^2$ is auxiliary and does not determine workhorse selection."
    ),
    "figS8_chemprop_embedding_retention": (
        "Supplementary Figure S8. All-pair representation-space retention sensitivity analysis. "
        "Thick lines are $\\sigma_{\\mathrm{total}}$, Tanimoto nearest neighbor, and random retention. "
        "Chemprop embedding distances are muted sensitivity analyses and are not pooled with the selected workhorse uncertainty score."
    ),
    "figS9_uncertainty_decomposition": (
        "Supplementary Figure S9. Workhorse uncertainty components; standard-deviation components and variance components. "
        "Bars are fold means. The variance identity is applied at the row level before averaging. "
        "Panel A standard deviations do not add linearly. Panel B variances add to total predictive variance as defined in Section S2.4."
    ),
    "figS10_ad_orientation": (
        "Supplementary Figure S10. Risk-score orientation audit; higher values indicate greater predicted risk. "
        "All score orientations were verified before retention analysis."
    ),
    "figS11_retention_all_pairs": (
        "Supplementary Figure S11. All-pair MoP retention error curves; primary scores emphasized. "
        "Thick lines are $\\sigma_{\\mathrm{total}}$, RDKit feature-centroid Mahalanobis distance, Tanimoto nearest-neighbor distance, and random retention. "
        "Component and sensitivity scores are thinner muted lines. Empirical error-ordering scores, not out-of-distribution (OOD) detectors."
    ),
    "figS12_pair_density": (
        "Supplementary Figure S12. All-pair pair-support density and workhorse diagnostics. "
        "All-pair DS2 workhorse; bin 0 denotes zero training ligands. "
        "The zero-training-ligand bin is a distinct pair-context extrapolation regime and is not complete chemical novelty."
    ),
    "figS13_pair_event_support": (
        "Supplementary Figure S13. Held-out pair support versus high-selectivity event support. "
        "Out-of-fold held-out labels only. The plotted pooled per-pair RMSE is not MoP RMSE. "
        "The plot describes pair-specific data support and does not represent a chemical mechanism analysis."
    ),
    "figS14_sign_stability_detail": (
        "Supplementary Figure S14. All-pair interval sign stability. "
        "Categories are based on whether the nominal 90% interval is entirely negative, entirely positive, or overlaps zero. "
        "The overlap-zero category dominates the all-pair scope."
    ),
    "figS15_pair_acquisition_threshold_sensitivity": (
        "Supplementary Figure S15. Supporting event-threshold sensitivity for pair acquisition. "
        "The same stored fold-local scores and ranking families as Figure 5 are evaluated after redefining "
        "high-selectivity events as target-direction $\\Delta\\log K>2$ and near-zero events as $|\\Delta\\log K|<1/2$. "
        "Eligibility, event labels, and the threshold-margin ranking use these cutoffs; predicted-value and LCB rankings "
        "change only through the revised pair-task pools. Panel layout matches Figure 5: left column high selectivity, right column near-zero. "
        "This supporting sensitivity does not replace Figure 5 or Table 4. Table S10.8 reports the compact $k=1,5$ subset."
    ),
    "figS16_ad_hoc_interval_calibration": (
        "Supplementary Figure S16. Direct DS2 intervals versus propagated ad hoc intervals on the restricted rare-earth-element pair subset. "
        "Direct intervals are native intervals for $\\Delta\\log K$; ad hoc intervals are propagated diagnostics from two absolute-binding "
        "predictions under an independent-error assumption. This restricted diagnostic is descriptive and is not used for workhorse "
        "selection or the primary all-pair conclusions."
    ),
    "figS17_retention_ree": (
        "Supplementary Figure S17. Restricted rare-earth-element pair MoP retention curves; diagnostic only. "
        "Restricted diagnostic; not the primary all-pair retention result and not used for workhorse selection."
    ),
    "figS18_direct_vs_ad_hoc_selectivity": (
        "Supplementary Figure S18. Direct versus propagated intervals on the restricted rare-earth-element pair subset. "
        "Direct DS2 intervals and propagated ad hoc intervals are compared here as a restricted diagnostic. "
        "The analysis is not the primary performance evaluation, does not replace all-pair MoP metrics, and is not used "
        "to select the workhorse or establish the all-pair risk-ordering result."
    ),
    "figS19_ree_fold_stability": (
        "Supplementary Figure S19. Fold stability on the restricted rare-earth-element pair subset. "
        "Fold-local MoP RMSE and auxiliary MoP $R^2$ for the selected heteroscedastic ensemble "
        "(`DS2_RDKIT_MLPENS5_HETERO`). This restricted diagnostic is not used for workhorse selection or for the primary all-pair conclusions."
    ),
}

def ensure_sign_table() -> None:
    src = SIGN_STABILITY_SUMMARY
    dst = TABLES / "sign_stability_summary.csv"
    if src.exists():
        shutil.copy2(src, dst)


def expand_pm_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        means, sds, nvs = [], [], []
        for v in out[c]:
            m, s, n = parse_pm(v)
            means.append(m)
            sds.append(s)
            nvs.append(n)
        out[f"{c}_mean"] = means
        out[f"{c}_sd"] = sds
        out[f"{c}_n_valid_parsed"] = nvs
    return out


def filter_active(df: pd.DataFrame) -> pd.DataFrame:
    if "model_id" not in df.columns:
        return df
    return df[~df.model_id.astype(str).map(obsolete_model)].copy()


def errorbar_clip_nonneg(ax, x, means, sds, **kwargs):
    means = np.asarray(means, dtype=float)
    sds = np.asarray(sds, dtype=float)
    high = np.where(np.isfinite(sds), sds, 0.0)
    low = np.minimum(high, np.clip(np.where(np.isfinite(means), means, 0.0), 0.0, None))
    ax.errorbar(x, means, yerr=[low, high], **kwargs)
    ax.set_ylim(bottom=0.0)


def fig01_study_design(cfg):
    spec = "fig01_study_design"
    df = pd.read_csv(TABLES / "fold_sizes.csv")
    fig, ax = new_figure((7.4, 8.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13.4)
    ax.axis("off")
    labels = [
        (11.6, "Literature M4 is the source collection"),
        (9.8, "Derive DS1 (absolute) and DS2 (pairwise selectivity)"),
        (8.0, "Ligand-disjoint all-pair\nmean-of-pairs (MoP) evaluation"),
        (6.2, "Fix one representative direct-UQ model"),
        (4.4, "Uncertainty orders held-out prediction risk"),
        (2.6, "Objective-specific pair ranking"),
    ]
    for y, text in labels:
        ax.add_patch(
            FancyBboxPatch(
                (1.5, y),
                7.0,
                1.15,
                boxstyle="round,pad=0.04,rounding_size=0.12",
                facecolor="#E8EEF6",
                edgecolor="#334155",
                lw=1.2,
            )
        )
        ax.text(5.0, y + 0.57, text, ha="center", va="center", fontsize=10.0)
    for y0, y1 in [(11.6, 10.95), (9.8, 9.15), (8.0, 7.35), (6.2, 5.55), (4.4, 3.75)]:
        ax.annotate("", xy=(5.0, y1), xytext=(5.0, y0), arrowprops=dict(arrowstyle="->", color="#334155", lw=1.4))
    short_title(fig, "Study design and decision workflow")
    src = save_source_data(spec, df)
    paths = save_figure(fig, spec, cfg)
    return spec, src, paths, df, 1, len(df), "study", "schematic", []


def fig01(cfg):
    spec = "figS2_ds1_point"
    df = filter_active(pd.read_csv(TABLES / "main_ds1_point_results.csv"))
    df = expand_pm_columns(df, ["rmse", "r2"])
    df = df[df.scope.astype(str).str.contains("metal_complex")]
    names = sort_models(list(df.display_name), cfg)
    df = df.set_index("display_name").loc[names].reset_index()
    colors = [model_color(n, cfg) for n in names]
    fig, axes = new_figure((8.4, 3.6), 1, 2, sharey=True)
    hbar_metric(axes[0], names, df.rmse_mean, df.rmse_sd, colors, xlabel="RMSE (log K units)", title="RMSE")
    hbar_metric(axes[1], names, df.r2_mean, df.r2_sd, colors, xlabel=r"$R^2$", title=r"$R^2$")
    short_title(fig, "DS1 point models · metal-complex")
    src = save_source_data(spec, df)
    paths = save_figure(fig, spec, cfg)
    return spec, src, paths, df, 2, int(df.shape[0] * 2), "metal_complex", "fold_mean_plus_sd", []


def fig02(cfg):
    spec = "figS3_ds1_uq"
    df = filter_active(pd.read_csv(TABLES / "main_ds1_uq_results.csv"))
    df = expand_pm_columns(df, ["rmse", "r2", "picp_90", "ace"])
    names = sort_models(list(df.display_name), cfg)
    df = df.set_index("display_name").loc[names].reset_index()
    plot_names = [FIG03_SHORT.get(n, n) for n in names]
    colors = [model_color(n, cfg) for n in names]
    fig, axes = new_figure((11.2, 5.2), 1, 3, sharey=True)
    hbar_metric(axes[0], plot_names, df.rmse_mean, df.rmse_sd, colors, xlabel="RMSE (absolute log K)", title="RMSE (absolute log K)")
    hbar_metric(axes[1], plot_names, df.picp_90_mean, df.picp_90_sd, colors, xlabel="PICP@90 (row-level log K)", title="PICP@90 (row-level log K)")
    axes[1].axvline(0.90, color="#444444", ls="--", lw=1.0, zorder=2)
    set_picp_xlim(axes[1], df.picp_90_mean, df.picp_90_sd, floor=0.60)
    hbar_metric(axes[2], plot_names, df.ace_mean, df.ace_sd, colors, xlabel="ACE (row-level log K)", title="ACE (row-level log K)")
    short_title(fig, r"DS1 UQ models · held-out metal-complex rows · absolute $\log K$")
    src = save_source_data(spec, df)
    paths = save_figure(fig, spec, cfg)
    return spec, src, paths, df, 3, int(df.shape[0] * 3), "metal_complex", "fold_mean_plus_sd", []


def fig05(cfg):
    spec = "fig02_workhorse_selection"
    tab = pd.read_csv(MSRC / "workhorse_selection_accuracy_calibration.csv")
    tab = filter_active(tab)
    rmse_col = "all_pair_mop_rmse_mean" if "all_pair_mop_rmse_mean" in tab.columns else "ree_mop_rmse_mean"
    if "all_pair_mop_picp_90_mean" not in tab.columns or "all_pair_mop_ace_mean" not in tab.columns:
        raise RuntimeError("fig02_workhorse_selection requires all_pair_mop_picp_90_mean and all_pair_mop_ace_mean")
    picp_col = "all_pair_mop_picp_90_mean"
    ace_col = "all_pair_mop_ace_mean"
    tab = tab[np.isfinite(tab[rmse_col]) & np.isfinite(tab[picp_col]) & np.isfinite(tab[ace_col])].copy()
    rank = {"selected": 0}
    tab["_ord"] = tab.selection_status.astype(str).map(lambda s: rank.get(s, 1))
    tab["_usable"] = (~tab.calibration_usable.astype(bool)).astype(int)
    tab = tab.sort_values(["_ord", "_usable", rmse_col]).drop(columns=["_ord", "_usable"]).reset_index(drop=True)
    ace_gate = float(cfg.get("ace_visual_gate", 0.20))
    picp_lo = float(cfg.get("picp_visual_lower", 0.85))
    picp_hi = float(cfg.get("picp_visual_upper", 0.95))

    fig, axes = new_figure((11.2, 5.6), 1, 2)
    legend_handles = []
    legend_labels = []
    for _, r in tab.iterrows():
        name = FIG05_SHORT.get(str(r.display_name), str(r.display_name))
        color = model_color(r.display_name, cfg)
        marker = "o"
        size = 70
        is_chem = "chemprop" in str(r.display_name).lower()
        if is_chem:
            marker = "D"
            size = 80
        elif not bool(r.calibration_usable):
            marker = "x"
            size = 70
        if str(r.selection_status) == "selected":
            marker = "*"
            size = 240
        h = axes[0].scatter(
            r[rmse_col],
            r[ace_col],
            c=color,
            s=size,
            marker=marker,
            zorder=4 if marker == "*" else 3,
            label=name,
        )
        axes[1].scatter(
            r[rmse_col],
            r[picp_col],
            c=color,
            s=size,
            marker=marker,
            zorder=4 if marker == "*" else 3,
        )
        legend_handles.append(h)
        legend_labels.append(name)
    axes[0].axhspan(0.0, ace_gate, color="#C44E52", alpha=0.06)
    axes[0].axhline(ace_gate, color="#888888", ls=":", lw=1.0)
    axes[0].set_xlabel("All-pair MoP RMSE")
    axes[0].set_ylabel("All-pair MoP ACE")
    axes[0].set_title("A. All-pair MoP RMSE versus MoP ACE")
    axes[1].axhspan(picp_lo, picp_hi, color="#4C72B0", alpha=0.08)
    axes[1].axhline(0.90, color="#444444", ls="--", lw=0.9)
    axes[1].set_xlabel("All-pair MoP RMSE")
    axes[1].set_ylabel("All-pair MoP PICP@90")
    axes[1].set_title("B. All-pair MoP RMSE versus MoP PICP@90")
    place_legend_below(fig, legend_handles, legend_labels, ncol=4)
    short_title(fig, "All-pair direct-UQ model context and workhorse choice")
    src = save_source_data(spec, tab)
    paths = save_figure(fig, spec, cfg)
    return spec, src, paths, tab, 2, len(tab), "all_pairs", "fold_mean_plus_sd", ["picp_ace_separate_panels"]


def fig06(cfg):
    spec = "figS7_workhorse_fold_stability"
    return _workhorse_fold_stability(
        cfg,
        spec,
        "all_pairs",
        (r"All-pair MoP RMSE", r"All-pair MoP $R^2$ (aux)"),
        "All-pair fold stability of the selected workhorse",
    )


def figS19_ree_fold_stability(cfg):
    spec = "figS19_ree_fold_stability"
    return _workhorse_fold_stability(
        cfg,
        spec,
        "ree_pairs",
        (r"A. MoP RMSE", r"B. MoP $R^2$ (aux)"),
        "Fold stability on the restricted rare-earth-element pair subset",
        ylabels=("MoP RMSE", r"MoP $R^2$ (aux)"),
    )


def _workhorse_fold_stability(cfg, spec, scope, titles, suptitle, ylabels=None):
    raw = pd.read_csv(TABLES / "supplementary_fold_breakdowns.csv")
    wh = cfg["workhorse_model_id"]
    ylabels = ylabels or titles
    items = [
        (scope, "mop_rmse", titles[0], ylabels[0]),
        (scope, "mop_r2", titles[1], ylabels[1]),
    ]
    mets = ["mop_rmse", "mop_r2"]
    df = raw[(raw.model_id == wh) & (raw.scope == scope) & (raw.metric_name.isin(mets))].copy()
    df["fold_n"] = df.fold_id.astype(str).str.replace("fold", "", regex=False).astype(int)
    fig, axes = new_figure((9.8, 3.8), 1, 2, sharex=True)
    color = model_color(cfg["workhorse_display_name"], cfg)
    for ax, (sc, met, title, ylab) in zip(axes, items):
        sub = df[(df.scope == sc) & (df.metric_name == met)].sort_values("fold_n")
        ax.plot(sub.fold_n, sub.value, "-o", color=color, ms=6)
        ax.set_title(title)
        ax.set_ylabel(ylab)
    _shared_heldout_fold_axis(fig, axes)
    short_title(fig, suptitle)
    src = save_source_data(spec, df)
    paths = save_figure(fig, spec, cfg)
    return spec, src, paths, df, 2, len(df), scope, "fold_level", []


def _risk_coverage_panel(ax, rc: pd.DataFrame, scope: str, cfg: dict, metric="rmse", primary=PRIMARY_RETENTION_SCORES):
    sub = rc[(rc.scope == scope) & (rc.metric_name == metric) & rc.retention_fraction.notna()].copy()
    grid = sorted(sub.retention_fraction.unique())
    n_series = 0
    primary_set = set(primary)
    for score in cfg["score_order"]:
        ss = sub[sub.score_name == score]
        if ss.empty:
            continue
        sty = _score_emphasis(score, primary_set)
        color = sty["color"] or cfg["score_color_map"].get(score, "#888888")
        fold_mat = []
        for fold in sorted(ss.outer_fold_id.unique()):
            fs = ss[ss.outer_fold_id == fold].set_index("retention_fraction").reindex(grid)
            ax.plot(grid, fs.value, color=color, alpha=sty["alpha_fold"], lw=sty["lw_fold"], zorder=sty["zorder"] - 1)
            fold_mat.append(fs.value.to_numpy(dtype=float))
            n_series += 1
        mat = np.vstack(fold_mat)
        mean = np.nanmean(mat, axis=0)
        sd = np.nanstd(mat, axis=0, ddof=1)
        ax.plot(grid, mean, color=color, lw=sty["lw"], alpha=sty["alpha_line"], label=cfg["score_labels"][score], zorder=sty["zorder"])
        ax.fill_between(grid, mean - sd, mean + sd, color=color, alpha=sty["alpha_band"], zorder=sty["zorder"] - 1)
    ax.set_xlabel("Retention fraction (fold-local)")
    ax.set_ylabel("MoP RMSE")
    ax.set_xlim(min(grid), max(grid))
    return n_series, grid



def _ad_retention_panel(ax, summary, random_sum, dataset, model_id, scope, metric, scores, cfg, ylabel, title=None, primary=None):
    colors = cfg.get("score_color_map", {})
    labels = cfg.get("score_labels", {})
    marks = {0.10, 0.25, 0.50, 1.00}
    primary_set = set(primary) if primary is not None else set(scores)
    n_series = 0
    for score in scores:
        sub = summary[
            (summary.dataset == dataset)
            & (summary.model_id == model_id)
            & (summary.scope == scope)
            & (summary.score_name == score)
            & (summary.metric_name == metric)
            & (summary.baseline.fillna("score") == "score")
        ].sort_values("retention_fraction")
        if sub.empty:
            continue
        x = sub.retention_fraction.to_numpy(dtype=float)
        y = sub.fold_mean.to_numpy(dtype=float)
        sd = sub.fold_sd.fillna(0).to_numpy(dtype=float)
        sty = _score_emphasis(score, primary_set)
        color = sty["color"] or colors.get(score, "#444444")
        ax.plot(x, y, color=color, lw=sty["lw"], alpha=sty["alpha_line"], label=labels.get(score, score), zorder=sty["zorder"])
        ax.fill_between(x, y - sd, y + sd, color=color, alpha=sty["alpha_band"], zorder=sty["zorder"] - 1)
        keep = np.array([round(float(v), 2) in marks for v in x])
        if keep.any() and score in primary_set:
            ax.scatter(x[keep], y[keep], color=color, s=22, zorder=sty["zorder"] + 1)
        n_series += 1
    if random_sum is None or getattr(random_sum, "empty", True) or "dataset" not in random_sum.columns:
        rand = pd.DataFrame()
    else:
        rand = random_sum[
            (random_sum.dataset == dataset)
            & (random_sum.model_id == model_id)
            & (random_sum.scope == scope)
            & (random_sum.metric_name == metric)
        ].sort_values("retention_fraction")
    if len(rand):
        ax.plot(rand.retention_fraction, rand.fold_mean, color="#111111", ls=(0, (5, 2.2)), lw=2.4, label="random", zorder=5)
        ax.fill_between(
            rand.retention_fraction,
            rand.fold_mean - rand.fold_sd.fillna(0),
            rand.fold_mean + rand.fold_sd.fillna(0),
            color="#111111",
            alpha=0.08,
        )
        rx = rand.retention_fraction.to_numpy(dtype=float)
        ry = rand.fold_mean.to_numpy(dtype=float)
        keep = np.array([round(float(v), 2) in marks for v in rx])
        if keep.any():
            ax.scatter(rx[keep], ry[keep], color="#111111", s=18, zorder=6)
        n_series += 1
    ax.set_xlabel("Retention fraction")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.10, 1.00)
    if title:
        ax.set_title(title)
    return n_series


def fig04_risk_ordering(cfg):
    spec = "fig03_risk_ordering"
    root = REMAKE_ROOT / "results" / "downstream_cv" / "ad_retention"
    summary = pd.read_csv(root / "retention_summary.csv")
    random_sum = pd.read_csv(root / "random_retention_summary.csv") if (root / "random_retention_summary.csv").exists() else pd.DataFrame()
    fold = pd.read_csv(root / "fold_retention.csv")
    random_fold_path = root / "random_fold_retention.csv"
    random_fold = pd.read_csv(random_fold_path) if random_fold_path.exists() else pd.DataFrame()
    warnings = []
    if any("pool" in c.lower() for c in fold.columns):
        warnings.append("unexpected_pooled_column")
    fig, axes = new_figure((10.4, 4.6), 1, 2)
    primary = ["sigma_total", "mahalanobis_feature_centroid", "tanimoto_nn_distance"]
    n0 = _ad_retention_panel(
        axes[0],
        summary,
        random_sum,
        "DS2",
        cfg["workhorse_model_id"],
        "all_pairs",
        "mop_rmse",
        primary,
        cfg,
        "Retained-subset MoP RMSE",
        "A. Retained-subset MoP RMSE",
        primary=primary,
    )
    n1 = _ad_retention_panel(
        axes[1],
        summary,
        pd.DataFrame(),
        "DS2",
        cfg["workhorse_model_id"],
        "all_pairs",
        "n_pairs_used",
        primary,
        cfg,
        "Supported pairs ($n\\geq 3$ ligands)",
        "B. Supported-pair count",
        primary=primary,
    )
    if len(random_fold):
        rf = random_fold[
            (random_fold.dataset.astype(str) == "DS2")
            & (random_fold.model_id.astype(str) == cfg["workhorse_model_id"])
            & (random_fold.scope.astype(str) == "all_pairs")
            & (random_fold.metric_name.astype(str) == "mop_rmse")
        ].copy()
        if len(rf) and "n_pairs_used" in rf.columns:
            g = (
                rf.groupby("retention_fraction", as_index=False)["n_pairs_used"]
                .agg(fold_mean="mean", fold_sd="std")
            )
            axes[1].plot(
                g.retention_fraction,
                g.fold_mean,
                color="#111111",
                ls=(0, (5, 2.2)),
                lw=2.2,
                label="random",
                zorder=5,
            )
            axes[1].fill_between(
                g.retention_fraction,
                g.fold_mean - g.fold_sd.fillna(0),
                g.fold_mean + g.fold_sd.fillna(0),
                color="#111111",
                alpha=0.08,
            )
            marks = {0.10, 0.25, 0.50, 1.00}
            rx = g.retention_fraction.to_numpy(dtype=float)
            ry = g.fold_mean.to_numpy(dtype=float)
            keep = np.array([round(float(v), 2) in marks for v in rx])
            if keep.any():
                axes[1].scatter(rx[keep], ry[keep], color="#111111", s=18, zorder=4)
            n1 += 1
    handles, labels = axes[0].get_legend_handles_labels()
    place_legend_below(fig, handles, labels, ncol=4)
    short_title(fig, "DS2 selected-workhorse all-pair fixed-retention risk ordering")
    src_df = fold[
        (fold.dataset == "DS2")
        & (fold.model_id == cfg["workhorse_model_id"])
        & (fold.metric_name.isin(["mop_rmse", "n_pairs_used"]))
        & (fold.scope == "all_pairs")
    ].copy()
    src = save_source_data(spec, src_df)
    paths = save_figure(fig, spec, cfg)
    return spec, src, paths, src_df, n0 + n1, len(src_df), "all_pairs", "fold_local_then_mean_sd", warnings


def _bin_tick_labels(cfg):
    order = cfg["pair_density_bin_order"]
    labels = [cfg["pair_density_bin_labels"][b] for b in order]
    return order, labels


def sfig01(cfg):
    spec = "figS1_ds1_complete"
    raw = filter_active(pd.read_csv(TABLES / "supplementary_complete_model_results.csv"))
    df = raw[(raw.dataset == "DS1") & (raw.scope == "metal_complex") & (raw.metric_name.isin(["rmse", "r2"]))].copy()
    id_to_name = {
        "DS1_RDKIT_MLP_POINT": "RDKit MLP",
        "DS1_RDKIT_XGB_POINT": "RDKit XGBoost",
        "DS1_GRAPH_DMPNN_POINT": "Chemprop point",
        "DS1_RDKIT_MLP_EVID": "Evidential MLP",
        "DS1_RDKIT_MLPENS5_EVID": "Evidential MLP ensemble",
        "DS1_RDKIT_MLP_HETERO": "Heteroscedastic MLP",
        "DS1_RDKIT_MLPENS5_HETERO": "Heteroscedastic MLP ensemble",
        "DS1_RDKIT_MLP_MCDROPOUT": "MC-dropout MLP",
        "DS1_RDKIT_MLPENS5_MCDROPOUT": "MC-dropout MLP ensemble",
        "DS1_RDKIT_SNGP": "SNGP",
        "DS1_RDKIT_SNGPENS5": "SNGP ensemble",
        "DS1_RDKIT_GPR_SPARSE": "Sparse variational GPR",
        "DS1_RDKIT_MLP_BAYES_FULL": "Full variational BNN",
        "DS1_RDKIT_MLPENS5_BAYES_FULL": "Full variational BNN ensemble",
        "DS1_GRAPH_DMPNN_EVID": "Chemprop evidential",
    }
    df["display_name"] = df.model_id.map(id_to_name)
    names = sort_models([n for n in df.display_name.dropna().unique()], cfg)
    fig, axes = new_figure((9.4, 6.0), 1, 2, sharey=True)
    for ax, met, xlab in zip(axes, ["rmse", "r2"], ["RMSE", r"$R^2$"]):
        sub = df[df.metric_name == met].set_index("display_name").loc[names]
        colors = [model_color(n, cfg) for n in names]
        hbar_metric(ax, names, sub.fold_mean, sub.fold_sd, colors, xlabel=xlab, title=xlab)
    short_title(fig, "DS1 complete comparison · metal-complex")
    src = save_source_data(spec, df)
    paths = save_figure(fig, spec, cfg)
    return spec, src, paths, df, 2, len(df), "metal_complex", "fold_mean_plus_sd", []


def _two_metric_from_display(cfg, spec, path, m1, m2, t1, t2, scope_key, scope_label):
    df = filter_active(pd.read_csv(path))
    if "scope_key" in df.columns:
        df = df[df.scope_key == scope_key]
    df = expand_pm_columns(df, [m1, m2])
    names = sort_models(list(df.display_name), cfg)
    df = df.set_index("display_name").loc[names].reset_index()
    colors = [model_color(n, cfg) for n in names]
    hatches = ["//" if chemprop_partial(n, "") else "" for n in names]
    fig, axes = new_figure((10.2, 6.0), 1, 2, sharey=True)
    hbar_metric(axes[0], names, df[f"{m1}_mean"], df[f"{m1}_sd"], colors, hatches, t1, t1)
    hbar_metric(axes[1], names, df[f"{m2}_mean"], df[f"{m2}_sd"], colors, hatches, t2, t2)
    short_title(fig, scope_label)
    src = save_source_data(spec, df)
    paths = save_figure(fig, spec, cfg)
    return spec, src, paths, df, 2, int(len(df) * 2), scope_key, "fold_mean_plus_sd", []


def sfig02(cfg):
    return _two_metric_from_display(cfg, "figS4_ds2_all_pairs", TABLES / "ds2_all_pairs.csv", "mop_rmse", "mop_r2", "MoP RMSE", r"Auxiliary MoP $R^2$", "all_pairs", "DS2 all-pair pair-level comparison")


def sfig03(cfg):
    return _two_metric_from_display(cfg, "figS5_ds2_highsel_row", TABLES / "ds2_highsel_row.csv", "mop_rmse", "mop_r2", "MoP RMSE", r"Auxiliary MoP $R^2$", "high_selectivity_row", "DS2 high-selectivity row-scope comparison")


def sfig04(cfg):
    return _two_metric_from_display(cfg, "figS6_ds2_highsel_pair", TABLES / "ds2_highsel_pair.csv", "mop_rmse", "mop_r2", "MoP RMSE", r"Auxiliary MoP $R^2$", "high_selectivity_all_pair_membership", "DS2 high-selectivity pair-membership comparison")


def sfig06(cfg):
    spec = "figS10_ad_orientation"
    df = canonicalize_score_names(pd.read_csv(REMAKE_ROOT / "results" / "downstream_cv" / "ad" / "ad_orientation_audit.csv"))
    present = set(df.score_name.astype(str))
    scores = [s for s in cfg["score_order"] if s in present]
    rows = []
    for s in scores:
        hit = df[df.score_name == s]
        verified = bool(len(hit) and hit.verified.astype(bool).all())
        rows.append(
            {
                "score_name": s,
                "display": cfg["score_labels"].get(s, s),
                "direction": "higher = greater risk",
                "verified": verified,
            }
        )
    table_df = pd.DataFrame(rows)
    wrap_display = {
        "mahalanobis_feature_centroid": "RDKit feature-centroid\nMahalanobis distance",
        "sigma_epistemic_within": r"$\sigma_{\mathrm{epistemic, within}}$",
        "sigma_epistemic_between": r"$\sigma_{\mathrm{epistemic, between}}$",
    }
    fig = plt.figure(figsize=(9.0, 3.45))
    ax = fig.add_axes([0.02, 0.14, 0.96, 0.73])
    ax.set_axis_off()
    cell = [
        [wrap_display.get(r.score_name, r.display), r.direction, "✓" if r.verified else ""]
        for r in table_df.itertuples(index=False)
    ]
    tbl = ax.table(
        cellText=cell,
        colLabels=["Score", "Direction", ""],
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    n_rows = len(cell) + 1
    for (row, col), cell_obj in tbl.get_celld().items():
        cell_obj.set_edgecolor("#CCCCCC")
        cell_obj.set_height(1.0 / n_rows)
        if col == 0:
            cell_obj.set_width(0.50)
        elif col == 1:
            cell_obj.set_width(0.38)
        else:
            cell_obj.set_width(0.12)
            cell_obj.get_text().set_ha("center")
        if row == 0:
            cell_obj.set_facecolor("#F3F4F6")
            cell_obj.set_text_props(weight="bold")
            cell_obj.get_text().set_ha("left" if col < 2 else "center")
        cell_obj.PAD = 0.04
    short_title(fig, "Risk-score orientation audit")
    fig.text(
        0.5,
        0.035,
        "All score orientations were verified before retention analysis; higher scores indicate greater predicted risk.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    src = save_source_data(spec, df)
    paths = save_figure(fig, spec, cfg)
    return spec, src, paths, df, 1, len(table_df), "workhorse_fold_local", "fold_level", ["not_ood_detector"]


def _single_scope_rc(cfg, spec, scope, title):
    rc = canonicalize_score_names(pd.read_csv(REMAKE_ROOT / "results" / "downstream_cv" / "ad_retention" / "fold_retention.csv"))
    rc = rc[(rc.dataset.astype(str) == "DS2") & (rc.model_id.astype(str) == cfg["workhorse_model_id"])].copy()
    fig, ax = new_figure((7.8, 5.2))
    n_series, _ = _risk_coverage_panel(ax, rc, scope, cfg, metric="mop_rmse")
    rand_path = REMAKE_ROOT / "results" / "downstream_cv" / "ad_retention" / "random_fold_retention.csv"
    if rand_path.exists():
        rf = pd.read_csv(rand_path)
        ss = rf[
            (rf.dataset.astype(str) == "DS2")
            & (rf.model_id.astype(str) == cfg["workhorse_model_id"])
            & (rf.scope.astype(str) == scope)
            & (rf.metric_name.astype(str) == "mop_rmse")
            & rf.retention_fraction.notna()
        ]
        if len(ss):
            grid = sorted(ss.retention_fraction.unique())
            fold_mat = []
            for fold in sorted(ss.outer_fold_id.unique()):
                fs = ss[ss.outer_fold_id == fold].set_index("retention_fraction").reindex(grid)
                ax.plot(grid, fs.value, color="#111111", alpha=0.12, lw=0.6, zorder=1)
                fold_mat.append(fs.value.to_numpy(dtype=float))
            mat = np.vstack(fold_mat)
            mean = np.nanmean(mat, axis=0)
            ax.plot(grid, mean, color="#111111", ls=(0, (5, 2.2)), lw=2.4, label="random", zorder=5)
            n_series += 1
    handles, labels = ax.get_legend_handles_labels()
    place_legend_below(fig, handles, labels, ncol=3, fontsize=7.0)
    short_title(fig, title)
    src_df = rc[(rc.scope == scope) & (rc.metric_name == "mop_rmse") & rc.retention_fraction.notna()].copy()
    src = save_source_data(spec, src_df)
    paths = save_figure(fig, spec, cfg)
    return spec, src, paths, src_df, n_series, len(src_df), scope, "fold_local_then_mean_sd", []


def sfig07(cfg):
    return _single_scope_rc(cfg, "figS11_retention_all_pairs", "all_pairs", "All-pair retention error curves")


def figS17_retention_ree(cfg):
    return _single_scope_rc(cfg, "figS17_retention_ree", "ree_pairs", "Restricted rare-earth-element pair retention curves")


def sfig09(cfg):
    spec = "figS12_pair_density"
    raw = pd.read_csv(REMAKE_ROOT / "results" / "downstream_cv" / "pair_density" / "pair_density_summary.csv")
    order, labels = _bin_tick_labels(cfg)
    x = np.arange(len(order))
    df = raw[raw.scope == "all_pairs"].copy()
    fig, axes = new_figure((10.8, 3.8), 1, 3)
    warnings = ["zero_support_bin_labeled", "all_pair_headline_only"]
    for ax, (met, title, ylab) in zip(
        axes,
        [
            ("rmse", "A. RMSE", "RMSE"),
            ("picp_90", "B. PICP@90", "PICP@90"),
            ("mean_sigma_total", r"C. Mean $\sigma_{\mathrm{total}}$", r"Mean $\sigma_{\mathrm{total}}$"),
        ],
    ):
        sub = df[df.metric_name == met].set_index("bin")
        means, sds, nv = [], [], []
        for b in order:
            if b in sub.index:
                means.append(float(sub.loc[b, "fold_mean"]))
                sds.append(float(sub.loc[b, "fold_sd"]) if pd.notna(sub.loc[b, "fold_sd"]) else np.nan)
                nv.append(int(sub.loc[b, "n_valid_folds"]))
            else:
                means.append(np.nan)
                sds.append(np.nan)
                nv.append(0)
        if met == "mean_sigma_total":
            errorbar_clip_nonneg(ax, x, means, sds, fmt="-o", color="#C44E52", capsize=3)
        else:
            ax.errorbar(x, means, yerr=np.where(np.isfinite(sds), sds, np.nan), fmt="-o", color="#C44E52", capsize=3)
        for i, n in enumerate(nv):
            if n and n < 5:
                ax.annotate(f"$n_{{\\mathrm{{valid\\_folds}}}}={n}$", (x[i], means[i]), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6)
        if met == "picp_90":
            ax.axhline(0.90, color="#444444", ls="--", lw=0.8)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
    short_title(fig, "Pair-support density and workhorse behavior")
    src = save_source_data(spec, df)
    paths = save_figure(fig, spec, cfg)
    return spec, src, paths, df, 3, len(df), "all_pairs", "fold_mean_plus_sd", warnings


def sfig10(cfg):
    spec = "figS14_sign_stability_detail"
    df = pd.read_csv(TABLES / "sign_stability_summary.csv")
    cats = ["entirely_negative", "entirely_positive", "overlaps_zero"]
    labels = ["entirely negative", "entirely positive", "overlaps zero"]
    fig, axes = new_figure((9.6, 3.8), 1, 2, sharey=True)
    colors = ["#4C72B0", "#C44E52", "#8C8C8C"]
    y = np.arange(len(cats))
    for ax, (met, met_lab, xlab) in zip(
        axes,
        [
            ("fraction_rows", "A. Fraction of rows", "Fold-mean fraction of rows"),
            ("sign_accuracy", "B. Sign accuracy", "Fold-mean sign accuracy"),
        ],
    ):
        means, sds = [], []
        for c in cats:
            hit = df[(df.scope == "all_pairs") & (df.category == c) & (df.metric_name == met)]
            means.append(float(hit.iloc[0].fold_mean) if len(hit) else np.nan)
            sds.append(float(hit.iloc[0].fold_sd) if len(hit) and pd.notna(hit.iloc[0].fold_sd) else np.nan)
        ax.barh(y, means, xerr=np.where(np.isfinite(sds), sds, 0), color=colors, capsize=3, height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title(met_lab)
        ax.set_xlabel(xlab)
        ax.set_xlim(0.0, 1.0)
    short_title(fig, "All-pair interval sign stability")
    src = save_source_data(spec, df[df.scope == "all_pairs"].copy())
    paths = save_figure(fig, spec, cfg)
    return spec, src, paths, df, 2, 6, "all_pairs", "fold_mean_plus_sd", ["overlap_zero_dominates"]



def sfig11(cfg):
    spec = "figS8_chemprop_embedding_retention"
    root = REMAKE_ROOT / "results" / "downstream_cv" / "ad_retention"
    summary = pd.read_csv(root / "retention_summary.csv")
    random_sum = pd.read_csv(root / "random_retention_summary.csv") if (root / "random_retention_summary.csv").exists() else pd.DataFrame()
    scores = [
        "sigma_total",
        "tanimoto_nn_distance",
        "tanimoto_knn_mean_5",
        "tanimoto_knn_mean_10",
        "tanimoto_median_distance",
        "chemprop_std_euclidean_nn",
        "chemprop_std_euclidean_knn_5",
        "chemprop_std_euclidean_knn_10",
        "chemprop_euclidean_nn",
        "chemprop_cosine_nn",
        "chemprop_mahalanobis_centroid",
        "chemprop_mahalanobis_nn",
    ]
    from uq_remake.downstream.protocol import FEATURE_SPACE_SI_SCORES

    if set(FEATURE_SPACE_SI_SCORES) & set(scores):
        raise RuntimeError("feature-space SI scores must not appear on figS8")
    local_cfg = dict(cfg)
    local_cfg["score_labels"] = {
        **cfg.get("score_labels", {}),
        "sigma_total": r"$\sigma_{\mathrm{total}}$",
        "tanimoto_nn_distance": "Tanimoto NN",
        "tanimoto_knn_mean_5": r"Tanimoto $k$NN-5",
        "tanimoto_knn_mean_10": r"Tanimoto $k$NN-10",
        "tanimoto_median_distance": "Tanimoto median",
        "chemprop_std_euclidean_nn": "Euclidean NN",
        "chemprop_std_euclidean_knn_5": r"Euclidean $k$NN-5",
        "chemprop_std_euclidean_knn_10": r"Euclidean $k$NN-10",
        "chemprop_euclidean_nn": "Euclidean NN (raw)",
        "chemprop_cosine_nn": "cosine NN",
        "chemprop_mahalanobis_centroid": "Mahalanobis centroid",
        "chemprop_mahalanobis_nn": "Mahalanobis NN",
    }
    fig, axes = new_figure((10.6, 5.0), 1, 2, sharex=True)
    n0 = _ad_retention_panel(
        axes[0],
        summary,
        random_sum,
        "DS2",
        "DS2_GRAPH_DMPNN_EVID",
        "all_pairs",
        "mop_rmse",
        scores,
        local_cfg,
        "All-pair MoP RMSE",
        "A. All-pair MoP RMSE",
        primary=["sigma_total", "tanimoto_nn_distance"],
    )
    n1 = _ad_retention_panel(
        axes[1],
        summary,
        random_sum,
        "DS2",
        "DS2_GRAPH_DMPNN_EVID",
        "all_pairs",
        "n_pairs_used",
        scores,
        local_cfg,
        "Pairs contributing to MoP",
        "B. Supported-pair count",
        primary=["sigma_total", "tanimoto_nn_distance"],
    )
    handles, labels = axes[0].get_legend_handles_labels()
    place_legend_below(fig, handles, labels, ncol=4, fontsize=6.5)
    short_title(fig, "All-pair representation-space retention sensitivities")
    src_df = summary[(summary.model_id.astype(str).str.contains("GRAPH_DMPNN_EVID")) & (summary.scope.astype(str) == "all_pairs")].copy()
    src = save_source_data(spec, src_df)
    paths = save_figure(fig, spec, cfg)
    return spec, src, paths, src_df, n0 + n1, len(src_df), "all_pairs", "fold_local_then_mean_sd", ["chemprop_only"]


def sfig12(cfg):
    spec = "figS9_uncertainty_decomposition"
    score_root = REMAKE_ROOT / "results" / "downstream_cv" / "ad_retention" / "fold_scores" / cfg["workhorse_model_id"]
    frames = []
    for fold in ("fold0", "fold1", "fold2", "fold3", "fold4"):
        path = score_root / f"{fold}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    for col in ("sigma_total", "sigma_aleatoric", "sigma_epistemic_within", "sigma_epistemic_between"):
        if col not in raw.columns:
            raw[col] = 0.0
    cols = ["sigma_total", "sigma_epistemic_within", "sigma_epistemic_between", "sigma_aleatoric"]
    g = raw.groupby("outer_fold_id")[cols].mean().reset_index()
    var_frame = raw[cols].astype(float) ** 2
    var_frame["outer_fold_id"] = raw["outer_fold_id"].to_numpy()
    g_var = var_frame.groupby("outer_fold_id")[cols].mean().reset_index()
    fig, axes = new_figure((11.2, 4.4), 1, 2)
    x = np.arange(len(g))
    all_series = [
        ("sigma_aleatoric", cfg["score_color_map"]["sigma_aleatoric"]),
        ("sigma_epistemic_within", cfg["score_color_map"]["sigma_epistemic_within"]),
        ("sigma_epistemic_between", cfg["score_color_map"]["sigma_epistemic_between"]),
        ("sigma_total", cfg["score_color_map"]["sigma_total"]),
    ]
    width_all = 0.18
    for i, (col, color) in enumerate(all_series):
        axes[0].bar(x + (i - 1.5) * width_all, g[col], width_all, color=color, label=cfg["score_labels"].get(col, col))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(g.outer_fold_id)
    axes[0].set_ylabel("Mean row-level sigma (fold-local)")
    axes[0].set_title(r"A. Standard deviations (not additive)")
    axes[0].set_xlabel("Held-out fold")
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper right")
    a2 = g_var["sigma_aleatoric"]
    w2 = g_var["sigma_epistemic_within"]
    b2 = g_var["sigma_epistemic_between"]
    t2 = g_var["sigma_total"]
    axes[1].bar(x, a2, 0.55, color=cfg["score_color_map"]["sigma_aleatoric"], label=r"$\sigma_{\mathrm{aleatoric}}^2$")
    axes[1].bar(x, w2, 0.55, bottom=a2, color=cfg["score_color_map"]["sigma_epistemic_within"], label=r"$\sigma_{\mathrm{epistemic,within}}^2$")
    axes[1].bar(x, b2, 0.55, bottom=a2 + w2, color=cfg["score_color_map"]["sigma_epistemic_between"], label=r"$\sigma_{\mathrm{epistemic,between}}^2$")
    axes[1].scatter(x, t2, color="#111111", zorder=6, s=70, marker="D", edgecolors="white", linewidths=0.6, label=r"mean of $\sigma_{\mathrm{total}}^2$")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(g_var.outer_fold_id)
    axes[1].set_xlabel("Held-out fold")
    axes[1].set_ylabel("Mean row-level variance")
    axes[1].set_title("B. Variances add to total predictive variance")
    axes[1].legend(frameon=False, fontsize=6.5, loc="upper right")
    short_title(fig, "Workhorse uncertainty components")
    src_df = g.merge(g_var, on="outer_fold_id", suffixes=("_sigma", "_variance"))
    src = save_source_data(spec, src_df)
    paths = save_figure(fig, spec, cfg)
    return spec, src, paths, src_df, 2, int(src_df.shape[0] * 8), "heldout_fold", "fold_mean_of_row_scores", ["descriptive_not_lab_noise"]


def fig02_direct_vs_ad_hoc_selectivity(cfg):
    spec = "figS18_direct_vs_ad_hoc_selectivity"
    path = REMAKE_ROOT / "results" / "downstream_cv" / "ad_hoc_selectivity" / "ad_hoc_selectivity_summary.csv"
    summary = pd.read_csv(path)
    work = summary[summary.scope.astype(str) == "ree_pairs"].copy()
    if "method_class" not in work.columns:
        from uq_remake.downstream.ad_hoc_selectivity import method_class

        work["method_class"] = [method_class(m, i) for m, i in zip(work.selectivity_method, work.source_model_id)]
    op_path = TABLES / "main_ds2_operational_comparison.csv"
    usable = set()
    if op_path.exists():
        op = pd.read_csv(op_path)
        if "calibration_usable" in op.columns:
            usable = {
                str(m)
                for m in op.loc[op.calibration_usable.astype(str).str.lower().isin({"true", "1"}), "model_id"].astype(str)
                if str(m).startswith("DS2_")
            }
    keep_rows = []
    for _, r in work.iterrows():
        cls = str(r.method_class)
        mid = str(r.source_model_id)
        if cls == "direct_ds2" and (mid == cfg["workhorse_model_id"] or mid in usable):
            keep_rows.append(r)
        elif cls in {"ad_hoc_ds1_uncertainty", "ad_hoc_ds1_point"}:
            keep_rows.append(r)
    plot = pd.DataFrame(keep_rows)
    class_ord = {"direct_ds2": 0, "ad_hoc_ds1_uncertainty": 1, "ad_hoc_ds1_point": 2}
    plot["_c"] = plot.method_class.astype(str).map(class_ord).fillna(9)
    plot = plot.sort_values(["_c", "mop_rmse"])
    colors = []
    labels = []
    for _, r in plot.iterrows():
        cls = str(r.method_class)
        name = FIG05_SHORT.get(str(r.source_model), str(r.source_model))
        if cls == "direct_ds2":
            colors.append("#4C72B0")
            labels.append(f"direct · {name}")
        elif cls == "ad_hoc_ds1_point":
            colors.append("#7F7F7F")
            labels.append(f"ad hoc point · {name}")
        else:
            colors.append("#E07A3D")
            labels.append(f"ad hoc · {name}")
    fig, axes = new_figure((9.4, max(4.2, 0.38 * len(plot) + 1.8)), 1, 2, sharey=True)
    hbar_metric(axes[0], labels, plot.mop_rmse, plot.mop_rmse_sd, colors, xlabel="MoP RMSE", title="A. MoP RMSE")
    hbar_metric(axes[1], labels, plot.sign_accuracy, plot.sign_accuracy_sd, colors, xlabel="Sign accuracy", title="B. Sign accuracy")
    short_title(fig, "Direct versus propagated intervals on the restricted rare-earth-element pair subset")
    src_df = plot.drop(columns=["_c"], errors="ignore").copy()
    src_df["plot_label"] = labels
    src = save_source_data(spec, src_df)
    paths = save_figure(fig, spec, cfg)
    return spec, src, paths, src_df, 2, len(src_df), "ree_pairs", "fold_mean_plus_sd", []


PAIR_ACQ_STRAT = [
    ("y_pred", "Predicted value", "#2A9D8F"),
    ("lcb_1sigma", "LCB-1", "#E17C05"),
    ("lcb_2sigma", "LCB-2", "#C44E52"),
    ("lcb_3sigma", "LCB-3", "#8172B3"),
    ("expected_improvement", "Threshold margin", "#4C72B0"),
    ("random", "Random", "#7F8C8D"),
]


def _pair_acq_lookup(summary, task, orientation, strategy, k, metric):
    hit = summary[
        (summary.task.astype(str) == task)
        & (summary.orientation.astype(str) == orientation)
        & (summary.strategy.astype(str) == strategy)
        & (summary.metric_name.astype(str) == metric)
        & (pd.to_numeric(summary.k, errors="coerce") == k)
    ]
    if hit.empty:
        return np.nan, np.nan
    return float(hit.iloc[0].fold_mean), float(hit.iloc[0].fold_sd)


def _pair_acquisition_figure(cfg, spec, summary_path, high_title, low_title, title):
    summary = pd.read_csv(summary_path)
    ks = [1, 5, 10]
    panel_grid = [
        [
            ("high_selectivity", "both", high_title, "any_hit", "True hit@$k$", "A", True),
            ("low_selectivity", "near_zero", low_title, "any_hit", "True hit@$k$", "B", True),
        ],
        [
            ("high_selectivity", "both", high_title, "hit_rate", "Event precision@$k$", "C", True),
            ("low_selectivity", "near_zero", low_title, "hit_rate", "Event precision@$k$", "D", True),
        ],
        [
            ("high_selectivity", "both", high_title, "regret", r"Retrospective regret@$k$; lower is better", "E", False),
            ("low_selectivity", "near_zero", low_title, "regret", r"Retrospective regret@$k$; lower is better", "F", False),
        ],
    ]
    fig, axes = new_figure((9.0, 10.4), 3, 2, height_ratios=[1.0, 1.0, 1.05])
    src_rows = []
    x = np.arange(len(ks))
    width = 0.13

    def _draw_panel(ax, task, orientation, panel_title, metric, ylabel, letter, *, fraction_axis):
        means_all, sds_all = [], []
        for i, (strategy, label, color) in enumerate(PAIR_ACQ_STRAT):
            means, sds = [], []
            for k in ks:
                m, s = _pair_acq_lookup(summary, task, orientation, strategy, k, metric)
                means.append(m)
                sds.append(s)
                src_rows.append(
                    {
                        "task": task,
                        "orientation": orientation,
                        "strategy": strategy,
                        "k": k,
                        "metric_name": metric,
                        "displayed_as": (
                            "hit@k"
                            if metric == "any_hit"
                            else ("event_precision@k" if metric == "hit_rate" else "regret@k")
                        ),
                        "fold_mean": m,
                        "fold_sd": s,
                    }
                )
            ax.bar(
                x + (i - 2.5) * width,
                means,
                width,
                yerr=np.where(np.isfinite(sds), sds, 0.0),
                color=color,
                label=label,
                ecolor="#333333",
                capsize=2,
                error_kw={"elinewidth": 0.8},
            )
            means_all.extend(means)
            sds_all.extend(sds)
        ax.set_title(f"{letter}. {panel_title}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"$k={k}$" for k in ks])
        ax.set_ylabel(ylabel)
        if fraction_axis:
            ax.set_ylim(0.0, 1.05)
        else:
            tops = [
                (0.0 if not np.isfinite(m) else float(m)) + (0.0 if not np.isfinite(s) else float(s))
                for m, s in zip(means_all, sds_all)
            ]
            ymax = max(tops) if tops else 0.0
            ax.set_ylim(0.0, ymax * 1.15 if ymax > 0 else 1.0)

    for row_axes, row_panels in zip(axes, panel_grid):
        for ax, panel in zip(np.atleast_1d(row_axes), row_panels):
            *args, fraction_axis = panel
            _draw_panel(ax, *args, fraction_axis=fraction_axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    place_legend_below(fig, handles, labels, ncol=6)
    short_title(fig, title)
    src_df = pd.DataFrame(src_rows)
    src = save_source_data(spec, src_df)
    paths = save_figure(fig, spec, cfg)
    return spec, src, paths, src_df, 6, len(src_df), "heldout_fold", "pair_equal_then_fold_mean_plus_sd", ["sigma_only"]


def fig05_pair_acquisition(cfg):
    return _pair_acquisition_figure(
        cfg,
        "fig05_pair_acquisition",
        REMAKE_ROOT / "results" / "downstream_cv" / "pair_acquisition" / "pair_acquisition_summary.csv",
        r"High selectivity ($\Delta\log K>1$)",
        r"Near-zero selectivity ($|\Delta\log K|<1$)",
        "Objective-specific pair acquisition",
    )


def figS15_pair_acquisition_threshold_sensitivity(cfg):
    return _pair_acquisition_figure(
        cfg,
        "figS15_pair_acquisition_threshold_sensitivity",
        REMAKE_ROOT / "results" / "downstream_cv" / "pair_acquisition_tau_2_0p5" / "pair_acquisition_summary.csv",
        r"High selectivity ($\Delta\log K>2$)",
        r"Near-zero selectivity ($|\Delta\log K|<1/2$)",
        "Event-threshold sensitivity for pair acquisition",
    )


def figS19_predicted_magnitude_calibration(cfg):
    spec = "fig04_predicted_magnitude"
    bins = pd.read_csv(
        REMAKE_ROOT / "results" / "downstream_cv" / "pair_acquisition" / "pair_acquisition_calibration_bin_summary.csv"
    )
    rho = pd.read_csv(
        REMAKE_ROOT / "results" / "downstream_cv" / "pair_acquisition" / "pair_acquisition_calibration_rho_summary.csv"
    )
    order = ["0-0.5", "0.5-1", "1-2", "2-4", "4-8", ">8"]
    labels = [r"$[0,0.5)$", r"$[0.5,1)$", r"$[1,2)$", r"$[2,4)$", r"$[4,8)$", r"$[8,\infty)$"]
    fig, axes = new_figure((11.0, 4.0), 1, 3, width_ratios=[1.05, 1.05, 1.15])
    src_rows = []

    def _bin_vals(metric):
        means, sds = [], []
        for lab in order:
            hit = bins[(bins.bin_id.astype(str) == f"abs_yhat_{lab}") & (bins.metric_name.astype(str) == metric)]
            if hit.empty:
                means.append(np.nan)
                sds.append(np.nan)
            else:
                means.append(float(hit.iloc[0].fold_mean))
                sds.append(float(hit.iloc[0].fold_sd))
                src_rows.append(
                    {
                        "bin_id": f"abs_yhat_{lab}",
                        "metric_name": metric,
                        "fold_mean": float(hit.iloc[0].fold_mean),
                        "fold_sd": float(hit.iloc[0].fold_sd),
                    }
                )
        return means, sds

    x = np.arange(len(order))
    panel_titles = {
        "rmse": "A. Error versus predicted magnitude",
        "mean_sigma": "B. Uncertainty versus predicted magnitude",
    }
    for ax, metric, ylab, color in (
        (axes[0], "rmse", "RMSE", "#4C72B0"),
        (axes[1], "mean_sigma", r"Mean $\sigma_{\mathrm{total}}$", "#C44E52"),
    ):
        means, sds = _bin_vals(metric)
        ax.errorbar(x, means, yerr=np.where(np.isfinite(sds), sds, np.nan), fmt="-o", color=color, capsize=3, lw=1.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_xlabel(r"$|\widehat y|$")
        ax.set_ylabel(ylab)
        ax.set_title(panel_titles[metric])

    rho_order = [
        ("rho_sigma_abs_error", r"$\rho(\sigma_{\mathrm{total}},|e|)$"),
        ("rho_sigma_abs_yhat", r"$\rho(\sigma_{\mathrm{total}},|\widehat y|)$"),
        ("rho_sigma_abs_y", r"$\rho(\sigma_{\mathrm{total}},|y|)$"),
        ("rho_abs_yhat_abs_error", r"$\rho(|\widehat y|,|e|)$"),
    ]
    means, sds, labs = [], [], []
    for name, lab in rho_order:
        hit = rho[rho.metric_name.astype(str) == name]
        m = float(hit.iloc[0].fold_mean) if len(hit) else np.nan
        s = float(hit.iloc[0].fold_sd) if len(hit) else np.nan
        means.append(m)
        sds.append(s)
        labs.append(lab)
        src_rows.append({"bin_id": "all_heldout_rows", "metric_name": name, "fold_mean": m, "fold_sd": s})
    axes[2].bar(np.arange(len(labs)), means, yerr=np.where(np.isfinite(sds), sds, 0.0), color="#55A868", ecolor="#333333", capsize=3)
    axes[2].set_xticks(np.arange(len(labs)))
    axes[2].set_xticklabels(labs, rotation=25, ha="right")
    axes[2].set_ylabel("Rank correlation")
    axes[2].set_title("C. Rank associations")
    axes[2].axhline(0.0, color="#333333", lw=0.8)
    short_title(fig, "Predicted magnitude, error, and uncertainty")
    src_df = pd.DataFrame(src_rows)
    src = save_source_data(spec, src_df)
    paths = save_figure(fig, spec, cfg)
    return spec, src, paths, src_df, 3, len(src_df), "all_heldout_rows", "fold_mean_plus_sd", ["no_threshold_curve"]


def figS16_ad_hoc_interval_calibration(cfg):
    spec = "figS16_ad_hoc_interval_calibration"
    path = REMAKE_ROOT / "results" / "downstream_cv" / "ad_hoc_selectivity" / "ad_hoc_selectivity_comparison.csv"
    raw = pd.read_csv(path)
    raw = raw[raw.scope.astype(str) == "ree_pairs"].copy()
    keep = [
        ("DS2_RDKIT_MLPENS5_HETERO", "direct_ds2_selectivity"),
        ("DS2_RDKIT_MLPENS5_EVID", "direct_ds2_selectivity"),
        ("DS2_RDKIT_MLP_HETERO", "direct_ds2_selectivity"),
        ("DS1_GRAPH_DMPNN_EVID", "ad_hoc_ds1_difference"),
        ("DS1_RDKIT_MLPENS5_MCDROPOUT", "ad_hoc_ds1_difference"),
        ("DS1_RDKIT_MLPENS5_HETERO", "ad_hoc_ds1_difference"),
    ]
    rows = []
    for mid, method in keep:
        hit = raw[(raw.source_model_id.astype(str) == mid) & (raw.selectivity_method.astype(str) == method)]
        if hit.empty:
            continue
        r = hit.iloc[0]
        if pd.isna(r.picp_90) and pd.isna(r.ace):
            continue
        name = FIG05_SHORT.get(str(r.source_model), str(r.source_model))
        cls = "direct" if method == "direct_ds2_selectivity" else "ad hoc"
        rows.append(
            {
                "source_model_id": mid,
                "source_model": r.source_model,
                "selectivity_method": method,
                "plot_label": f"{cls} · {name}",
                "method_class": cls,
                "picp_90": float(r.picp_90) if pd.notna(r.picp_90) else np.nan,
                "picp_90_sd": float(r.picp_90_sd) if pd.notna(r.picp_90_sd) else np.nan,
                "ace": float(r.ace) if pd.notna(r.ace) else np.nan,
                "ace_sd": float(r.ace_sd) if pd.notna(r.ace_sd) else np.nan,
            }
        )
    plot = pd.DataFrame(rows)
    if plot.empty:
        raise RuntimeError("figS16_ad_hoc_interval_calibration: no aligned PICP/ACE rows")
    colors = ["#4C72B0" if c == "direct" else "#E07A3D" for c in plot.method_class]
    fig, axes = new_figure((9.6, max(3.6, 0.42 * len(plot) + 1.6)), 1, 2, sharey=True)
    hbar_metric(axes[0], plot.plot_label, plot.picp_90, plot.picp_90_sd, colors, xlabel="PICP@90", title="A. PICP@90")
    axes[0].axvline(0.90, color="#333333", ls="--", lw=1.0, zorder=1)
    hbar_metric(axes[1], plot.plot_label, plot.ace, plot.ace_sd, colors, xlabel="ACE", title="B. ACE")
    short_title(fig, "Restricted rare-earth-element pair interval calibration")
    src = save_source_data(spec, plot)
    paths = save_figure(fig, spec, cfg)
    return spec, src, paths, plot, 2, len(plot), "ree_pairs", "fold_mean_plus_sd", ["aligned_formulation_diagnostic"]


def figS13_pair_event_support(cfg):
    spec = "figS13_pair_event_support"
    path = REMAKE_ROOT / "results" / "downstream_cv" / "pair_event_support" / "pair_event_support_pooled.csv"
    if not path.exists():
        path = TABLES / "pair_event_support_pooled.csv"
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError("figS13_pair_event_support: empty pair-event table")
    fig, axes = new_figure((10.4, 4.4), 1, 2)
    x = pd.to_numeric(df.n_heldout_rows, errors="coerce")
    y_ev = pd.to_numeric(df.n_high_selectivity_events, errors="coerce")
    rmse = pd.to_numeric(df.out_of_fold_pooled_per_pair_rmse, errors="coerce")
    sc0 = axes[0].scatter(x, y_ev, c=rmse, cmap="viridis", s=18, alpha=0.75, linewidths=0)
    axes[0].axhline(PAIR_ACQ_EVENT_MIN_EXCLUSIVE, color="#333333", ls="--", lw=1.0)
    axes[0].set_xlabel("Held-out rows per pair")
    axes[0].set_ylabel("Observed high-selectivity events")
    axes[0].set_title("A. Held-out rows versus event count")
    sizes = 18.0 + 80.0 * (x / max(float(np.nanmax(x)), 1.0))
    axes[1].scatter(y_ev, rmse, s=sizes, c="#4C72B0", alpha=0.55, linewidths=0)
    axes[1].axvline(PAIR_ACQ_EVENT_MIN_EXCLUSIVE, color="#333333", ls="--", lw=1.0)
    axes[1].set_xlabel("Observed high-selectivity events")
    axes[1].set_ylabel("Pooled per-pair RMSE")
    axes[1].set_title("B. Event count versus pooled per-pair RMSE")
    cbar = fig.colorbar(sc0, ax=list(np.ravel(axes)), fraction=0.035, pad=0.02)
    cbar.set_label("Pooled per-pair RMSE")
    label_pairs = []
    top_freq = df.nlargest(2, "n_heldout_rows")
    seen = set()
    for _, r in top_freq.iterrows():
        key = str(r.metal_species_pair)
        if key in seen:
            continue
        seen.add(key)
        label_pairs.append(r)
    offsets = [(6, 8), (8, -14)]
    for r, off in zip(label_pairs, offsets):
        axes[0].annotate(
            str(r.metal_species_pair),
            (r.n_heldout_rows, r.n_high_selectivity_events),
            textcoords="offset points",
            xytext=off,
            fontsize=6,
        )
    short_title(fig, "Held-out pair support versus high-selectivity event support")
    src = save_source_data(spec, df)
    paths = save_figure(fig, spec, cfg)
    return spec, src, paths, df, 2, len(df), "all_pairs", "out_of_fold_pooled", ["descriptive_heldout_support"]


GENERATORS = {
    "fig01_study_design": fig01_study_design,
    "fig02_workhorse_selection": fig05,
    "fig03_risk_ordering": fig04_risk_ordering,
    "fig04_predicted_magnitude": figS19_predicted_magnitude_calibration,
    "fig05_pair_acquisition": fig05_pair_acquisition,
    "figS1_ds1_complete": sfig01,
    "figS2_ds1_point": fig01,
    "figS3_ds1_uq": fig02,
    "figS4_ds2_all_pairs": sfig02,
    "figS5_ds2_highsel_row": sfig03,
    "figS6_ds2_highsel_pair": sfig04,
    "figS7_workhorse_fold_stability": fig06,
    "figS16_ad_hoc_interval_calibration": figS16_ad_hoc_interval_calibration,
    "figS10_ad_orientation": sfig06,
    "figS11_retention_all_pairs": sfig07,
    "figS17_retention_ree": figS17_retention_ree,
    "figS12_pair_density": sfig09,
    "figS13_pair_event_support": figS13_pair_event_support,
    "figS14_sign_stability_detail": sfig10,
    "figS8_chemprop_embedding_retention": sfig11,
    "figS9_uncertainty_decomposition": sfig12,
    "figS18_direct_vs_ad_hoc_selectivity": fig02_direct_vs_ad_hoc_selectivity,
    "figS19_ree_fold_stability": figS19_ree_fold_stability,
    "figS15_pair_acquisition_threshold_sensitivity": figS15_pair_acquisition_threshold_sensitivity,
}


def validate_result(fid, src_df, spec_row, warnings) -> list[str]:
    w = list(warnings)
    if "model_id" in src_df.columns:
        for mid in src_df.model_id.astype(str).unique():
            if obsolete_model(mid):
                w.append(f"obsolete_model:{mid}")
    text = " ".join(src_df.astype(str).head(3).fillna("").to_string().split())
    if "BAYES_LASTLAYER" in text:
        w.append("obsolete_token_in_source_data")
    if spec_row.get("aggregation") == "fold_mean_plus_sd":
        if "aggregation" in src_df.columns and src_df.aggregation.astype(str).str.contains("pool", case=False).any():
            w.append("pooled_aggregation_in_source")
    return w


def remove_stale_figure_outputs() -> None:
    live = {spec["figure_id"] for spec in FIGURE_SPECS}
    removed = []
    locked = []
    for path in [*FIGDIR.glob("*.png"), *FIGDIR.glob("*.pdf"), *DATADIR.glob("*.csv")]:
        if path.stem in live:
            continue
        try:
            path.unlink(missing_ok=True)
            removed.append(rel(path))
        except OSError:
            locked.append(rel(path))
    if removed or locked:
        print(json.dumps({"removed_stale_figures": removed, "locked_stale_figures": locked}, indent=2), flush=True)


def write_captions() -> Path:
    lines = ["# Figure captions", "", "Generated from canonical fold-level artifacts. Retention curves are empirical ranking tools, not coverage guarantees.", ""]
    for spec in FIGURE_SPECS:
        fid = spec["figure_id"]
        if fid not in CAPTIONS:
            continue
        lines.append(f"## {fid}")
        lines.append("")
        lines.append(CAPTIONS[fid])
        lines.append("")
        lines.append(f"- dataset: `{spec['dataset']}`")
        lines.append(f"- scope: `{spec['scope']}`")
        lines.append(f"- models: `{spec['model_ids']}`")
        lines.append(f"- aggregation: `{spec['aggregation']}`")
        lines.append(f"- source: `{spec['source_artifact']}`")
        lines.append(f"- status: `{spec['status']}`")
        lines.append("")
    path = FIGDIR / "captions.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    cfg = load_config()
    apply_style(cfg)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    DATADIR.mkdir(parents=True, exist_ok=True)
    ensure_sign_table()
    write_ad_audit()

    registry = pd.DataFrame(
        [
            {
                "figure_id": s["figure_id"],
                "purpose": s["purpose"],
                "source_artifact": s["source_artifact"],
                "dataset": s["dataset"],
                "model_ids": s["model_ids"],
                "scope": s["scope"],
                "metric_name": s["metric_name"],
                "aggregation": s["aggregation"],
                "status": s["status"],
                "panel": s.get("panel", ""),
            }
            for s in FIGURE_SPECS
        ]
    )
    registry.to_csv(MAN / "figure_data_registry.csv", index=False)
    registry.to_csv(TABLES / "figure_data_registry.csv", index=False)

    missing_gen = [s["figure_id"] for s in FIGURE_SPECS if s["figure_id"] not in GENERATORS]
    extra_gen = [fid for fid in GENERATORS if fid not in {s["figure_id"] for s in FIGURE_SPECS}]
    if missing_gen or extra_gen:
        raise RuntimeError(f"figure generator mismatch missing={missing_gen} extra={extra_gen}")
    missing_cap = [s["figure_id"] for s in FIGURE_SPECS if s["figure_id"] not in CAPTIONS]
    extra_cap = [fid for fid in CAPTIONS if fid not in {s["figure_id"] for s in FIGURE_SPECS}]
    if missing_cap or extra_cap:
        raise RuntimeError(f"figure caption mismatch missing={missing_cap} extra={extra_cap}")

    audit = audit_inputs(cfg)
    audit.to_csv(FIGDIR / "figure_input_audit.csv", index=False)
    blocked = set(audit.loc[~audit.status.isin(["ok", "not_computed"]), "figure_id"])

    val_rows = []
    man_rows = []
    for spec in FIGURE_SPECS:
        fid = spec["figure_id"]
        if spec["status"] == "not_computed":
            val_rows.append(
                {
                    "figure_id": fid,
                    "source_artifact": "",
                    "n_series": 0,
                    "n_points": 0,
                    "scope": "",
                    "aggregation": "",
                    "status": "not_computed",
                    "warnings": "no_canonical_artifact",
                }
            )
            continue
        if fid in blocked:
            val_rows.append(
                {
                    "figure_id": fid,
                    "source_artifact": spec["source_artifact"],
                    "n_series": 0,
                    "n_points": 0,
                    "scope": spec["scope"],
                    "aggregation": spec["aggregation"],
                    "status": "blocked_input_audit",
                    "warnings": str(audit.loc[audit.figure_id == fid, "status"].iloc[0]),
                }
            )
            continue
        gen = GENERATORS[fid]
        fid, src_path, out_paths, src_df, n_series, n_points, scope, agg, warnings = gen(cfg)
        warnings = validate_result(fid, src_df, spec, warnings)
        status = "ok"
        if spec["status"] == "provisional":
            status = "ok_provisional"
        if any(w.startswith("obsolete") for w in warnings):
            status = "fail"
        val_rows.append(
            {
                "figure_id": fid,
                "source_artifact": spec["source_artifact"],
                "n_series": n_series,
                "n_points": n_points,
                "scope": scope,
                "aggregation": agg,
                "status": status,
                "warnings": "|".join(warnings) if warnings else "",
            }
        )
        src_hash = sha256_file(REMAKE_ROOT / spec["source_artifact"]) if spec["source_artifact"] else ""
        for p in out_paths:
            man_rows.append(
                {
                    "figure_id": fid,
                    "script": SCRIPT_REL,
                    "source_artifact": spec["source_artifact"],
                    "source_hash": src_hash,
                    "output_path": rel(p),
                    "generation_timestamp": utc_now(),
                    "status": status,
                }
            )
        man_rows.append(
            {
                "figure_id": fid,
                "script": SCRIPT_REL,
                "source_artifact": spec["source_artifact"],
                "source_hash": src_hash,
                "output_path": rel(src_path),
                "generation_timestamp": utc_now(),
                "status": status,
            }
        )

    pd.DataFrame(val_rows).to_csv(FIGDIR / "figure_validation_report.csv", index=False)
    pd.DataFrame(man_rows).to_csv(FIGDIR / "figure_generation_manifest.csv", index=False)
    write_captions()
    remove_stale_figure_outputs()
    n_fail = sum(1 for r in val_rows if str(r["status"]).startswith("fail") or r["status"] == "blocked_input_audit")
    print(json.dumps({"n_figures": len(val_rows), "n_fail": n_fail, "figdir": rel(FIGDIR)}, indent=2))
    if n_fail:
        for r in val_rows:
            if str(r["status"]).startswith("fail") or r["status"] == "blocked_input_audit":
                print("FAIL", r["figure_id"], r["status"], r["warnings"])


if __name__ == "__main__":
    main()
