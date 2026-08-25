"""Write frozen model/training audit artifacts (no workhorse winner)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from uq_remake import REMAKE_ROOT
from uq_remake.training.protocol import canonical_model_id, load_active_grid_cfg, utc_now

UQ_METHOD = {
    "POINT_MLP": "none",
    "POINT_XGB": "none",
    "CHEMPROP_POINT_NATIVE": "none",
    "EVID_MLP": "nig_evidential",
    "EVID_MLPENS5": "nig_evidential_ensemble",
    "HETERO_MLP": "heteroscedastic_log_var_mc",
    "HETERO_MLPENS5": "ensemble_law_of_total_variance",
    "MCDROPOUT_MLP": "mcdropout_global_observation_noise",
    "MCDROPOUT_MLPENS5": "ensemble_law_of_total_variance",
    "CHEMPROP_EVIDENTIAL_NATIVE": "chemprop_native_nig_evidential",
    "BAYES_FULL": "variational_weight_posterior",
    "BAYES_FULLENS5": "variational_weight_posterior_plus_ensemble",
    "SNGP": "sngp_gp_posterior",
    "SNGPENS5": "sngp_gp_posterior_plus_ensemble",
    "GPR": "exact_gp_posterior",
    "GPR_SPARSE": "sparse_variational_gp",
}


def _uq_head(fam: str, is_point: bool) -> str:
    if is_point:
        return "POINT"
    if "BAYES" in fam:
        return "BAYES_FULL"
    if fam.startswith("SNGP"):
        return "SNGP"
    if fam.startswith("GPR"):
        return "GPR"
    if "EVID" in fam or "EVIDENTIAL" in fam:
        return "EVID"
    if "HETERO" in fam:
        return "HETERO"
    if "MCDROPOUT" in fam:
        return "MCDROPOUT"
    return fam


def _arch(fam: str, is_graph: bool, is_ens: bool) -> str:
    if is_graph:
        return "DMPNN"
    if "XGB" in fam:
        return "XGB"
    if fam.startswith("SNGP"):
        return "SNGPENS5" if is_ens else "SNGP"
    if fam == "GPR_SPARSE":
        return "GPR_SPARSE"
    if fam == "GPR":
        return "GPR"
    if is_ens:
        return "MLPENS5"
    return "MLP"

AUDIT = REMAKE_ROOT / "results" / "model_audit"


def _write_csv(name: str, rows: list[dict]) -> Path:
    AUDIT.mkdir(parents=True, exist_ok=True)
    path = AUDIT / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_workhorse_rule() -> Path:
    from uq_remake.metrics.workhorse import write_rule

    return write_rule()


def write_static_audit() -> dict[str, str]:
    cfg = load_active_grid_cfg()
    write_workhorse_rule()
    families = list(cfg["active_families"])
    inventory = []
    active = []
    eligibility = []
    for ds in ("DS1", "DS2"):
        for fam in families:
            mid = canonical_model_id(ds, fam)
            is_point = fam in {"POINT_MLP", "POINT_XGB", "CHEMPROP_POINT_NATIVE"}
            is_ens = "ENS5" in fam
            is_graph = fam.startswith("CHEMPROP")
            is_bnn = fam.startswith("BAYES_FULL")
            is_gpr_exact = fam == "GPR"
            row = {
                "dataset": ds,
                "family": fam,
                "model_id": mid,
                "active": not is_gpr_exact,
                "n_folds": 5,
                "ensemble_size": 5 if is_ens else 1,
                "input": "GRAPH" if is_graph else "RDKIT",
                "arch": _arch(fam, is_graph, is_ens),
                "uq_head": _uq_head(fam, is_point),
                "uncertainty_status": "unavailable" if is_point else ("computationally_infeasible_exact" if is_gpr_exact else "valid_when_trained"),
                "uncertainty_method": UQ_METHOD[fam],
                "chemprop_mc_dropout": False,
                "called_bnn": is_bnn,
                "called_weight_posterior": is_bnn,
                "pca": False,
                "workhorse_eligible": (not is_point) and (not is_gpr_exact),
                "uq_tiebreakers_eligible": (not is_point) and (not is_gpr_exact),
            }
            inventory.append(row)
            active.append(row)
            eligibility.append(
                {
                    "model_id": mid,
                    "dataset": ds,
                    "family": fam,
                    "eligible": (not is_point) and (not is_gpr_exact),
                    "chemprop_native_evidential_eligible": fam == "CHEMPROP_EVIDENTIAL_NATIVE",
                    "variational_bnn_eligible": is_bnn,
                    "sngp_eligible": fam.startswith("SNGP"),
                    "gpr_sparse_eligible": fam == "GPR_SPARSE",
                    "exact_gpr_eligible": False,
                    "reason": (
                        "exact GPR infeasible at DS1/DS2 n_train; use GPR_SPARSE"
                        if is_gpr_exact
                        else "active shared grid; winner not computed until five folds are complete"
                    ),
                }
            )

    inventory.extend(
        [
            {
                "dataset": "DS1+DS2",
                "family": "CHEMPROP_MC_DROPOUT",
                "model_id": "DS*_GRAPH_DMPNN_MCDROPOUT",
                "active": False,
                "n_folds": 0,
                "ensemble_size": 0,
                "input": "GRAPH",
                "arch": "DMPNN",
                "uq_head": "MCDROPOUT",
                "uncertainty_status": "excluded",
                "uncertainty_method": "not_used",
                "chemprop_mc_dropout": True,
                "called_bnn": False,
                "called_weight_posterior": False,
                "pca": False,
                "workhorse_eligible": False,
                "uq_tiebreakers_eligible": False,
            }
        ]
    )

    removed = [
        {
            "model": "BAYES_LASTLAYER / BayesFullLinear",
            "src": "removed last-layer Bayesian linear (not the variational BNN)",
            "archive": "permanently_deleted",
            "reason": "last-layer Bayesian linear; not the full variational BNN",
            "label": "archived_not_part_of_current_grid",
        },
        {
            "model": "old capped GPR (epistemic forced to 0)",
            "src": "previous train_gpr",
            "archive": "permanently_deleted",
            "reason": "replaced by exact GPR (tests) and GPR_SPARSE (trained)",
            "label": "replaced_by_GPR_SPARSE",
        },
        {
            "model": "old DS2-only SNGP artifacts",
            "src": "previous DS2-only run",
            "archive": "permanently_deleted",
            "reason": "replaced by shared DS1/DS2 SNGP in src/uq_remake/models/sngp.py",
            "label": "replaced_by_shared_SNGP",
        },
        {
            "model": "ranking-aux (*_RANK)",
            "src": "src/uq_remake/models/ranking_aux.py",
            "archive": "permanently_deleted",
            "reason": "DS2-only exploratory auxiliary loss; not in shared grid",
            "label": "archived_not_part_of_current_grid",
        },
        {
            "model": "Chemprop MC-dropout / GRAPH_DMPNN_BAYES",
            "src": "src/uq_remake/models/chemprop_dmpnn.py:predict_dmpnn_dropout_uq (raises)",
            "archive": "permanently_deleted",
            "reason": "excluded entirely; only native Chemprop evidential is in the grid",
            "label": "archived_not_part_of_current_grid",
        },
        {
            "model": "custom/residual Chemprop UQ",
            "src": "removed residual Chemprop UQ runner",
            "archive": "results/_publish/ (old numeric; invalid)",
            "reason": "residual-scale ale/epi split is not native Chemprop UQ",
            "label": "invalid_do_not_mix_with_corrected",
        },
        {
            "model": "CPEMB UQ families",
            "src": "removed CPEMB UQ grid",
            "archive": "n/a",
            "reason": "DS2 has no CPEMB equivalent; shared grid is RDKIT+GRAPH only",
            "label": "archived_not_part_of_current_grid",
        },
        {
            "model": "old BayesMLP (MSE + dropout, sigma_ale=1e-3)",
            "src": "src/uq_remake/models/uq_tabular.py (replaced by MCDropoutMLP)",
            "archive": "n/a",
            "reason": "was not a BNN; observation noise was a dummy floor",
            "label": "replaced_by_MCDROPOUT_MLP",
        },
    ]

    schema = [
        {
            "family": "POINT_*",
            "uncertainty_status": "unavailable",
            "fill_policy": "NaN; never 0 / residual SD / constant floor",
            "aleatoric": "unavailable",
            "epistemic": "unavailable",
            "interval_method": "",
        },
        {
            "family": "MCDROPOUT_MLP",
            "uncertainty_status": "valid",
            "fill_policy": "learned global log observation variance; MC var of mu (ddof=0)",
            "aleatoric": "exp(s), s=log sigma_a^2 learned globally",
            "epistemic": "Var_t[mu_t] over MC dropout",
            "interval_method": "normal_moment_approximation",
        },
        {
            "family": "HETERO_MLP",
            "uncertainty_status": "valid",
            "fill_policy": "average variances, never average standard deviations",
            "aleatoric": "E_t[exp(s_t(x))]",
            "epistemic": "Var_t[mu_t]",
            "interval_method": "normal_moment_approximation",
        },
        {
            "family": "EVID_MLP",
            "uncertainty_status": "valid",
            "fill_policy": "NIG moments; not a weight posterior",
            "aleatoric": "beta/(alpha-1)",
            "epistemic": "beta/(nu (alpha-1))",
            "interval_method": "normal_moment_approximation",
        },
        {
            "family": "*_MLPENS5",
            "uncertainty_status": "valid",
            "fill_policy": "law of total variance; within/between saved separately",
            "aleatoric": "mean of member aleatoric variances",
            "epistemic": "within-member + between-member disagreement (ddof=0)",
            "interval_method": "normal_moment_approximation",
        },
        {
            "family": "CHEMPROP_EVIDENTIAL_NATIVE",
            "uncertainty_status": "valid",
            "fill_policy": "native Chemprop return_unc_parameters; no residual proxy",
            "aleatoric": "beta/(alpha-1)",
            "epistemic": "beta/(lambda (alpha-1)); lambda=nu",
            "interval_method": "normal_moment_approximation",
        },
        {
            "family": "BAYES_FULL",
            "uncertainty_status": "valid",
            "fill_policy": "variational posterior samples; aleatoric = exp(log_sigma2_obs)",
            "aleatoric": "learned homoscedastic exp(s_obs)",
            "epistemic": "Var_s[f_theta_s(x)] over weight samples; single-model within=between=0",
            "interval_method": "empirical_posterior_predictive",
        },
        {
            "family": "SNGP",
            "uncertainty_status": "valid",
            "fill_policy": "GP posterior over RFF readout; spectral-normalized backbone is point-estimated",
            "aleatoric": "learned homoscedastic exp(s_obs)",
            "epistemic": "phi^T Lambda^{-1} phi from training-only precision",
            "interval_method": "gaussian_predictive_moment",
        },
        {
            "family": "GPR / GPR_SPARSE",
            "uncertainty_status": "valid_when_trained",
            "fill_policy": "exact GP if n_train<=2048 else SVGP; variances inverse-scaled by target scale^2",
            "aleatoric": "GP observation noise in original units",
            "epistemic": "latent GP variance in original units",
            "interval_method": "gaussian_predictive_moment",
        },
        {
            "family": "CHEMPROP_MC_DROPOUT",
            "uncertainty_status": "excluded",
            "fill_policy": "not trained",
            "aleatoric": "n/a",
            "epistemic": "n/a",
            "interval_method": "n/a",
        },
    ]

    invalid = [
        {
            "artifact": "old numeric CSVs from superseded OOF grids",
            "status": "invalid",
            "reason": "old features + residual Chemprop UQ + BayesMLP dummy aleatoric",
        },
        {
            "artifact": "Chemprop MC-dropout predictions",
            "status": "excluded",
            "reason": "not in active grid",
        },
        {
            "artifact": "POINT model sigma fields",
            "status": "unavailable",
            "reason": "no UQ head; fields are NaN",
        },
        {
            "artifact": "fast-mode checkpoints copied into full",
            "status": "do_not_reuse",
            "reason": "active grid trains fresh under checkpoints/active/",
        },
    ]

    paths = {
        "model_inventory.csv": str(_write_csv("model_inventory.csv", inventory)),
        "active_grid.csv": str(_write_csv("active_grid.csv", active)),
        "removed_models.csv": str(_write_csv("removed_models.csv", removed)),
        "uncertainty_schema.csv": str(_write_csv("uncertainty_schema.csv", schema)),
        "workhorse_eligibility.csv": str(_write_csv("workhorse_eligibility.csv", eligibility)),
        "invalid_or_unavailable_uq.csv": str(_write_csv("invalid_or_unavailable_uq.csv", invalid)),
        "workhorse_selection_rule.json": str(AUDIT / "workhorse_selection_rule.json"),
    }
    report = AUDIT / "model_training_audit_report.md"
    report.write_text(_report_text(cfg), encoding="utf-8")
    paths["model_training_audit_report.md"] = str(report)
    paths["written_utc"] = utc_now()
    return paths


def _report_text(cfg: dict) -> str:
    families = ", ".join(cfg["active_families"])
    return f"""# Model / training audit (active shared grid)

Written: {utc_now()}

## Scope

This task rebuilds the **shared DS1/DS2** model, training, uncertainty, ensemble, seed, checkpoint, and prediction protocol. Stage 4/5 metrics, bootstrap CIs, calibration tables, and manuscript figures are **out of scope**. The workhorse **rule** is frozen; the winner is **not** computed here.

## Active families (identical on DS1 and DS2)

{families}

Canonical IDs: `DS{{1,2}}_RDKIT_MLP_{{POINT,EVID,HETERO,MCDROPOUT,BAYES_FULL}}`, `DS{{1,2}}_RDKIT_MLPENS5_{{EVID,HETERO,MCDROPOUT,BAYES_FULL}}`, `DS{{1,2}}_RDKIT_{{SNGP,SNGPENS5,GPR,GPR_SPARSE}}`, `DS{{1,2}}_GRAPH_DMPNN_{{POINT,EVID}}`.

Chemprop MC-dropout is **not included**. Restored active UQ families: FixedPriorVariationalBNN (`BAYES_FULL`), SNGP, and GPR_SPARSE (exact GPR is implemented but infeasible at DS1/DS2 n_train).

## Protocol (shared)

- 5 ligand-disjoint outer folds; ligand-disjoint validation
- Ensemble size 5; MC-dropout passes 30; variance ddof=0
- Seed: `42 + fold_id*1000 + member_id*10 + model_offset`
- No PCA; no imputation; no silent zero-fill of missing UQ
- Hyperparameters from train/val only (never outer-test PICP/ACE/Stage4/5)
- Evidential λ for tabular MLP: val NLL over {{0.01, 0.2}} on the single model; reused for that fold's ensemble
- Chemprop evidential regularization: 0.2 (declared a priori; not tuned on test)
- H+ policy: `train_with_H`. Predict **all** fold test rows; tag `family` so later metrics can restrict to metal_complex
- Intervals: `normal_moment_approximation`. NIG parameters are saved for later Student-t
- New artifacts only: `checkpoints/active/` and `results/cv_metrics/`. Do not mix with retired prediction dumps.

## Uncertainty terminology

- **MC-dropout MLP** is not a BNN. Epistemic = MC variance of the mean. Aleatoric = learned global observation variance.
- **BAYES_FULL** is a fixed-prior full variational BNN. Epistemic = posterior weight-sample variance of the mean. Aleatoric = learned `exp(log_sigma2_obs)`.
- **SNGP** is not a full BNN. Epistemic = GP posterior variance over random Fourier features. Backbone is spectrally normalized and point-estimated.
- **GPR / GPR_SPARSE** is a kernel GP, not a neural model.
- **Evidential** (tabular and Chemprop) is not a weight posterior. Epistemic = NIG `β/(ν(α-1))`.
- **Heteroscedastic** averages **variances**, never standard deviations.
- **Ensembles** use the law of total variance. Between-member disagreement is saved separately and is not automatically "total epistemic" by itself; total epistemic = within + between.

## Removed / archived (not used for workhorse selection)

BAYES_LASTLAYER, ranking-aux, CPEMB UQ, custom/residual Chemprop UQ, Chemprop MC-dropout, old BayesMLP dummy `σ_a=1e-3`, old capped GPR with epistemic forced to 0.

Native Chemprop evidential, BAYES_FULL, SNGP, and GPR_SPARSE **are eligible** when all five folds are valid. Exact GPR is documented infeasible at this n_train. Chemprop MC-dropout is **not**.
"""


def append_csv(path: Path, row: dict) -> None:
    """Append one JSON-like row to a CSV without pandas concat of mismatched NA frames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if path.exists() and path.stat().st_size > 1:
        try:
            prev = pd.read_csv(path)
            if len(prev):
                rows.extend(prev.to_dict(orient="records"))
        except Exception:
            rows = []
    rows.append({k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in row.items()})
    pd.DataFrame.from_records(rows).to_csv(path, index=False)


if __name__ == "__main__":
    print(json.dumps(write_static_audit(), indent=2))
