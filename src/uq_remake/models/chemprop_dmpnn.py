"""Train / predict / fingerprint Chemprop D-MPNN POINT models (shared config)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

from uq_remake import REMAKE_ROOT, REPO_ROOT


def _ensure_rdbase() -> None:
    if "RDBASE" not in os.environ or not Path(os.environ["RDBASE"]).exists():
        cand = REPO_ROOT / ".venv311" / "Lib" / "site-packages" / "rdkit"
        if cand.exists():
            os.environ["RDBASE"] = str(cand)


def _stub_descriptastorus() -> None:
    """Avoid chemprop import crash when descriptastorus RDKit2DNormalized fails to init.

    We always supply external `--features_path` / features matrices, so the built-in
    rdkit_2d(_normalized) generators are unused. Newer RDKit dropped Descriptors.fr_*
    names that descriptastorus still instantiates at import time.
    """

    class _RDKit2D:
        NAME = "RDKit2DStub"

        def __init__(self, *args, **kwargs):
            pass

        def process(self, smiles):
            raise RuntimeError("descriptastorus stub: extra features come from --features_path")

    stub_pkg = types.ModuleType("descriptastorus")
    stub_desc = types.ModuleType("descriptastorus.descriptors")
    stub_rd = types.ModuleType("descriptastorus.descriptors.rdDescriptors")
    stub_norm = types.ModuleType("descriptastorus.descriptors.rdNormalizedDescriptors")
    stub_rd.RDKit2D = _RDKit2D
    stub_norm.RDKit2DNormalized = _RDKit2D
    stub_desc.rdDescriptors = stub_rd
    stub_desc.rdNormalizedDescriptors = stub_norm
    stub_desc.MakeGenerator = None
    stub_pkg.descriptors = stub_desc
    sys.modules["descriptastorus"] = stub_pkg
    sys.modules["descriptastorus.descriptors"] = stub_desc
    sys.modules["descriptastorus.descriptors.rdDescriptors"] = stub_rd
    sys.modules["descriptastorus.descriptors.rdNormalizedDescriptors"] = stub_norm


def _chemprop_env() -> dict:
    env = os.environ.copy()
    src = str(REPO_ROOT / "src")
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not prev else src + os.pathsep + prev
    env["PYTHONUTF8"] = "1"
    return env


def _chemprop_cmd(tool: str, cli_args: list[str]) -> list[str]:
    launcher = Path(__file__).with_name("chemprop_cli.py")
    return [sys.executable, "-u", str(launcher), tool, *cli_args]


def chemprop_version() -> str:
    try:
        import chemprop

        return str(getattr(chemprop, "__version__", "unknown"))
    except Exception:
        return "unknown"


def chemprop_identity_path(checkpoint_dir: Path) -> Path:
    return Path(checkpoint_dir) / "checkpoint_identity.json"


def write_chemprop_identity(
    checkpoint_dir: Path,
    *,
    dataset: str,
    fold: str,
    member_id: int,
    model_id: str,
) -> None:
    """Record that Chemprop's internal fold_0 is not the project fold index."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "fold": fold,
        "member_id": int(member_id),
        "model_id": model_id,
        "chemprop_internal_path": "fold_0/model_0/model.pt",
        "note": (
            "Chemprop always writes fold_0/model_0/model.pt inside our fold-specific "
            "save_dir. That internal fold_0 is not the project fold index."
        ),
        "chemprop_version": chemprop_version(),
    }
    chemprop_identity_path(checkpoint_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_chemprop_model_pt(
    checkpoint_dir: Path,
    *,
    dataset: str | None = None,
    fold: str | None = None,
    member_id: int | None = None,
    model_id: str | None = None,
) -> Path:
    """Resolve native Chemprop weights from explicit experiment metadata.

    Never glob a repo-wide model.pt. Chemprop's nested fold_0 is internal.
    """
    checkpoint_dir = Path(checkpoint_dir)
    expected = checkpoint_dir / "fold_0" / "model_0" / "model.pt"
    ident_path = chemprop_identity_path(checkpoint_dir)
    if ident_path.exists() and any(v is not None for v in (dataset, fold, member_id, model_id)):
        saved = json.loads(ident_path.read_text(encoding="utf-8"))
        checks = {
            "dataset": dataset,
            "fold": fold,
            "member_id": None if member_id is None else int(member_id),
            "model_id": model_id,
        }
        for key, want in checks.items():
            if want is None:
                continue
            got = saved.get(key)
            if key == "member_id" and got is not None:
                got = int(got)
            if str(got) != str(want):
                raise RuntimeError(
                    f"Chemprop checkpoint identity mismatch at {checkpoint_dir}: "
                    f"saved {key}={got!r}, expected {want!r}"
                )
    if not expected.exists():
        raise FileNotFoundError(
            f"No Chemprop model.pt at {expected}. "
            "Do not fall back to a global fold_0/model_0/model.pt search."
        )
    return expected


def _load_chemprop_test_loader(
    test_data: Path,
    test_features: Path,
    model_path: Path,
    batch_size: int = 256,
):
    """Load MoleculeDataLoader with feature scaler + graph cache enabled."""
    import torch
    from chemprop.data import MoleculeDataLoader, get_data, set_cache_graph
    from chemprop.utils import load_scalers

    set_cache_graph(True)
    _, features_scaler, *_rest = load_scalers(model_path)
    data = get_data(
        path=str(test_data),
        smiles_columns=["smiles"],
        target_columns=None,
        features_path=[str(test_features)],
        skip_invalid_smiles=False,
    )
    if features_scaler is not None:
        data.normalize_features(features_scaler)
    # Warm graph cache once.
    _ = data.batch_graph()
    loader = MoleculeDataLoader(dataset=data, batch_size=int(batch_size), num_workers=0)
    return data, loader


def nig_predictive_from_params(mu, lam, alpha, beta, var_floor: float = 0.0) -> dict:
    """NIG moments with per-row validity. Invalid rows get NaN variances, not a crash."""
    mu = np.asarray(mu, dtype=float).reshape(-1)
    lam = np.asarray(lam, dtype=float).reshape(-1)
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    beta = np.asarray(beta, dtype=float).reshape(-1)
    n = int(len(mu))
    if not (len(lam) == len(alpha) == len(beta) == n):
        raise ValueError(
            f"NIG length mismatch: mu={n} lambda={len(lam)} alpha={len(alpha)} beta={len(beta)}"
        )
    valid = (
        np.isfinite(mu)
        & np.isfinite(lam)
        & np.isfinite(alpha)
        & np.isfinite(beta)
        & (lam > 0)
        & (alpha > 1.0)
        & (beta > 0)
    )
    nan = np.full(n, np.nan, dtype=float)
    var_a = nan.copy()
    var_e = nan.copy()
    floor = float(var_floor)
    var_a[valid] = np.maximum(beta[valid] / (alpha[valid] - 1.0), floor)
    var_e[valid] = np.maximum(beta[valid] / (lam[valid] * (alpha[valid] - 1.0)), floor)
    var_t = var_a + var_e
    z = np.zeros(n, dtype=float)
    return {
        "mu": mu,
        "var_aleatoric": var_a,
        "var_epistemic": var_e,
        "var_epistemic_within": var_e,
        "var_epistemic_between": z,
        "var_epistemic_total": var_e,
        "var_total": var_t,
        "sigma_aleatoric": np.sqrt(var_a),
        "sigma_epistemic": np.sqrt(var_e),
        "sigma_epistemic_within": np.sqrt(var_e),
        "sigma_epistemic_between": z.copy(),
        "sigma_epistemic_total": np.sqrt(var_e),
        "sigma_total": np.sqrt(var_t),
        "nig_valid": valid,
        "n_invalid_nig": int((~valid).sum()),
        "n_valid_nig": int(valid.sum()),
    }


def predict_dmpnn_evidential_uq(
    test_data: Path,
    test_features: Path,
    checkpoint_dir: Path,
    preds_path: Path | None = None,
    batch_size: int = 256,
    n_expected: int | None = None,
    dataset: str | None = None,
    fold: str | None = None,
    member_id: int | None = None,
    model_id: str | None = None,
) -> dict[str, np.ndarray]:
    """Native Chemprop evidential UQ: one forward with NIG parameters.

    Returns μ and σ components from Soleimany et al.:
      σ_a² = β/(α-1),  σ_e² = β/(ν(α-1)),  σ_tot² = σ_a² + σ_e²
    (Chemprop's λ is the NIG evidence / precision parameter ν.)

    Chemprop MC-dropout is not implemented here and is not part of the active grid.
    """
    _ensure_rdbase()
    _stub_descriptastorus()
    import torch
    from chemprop.train.predict import predict
    from chemprop.utils import load_args, load_checkpoint, load_scalers

    model_path = resolve_chemprop_model_pt(
        Path(checkpoint_dir),
        dataset=dataset,
        fold=fold,
        member_id=member_id,
        model_id=model_id,
    )
    train_args = load_args(model_path)
    if str(getattr(train_args, "loss_function", "")).lower() != "evidential":
        raise ValueError(f"Checkpoint loss_function={train_args.loss_function!r}; expected evidential")
    scaler, *_ = load_scalers(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(model_path, device=device)
    data, loader = _load_chemprop_test_loader(test_data, test_features, model_path, batch_size=batch_size)
    n_input = len(data)
    if n_expected is not None and int(n_input) != int(n_expected):
        raise ValueError(
            f"Chemprop test loader has {n_input} rows, expected {n_expected}. "
            "Do not truncate predictions."
        )
    preds, lambdas, alphas, betas = predict(
        model=model,
        data_loader=loader,
        scaler=scaler,
        return_unc_parameters=True,
    )
    mu = np.asarray(preds, dtype=float).reshape(-1)
    lam = np.asarray(lambdas, dtype=float).reshape(-1)
    alpha = np.asarray(alphas, dtype=float).reshape(-1)
    beta = np.asarray(betas, dtype=float).reshape(-1)
    if not (len(mu) == len(lam) == len(alpha) == len(beta) == n_input):
        raise ValueError(
            f"Chemprop evidential output length mismatch: "
            f"mu={len(mu)} lambda={len(lam)} alpha={len(alpha)} beta={len(beta)} input={n_input}"
        )
    if n_expected is not None and len(mu) != int(n_expected):
        raise ValueError(f"Chemprop evidential preds {len(mu)} != expected {n_expected}")
    moments = nig_predictive_from_params(mu, lam, alpha, beta)
    if moments["n_invalid_nig"]:
        print(
            f"  [chemprop_evidential_invalid_nig] {moments['n_invalid_nig']}/{len(mu)} rows "
            f"(lambda>0, alpha>1, beta>0); variances set to NaN for those rows",
            flush=True,
        )
    out = {
        "mu": mu,
        "y_pred": mu,
        "gamma": mu,
        "nu": lam,
        "alpha": alpha,
        "beta": beta,
        "nig_lambda": lam,
        "nig_alpha": alpha,
        "nig_beta": beta,
        "var_aleatoric": moments["var_aleatoric"],
        "var_epistemic": moments["var_epistemic"],
        "var_epistemic_within": moments["var_epistemic_within"],
        "var_epistemic_between": moments["var_epistemic_between"],
        "var_epistemic_total": moments["var_epistemic_total"],
        "var_total": moments["var_total"],
        "sigma_aleatoric": moments["sigma_aleatoric"],
        "sigma_epistemic": moments["sigma_epistemic"],
        "sigma_epistemic_within": moments["sigma_epistemic_within"],
        "sigma_epistemic_between": moments["sigma_epistemic_between"],
        "sigma_epistemic_total": moments["sigma_epistemic_total"],
        "sigma_total": moments["sigma_total"],
        "nig_valid": moments["nig_valid"],
        "n_invalid_nig": moments["n_invalid_nig"],
        "n_valid_nig": moments["n_valid_nig"],
        "uncertainty_method": "chemprop_native_nig_evidential",
        "interval_method": "normal_moment_approximation",
        "chemprop_version": chemprop_version(),
    }
    if preds_path is not None:
        preds_path = Path(preds_path)
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        smiles = pd.read_csv(test_data)["smiles"].to_numpy()
        if len(smiles) != len(mu):
            raise ValueError(f"Cannot write evidential preds: smiles {len(smiles)} != mu {len(mu)}")
        pd.DataFrame(
            {
                "smiles": smiles,
                "logK": mu,
                "sigma_aleatoric": out["sigma_aleatoric"],
                "sigma_epistemic": out["sigma_epistemic"],
                "sigma_total": out["sigma_total"],
                "var_aleatoric": var_a,
                "var_epistemic": var_e,
                "var_total": var_t,
                "nig_lambda": lam,
                "nig_alpha": alpha,
                "nig_beta": beta,
                "uq_method": "evidential_nig",
                "chemprop_version": chemprop_version(),
            }
        ).to_csv(preds_path, index=False)
    return out


def predict_dmpnn_dropout_uq(*_a, **_k):
    raise RuntimeError(
        "Chemprop MC-dropout is not part of the active grid. "
        "Use native Chemprop evidential (predict_dmpnn_evidential_uq) only."
    )


def load_shared_cfg() -> dict:
    return json.loads((REMAKE_ROOT / "configs" / "shared_dmpnn.json").read_text(encoding="utf-8"))


def train_dmpnn_point(
    train_data: Path,
    train_features: Path,
    val_data: Path | None = None,
    val_features: Path | None = None,
    save_dir: Path | None = None,
    seed: int = 42,
    extra_args: list[str] | None = None,
    architecture: dict | None = None,
    num_folds: int = 1,
) -> dict:
    """Train blank Chemprop regression D-MPNN with shared depth/width.

    ``architecture`` overrides shared hidden size / depth / dropout / epochs /
    batch size / learning-rate fields when reproducing a published Chemprop config.
    """
    _ensure_rdbase()
    cfg = load_shared_cfg()
    if save_dir is None:
        raise ValueError("save_dir is required")
    save_dir.mkdir(parents=True, exist_ok=True)
    arch = architecture or {}
    cmd = _chemprop_cmd(
        "train",
        [
        "--data_path",
        str(train_data),
        "--features_path",
        str(train_features),
        "--dataset_type",
        "regression",
        "--save_dir",
        str(save_dir),
        "--epochs",
        str(int(arch.get("epochs", cfg["epochs"]))),
        "--batch_size",
        str(int(arch.get("batch_size", cfg["batch_size"]))),
        "--hidden_size",
        str(int(arch.get("hidden_size", cfg["dmpnn_hidden_size"]))),
        "--depth",
        str(int(arch.get("depth", cfg["dmpnn_depth"]))),
        "--ffn_num_layers",
        str(int(arch.get("ffn_num_layers", cfg["dmpnn_ffn_num_layers"]))),
        "--ffn_hidden_size",
        str(int(arch.get("ffn_hidden_size", cfg["dmpnn_ffn_hidden_size"]))),
        "--dropout",
        str(float(arch.get("dropout", cfg["dmpnn_dropout_point"]))),
        "--init_lr",
        str(float(arch.get("init_lr", cfg["init_lr"]))),
        "--max_lr",
        str(float(arch.get("max_lr", cfg["max_lr"]))),
        "--final_lr",
        str(float(arch.get("final_lr", cfg["final_lr"]))),
        "--warmup_epochs",
        str(float(arch.get("warmup_epochs", cfg["warmup_epochs"]))),
        "--seed",
        str(int(arch.get("seed", seed))),
        "--pytorch_seed",
        str(int(arch.get("pytorch_seed", seed))),
        "--metric",
        "rmse",
        "--extra_metrics",
        "mae",
        "r2",
        "--save_preds",
        # Windows + CUDA: dataloader workers can hang mid-epoch/validation.
        "--num_workers",
        "0",
        ],
    )
    if int(num_folds) > 1:
        cmd.extend(["--num_folds", str(int(num_folds))])
    elif val_data is not None and val_features is not None:
        cmd.extend(
            [
                "--separate_val_path",
                str(val_data),
                "--separate_val_features_path",
                str(val_features),
            ]
        )
    if extra_args:
        cmd.extend(extra_args)

    log_path = save_dir / "train_stdout.log"
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=_chemprop_env(),
        )
    elapsed = time.perf_counter() - t0
    meta = {
        "returncode": proc.returncode,
        "train_seconds": elapsed,
        "cmd": cmd,
        "log_path": str(log_path),
    }
    (save_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"chemprop train failed ({proc.returncode}); see {log_path}")
    return meta


def predict_dmpnn_point(
    test_data: Path,
    test_features: Path,
    checkpoint_dir: Path,
    preds_path: Path,
) -> np.ndarray:
    _ensure_rdbase()
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = _chemprop_cmd(
        "predict",
        [
            "--test_path",
            str(test_data),
            "--features_path",
            str(test_features),
            "--checkpoint_dir",
            str(checkpoint_dir),
            "--preds_path",
            str(preds_path),
        ],
    )
    log_path = preds_path.with_suffix(".log")
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=logf, stderr=subprocess.STDOUT, env=_chemprop_env())
    if proc.returncode != 0:
        raise RuntimeError(f"chemprop predict failed ({proc.returncode}); see {log_path}")
    pred_df = pd.read_csv(preds_path)
    # Chemprop writes smiles + prediction column(s)
    pred_cols = [c for c in pred_df.columns if c.lower() not in {"smiles", "logk"}]
    if not pred_cols:
        pred_cols = [c for c in pred_df.columns if c != "smiles"]
    return pred_df[pred_cols[0]].to_numpy(dtype=float)


def extract_fingerprints(
    data_path: Path,
    features_path: Path,
    checkpoint_dir: Path,
    out_npz: Path,
    record_ids: list[str],
) -> Path:
    """Save Chemprop learned fingerprints (pre-FFN graph embeddings) keyed by record_id."""
    _ensure_rdbase()
    out_csv = out_npz.with_suffix(".csv")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    cmd = _chemprop_cmd(
        "fingerprint",
        [
            "--test_path",
            str(data_path),
            "--features_path",
            str(features_path),
            "--checkpoint_dir",
            str(checkpoint_dir),
            "--preds_path",
            str(out_csv),
            "--fingerprint_type",
            "MPN",
        ],
    )
    log_path = out_npz.with_suffix(".fp.log")
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=logf, stderr=subprocess.STDOUT, env=_chemprop_env())
    if proc.returncode != 0:
        cmd2 = [c for c in cmd if c != "--fingerprint_type" and c != "MPN"]
        with open(log_path, "a", encoding="utf-8") as logf:
            logf.write("\n--- retry without fingerprint_type ---\n")
            proc = subprocess.run(cmd2, cwd=str(REPO_ROOT), stdout=logf, stderr=subprocess.STDOUT, env=_chemprop_env())
        if proc.returncode != 0:
            raise RuntimeError(f"chemprop fingerprint failed; see {log_path}")

    fp_df = pd.read_csv(out_csv)
    # Drop smiles / non-numeric
    num_cols = [c for c in fp_df.columns if c.lower() != "smiles"]
    emb = fp_df[num_cols].to_numpy(dtype=float)
    if len(record_ids) != len(emb):
        raise ValueError(f"fingerprint rows {len(emb)} != record_ids {len(record_ids)}")
    np.savez_compressed(out_npz, embeddings=emb, record_ids=np.asarray(record_ids), columns=np.asarray(num_cols))
    return out_npz
