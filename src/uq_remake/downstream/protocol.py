"""Downstream CV protocol constants for the selected DS2 workhorse."""
from __future__ import annotations

import hashlib
from pathlib import Path

from uq_remake import REMAKE_ROOT
from uq_remake.metrics.cv_protocol import EVALUATION_PARTITION, FOLD_SD_DDOF, MIN_PAIR_N, PAIR_COLUMN

DATASET = "DS2"
FAMILY = "HETERO_MLP"
MODEL_ID = "DS2_RDKIT_MLPENS5_HETERO"
MODEL_SHORT = "MLPENS5_HETERO"
FOLDS = ["fold0", "fold1", "fold2", "fold3", "fold4"]
N_MEMBERS = 5
HIDDEN_DIMS = [512, 512, 256]
LAMBDA_REG = 0.01
RETENTION_FRACTIONS = (0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
PRIMARY_RETENTION = 0.50
HIGH_CONFIDENCE_RETENTION = 0.10
NARROW_SHORTLIST_RETENTION = 0.25
RANDOM_N_REPEATS = 20
RANDOM_SEED = 0
TANIMOTO_K = (1, 5, 10)
EMBEDDING_K = (1, 5, 10)
FEATURE_KNN_K = (5, 10)
FINGERPRINT_TYPE = "morgan"
FINGERPRINT_RADIUS = 2
FINGERPRINT_N_BITS = 2048
COVARIANCE_ESTIMATOR = "ledoit_wolf"
CHEMPROP_MAHALANOBIS_DEFAULT = "centroid"
PAIR_DENSITY_BINS = (
    ("0", 0, 0),
    ("1-5", 1, 5),
    ("6-10", 6, 10),
    ("11-25", 11, 25),
    ("26-50", 26, 50),
    (">50", 51, 10**9),
)
Z90 = 1.6448536269514722

DOWNSTREAM = REMAKE_ROOT / "results" / "downstream_cv"
AD_DIR = DOWNSTREAM / "ad"
AD_RETENTION_DIR = DOWNSTREAM / "ad_retention"
AD_HOC_DIR = DOWNSTREAM / "ad_hoc_selectivity"
PAIR_ACQ_DIR = DOWNSTREAM / "pair_acquisition"
PAIR_ACQ_SENS_DIR = DOWNSTREAM / "pair_acquisition_tau_2_0p5"
PAIR_ACQ_SENS_SPEC = REMAKE_ROOT / "configs" / "experiments" / "pair_acquisition_threshold_sensitivity_spec.yaml"
PAIR_EVENT_DIR = DOWNSTREAM / "pair_event_support"
PAIR_DIR = DOWNSTREAM / "pair_density"
SIGN_DIR = DOWNSTREAM / "sign_stability"
CORR_DIR = DOWNSTREAM / "ad_distance_error_corr"
DS1_WORKHORSE = "DS1_RDKIT_MLPENS5_EVID"
DS2_WORKHORSE = MODEL_ID
DS1_CHEMPROP = "DS1_GRAPH_DMPNN_EVID"
DS2_CHEMPROP = "DS2_GRAPH_DMPNN_EVID"

WORKHORSE_EXPECTED = {
    "mop_rmse": (2.078, 0.130),
    "sign_accuracy_all": (0.822, 0.017),
}

WORKHORSE_PRIMARY_SCORES = (
    "sigma_total",
    "tanimoto_nn_distance",
    "mahalanobis_feature_centroid",
)
WORKHORSE_SENSITIVITY_SCORES = (
    "sigma_epistemic_within",
    "sigma_epistemic_between",
    "sigma_aleatoric",
    "tanimoto_knn_mean_5",
    "tanimoto_knn_mean_10",
    "tanimoto_median_distance",
    "pair_density_risk",
)
CHEMPROP_PRIMARY_SCORES = (
    "sigma_total",
    "tanimoto_nn_distance",
    "chemprop_std_euclidean_nn",
    "chemprop_mahalanobis_centroid",
)
CHEMPROP_SENSITIVITY_SCORES = (
    "chemprop_cosine_nn",
    "chemprop_euclidean_nn",
    "chemprop_mahalanobis_nn",
    "chemprop_std_euclidean_knn_5",
    "chemprop_std_euclidean_knn_10",
    "tanimoto_knn_mean_5",
    "tanimoto_knn_mean_10",
    "tanimoto_median_distance",
)
CHEMPROP_ONLY_SCORES = (
    "chemprop_std_euclidean_nn",
    "chemprop_mahalanobis_centroid",
    "chemprop_cosine_nn",
    "chemprop_euclidean_nn",
    "chemprop_mahalanobis_nn",
    "chemprop_std_euclidean_knn_5",
    "chemprop_std_euclidean_knn_10",
)
AD_SCORES = WORKHORSE_PRIMARY_SCORES + WORKHORSE_SENSITIVITY_SCORES
PRIMARY_SCORE_SET = set(WORKHORSE_PRIMARY_SCORES + CHEMPROP_PRIMARY_SCORES)
# Ligand-level AD distances for the SI distance-vs-|error| diagnostic.
# Feature-space extras are Spearman/Pearson SI only and are not plotted on
# retention-curve figures.
FEATURE_SPACE_SI_SCORES = (
    "mahalanobis_feature_nn",
    "mahalanobis_feature_knn_mean_5",
    "mahalanobis_feature_knn_mean_10",
    "euclidean_feature_nn",
    "euclidean_feature_median",
)
DISTANCE_ERROR_CORR_SCORES = (
    "tanimoto_nn_distance",
    "mahalanobis_feature_centroid",
    "tanimoto_median_distance",
    "tanimoto_knn_mean_5",
    "tanimoto_knn_mean_10",
) + FEATURE_SPACE_SI_SCORES
CHEMPROP_DISTANCE_ERROR_CORR_SCORES = DISTANCE_ERROR_CORR_SCORES + (
    "chemprop_std_euclidean_nn",
    "chemprop_std_euclidean_knn_5",
    "chemprop_std_euclidean_knn_10",
    "chemprop_euclidean_nn",
    "chemprop_cosine_nn",
    "chemprop_mahalanobis_centroid",
    "chemprop_mahalanobis_nn",
)
DISTANCE_ERROR_CORR_SCOPES = ("all_pairs", "ree_pairs")
DISTANCE_ERROR_CORR_MIN_N = 10

# Sigma-only pair acquisition. Distances are not used.
# A pair-task is eligible when n_event > PAIR_ACQ_EVENT_MIN_EXCLUSIVE.
PAIR_ACQ_THRESHOLD = 1.0
PAIR_ACQ_SENS_HIGH_THRESHOLD = 2.0
PAIR_ACQ_SENS_LOW_THRESHOLD = 0.5
PAIR_ACQ_EVENT_MIN_EXCLUSIVE = 1


def pair_acq_event_elig_text() -> str:
    n = int(PAIR_ACQ_EVENT_MIN_EXCLUSIVE)
    noun = "event" if n == 1 else "events"
    return f"more than {n} held-out {noun}"
PAIR_ACQ_TOP_K = (1, 5, 10)
PAIR_ACQ_STRATEGIES = (
    "y_pred",
    "lcb_1sigma",
    "lcb_2sigma",
    "lcb_3sigma",
    "expected_improvement",
)
PAIR_ACQ_RANDOM_STRATEGY = "random"
PAIR_ACQ_RANDOM_REPEATS = RANDOM_N_REPEATS
PAIR_ACQ_RANDOM_SEED = RANDOM_SEED
PAIR_ACQ_TASKS = ("high_selectivity", "low_selectivity")

# Manuscript-facing labels. Internal CSV score_name values stay unchanged.
SCORE_DISPLAY_NAMES = {
    "sigma_total": r"$\sigma_{\mathrm{total}}$",
    "sigma_aleatoric": r"$\sigma_{\mathrm{aleatoric}}$",
    "sigma_epistemic_within": r"$\sigma_{\mathrm{epistemic,within}}$",
    "sigma_epistemic_between": r"$\sigma_{\mathrm{epistemic,between}}$",
    "tanimoto_nn_distance": "Tanimoto nearest-neighbor distance",
    "tanimoto_knn_mean_5": r"Tanimoto $k$NN mean, $k=5$",
    "tanimoto_knn_mean_10": r"Tanimoto $k$NN mean, $k=10$",
    "tanimoto_median_distance": "Tanimoto median distance",
    "mahalanobis_feature_centroid": "RDKit feature-centroid Mahalanobis distance",
    "mahalanobis_joint": "RDKit feature-centroid Mahalanobis distance",
    "mahalanobis_feature_nn": "RDKit feature Mahalanobis-NN",
    "mahalanobis_feature_knn_mean_5": r"RDKit feature Mahalanobis $k$NN mean, $k=5$",
    "mahalanobis_feature_knn_mean_10": r"RDKit feature Mahalanobis $k$NN mean, $k=10$",
    "euclidean_feature_nn": "RDKit feature Euclidean-NN",
    "euclidean_feature_median": "RDKit feature median Euclidean distance",
    "pair_density_risk": "pair-density risk",
    "chemprop_std_euclidean_nn": "Chemprop Euclidean-NN",
    "chemprop_euclidean_nn": "Chemprop Euclidean-NN (unstandardized)",
    "chemprop_std_euclidean_knn_5": r"Chemprop Euclidean-$k$NN, $k=5$",
    "chemprop_std_euclidean_knn_10": r"Chemprop Euclidean-$k$NN, $k=10$",
    "chemprop_mahalanobis_centroid": "Chemprop embedding-centroid Mahalanobis",
    "chemprop_mahalanobis_nn": "Chemprop embedding Mahalanobis-NN",
    "chemprop_cosine_nn": "Chemprop cosine-NN",
    "random_retention": "random retention",
}
SCORE_METHOD_NAMES = {
    "mahalanobis_feature_centroid": "rdkit_feature_mahalanobis_centroid",
    "mahalanobis_joint": "rdkit_feature_mahalanobis_centroid",
    "mahalanobis_feature_nn": "rdkit_feature_mahalanobis_nn",
    "mahalanobis_feature_knn_mean_5": "rdkit_feature_mahalanobis_knn_mean_k5",
    "mahalanobis_feature_knn_mean_10": "rdkit_feature_mahalanobis_knn_mean_k10",
    "euclidean_feature_nn": "rdkit_feature_euclidean_nn",
    "euclidean_feature_median": "rdkit_feature_euclidean_median",
    "chemprop_mahalanobis_centroid": "chemprop_embedding_mahalanobis_centroid",
    "chemprop_mahalanobis_nn": "chemprop_embedding_mahalanobis_nn",
    "chemprop_std_euclidean_nn": "chemprop_embedding_euclidean_nn_standardized",
    "chemprop_std_euclidean_knn_5": "chemprop_embedding_euclidean_knn_mean_standardized_k5",
    "chemprop_std_euclidean_knn_10": "chemprop_embedding_euclidean_knn_mean_standardized_k10",
}
FAMILY_DISPLAY_NAMES = {
    "workhorse": "Heteroscedastic MLP ensemble",
    "chemprop": "Chemprop evidential",
}


def model_display_name(model_id: str) -> str:
    from uq_remake.metrics.workhorse import DISPLAY_NAMES

    mid = str(model_id)
    if mid in DISPLAY_NAMES:
        return DISPLAY_NAMES[mid]
    if mid.startswith("DS1_"):
        return DISPLAY_NAMES.get("DS2_" + mid[4:], mid)
    return mid


def score_display_name(score_name: str) -> str:
    key = str(score_name)
    return SCORE_DISPLAY_NAMES.get(key, key)


def family_display_name(family: str, model_id: str | None = None) -> str:
    if model_id:
        return model_display_name(model_id)
    key = str(family)
    return FAMILY_DISPLAY_NAMES.get(key, key)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path | str | None) -> str:
    if path is None:
        return ""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REMAKE_ROOT.resolve()))
    except Exception:
        return str(p)
