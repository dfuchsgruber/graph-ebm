from enum import StrEnum, unique
from typing import Any, TypeAlias, TypedDict

from graph_uq.experiment import experiment


@unique
class UncertaintyModelType(StrEnum):
    ENERGY = "energy"
    GNN_SAFE = "gnn_safe"
    ENTROPY = "entropy"
    MAX_SCORE = "max_score"
    GAUSSIAN_PER_CLASS = "gaussian_per_class"
    MUTUAL_INFORMATION = "mutual_information"
    PREDICTED_VARIANCE = "predicted_variance"
    TOTAL_VARIANCE = "total_variance"

    # Structural
    APPR_DISTANCE = "appr_distance"
    APPR_DIFFUSION = "appr_diffusion"

    # Feature based
    FEATURE_DISTANCE = "feature_distance"

    # Evidential methods
    EVIDENCE = "evidence"

    # Ours
    CORRECTED_CONDITIONAL_EBM = "corrected_conditional_ebm"
    MULTI_SCALE_CORRECTED_CONDITIONAL_EBM = "multi_scale_corrected_conditional_ebm"

    # HEAT
    HEAT = "heat"
    HEAT_COMPOSITE = "heat_composite"


@unique
class CovarianceType(StrEnum):
    FULL = "full"
    DIAGONAL = "diagonal"
    IDENTITY = "identity"
    ISOTROPIC = "isotropic"


@unique
class EnergyAggregation(StrEnum):
    LOGSUMEXP = "logsumexp"
    MINUS_LOGSUMEXP = "minus_logsumexp"
    SUM = "sum"


UncertaintiesConfig: TypeAlias = dict[str, Any]


@experiment.config
def default_uncertainty_config():
    uncertainty = dict()
