import copy

from typeguard import typechecked

from graph_uq.config.uncertainty import UncertaintiesConfig, UncertaintyModelType
from graph_uq.uncertainty.appr_diffusion import UncertaintyModelAPPRDiffusion
from graph_uq.uncertainty.appr_distance import UncertaintyModelAPPRDistance
from graph_uq.uncertainty.base import BaseUncertaintyModel
from graph_uq.uncertainty.corrected_conditional_ebm import (
    UncertaintyModelCorrectedConditionalEBM,
)
from graph_uq.uncertainty.energy import UncertaintyModelEnergy
from graph_uq.uncertainty.entropy import UncertaintyModelEntropy
from graph_uq.uncertainty.evidence import UncertaintyModelEvidence
from graph_uq.uncertainty.feature_distance import UncertaintyModelFeatureDistance
from graph_uq.uncertainty.gnn_safe import UncertaintyModelGNNSafe
from graph_uq.uncertainty.heat.composite import UncertaintyModelHeatComposite
from graph_uq.uncertainty.heat.ebm import UncertaintyModelHeat
from graph_uq.uncertainty.latent_space import UncertaintyModelGaussianPerClass
from graph_uq.uncertainty.max_score import UncertaintyModelMaxScore
from graph_uq.uncertainty.multi_scale_corrected_conditional_ebm import (
    UncertaintyModelMultiScaleCorrectedConditionalEBM,
)
from graph_uq.uncertainty.mutual_information import UncertaintyModelMutualInformation
from graph_uq.uncertainty.variance import UncertaintyModelVariance


@typechecked
def get_uncertainty_models(
    uncertainty: UncertaintiesConfig,
) -> dict[str, BaseUncertaintyModel]:
    """Returns the uncertainty models.

    Args:
        uncertainty (dict[str, Any]): The uncertainty configuration, containing configuration for each uncertainty model

    Returns:
        dict[str, BaseUncertaintyModel]: The uncertainty models
    """
    uncertainty_models = {}
    for name, config in uncertainty.items():
        config = copy.copy(
            config
        )  # we pop `type_` later, so we need to copy the config
        match UncertaintyModelType(config.pop("type_")):
            case UncertaintyModelType.ENERGY:
                uncertainty_models[name] = UncertaintyModelEnergy(**config)
            case UncertaintyModelType.ENTROPY:
                uncertainty_models[name] = UncertaintyModelEntropy(**config)
            case UncertaintyModelType.MAX_SCORE:
                uncertainty_models[name] = UncertaintyModelMaxScore(**config)
            case UncertaintyModelType.GAUSSIAN_PER_CLASS:
                uncertainty_models[name] = UncertaintyModelGaussianPerClass(**config)
            case UncertaintyModelType.MUTUAL_INFORMATION:
                uncertainty_models[name] = UncertaintyModelMutualInformation(**config)
            case UncertaintyModelType.PREDICTED_VARIANCE:
                uncertainty_models[name] = UncertaintyModelVariance(**config)
            case UncertaintyModelType.APPR_DISTANCE:
                uncertainty_models[name] = UncertaintyModelAPPRDistance(**config)
            case UncertaintyModelType.APPR_DIFFUSION:
                uncertainty_models[name] = UncertaintyModelAPPRDiffusion(**config)
            case UncertaintyModelType.FEATURE_DISTANCE:
                uncertainty_models[name] = UncertaintyModelFeatureDistance(**config)
            case UncertaintyModelType.EVIDENCE:
                uncertainty_models[name] = UncertaintyModelEvidence(**config)
            case UncertaintyModelType.GNN_SAFE:
                uncertainty_models[name] = UncertaintyModelGNNSafe(**config)
            case UncertaintyModelType.CORRECTED_CONDITIONAL_EBM:
                uncertainty_models[name] = UncertaintyModelCorrectedConditionalEBM(
                    **config
                )
            case UncertaintyModelType.MULTI_SCALE_CORRECTED_CONDITIONAL_EBM:
                uncertainty_models[name] = (
                    UncertaintyModelMultiScaleCorrectedConditionalEBM(**config)
                )
            case UncertaintyModelType.HEAT:
                uncertainty_models[name] = UncertaintyModelHeat(**config)
            case UncertaintyModelType.HEAT_COMPOSITE:
                uncertainty_models[name] = UncertaintyModelHeatComposite(**config)
            case type_:
                raise ValueError(f"Unknown uncertainty model type {type_}")
    return uncertainty_models
