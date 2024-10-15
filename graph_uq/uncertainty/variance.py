from typing import Any, Dict

import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor
from typeguard import typechecked

from graph_uq.config.uncertainty import UncertaintyModelType
from graph_uq.data import Data
from graph_uq.metric import Metric
from graph_uq.model.base import BaseModel
from graph_uq.model.prediction import Prediction
from graph_uq.uncertainty.base import BaseUncertaintyModel


class UncertaintyModelVariance(BaseUncertaintyModel):
    """Uncertainty model that relates to the variance of the MC predictions."""

    def __init__(self, eps: float = 1e-12, diffusion: Dict[str, Any] = {}):
        super().__init__(diffusion=diffusion)
        self.eps = eps

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, data: Data, model: BaseModel, prediction: Prediction
    ) -> dict[Metric, Float[Tensor, "num_nodes"]]:
        """Computes the uncertainty estimates of this model (variance)

        Args:
            data (Data): The data for which to make uncertainty estimates
            model (BaseModel): The model for which to make uncertainty estimates
            prediction (Prediction): The prediction of the model on the data. For most estimators, this should be used instead using `model` and `data` directly.

        Returns:
            dict[Metric, Float[Tensor, 'num_nodes']]: The uncertainty estimates, higher means more uncertain
        """
        uncertainties = {}
        for propagated in (True, False):
            probabilities = prediction.get_probabilities(propagated=propagated)
            if probabilities is not None and probabilities.size(0) > 1:
                variance = probabilities.var(0)

                # Variance of all classes
                uncertainties[
                    Metric(
                        propagated=propagated,
                        uncertainty_model_type=UncertaintyModelType.TOTAL_VARIANCE,
                    )
                ] = variance.sum(-1)

                # Variance of the predicted class
                predicted_labels = prediction.get_predictions(propagated=propagated)
                uncertainties[
                    Metric(
                        propagated=propagated,
                        uncertainty_model_type=UncertaintyModelType.PREDICTED_VARIANCE,
                    )
                ] = variance[torch.arange(variance.size(0)), predicted_labels]
        return uncertainties
