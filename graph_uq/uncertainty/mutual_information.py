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


class UncertaintyModelMutualInformation(BaseUncertaintyModel):
    """Uncertainty model that relates to the mutual information."""

    uncertainty_model_type: UncertaintyModelType = (
        UncertaintyModelType.MUTUAL_INFORMATION
    )

    def __init__(self, eps: float = 1e-12, diffusion: Dict[str, Any] = {}):
        super().__init__(diffusion=diffusion)
        self.eps = eps

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, data: Data, model: BaseModel, prediction: Prediction
    ) -> dict[Metric, Float[Tensor, "num_nodes"]]:
        """Computes the uncertainty estimates of this model (mutual information): E_theta[ H[ p(y, theta) ] ] - H[ E_theta[ p(y, theta) ] ]

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
                # Compute the expected entropy, i.e. average the entropies over the Monte Carlo samples
                expected_entropy: Float[Tensor, "num_nodes"] = (
                    -(probabilities * torch.log(probabilities + self.eps))
                    .sum(-1)
                    .mean(0)
                )
                expected_probabilities = probabilities.mean(0)
                predictive_entropy = -(
                    expected_probabilities
                    * torch.log(expected_probabilities + self.eps)
                ).sum(-1)
                uncertainties[
                    Metric(
                        propagated=propagated,
                        uncertainty_model_type=self.uncertainty_model_type,
                    )
                ] = predictive_entropy - expected_entropy
        return uncertainties
