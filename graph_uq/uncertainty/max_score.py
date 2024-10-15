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


class UncertaintyModelMaxScore(BaseUncertaintyModel):
    """Uncertainty model that relates to the maximum score."""

    uncertainty_model_type: UncertaintyModelType = UncertaintyModelType.MAX_SCORE

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, data: Data, model: BaseModel, prediction: Prediction
    ) -> dict[Metric, Float[Tensor, "num_nodes"]]:
        """Computes the uncertainty estimates of this model (max score)

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
            if probabilities is not None:
                probabilities = probabilities.mean(0)
                predicted_labels = prediction.get_predictions(propagated=propagated)
                max_score = probabilities[
                    torch.arange(probabilities.size(0)), predicted_labels
                ]
                uncertainties[
                    Metric(
                        propagated=propagated,
                        uncertainty_model_type=self.uncertainty_model_type,
                    )
                ] = 1 - max_score
        return uncertainties
