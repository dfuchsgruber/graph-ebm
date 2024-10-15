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


class UncertaintyModelEvidence(BaseUncertaintyModel):
    """Uncertainty model that relates to evidence"""

    uncertainty_model_type: UncertaintyModelType = UncertaintyModelType.EVIDENCE

    def __init__(self, diffusion: Dict[str, Any] = {}):
        super().__init__(diffusion=diffusion)

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, data: Data, model: BaseModel, prediction: Prediction
    ) -> dict[Metric, Float[Tensor, " num_nodes"]]:
        """Computes the total evidence for a sample (the sum of evidences for all classes)

        Args:
            data (Data): The data for which to make uncertainty estimates
            model (BaseModel): The model for which to make uncertainty estimates
            prediction (Prediction): The prediction of the model on the data. For most estimators, this should be used instead using `model` and `data` directly.

        Returns:
            dict[Metric, Float[Tensor, 'num_nodes']]: The uncertainty estimates
        """
        uncertainties = {}
        for propagated in (True, False):
            evidence = prediction.get_evidence(propagated=propagated)
            if evidence is not None:
                uncertainties[
                    Metric(
                        propagated=propagated,
                        uncertainty_model_type=self.uncertainty_model_type,
                    )
                ] = -evidence.sum(-1).mean(0)
        return uncertainties
