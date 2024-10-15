from typing import Any

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


class UncertaintyModelEnergy(BaseUncertaintyModel):
    """Energy-based uncertainty model."""

    uncertainty_model_type: UncertaintyModelType = UncertaintyModelType.ENERGY

    def __init__(self, temperature: float = 1.0, diffusion: dict[str, Any] = {}):
        super().__init__(diffusion=diffusion)
        self.temperature = temperature

    @staticmethod
    @jaxtyped(typechecker=typechecked)
    def energy(
        logits: Float[Tensor, "*batch d"], temperature: float
    ) -> Float[Tensor, " *batch"]:
        """Computes the energy of the logits.

        Args:
            logits (Float[Tensor, 'num_nodes ...']): The logits
            temperature (float): The temperature

        Returns:
            Float[Tensor, 'num_nodes']: The energy
        """
        return -temperature * torch.logsumexp(logits / temperature, dim=-1)

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, data: Data, model: BaseModel, prediction: Prediction
    ) -> dict[Metric, Float[Tensor, " num_nodes"]]:
        """Computes the uncertainty estimates of this model (energy).

        Args:
            data (Data): The data for which to make uncertainty estimates
            model (BaseModel): The model for which to make uncertainty estimates
            prediction (Prediction): The prediction of the model on the data. For most estimators, this should be used instead using `model` and `data` directly.

        Returns:
            dict[Metric, Float[Tensor, 'num_nodes']]: The energy estimates
        """
        uncertainties = {}
        for propagated in (True, False):
            logits = prediction.get_logits(propagated=propagated)
            if logits is not None:
                energy = self.energy(logits, self.temperature).mean(0)
                uncertainties[
                    Metric(
                        propagated=propagated,
                        uncertainty_model_type=self.uncertainty_model_type,
                    )
                ] = energy
        return uncertainties
