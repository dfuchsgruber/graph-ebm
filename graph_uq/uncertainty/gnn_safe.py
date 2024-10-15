from typing import Any

import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul
from typeguard import typechecked

from graph_uq.config.uncertainty import UncertaintyModelType
from graph_uq.data import Data
from graph_uq.metric import Metric
from graph_uq.model.base import BaseModel
from graph_uq.model.prediction import Prediction
from graph_uq.uncertainty.energy import UncertaintyModelEnergy


class UncertaintyModelGNNSafe(UncertaintyModelEnergy):
    """Uncertainty model that relates to diffused energy"""

    def __init__(
        self,
        *args,
        add_self_loops: bool = False,
        alpha: float = 0.5,
        num_iterations: int = 2,
        diffusion: dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__(diffusion=diffusion)
        self.add_self_loops = add_self_loops
        self.alpha = alpha
        self.num_iterations = num_iterations

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, data: Data, model: BaseModel, prediction: Prediction
    ) -> dict[Metric, Float[Tensor, " num_nodes"]]:
        """Computes the uncertainty estimates of this model (predictive entropy): E_theta[ H[ p(y, theta) ] ]

        Args:
            data (Data): The data for which to make uncertainty estimates
            model (BaseModel): The model for which to make uncertainty estimates
            prediction (Prediction): The prediction of the model on the data. For most estimators, this should be used instead using `model` and `data` directly.

        Returns:
            dict[Metric, Float[Tensor, 'num_nodes']]: The uncertainty estimates
        """
        uncertainties = {}
        for propagated in (True, False):
            logits = prediction.get_logits(propagated=propagated)
            if logits is not None:
                energy = self.energy(logits, self.temperature).mean(0)
                diffused = data.label_propagation_diffusion(
                    energy.unsqueeze(-1),
                    self.num_iterations,
                    alpha=self.alpha,
                    add_self_loops=False,
                )[-1].squeeze(-1)
                uncertainties[
                    Metric(
                        propagated=propagated,
                        uncertainty_model_type=self.uncertainty_model_type,
                    )
                ] = diffused
        return uncertainties
