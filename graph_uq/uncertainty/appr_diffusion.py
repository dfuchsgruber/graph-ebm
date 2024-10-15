import itertools
from typing import Any, Dict

import torch
import torch_scatter
from jaxtyping import Float, jaxtyped
from torch import Tensor
from torch_geometric.nn.conv import APPNP
from typeguard import typechecked

from graph_uq.config.data import DatasetSplit
from graph_uq.config.uncertainty import UncertaintyModelType
from graph_uq.data import Data
from graph_uq.metric import Metric
from graph_uq.model.base import BaseModel
from graph_uq.model.prediction import Prediction
from graph_uq.uncertainty.base import BaseUncertaintyModel


class UncertaintyModelAPPRDiffusion(BaseUncertaintyModel):
    """Uncertainty model that measures in terms of structural similarity by diffusing a signal on the training nodes"""

    def __init__(
        self, alpha: float = 0.2, num_steps: int = 10, diffusion: Dict[str, Any] = {}
    ):
        self.teleport_probability = alpha
        self.num_page_rank_iterations = num_steps
        super().__init__(diffusion=diffusion)

    @torch.no_grad()
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
        signal = data.get_mask(DatasetSplit.TRAIN).unsqueeze(-1).float()
        appnp = APPNP(self.num_page_rank_iterations, self.teleport_probability)
        diffused = appnp(
            signal, data.edge_index, edge_weight=getattr(data, "edge_weight", None)
        )
        uncertainties[
            Metric(uncertainty_model_type=UncertaintyModelType.APPR_DIFFUSION)
        ] = -diffused.squeeze(1)
        return uncertainties
