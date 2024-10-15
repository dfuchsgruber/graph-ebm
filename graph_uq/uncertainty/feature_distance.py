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


class UncertaintyModelFeatureDistance(BaseUncertaintyModel):
    """Uses feature distance to train for uncertainty."""

    def __init__(
        self,
        num_diffusion_steps: int,
        normalize: bool = True,
        add_self_loops: bool = True,
        diffusion: Dict[str, Any] = {},
    ):
        super().__init__(diffusion=diffusion)
        self.num_diffusion_steps = num_diffusion_steps
        self.normalize = normalize
        self.add_self_loops = add_self_loops

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

        # APPR scores
        x = data.get_diffused_nodes_features(
            self.num_diffusion_steps,
            normalize=self.normalize,
            add_self_loops=self.add_self_loops,
            cache=True,
        )
        difference: Float[torch.Tensor, "num_nodes num_train num_features"] = (
            x[:, None, :] - x[data.get_mask(DatasetSplit.TRAIN)][None, :, :]
        )
        for norm in (0, 1, 2, float("inf")):
            distance = difference.norm(p=norm, dim=-1)
            uncertainties[
                Metric(
                    uncertainty_model_type=UncertaintyModelType.FEATURE_DISTANCE,
                    suffix=f"p{norm}_mean",
                )
            ] = distance.mean(-1)
            uncertainties[
                Metric(
                    uncertainty_model_type=UncertaintyModelType.FEATURE_DISTANCE,
                    suffix=f"p{norm}_min",
                )
            ] = distance.min(-1).values
            uncertainties[
                Metric(
                    uncertainty_model_type=UncertaintyModelType.FEATURE_DISTANCE,
                    suffix=f"p{norm}_max",
                )
            ] = distance.max(-1).values

            distance_sorted = torch.sort(distance, dim=-1).values
            for k in (2, 5, 10):
                uncertainties[
                    Metric(
                        uncertainty_model_type=UncertaintyModelType.FEATURE_DISTANCE,
                        suffix=f"p{norm}_top{k}_mean",
                    )
                ] = distance_sorted[:, :k].mean(-1)

        return uncertainties
