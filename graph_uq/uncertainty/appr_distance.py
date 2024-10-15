import itertools
from typing import Any, Dict

import torch
import torch_scatter
from jaxtyping import Float, jaxtyped
from torch import Tensor
from typeguard import typechecked

from graph_uq.config.data import DatasetSplit
from graph_uq.config.uncertainty import UncertaintyModelType
from graph_uq.data import Data
from graph_uq.metric import Metric
from graph_uq.model.base import BaseModel
from graph_uq.model.prediction import Prediction
from graph_uq.uncertainty.base import BaseUncertaintyModel


class UncertaintyModelAPPRDistance(BaseUncertaintyModel):
    """Uncertainty model that measures in terms of structural similarity of APPR matrix distances to training nodes."""

    def __init__(
        self, alpha: float = 0.2, num_steps: int = 10, diffusion: Dict[str, Any] = {}
    ):
        self.teleport_probability = alpha
        self.num_page_rank_iterations = num_steps
        super().__init__(diffusion=diffusion)

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
        appr_matrix = data.get_log_appr_matrix(
            teleport_probability=self.teleport_probability,
            num_iterations=self.num_page_rank_iterations,
        ).exp()
        appr_distances_to_train = appr_matrix[:, data.get_mask(DatasetSplit.TRAIN)]
        uncertainties |= {
            Metric(
                uncertainty_model_type=UncertaintyModelType.APPR_DISTANCE, suffix="mean"
            ): appr_distances_to_train.mean(-1),
            Metric(
                uncertainty_model_type=UncertaintyModelType.APPR_DISTANCE, suffix="max"
            ): appr_distances_to_train.max(-1).values,
            Metric(
                uncertainty_model_type=UncertaintyModelType.APPR_DISTANCE, suffix="min"
            ): appr_distances_to_train.min(-1).values,
        }
        # Aggregate for each training class separately
        y_train = data.y[data.get_mask(DatasetSplit.TRAIN)]
        for aggr_in_class, aggr_classes in itertools.product(
            ("mean", "max", "min"), repeat=2
        ):
            match aggr_in_class:
                case "mean":
                    aggr_by_class = torch_scatter.scatter_mean(
                        appr_distances_to_train, y_train, dim=1
                    )
                case "max":
                    aggr_by_class = torch_scatter.scatter_max(
                        appr_distances_to_train, y_train, dim=1
                    )[0]
                case "min":
                    aggr_by_class = torch_scatter.scatter_min(
                        appr_distances_to_train, y_train, dim=1
                    )[0]
                case _:
                    raise ValueError(aggr_in_class)
            match aggr_classes:
                case "mean":
                    uncertainty = aggr_by_class.mean(1)
                case "max":
                    uncertainty = aggr_by_class.max(1).values
                case "min":
                    uncertainty = aggr_by_class.min(1).values
            uncertainties |= {
                Metric(
                    uncertainty_model_type=UncertaintyModelType.APPR_DISTANCE,
                    suffix=f"by_class_{aggr_in_class}_{aggr_classes}",
                ): uncertainty,
            }
        return uncertainties
