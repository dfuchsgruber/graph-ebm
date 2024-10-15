from abc import abstractmethod
from copy import deepcopy
from typing import Any

from jaxtyping import Bool, Float, jaxtyped
from torch import Tensor
from typeguard import typechecked

from graph_uq.config.data import DatasetSplit, DiffusionType, DistributionType
from graph_uq.data import Data
from graph_uq.evaluation.result import EvaluationResult
from graph_uq.evaluation.uncertainty import binary_classification
from graph_uq.metric import Metric, UncertaintyMetric
from graph_uq.model.base import BaseModel
from graph_uq.model.prediction import Prediction


class BaseUncertaintyModel:
    """Base class for uncertainty models."""

    def __init__(self, diffusion: dict[str, Any] = {}):
        self.diffusion_config = diffusion

    @jaxtyped(typechecker=typechecked)
    def get_diffused_uncertainties(
        self,
        data: Data,
        uncertainty: Float[Tensor, " num_nodes"],
    ) -> dict[Metric, Float[Tensor, " num_nodes"]]:
        """Diffuses the uncertainty according to a fixed type."""
        result = {}
        for diffusion_name, diffusion_config in self.diffusion_config.items():
            diffusion_config = deepcopy(diffusion_config)
            steps = diffusion_config.pop("steps")
            result |= {
                Metric(
                    uncertainty_diffusion_num_steps=k,
                    uncertainty_diffusion=diffusion_name,
                ): diffused_uncertainty.squeeze(-1)
                for k, diffused_uncertainty in enumerate(
                    data.diffusion(
                        uncertainty.unsqueeze(-1), max(steps) + 1, **diffusion_config
                    )
                )
                if k > 0 and k in steps
            }
        return result

    def fit(self, data: Data, model: BaseModel, prediction: Prediction):
        """Fits the uncertainty model."""
        ...

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, data: Data, model: BaseModel, prediction: Prediction
    ) -> dict[Metric, Float[Tensor, " num_nodes"]]:
        """Computes the uncertainty estimates of this model.

        Args:
            data (Data): The data for which to make uncertainty estimates
            model (BaseModel): The model for which to make uncertainty estimates
            prediction (Prediction): The prediction of the model on the data. For most estimators, this should be used instead using `model` and `data` directly.

        Returns:
            dict[Metric, Float[Tensor, ' num_nodes']]: The uncertainty estimates
        """
        raise NotImplementedError

    @jaxtyped(typechecker=typechecked)
    def ood_detection(
        self,
        mask: Bool[Tensor, " num_nodes"],
        data: Data,
        model: BaseModel,
        prediction: Prediction,
        uncertainty: Float[Tensor, " num_nodes"],
    ) -> EvaluationResult:
        """Evaluates ood detection performance of this model for a subset of nodes.

        Args:
            mask (Bool[Tensor, ' num_nodes']): The mask of nodes for which to evaluate uncertainty estimates
            data (Data): The data for which to evaluate uncertainty estimates
            model (BaseModel): The model for which to make uncertainty estimates
            prediction (Prediction): The prediction of the model on the data. For most estimators, this should be used instead using `model` and `data` directly.
            uncertainty (Float[Tensor, ' num_nodes']): The uncertainty estimates

        Returns:
            EvaluationResult: The uncertainty metrics
        """
        # ood detection
        is_ood = data.get_distribution_mask(DistributionType.OOD)[mask]
        metrics = binary_classification(uncertainty[mask], is_ood)
        return EvaluationResult(
            metrics=metrics,
            # plots={
            #     Metric(plot_type=PlotType.UNCERTAINTY_DISTRIBUTION) : plot_uncertainty_distribution(
            #         uncertainty[mask].detach().numpy(), is_ood.long().detach().numpy(),
            #         label_names={1 : 'Out-of-distribution', 0 : 'In-distribution'},
            #         title=', '.join((f'{key} : {value:.4f}' for key, value in metrics.items())),
            #         ),
            #     }
        ).extend_metrics(
            Metric(distribution_shift_metric=UncertaintyMetric.OOD_DETECTION)
        )

    @jaxtyped(typechecker=typechecked)
    def misclassification_detection(
        self,
        mask: Bool[Tensor, " num_nodes"],
        data: Data,
        model: BaseModel,
        prediction: Prediction,
        uncertainty: Float[Tensor, " num_nodes"],
    ) -> EvaluationResult:
        """Evaluates misclassification detection performance of this model for a subset of nodes.

        Args:
            mask (Bool[Tensor, ' num_nodes']): The mask of nodes for which to evaluate uncertainty estimates
            data (Data): The data for which to evaluate uncertainty estimates
            model (BaseModel): The model for which to make uncertainty estimates
            prediction (Prediction): The prediction of the model on the data. For most estimators, this should be used instead using `model` and `data` directly.
            uncertainty (Float[Tensor, ' num_nodes']): The uncertainty estimates

        Returns:
            EvaluationResult: The uncertainty metrics
        """
        metrics = {}
        for targets_propagated in (True,):  # omit False for now
            predicted_labels = prediction.get_predictions(propagated=targets_propagated)
            if predicted_labels is not None:
                misclassified = predicted_labels != data.y
                for distribution_type in DistributionType:
                    distribution_shift_mask = mask & data.get_distribution_mask(
                        distribution_type
                    )
                    metrics |= {
                        metric
                        + Metric(
                            targets_propagated=targets_propagated,
                            distribution_shift_metric=UncertaintyMetric.MISCLASSIFICATION_DETECTION,
                            dataset_distribution=distribution_type,
                        ): value
                        for metric, value in binary_classification(
                            uncertainty[distribution_shift_mask],
                            misclassified[distribution_shift_mask],
                        ).items()
                    }

        return EvaluationResult(metrics=metrics)

    @jaxtyped(typechecker=typechecked)
    def evaluate_mask(
        self,
        mask: Bool[Tensor, " num_nodes"],
        data: Data,
        model: BaseModel,
        prediction: Prediction,
        uncertainty: Float[Tensor, " num_nodes"],
    ) -> EvaluationResult:
        """Evaluates the uncertainty estimates of this model for a subset of nodes.

        Args:
            mask (Bool[Tensor, ' num_nodes']): The mask of nodes for which to evaluate uncertainty estimates
            data (Data): The data for which to evaluate uncertainty estimates
            model (BaseModel): The model for which to make uncertainty estimates
            prediction (Prediction): The prediction of the model on the data. For most estimators, this should be used instead using `model` and `data` directly.
            uncertainty (Float[Tensor, ' num_nodes']): The uncertainty estimates

        Returns:
            EvaluationResult: The uncertainty metrics
        """
        return self.ood_detection(
            mask, data, model, prediction, uncertainty
        ) + self.misclassification_detection(mask, data, model, prediction, uncertainty)

    @jaxtyped(typechecker=typechecked)
    def evaluate(
        self,
        data: Data,
        model: BaseModel,
        prediction: Prediction,
        uncertainty: Float[Tensor, " num_nodes"],
    ) -> EvaluationResult:
        """Evaluates the uncertainty estimates of this model.

        Args:
            data (Data): The data for which to evaluate uncertainty estimates
            model (BaseModel): The model for which to make uncertainty estimates
            prediction (Prediction): The prediction of the model on the data. For most estimators, this should be used instead using `model` and `data` directly.
            uncertainty (Float[Tensor, ' num_nodes']): The uncertainty estimates

        Returns:
            EvaluationResult: The uncertainty metrics
        """
        result = EvaluationResult()
        result.uncertainties[Metric()] = uncertainty.clone()

        for split in DatasetSplit:
            mask = data.get_mask(split)
            result += self.evaluate_mask(
                mask, data, model, prediction, uncertainty
            ).extend_metrics(Metric(dataset_split=split))

        for metric, uncertainty_diffused in self.get_diffused_uncertainties(
            data, uncertainty
        ).items():
            result.uncertainties[metric] = uncertainty_diffused
            for split in DatasetSplit:
                mask = data.get_mask(split)
                result += self.evaluate_mask(
                    mask, data, model, prediction, uncertainty_diffused
                ).extend_metrics(Metric(dataset_split=split) + metric)

        return result
