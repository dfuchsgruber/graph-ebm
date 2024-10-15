import logging
from copy import deepcopy

import torch.distributions as D
import torch.nn.functional as F
from jaxtyping import Float, Shaped, jaxtyped
from torch import Tensor
from typeguard import typechecked

from graph_uq.config.trainer import LossFunctionType, TrainerConfig
from graph_uq.data.data import Data
from graph_uq.logging.logger import Logger
from graph_uq.metric import DatasetSplit, Metric, MetricName
from graph_uq.model.prediction import Prediction
from graph_uq.model.sgnn import SGNN
from graph_uq.trainer.sgd import SGDTrainer


class SGNNTrainer(SGDTrainer):
    """Trainer that trains a SGNN."""

    def __init__(self, config: TrainerConfig):
        super().__init__(config)
        self.kl_divergence_loss_weight = config["kl_divergence_loss_weight"]
        self.teacher_config = deepcopy(config["teacher"])

    def bayesian_risk(
        self, batch: Data, prediction: Prediction, which: str
    ) -> Float[Tensor, "mask_size"]:
        """Computes the Bayesian risk loss according to [1], Equation(9).

        Args:
            batch (Data): The batch of data.
            prediction (Prediction): The prediction.
            which (str): The dataset split.

        Returns:
            Float[Tensor, 'mask_size']: The Bayesian risk loss.

        [1] https://proceedings.neurips.cc/paper_files/paper/2020/file/968c9b4f09cbb7d7925f38aea3484111-Paper.pdf
        """
        assert prediction.alpha is not None
        alpha = prediction.alpha.mean(0)[batch.get_mask(which)]  # average over samples
        alpha_0 = alpha.sum(-1, keepdim=True)
        probabilities = alpha / alpha_0
        labels_one_hot = F.one_hot(
            batch.y[batch.get_mask(which)], num_classes=alpha.size(-1)
        )
        squared_error = (probabilities - labels_one_hot).pow(2)
        variance = probabilities * (1 - probabilities) / (alpha_0 + 1)
        return (squared_error + variance).sum(-1)

    @jaxtyped(typechecker=typechecked)
    def loss(
        self,
        batch: Data,
        prediction: Prediction,
        epoch_idx: int,
        which: str,
        loss_weights: Float[Tensor, "mask_size"] | None = None,
    ) -> dict[Metric, float | int | Shaped[Tensor, ""]]:
        """Computes the entire training objective"""

        result = {}
        match LossFunctionType(self.loss_function_type):
            case LossFunctionType.BAYESIAN_RISK:
                loss = self.bayesian_risk(batch, prediction, which)
                if loss_weights is not None:
                    loss *= loss_weights
                loss = loss.mean()
            case _:
                raise NotImplementedError(
                    f"Loss function type {self.loss_function_type} not implemented"
                )

        if self.kl_divergence_loss_weight != 0:
            assert prediction.alpha_prior is not None
            assert prediction.alpha is not None
            prior = D.Dirichlet(
                prediction.alpha_prior[..., : prediction.alpha.size(-1)]
            )
            predicted = D.Dirichlet(prediction.alpha)
            kl_divergence = D.kl.kl_divergence(predicted, prior).mean(-1).sum()
            result[
                Metric(
                    name=MetricName.KL_DIVERGENCE,
                    dataset_split=DatasetSplit(which),
                    propagated=self.train_with_propagated_prediction,
                )
            ] = kl_divergence.item()
            loss += self.kl_divergence_loss_weight * kl_divergence

        if prediction.teacher_probabilities is not None:
            categorical_teacher = D.Categorical(prediction.teacher_probabilities)
            predicted = D.Categorical(
                prediction.get_probabilities(
                    propagated=self.train_with_propagated_prediction
                )
            )
            distillation_loss = (
                D.kl.kl_divergence(predicted, categorical_teacher).mean(-1).sum()
            )
            lambda_teacher = min(1.0, (epoch_idx + 1) / 200)
            result[
                Metric(
                    name=MetricName.DISTIALLATION_LOSS,
                    dataset_split=DatasetSplit(which),
                    propagated=self.train_with_propagated_prediction,
                )
            ] = distillation_loss.item()
            loss += lambda_teacher * distillation_loss

        return result | {
            Metric(
                name=MetricName.LOSS,
                dataset_split=DatasetSplit(which),
                propagated=self.train_with_propagated_prediction,
            ): loss
        }

    @jaxtyped(typechecker=typechecked)
    def fit(
        self, model: SGNN, data: Data, logger: Logger
    ) -> dict[Metric, list[float | int | Shaped[Tensor, ""]]]:
        """Performs training with potentiall a teacher model.

        Args:
            model (SGNN): The model to train
            data (Data): The data to train on

        Returns:
            dict[Metric, list[float | int | Shaped[Tensor, '']]]: The metrics
        """
        metrics = {}

        assert (model.teacher is None and self.teacher_config is None) or (
            model.teacher is not None and self.teacher_config is not None
        ), "Either both or none of model.teacher and teacher must be specified"
        if self.teacher_config is not None:
            assert model.teacher is not None
            from graph_uq.trainer.build import get_trainer

            trainer = get_trainer(self.teacher_config)
            if self.verbose:
                logging.info("Training teacher model")
            metrics |= {
                k + Metric(teacher="teacher"): v
                for k, v in trainer.fit(model.teacher, data, logger).items()
            }

        if self.verbose:
            logging.info("Training student model")
        metrics |= super().fit(model, data, logger)
        return metrics
