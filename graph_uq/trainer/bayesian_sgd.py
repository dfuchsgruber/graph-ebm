import logging

import torch.nn.functional as F
from jaxtyping import Float, Shaped, jaxtyped
from torch import Tensor
from typeguard import typechecked

from graph_uq.config.trainer import LossFunctionType, TrainerConfig
from graph_uq.data.data import Data
from graph_uq.metric import DatasetSplit, Metric, MetricName
from graph_uq.model.prediction import Prediction
from graph_uq.trainer.sgd import SGDTrainer


class BayesianSGDTrainer(SGDTrainer):
    """Trainer for Bayesian GNNs that uses SGD."""

    def __init__(self, config: TrainerConfig, *args, **kwargs):
        super().__init__(config)
        self.kl_divergence_loss_weight = config["kl_divergence_loss_weight"]

    @jaxtyped(typechecker=typechecked)
    def loss(
        self,
        batch: Data,
        prediction: Prediction,
        epoch_idx: int,
        which: str,
        loss_weights: Float[Tensor, " mask_size"] | None = None,
    ) -> dict[Metric, float | int | Shaped[Tensor, ""]]:
        """Computes the entire training objective"""
        result = {}
        logits = prediction.get_logits(propagated=self.train_with_propagated_prediction)
        assert logits is not None, "Can not compute loss without logits"
        if logits.size(0) > 1 and logits.grad is not None:
            logging.warn(
                f"Using average logits (over {logits.size(0)} samples) for loss function. This is not a good idea..."
            )
        logits = logits.mean(0)  # average over samples
        mask = batch.get_mask(which)
        match LossFunctionType(self.loss_function_type):
            case LossFunctionType.CROSS_ENTROPY_AND_KL_DIVERGENCE:
                loss = F.cross_entropy(logits[mask], batch.y[mask], reduction="none")
                if loss_weights is not None:
                    loss *= loss_weights
                loss = loss.mean()
                assert (
                    prediction.kl_divergence is not None
                ), "Can not compute loss without KL divergence"
                kl_divergence = prediction.kl_divergence.mean()
                loss += self.kl_divergence_loss_weight * kl_divergence
                result[
                    Metric(
                        name=MetricName.KL_DIVERGENCE,
                        dataset_split=DatasetSplit(which),
                        propagated=self.train_with_propagated_prediction,
                    )
                ] = kl_divergence
            case type_:
                raise ValueError(f"Unknown loss function type {type_}")

        return result | {
            Metric(
                name=MetricName.LOSS,
                dataset_split=DatasetSplit(which),
                propagated=self.train_with_propagated_prediction,
            ): loss
        }
