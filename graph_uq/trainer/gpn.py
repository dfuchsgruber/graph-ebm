from typing import Iterable

import torch
import torch.nn.functional as F
from jaxtyping import Float, Shaped, jaxtyped
from torch import Tensor
from tqdm import tqdm
from typeguard import typechecked

from graph_uq.config.data import DatasetSplit
from graph_uq.config.trainer import GPNWarmup, TrainerConfig
from graph_uq.data import Data
from graph_uq.logging.logger import Logger
from graph_uq.metric import Metric, MetricName
from graph_uq.model.gpn import GraphPosteriorNetwork
from graph_uq.model.prediction import Prediction
from graph_uq.trainer.loss import entropy_regularization, uce_loss
from graph_uq.trainer.sgd import SGDTrainer


class GPNTrainer(SGDTrainer):
    """Trainer to train a GPN model"""

    def __init__(self, config: TrainerConfig):
        super().__init__(config)
        self.learning_rate_flow = config["learning_rate_flow"]
        self.weight_decay_flow = config["weight_decay_flow"]
        self.entropy_regularization_loss_weight = config[
            "entropy_regularization_loss_weight"
        ]
        self.warmup = config["warmup"]
        self.num_warmup_epochs = config["num_warmup_epochs"]
        self.learning_rate_warmup = config["learning_rate_warmup"]
        self.weight_decay_warmup = config["weight_decay_warmup"]

    @typechecked
    def get_optimizer(self, model: GraphPosteriorNetwork) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            [
                {
                    "params": model.flow_parameters,
                    "lr": self.learning_rate_flow,
                    "weight_decay": self.weight_decay_flow,
                },
                {
                    "params": model.non_flow_parameters,
                    "lr": self.learning_rate,
                    "weight_decay": self.weight_decay,
                },
            ]
        )

    @typechecked
    def get_warmup_flow_optimizer(
        self,
        model: GraphPosteriorNetwork,
    ) -> torch.optim.Optimizer:
        """Gets the optimizer for the flow component of the model"""
        return torch.optim.Adam(
            model.flow_parameters,
            lr=self.learning_rate_warmup,
            weight_decay=self.weight_decay_warmup,
        )

    @jaxtyped(typechecker=typechecked)
    def entropy_regularization(
        self,
        batch: Data,
        prediction: Prediction,
        which: DatasetSplit,
        loss_weights: Float[Tensor, "mask_size"] | None = None,
        approximate: bool = True,
    ) -> Float[Tensor, ""]:
        """Computes the regularizer for the entropy of the outputted distribution."""
        mask = batch.get_mask(which) & (batch.y < prediction.num_classes)
        alpha = prediction.alpha
        assert alpha is not None and alpha.size(0) == 1
        alpha = alpha.mean(0)
        reg = entropy_regularization(alpha[mask], approximate=approximate)
        if loss_weights is not None:
            reg *= loss_weights
        return reg.sum()

    @jaxtyped(typechecker=typechecked)
    def uce_loss(
        self,
        batch: Data,
        prediction: Prediction,
        epoch_idx: int,
        which: DatasetSplit,
        loss_weights: Float[Tensor, "mask_size"] | None = None,
    ) -> Float[Tensor, ""]:
        """Computes the uncertainty cross entropy loss (UCE loss)"""
        alpha = prediction.alpha
        assert alpha is not None and alpha.size(0) == 1
        alpha = alpha.mean(0)
        mask = batch.get_mask(which) & (batch.y < prediction.num_classes)
        uce = uce_loss(alpha[mask], batch.y[mask])
        if loss_weights is not None:
            uce *= loss_weights
        return uce.sum()

    @jaxtyped(typechecker=typechecked)
    def loss(
        self,
        batch: Data,
        prediction: Prediction,
        epoch_idx: int,
        which: DatasetSplit,
        loss_weights: Float[Tensor, "mask_size"] | None = None,
        approximate: bool = True,
    ) -> dict[Metric, float | int | Shaped[Tensor, ""]]:
        """Computes the entire training objective, i.e. UCE loss and entropy regularization."""
        uce_loss = self.uce_loss(
            batch, prediction, epoch_idx, which, loss_weights=loss_weights
        )
        entropy_reg = self.entropy_regularization(
            batch, prediction, which, loss_weights=loss_weights, approximate=approximate
        )
        total_loss = uce_loss + self.entropy_regularization_loss_weight * entropy_reg
        return {
            Metric(
                name=MetricName.LOSS, dataset_split=which, propagated=True
            ): total_loss,
            Metric(
                name=MetricName.UCE_LOSS, dataset_split=which, propagated=True
            ): uce_loss,
            Metric(
                name=MetricName.ENTROPY_REGULARIZATION_LOSS,
                dataset_split=which,
                propagated=True,
            ): entropy_reg,
        }

    def cross_entropy_loss(
        self,
        batch: Data,
        prediction: Prediction,
        epoch_idx: int,
        which: DatasetSplit,
        loss_weights: Float[Tensor, "mask_size"] | None = None,
    ) -> dict[Metric, float | int | Shaped[Tensor, ""]]:
        """Computes the normal cross entropy loss when only warming up the encoder"""
        probabilities = prediction.get_probabilities(propagated=True)
        assert probabilities is not None and probabilities.size(0) == 1
        log_probs = probabilities.mean(0).log()  # type: ignore
        mask = batch.get_mask(which) & batch.y < prediction.num_classes
        ce_loss = F.nll_loss(log_probs[mask], batch.y[mask], reduction="none")
        if loss_weights is not None:
            ce_loss *= loss_weights
        ce_loss = ce_loss.mean()
        return {
            Metric(name=MetricName.LOSS, dataset_split=which): ce_loss,
        }

    @jaxtyped(typechecker=typechecked)
    def warmup_step(
        self,
        batch: Data,
        prediction: Prediction,
        epoch_idx: int,
        which: DatasetSplit = DatasetSplit.TRAIN,
        loss_weights: Float[Tensor, "mask_size"] | None = None,
    ) -> dict[Metric, float | int | Shaped[Tensor, ""]]:
        """One step of warmup training"""
        match self.warmup:
            case GPNWarmup.ENCODER:
                return self.cross_entropy_loss(
                    batch, prediction, epoch_idx, which=which, loss_weights=loss_weights
                )
            case GPNWarmup.FLOW:
                return self.loss(
                    batch, prediction, epoch_idx, which=which, loss_weights=loss_weights
                )
            case _:
                raise ValueError(self.warmup)

    @jaxtyped(typechecker=typechecked)
    def warmup_epoch_loop(
        self,
        model: GraphPosteriorNetwork,
        batch: Data,
        epoch_idx: int,
        optimizer: torch.optim.Optimizer,
    ) -> dict[Metric, float | int | Shaped[Tensor, ""]]:
        """Performs one epoch of warm-up training"""
        warmup_metrics = dict()
        model = model.train()

        optimizer.zero_grad()
        prediction: Prediction = model.predict(batch)
        warmup_metrics |= self.warmup_step(batch, prediction, epoch_idx)
        loss = warmup_metrics[
            Metric(
                name=MetricName.LOSS, dataset_split=DatasetSplit.TRAIN, propagated=True
            )
        ]
        loss.backward()
        optimizer.step()

        model = model.eval()
        warmup_metrics |= self.warmup_step(
            batch, prediction, epoch_idx, which=DatasetSplit.VAL
        )

        return warmup_metrics

    @typechecked
    def setup_warmup_epoch_iterator(
        self,
    ) -> Iterable[int]:
        self.progress_bar = tqdm(
            range(self.num_warmup_epochs), disable=not self.has_progress_bar
        )
        return self.progress_bar

    @jaxtyped(typechecker=typechecked)
    def fit(
        self,
        model: GraphPosteriorNetwork,
        data: Data,
        logger: Logger,
    ):
        """Performs training of a model with SGD over different epochs.

        Args:
            batch (Data): The data on which to train on
            training_config (SGDTrainingConfig): Configuration for training

        """
        model, data = (
            self.transfer_model_to_device(model),
            self.transfer_data_to_device(data),
        )  # type: ignore

        # Warmup training for the flow
        match self.warmup:
            case GPNWarmup.FLOW:
                warmup_optimizer = self.get_warmup_flow_optimizer(model)
            case GPNWarmup.ENCODER:
                warmup_optimizer = self.get_optimizer(model)
            case _:
                raise ValueError(f"Warmup type {self.warmup} not supported.")

        for warmup_epoch in self.setup_warmup_epoch_iterator():
            self.warmup_epoch = warmup_epoch
            warmup_metrics = self.warmup_epoch_loop(
                model, data, warmup_epoch, warmup_optimizer
            )
            loss = warmup_metrics.get(
                Metric(name=MetricName.LOSS, dataset_split=DatasetSplit.TRAIN), None
            )
            if loss:
                self.update_progress_bar(
                    "Warmup: "
                    + ", ".join(
                        f"{key} : {value:.3f}"
                        for key, value in {
                            "train loss": warmup_metrics.get(
                                Metric(
                                    name=MetricName.LOSS,
                                    dataset_split=DatasetSplit.TRAIN,
                                ),
                                None,
                            ),
                            "val loss": warmup_metrics.get(
                                Metric(
                                    name=MetricName.LOSS, dataset_split=DatasetSplit.VAL
                                ),
                                None,
                            ),
                        }.items()
                    )
                )

            warmup_metrics_to_log = {
                f"warmup/{key}": value for key, value in warmup_metrics.items()
            }
            self.log(
                logger,
                warmup_metrics_to_log,
                self.warmup_epoch,
            )

        # Normal training using SGD: `self.loss` is overridden with the UCE loss that is used to train GPN
        return super().fit(model, data, logger)
