import logging
from collections import defaultdict
from copy import deepcopy
from typing import Iterable

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int, Shaped, jaxtyped
from torch import Tensor
from tqdm import tqdm
from typeguard import typechecked

from graph_uq.config.data import DatasetSplit
from graph_uq.config.trainer import (
    LossFunctionType,
    TrainerConfig,
)
from graph_uq.data.data import Data
from graph_uq.logging.logger import Logger
from graph_uq.metric import Metric, MetricName
from graph_uq.model.base import BaseModel
from graph_uq.model.prediction import Prediction
from graph_uq.trainer.base import BaseTrainer
from graph_uq.trainer.early_stopping import EarlyStopping
from graph_uq.trainer.edge_reconstruction import EdgeReconstructionTrainerMixin
from graph_uq.util.tensor import apply_to_nested_tensors


class SGDTrainer(EdgeReconstructionTrainerMixin, BaseTrainer):
    def __init__(self, config: TrainerConfig):
        BaseTrainer.__init__(self, config)
        EdgeReconstructionTrainerMixin.__init__(
            self, config["edge_reconstruction_loss"]
        )
        self.learning_rate = config["learning_rate"]
        self.weight_decay = config["weight_decay"]
        self.early_stopping_config = deepcopy(config["early_stopping"])
        self.max_epochs = config["max_epochs"]
        self.min_epochs = config["min_epochs"]
        self.has_progress_bar = config["progress_bar"]
        self.use_gpu = config["use_gpu"]
        self.log_every_epoch = config["log_every_epoch"]
        self.commit_every_epoch = config["commit_every_epoch"]
        self.train_with_propagated_prediction = config[
            "train_with_propagated_prediction"
        ]
        self.loss_function_type = config["loss_function_type"]

    @typechecked
    def get_optimizer(self, model: BaseModel) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def setup_early_stopping(self):
        self.early_stopping = EarlyStopping(self.early_stopping_config)

    @typechecked
    def should_stop(self, epoch_idx: int) -> bool:
        if epoch_idx < self.min_epochs:
            return False
        elif epoch_idx >= self.max_epochs:
            return True
        else:
            return self.early_stopping.should_stop

    @jaxtyped(typechecker=typechecked)
    def get_loss_weights(
        self, labels: Int[Tensor, " num_nodes"]
    ) -> Float[Tensor, " num_nodes"]:
        return torch.ones_like(labels, dtype=torch.float)

    @typechecked
    def setup_epoch_iterator(self) -> Iterable[int]:
        self.progress_bar = tqdm(
            range(self.max_epochs), disable=not self.has_progress_bar
        )
        return self.progress_bar

    @typechecked
    def update_progress_bar(self, message: str):
        if self.has_progress_bar:
            self.progress_bar.set_description(message)

    @typechecked
    def transfer_data_to_device(self, data: Data) -> Data:
        if self.use_gpu and torch.cuda.is_available():
            data = data.cuda()
        return data

    @typechecked
    def transfer_model_to_device(self, model: BaseModel) -> BaseModel:
        if self.use_gpu and torch.cuda.is_available():
            model = model.cuda()
        return model

    @typechecked
    def should_log_in_epoch(self, epoch_idx: int) -> bool:
        if self.log_every_epoch is None or self.log_every_epoch <= 0:
            return False
        else:
            return (epoch_idx % self.log_every_epoch) == 0

    @jaxtyped(typechecker=typechecked)
    def log(
        self,
        logger: Logger,
        metrics: dict[str, float | int | Shaped[Tensor, ""]],
        epoch_idx: int,
    ):
        if self.commit_every_epoch is None or self.commit_every_epoch < 1:
            commit = False
        else:
            commit = epoch_idx % self.commit_every_epoch == 0
        # Note that the values that are not commited may never be logged
        if self.should_log_in_epoch(epoch_idx):
            logger.log(metrics, step=epoch_idx, commit=commit)

    @jaxtyped(typechecker=typechecked)
    def loss(
        self,
        batch: Data,
        prediction: Prediction,
        epoch_idx: int,
        which: str,
        *args,
        loss_weights: Float[Tensor, "mask_size"] | None = None,
        **kwargs,
    ) -> dict[Metric, float | int | Shaped[Tensor, ""]]:
        """Computes the entire training objective"""
        logits = prediction.get_logits(propagated=self.train_with_propagated_prediction)
        assert logits is not None, "Can not compute loss without logits"
        if logits.size(0) > 1 and logits.grad is not None:
            logging.warn(
                f"Using average logits (over {logits.size(0)} samples) for loss function. This is not a good idea..."
            )
        logits = logits.mean(0)  # average over samples
        mask = batch.get_mask(which)
        match LossFunctionType(self.loss_function_type):
            case LossFunctionType.CROSS_ENTROPY:
                loss = F.cross_entropy(logits[mask], batch.y[mask], reduction="none")
                if loss_weights is not None:
                    loss *= loss_weights
                loss = loss.mean()
            case type_:
                raise ValueError(f"Unknown loss function type {type_}")

        result = {}

        if self.has_edge_reconstruction_loss:
            edge_reconstruction_loss = self.edge_reconstruction_loss(batch, prediction)
            result |= {
                Metric(
                    name=MetricName.EDGE_RECONSTRUCTION_LOSS,
                    dataset_split=DatasetSplit(which),
                ): edge_reconstruction_loss
            }
            loss += edge_reconstruction_loss * self.edge_reconstruction_loss_weight

        return result | {
            Metric(
                name=MetricName.LOSS,
                dataset_split=DatasetSplit(which),
                propagated=self.train_with_propagated_prediction,
            ): loss
        }

    @jaxtyped(typechecker=typechecked)
    def any_loss_step(
        self,
        batch: Data,
        prediction: Prediction,
        epoch_idx: int,
        which: str,
        *args,
        loss_weights: Float[Tensor, "mask_size"] | None = None,
        **kwargs,
    ) -> dict[Metric, float | int | Shaped[Tensor, ""]]:
        """Calculates the loss for a prediction."""
        return self.loss(
            batch,
            prediction,
            epoch_idx,
            which,
            *args,
            loss_weights=loss_weights,
            **kwargs,
        )

    @jaxtyped(typechecker=typechecked)
    def any_steps(
        self,
        which: Iterable[DatasetSplit],
        batch: Data,
        prediction: Prediction,
        epoch_idx: int,
        *args,
        **kwargs,
    ) -> dict[Metric, float | int | Shaped[torch.Tensor, ""]]:
        """Performs a step on sequences of dataset splits.

        Args:
            which (DatasetSplit): on which split to perform the step
            batch (Data): the batch to predict for
            prediction (Prediction): the model predictions made on this batch
            epoch_idx (int): which epoch

        Returns:
            dict[Metric, float | int | Shaped[torch.Tensor, '']]: train metrics, including the loss
        """
        metrics = super().any_steps(
            which, batch, prediction, epoch_idx, *args, **kwargs
        )
        for split in which:
            loss_weights = self.get_loss_weights(batch.y[batch.get_mask(split)])
            with torch.set_grad_enabled(split == DatasetSplit.TRAIN):
                metrics |= self.any_loss_step(
                    batch,
                    prediction,
                    epoch_idx,
                    split,
                    *args,
                    loss_weights=loss_weights,
                    **kwargs,
                )
        return metrics

    @jaxtyped(typechecker=typechecked)
    def epoch_loop(
        self,
        model: BaseModel,
        batch: Data,
        epoch_idx: int,
        optimizer: torch.optim.Optimizer,
        *args,
        **kwargs,
    ) -> dict[Metric, float | int | Shaped[Tensor, ""]]:
        """Overridable iteration of one training epoch.

        Args:
            model (BaseModel): the model to train
            data (Data): the dataset on which to train
            epoch_idx (int): in which epoch
            optimizer (torch.optim.Optimizer): the optimizer

        Returns:
            dict[Metric, float | int | Shaped[Tensor, '']]: Metrics for this epoch
        """
        epoch_metrics = dict()
        model = model.train()

        optimizer.zero_grad()
        prediction: Prediction = model.predict(batch)
        epoch_metrics |= self.any_steps(
            [DatasetSplit.TRAIN], batch, prediction, epoch_idx, *args, **kwargs
        )
        loss = epoch_metrics[
            Metric(
                name=MetricName.LOSS, dataset_split=DatasetSplit.TRAIN, propagated=True
            )
        ]
        loss.backward()
        optimizer.step()

        model = model.eval()
        if model.prediction_changes_at_eval:
            with torch.no_grad():
                prediction = model.predict(
                    batch, num_samples=1
                )  # we only use one sample for evaluating during training, TODO: make this configurable
        epoch_metrics |= self.any_steps(
            [split for split in DatasetSplit if split != DatasetSplit.TRAIN],
            batch,
            prediction,
            epoch_idx,
            *args,
            **kwargs,
        )
        return apply_to_nested_tensors(
            epoch_metrics, lambda tensor: tensor.detach().cpu()
        )

    @jaxtyped(typechecker=typechecked)
    def fit(
        self, model: BaseModel, data: Data, logger: Logger, *args, **kwargs
    ) -> dict[Metric, list[float | int | Shaped[Tensor, ""]]]:
        """Performs training of a model with SGD over different epochs.

        Args:
            model (BaseModel): the model to train
            data (Data): the dataset on which to train
        """
        if self.verbose:
            logging.info(f"Training on {data}")
        model, data = (
            self.transfer_model_to_device(model),
            self.transfer_data_to_device(data),
        )

        optimizer = self.get_optimizer(model)
        self.setup_early_stopping()

        epoch_metrics = defaultdict(list)
        epoch_iterator = self.setup_epoch_iterator()

        for epoch_idx in epoch_iterator:
            self.current_epoch = epoch_idx
            metrics = self.epoch_loop(
                model, data, self.current_epoch, optimizer, *args, **kwargs
            )
            for name, value in metrics.items():
                epoch_metrics[name].append(value)
            loss = metrics.get(
                Metric(name=MetricName.LOSS, dataset_split=DatasetSplit.TRAIN), None
            )
            if loss:
                self.update_progress_bar(
                    ", ".join(
                        f"{key} : {value:.3f}"
                        for key, value in {
                            "train loss": metrics.get(
                                Metric(
                                    name=MetricName.LOSS,
                                    dataset_split=DatasetSplit.TRAIN,
                                ),
                                None,
                            ),
                            "val loss": metrics.get(
                                Metric(
                                    name=MetricName.LOSS, dataset_split=DatasetSplit.VAL
                                ),
                                None,
                            ),
                            "train acc": metrics.get(
                                Metric(
                                    name=MetricName.ACCURACY,
                                    dataset_split=DatasetSplit.TRAIN,
                                ),
                                None,
                            ),
                            "val acc": metrics.get(
                                Metric(
                                    name=MetricName.ACCURACY,
                                    dataset_split=DatasetSplit.VAL,
                                ),
                                None,
                            ),
                        }.items()
                    )
                )

            self.log(
                logger, {str(k): v for k, v in metrics.items()}, self.current_epoch
            )
            self.early_stopping.step(metrics, self.current_epoch, model)
            if self.should_stop(self.current_epoch):
                if self.verbose:
                    logging.info(f"Early stopping after {self.current_epoch} epochs.")
                break

        # Restore the best model
        if self.early_stopping.best_state is not None:
            model.load_state_dict(self.early_stopping.best_state)

        return dict(epoch_metrics)
