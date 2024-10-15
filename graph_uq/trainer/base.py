from typing import Iterable

import torch
from jaxtyping import Shaped, jaxtyped
from torch import Tensor
from typeguard import typechecked

from graph_uq.config.trainer import TrainerConfig
from graph_uq.data.data import Data
from graph_uq.evaluation.classificiation import evaluate_classification
from graph_uq.logging.logger import Logger
from graph_uq.metric import DatasetSplit, Metric
from graph_uq.model.base import BaseModel
from graph_uq.model.prediction import Prediction


class BaseTrainer:
    """Base class for model training."""

    def __init__(self, config: TrainerConfig, *args, **kwargs):
        self.verbose = config["verbose"]

    @torch.no_grad()
    @jaxtyped(typechecker=typechecked)
    def any_steps(
        self,
        which: Iterable[str],
        batch: Data,
        prediction: Prediction,
        epoch_idx: int,
        *args,
        **kwargs,
    ) -> dict[Metric, float | int | Shaped[torch.Tensor, ""]]:
        """Computes classification metrics on a sequence of splits.

        Args:
            batch (Data): the batch to predict for
            prediction (Prediction): the model predictions made on this batch
            epoch_idx (int): which epoch

        Returns:
            dict[Metric, float | int | Shaped[torch.Tensor, '']]: test metrics
        """
        metrics = dict()
        for split in which:
            mask = batch.get_mask(split)
            for metric, value in evaluate_classification(
                batch, prediction, mask, batch.num_classes_train
            ).items():
                metrics |= {metric + Metric(dataset_split=DatasetSplit(split)): value}
        return metrics

    @jaxtyped(typechecker=typechecked)
    def fit(
        self, model: BaseModel, data: Data, logger: Logger, *args, **kwargs
    ) -> dict[Metric, list[float | int | Shaped[Tensor, ""]]]:
        """Fits the model to a dataset.

        Args:
            model (BaseModel): The model to fit
            data (Data): The dataset to fit to
            logger (Logger): The logger to use
        """
        return {}

    def transfer_model_to_device(self, model: BaseModel) -> BaseModel:
        return model

    def transfer_data_to_device(self, data: Data) -> Data:
        return data
