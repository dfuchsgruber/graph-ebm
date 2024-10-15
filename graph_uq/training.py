from jaxtyping import jaxtyped
from typeguard import typechecked

from graph_uq.config.trainer import TrainerConfig
from graph_uq.data.data import Data
from graph_uq.logging.logger import Logger
from graph_uq.metric import Metric, MetricValue
from graph_uq.model.base import BaseModel, Ensemble
from graph_uq.trainer.build import get_trainer


@jaxtyped(typechecker=typechecked)
def train_model(
    config: TrainerConfig, data: Data, model: BaseModel, logger: Logger
) -> dict[Metric, list[MetricValue]]:
    """Train a model.

    Args:
        data (Data): The data
        model (BaseModel): The model

    Returns:
        dict[Metric, list[float | int | Shaped[Tensor, '']]]: The metrics for each epoch
    """
    if isinstance(model, Ensemble):
        # Train each model in the ensemble
        for model_ in model.models:
            trainer = get_trainer(config)
            metrics = trainer.fit(model_, data, logger)
    else:
        trainer = get_trainer(config)
        metrics = trainer.fit(model, data, logger)
    return metrics
