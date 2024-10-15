from typeguard import typechecked

from graph_uq.config.trainer import TrainerConfig, TrainerType
from graph_uq.trainer.base import BaseTrainer
from graph_uq.trainer.bayesian_sgd import BayesianSGDTrainer
from graph_uq.trainer.gpn import GPNTrainer
from graph_uq.trainer.none import NoTrainer
from graph_uq.trainer.sgd import SGDTrainer
from graph_uq.trainer.sgnn import SGNNTrainer


@typechecked
def get_trainer(config: TrainerConfig, *args, **kwargs) -> BaseTrainer:
    """Get a trainer.

    Args:
        type_ (str): The type of trainer to get

    Returns:
        BaseTrainer: The trainer
    """
    match TrainerType(config["type_"]):
        case TrainerType.SGD:
            return SGDTrainer(config, *args, **kwargs)
        case TrainerType.GPN:
            return GPNTrainer(config, *args, **kwargs)
        case TrainerType.NONE:
            return NoTrainer(config, *args, **kwargs)
        case TrainerType.BayesianSGD:
            return BayesianSGDTrainer(config, *args, **kwargs)
        case TrainerType.SGNN:
            return SGNNTrainer(config, *args, **kwargs)
        case type_:
            raise ValueError(f"Unknown trainer type: {type_}")
