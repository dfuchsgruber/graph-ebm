from pathlib import Path
from typing import TypedDict

from graph_uq.experiment import experiment

from . import default  # noqa: F401, F81
from .data import DataConfig
from .evaluation import EvaluationConfig
from .logging import LoggingConfig
from .model import ModelConfig
from .plot import PlottingConfig
from .trainer import TrainerConfig
from .uncertainty import UncertaintiesConfig


class Config(TypedDict):
    data: DataConfig
    trainer: TrainerConfig
    evaluation: EvaluationConfig
    plot: PlottingConfig
    logging: LoggingConfig
    model: ModelConfig
    uncertainty: UncertaintiesConfig

    # seml
    db_collection: str | None
    overwrite: str | None


@experiment.config
def default_config(logging: LoggingConfig):
    db_collection = overwrite = None  # noqa: F841
    output_base_dir = str(Path("runs") / str(db_collection) / str(overwrite))  # noqa: F841

    if logging["wandb"]["name"] is None:
        logging["wandb"]["name"] = f"{db_collection}_{overwrite}"

    logging["output_dir"] = str(Path(output_base_dir) / "logs")
