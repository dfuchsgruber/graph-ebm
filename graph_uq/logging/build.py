from typeguard import typechecked

from graph_uq.config.logging import LoggerType, LoggingConfig
from graph_uq.logging.logger import Logger
from graph_uq.logging.memory import MemoryLogger
from graph_uq.logging.tinydb import TinyDBLogger
from graph_uq.logging.wandb import WandbLogger


@typechecked
def get_logger(config: LoggingConfig) -> Logger:
    match config["logger"]:
        case LoggerType.WANDB:
            return WandbLogger(config["wandb"])
        case LoggerType.TINYDB:
            return TinyDBLogger(config["output_dir"])
        case LoggerType.MEMORY:
            return MemoryLogger(config["output_dir"])
        case logger:
            raise ValueError(f"Unknown logger type {logger}")
