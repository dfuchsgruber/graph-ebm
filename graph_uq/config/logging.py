from enum import StrEnum, unique
from pathlib import Path
from typing import Any, TypedDict

from graph_uq.experiment import experiment


@unique
class LoggerType(StrEnum):
    WANDB = "wandb"
    TINYDB = "tinydb"
    MEMORY = "memory"


class WandbConfig(TypedDict):
    id: str | None
    entity: str | None
    project: str
    group: str
    mode: str | None
    name: str | None
    tags: list[str] | None
    dir: str | None
    log_internal_dir: str
    cache_dir: str


class LoggingConfig(TypedDict):
    bar_plots: dict[str, Any]
    embeddings: dict[str, Any]
    output_dir: str
    logger: LoggerType | str
    wandb: WandbConfig


default_wandb_config = WandbConfig(  # noqa: F841
    id=None,
    entity=None,
    project="graph_uq_inductive_biases",
    group="test",
    mode=None,
    name=None,
    tags=None,
    dir=None,
    log_internal_dir=str(
        Path("/nfs/staff-ssd/fuchsgru/wandb/null")
    ),  # this is a dummy directory that links to /dev/null to not save the internal logs
    cache_dir=str(Path("wandb_cache")),
)

default_evaluation_config = LoggingConfig(
    bar_plots=dict(),  # | log_default_classification() | log_default_misclassification_data_train(),
    embeddings=dict(),
    logger=LoggerType.MEMORY,
    output_dir=str(Path(".")),
    wandb=default_wandb_config,
)


@experiment.config
def _default_evaluation_config():
    logging = default_evaluation_config
