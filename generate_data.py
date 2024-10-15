from typing import Any

from rich.console import Console
from rich.table import Table
from typeguard import typechecked

from graph_uq.config import Config
from graph_uq.config.data import DataConfig
from graph_uq.data.registry import DatasetRegistry
from graph_uq.experiment import experiment


@experiment.config
def generate_data_config():
    force_rebuild = False  # noqa: F841


@experiment.capture()  # type: ignore
def force_rebuild(force_rebuild) -> bool:
    return force_rebuild


def get_datasets(config: Config):
    registry = DatasetRegistry(
        config["data"]["registry"]["database_path"],
        config["data"]["registry"]["lockfile_path"],
        config["data"]["registry"]["storage_path"],
    )
    assert config["data"]["precomputed"] is not None
    for split_idx in range(config["data"]["num_splits"]):
        registry.get_dataset_or_build(
            config["data"]["precomputed"],
            split_idx,
            config,
            force_rebuild=False,
        )


@experiment.command
def clean_registry(data: DataConfig):
    """Clean the registry."""
    DatasetRegistry(
        data["registry"]["database_path"],
        data["registry"]["lockfile_path"],
        data["registry"]["storage_path"],
    ).clean()


@experiment.command
def list_registry(data: DataConfig):
    """Lists the registry."""
    DatasetRegistry(
        data["registry"]["database_path"],
        data["registry"]["lockfile_path"],
        data["registry"]["storage_path"],
    ).list()


@experiment.automain
def main(_config: Config, _seed: int):
    """Generates dataset and distribution shifts for different experiments."""
    get_datasets(_config)
