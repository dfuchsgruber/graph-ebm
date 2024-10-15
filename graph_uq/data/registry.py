from pathlib import Path
from typing import Any, NamedTuple

import torch

from graph_uq.config import Config
from graph_uq.data.build import apply_distribution_shift_and_split, get_base_data
from graph_uq.data.data import Data
from graph_uq.data.dataset import Dataset
from graph_uq.experiment import experiment
from graph_uq.registry import StorageRegistry
from graph_uq.util.sample import MaskSamplingException


class DatasetRegistryKey(NamedTuple):
    name: str
    split_idx: int


class CachedDataset(NamedTuple):
    dataset: Dataset
    config: dict[str, Any]


class DatasetRegistry(StorageRegistry[DatasetRegistryKey, CachedDataset]):
    """Class for storing pre-computed datasets and splits."""

    def __init__(self, database_path: str, lockfile_path: str, storage_path: str):
        super().__init__(
            database_path=database_path,
            lockfile_path=lockfile_path,
            storage_path=storage_path,
        )

    def serialize(self, value: CachedDataset, path: Path):
        torch.save(
            {
                "dataset": value.dataset,
                "config": value.config,
            },
            path,
        )

    def deserialize(self, path: Path) -> CachedDataset:
        deserialized = torch.load(path)
        return CachedDataset(
            dataset=deserialized["dataset"],
            config=deserialized["config"],
        )

    def get_dataset(
        self, name: str, split_idx: int, _config: dict[str, Any]
    ) -> CachedDataset:
        """Get a dataset from the registry."""
        key = DatasetRegistryKey(name=name, split_idx=split_idx)
        if key not in self:
            raise ValueError(
                f"Dataset {name} with split index {split_idx} not found in registry."
            )
        else:
            return self[key]

    def get_dataset_or_build(
        self,
        name: str,
        split_idx: int,
        config: Config,
        force_rebuild: bool = False,
        max_num_split_attempts: int = 1,
    ) -> tuple[Dataset, dict[str, Any]]:
        """Get a dataset from the registry."""
        key = DatasetRegistryKey(name=name, split_idx=split_idx)
        if key not in self or force_rebuild:
            # Build the dataset
            base_data: Data = get_base_data(config["data"])
            for _ in range(max_num_split_attempts):
                try:
                    dataset = apply_distribution_shift_and_split(
                        base_data, config["data"]
                    )
                    break
                except MaskSamplingException:
                    continue
            else:
                raise ValueError("Could not split the dataset.")
            self[key] = CachedDataset(dataset=dataset, config=dict(**config["data"]))

            return dataset, dict(**config["data"])

        else:
            return self[key]
