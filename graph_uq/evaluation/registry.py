from pathlib import Path
from typing import Any, NamedTuple

import torch

from graph_uq.data.registry import DatasetRegistryKey
from graph_uq.evaluation.result import EvaluationResult
from graph_uq.experiment import experiment
from graph_uq.metric import Metric
from graph_uq.model.registry import ModelRegistryKey
from graph_uq.registry import StorageRegistry
from graph_uq.util.seml import to_json_serializable


class EvaluationRegistryKey(NamedTuple):
    name: str


class CachedEvaluationResult(NamedTuple):
    result: EvaluationResult
    config: dict[str, Any]


class EvaluationRegistry(
    StorageRegistry[EvaluationRegistryKey, CachedEvaluationResult]
):
    def generate_path(self, key: EvaluationRegistryKey) -> Path:
        return self.storage_path / f"{self.key_fn(key)}.pt"

    @classmethod
    def key_fn(cls, key: EvaluationRegistryKey) -> str:
        return "_".join(map(str, key))

    @experiment.capture(prefix="evaluation.registry")  # type: ignore
    def __init__(self, database_path: str, lockfile_path: str, storage_path: str):
        super().__init__(database_path, lockfile_path, storage_path, self.key_fn)

    def serialize(self, value: CachedEvaluationResult, path: Path):
        torch.save(
            (
                {
                    "metrics": {
                        metric.serialize(): v
                        for metric, v in value.result.metrics.items()
                    },
                    "uncertainties": {
                        metric.serialize(): v
                        for metric, v in value.result.uncertainties.items()
                    },
                    "dataset_node_keys": value.result.dataset_node_keys,
                    "masks": {
                        metric.serialize(): v
                        for metric, v in value.result.masks.items()
                    },
                },
                to_json_serializable(value.config),
            ),
            path,
        )

    def deserialize(self, path: Path) -> CachedEvaluationResult:
        data, config = torch.load(path)
        return CachedEvaluationResult(
            result=EvaluationResult(
                metrics={Metric.deserialize(k): v for k, v in data["metrics"].items()},
                dataset_node_keys=data["dataset_node_keys"],
                masks={Metric.deserialize(k): v for k, v in data["masks"].items()},
                uncertainties={
                    Metric.deserialize(k): v for k, v in data["uncertainties"].items()
                },
            ),
            config=config,
        )
