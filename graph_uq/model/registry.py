from pathlib import Path
from typing import Any, NamedTuple

import torch
from typeguard import typechecked

from graph_uq.config import Config
from graph_uq.data.data import Data
from graph_uq.logging.logger import Logger
from graph_uq.metric import Metric, MetricValue
from graph_uq.model.base import BaseModel, Ensemble
from graph_uq.model.build import get_model
from graph_uq.registry import StorageRegistry
from graph_uq.training import train_model


class ModelRegistryKey(NamedTuple):
    name: str
    init_idx: int
    ensemble_idx: int | None


class CachedModel(NamedTuple):
    state_dict: dict[str, torch.Tensor]
    config: dict[str, Any]
    result: dict[Metric, list[MetricValue]]


class ModelRegistry(StorageRegistry[ModelRegistryKey, CachedModel]):
    """Class for storing trained models."""

    @staticmethod
    def _key_fn(key: ModelRegistryKey) -> str:
        return f"{key.name}_{key.init_idx}" + (
            "" if key.ensemble_idx is None else f"_{key.ensemble_idx}"
        )

    @typechecked
    def __init__(self, database_path: str, lockfile_path: str, storage_path: str):
        super().__init__(
            database_path=database_path,
            lockfile_path=lockfile_path,
            storage_path=storage_path,
            key_fn=self._key_fn,
        )

    def generate_path(self, key: ModelRegistryKey) -> Path:
        return self.generate_path_from_str(
            f"{key.name}_{key.init_idx}"
            + ("" if key.ensemble_idx is None else f"_{key.ensemble_idx}")
        )

    def serialize(self, value: CachedModel, path: Path):
        torch.save(
            {
                "state_dict": value.state_dict,
                "config": value.config,
                "result": value.result,
            },
            path,
        )

    def deserialize(self, path: Path) -> CachedModel:
        deserialized = torch.load(path, map_location="cpu")
        return CachedModel(
            state_dict=deserialized["state_dict"],
            config=deserialized["config"],
            result=deserialized["result"],
        )

    @typechecked
    def get_model(
        self, name: str, init_idx: int, ensemble_idx: int | None = None
    ) -> CachedModel:
        key = ModelRegistryKey(name=name, init_idx=init_idx, ensemble_idx=ensemble_idx)
        assert key in self, f"Model {name} not found."
        return self[key]

    @typechecked
    def get_model_or_train(
        self,
        data: Data,
        logger: Logger | None,
        name: str,
        init_idx: int,
        config: Config,
        force_retrain: bool = False,
        allow_train: bool = True,
    ) -> tuple[Ensemble, dict[Metric, list[MetricValue]], dict[str, Any]]:
        """Get a model from the registry."""

        models = []

        for ensemble_idx in range(config["model"]["num_ensemble_members"]):
            model: BaseModel = get_model(config["model"], data).cpu()
            key = ModelRegistryKey(
                name=name,
                init_idx=init_idx,
                ensemble_idx=None
                if config["model"]["num_ensemble_members"] == 1
                else ensemble_idx,
            )
            if key not in self or force_retrain:
                assert logger is not None
                assert allow_train, "Training is not allowed."
                result = train_model(config["trainer"], data, model, logger)
                cached = CachedModel(
                    state_dict=model.state_dict(),
                    config=dict(**config),
                    result=result,
                )
                self[key] = cached
            else:
                cached = self[key]
                model.load_state_dict(cached.state_dict)
            models.append(model)

        return Ensemble(config["model"], models), cached.result, cached.config
