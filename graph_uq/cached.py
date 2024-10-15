import logging
from typing import Any

from graph_uq.config.data import DataConfig, DistributionShiftType
from graph_uq.config.model import ModelConfig
from graph_uq.data.distribution_shift import (
    apply_feature_perturbation,
)
from graph_uq.data.registry import CachedDataset, DatasetRegistry
from graph_uq.experiment import experiment
from graph_uq.model.base import BaseModel, Ensemble
from graph_uq.model.build import get_model
from graph_uq.model.registry import CachedModel, ModelRegistry
from graph_uq.util.dict import merge_dicts


def get_cached_dataset(
    data_config: DataConfig,
    precomputed: str,
    split_idx: int,
    feature_perturbation_shifts: dict[str, dict[str, Any]] = {},
) -> CachedDataset:
    """Gets a dataset and applies feature perturbations if necessary.

    Args:
        precomputed (str): the precomputed dataset
        split_idx (int): the split index
        feature_perturbation_shifts (dict[str, dict[str, Any]], optional): the shifts to apply . Defaults to {}.

    Returns:
        CachedDataset: a cached dataset
    """
    cached: CachedDataset = DatasetRegistry(
        data_config["registry"]["database_path"],
        data_config["registry"]["lockfile_path"],
        data_config["registry"]["storage_path"],
    ).get_dataset(precomputed, split_idx, {})

    if "overwrite" in cached.config:
        # old versioning bug that I saved the whole dataset...
        cached = CachedDataset(dataset=cached.dataset, config=cached.config["data"])

    if (
        cached.config["distribution_shift"]["type_"]
        == DistributionShiftType.FEATURE_PERTURBATIONS
    ):
        base = cached.dataset.data_shifted["base"]

        for name, config in feature_perturbation_shifts.items():
            assert name not in cached.dataset.data_shifted
            cached.dataset.data_shifted[name] = apply_feature_perturbation(
                data_config, base, config
            )

    return cached


def get_cached_model(
    model_config: ModelConfig,
    pretrained: str,
    init_idx: int,
    ensemble_idx: int | None = None,
) -> CachedModel:
    """Gets a cached model.

    Args:
        pretrained (str): the pretrained model
        init_idx (int): the initialization index

    Returns:
        CachedModel: a cached model
    """
    return ModelRegistry(
        model_config["registry"]["database_path"],
        model_config["registry"]["lockfile_path"],
        model_config["registry"]["storage_path"],
    ).get_model(pretrained, init_idx, ensemble_idx=ensemble_idx)


def load_cached_model(
    model_config: ModelConfig, pretrained: str, init_idx: int, dataset: CachedDataset
) -> tuple[BaseModel, dict[str, Any]]:
    """Loads a cached model.

    Args:
        pretrained (str): the pretrained model
        init_idx (int): the initialization index
        dataset (CachedDataset): the dataset
    """
    ensemble_members = []
    for ensemble_idx in range(model_config["num_ensemble_members"]):
        cached_model = get_cached_model(
            model_config,
            pretrained,
            init_idx,
            ensemble_idx if model_config["num_ensemble_members"] > 1 else None,
        )
        model = get_model(
            ModelConfig(**merge_dicts(model_config, cached_model.config["model"])),
            dataset.dataset.data_train,
        ).eval()
        result = model.load_state_dict(cached_model.state_dict, strict=False)
        assert len(result.missing_keys) == 0, f"Missing keys: {result.missing_keys}"
        ensemble_members.append(model)
    model = Ensemble(model_config, ensemble_members)
    return model, cached_model.config
