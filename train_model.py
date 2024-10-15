from typing import Any

from typeguard import typechecked

from graph_uq.config import Config
from graph_uq.config.data import DataConfig
from graph_uq.config.evaluation import EvaluationConfig
from graph_uq.config.logging import LoggingConfig
from graph_uq.config.model import ModelConfig
from graph_uq.config.plot import PlottingConfig
from graph_uq.config.trainer import TrainerConfig
from graph_uq.config.uncertainty import UncertaintiesConfig
from graph_uq.data.data import Data
from graph_uq.data.dataset import Dataset
from graph_uq.data.registry import DatasetRegistry, DatasetRegistryKey
from graph_uq.experiment import experiment
from graph_uq.logging.build import get_logger
from graph_uq.logging.logger import Logger
from graph_uq.model.base import BaseModel
from graph_uq.model.registry import ModelRegistry
from graph_uq.util.environment import setup_environment
from graph_uq.util.seed import set_seed
from graph_uq.util.seml import setup_experiment, to_json_serializable


@typechecked
def get_dataset(
    precomputed: str, _split_idx: int, config: Config
) -> tuple[Dataset, dict[str, Any]]:
    registry = DatasetRegistry(
        config["data"]["registry"]["database_path"],
        config["data"]["registry"]["lockfile_path"],
        config["data"]["registry"]["storage_path"],
    )
    assert (
        DatasetRegistryKey(
            name=precomputed,
            split_idx=_split_idx,
        )
        in registry
    ), f"Dataset {precomputed, _split_idx} not found."
    dataset = registry.get_dataset_or_build(
        precomputed, _split_idx, config, force_rebuild=False
    )
    return dataset


@typechecked
def get_model(
    data_train: Data,
    logger: Logger,
    _storage_name: str,
    _init_idx: int,
    config: Config,
) -> tuple[BaseModel, dict[Any, list[Any]], dict[str, Any]]:
    registry = ModelRegistry(
        config["model"]["registry"]["database_path"],
        config["model"]["registry"]["lockfile_path"],
        config["model"]["registry"]["storage_path"],
    )
    return registry.get_model_or_train(
        data_train,
        logger,
        _storage_name,
        _init_idx,
        config,
        force_retrain=True,
    )


@experiment.command
def clean_registry(model: ModelConfig):
    """Clean the registry."""
    ModelRegistry(
        model["registry"]["database_path"],
        model["registry"]["lockfile_path"],
        model["registry"]["storage_path"],
    ).clean()  # type: ignore


@experiment.command
def list_registry(model: ModelConfig):
    """Lists the registry."""
    ModelRegistry(
        model["registry"]["database_path"],
        model["registry"]["lockfile_path"],
        model["registry"]["storage_path"],
    ).list()  # type: ignore


@experiment.capture(prefix="data")  # type: ignore
def get_num_splits(num_splits: int) -> int:
    return num_splits


@experiment.command
def delete(model: ModelConfig):
    registry = ModelRegistry(
        model["registry"]["database_path"],
        model["registry"]["lockfile_path"],
        model["registry"]["storage_path"],
    )
    from tinydb import Query, TinyDB

    with registry.lock:
        db = TinyDB(registry.database_path)
        for item in db:
            key = item["key"]
            if "feature_perturbations" in key:
                query = Query()
                db.remove(query.key == key)
                print("Deleted", key)


def main(_config: Config, seed: int):
    """Trains a model and saves the weights in a cache."""
    setup_environment()
    setup_experiment()
    _ = set_seed(seed)
    logger = get_logger(_config["logging"])

    logger.log_configuration(dict(**_config))

    results_all = []
    for split_idx in range(_config["data"]["num_splits"]):
        results_splits = []
        assert _config["data"]["precomputed"] is not None
        dataset, _ = get_dataset(
            precomputed=_config["data"]["precomputed"],
            _split_idx=split_idx,
            config=_config,
        )

        for init_idx in range(_config["model"]["num_inits"]):
            model, results, config = get_model(
                dataset.data_train,
                logger,
                config=_config,
                _init_idx=init_idx,
                _storage_name=f"{_config['model']['pretrained']}_{split_idx}",
            )
            results_splits.append(
                {str(metric): value for metric, value in results.items()}
            )
        results_all.append(results_splits)
    return to_json_serializable(results_all)


@experiment.automain
@typechecked
def _main(
    data: DataConfig,
    trainer: TrainerConfig,
    evaluation: EvaluationConfig,
    plot: PlottingConfig,
    logging: LoggingConfig,
    model: ModelConfig,
    uncertainty: UncertaintiesConfig,
    db_collection: str | None,
    overwrite: str | None,
    seed: int,
):
    """Automain to capture all keys in `Config`."""
    main(
        Config(
            trainer=trainer,
            evaluation=evaluation,
            plot=plot,
            logging=logging,
            data=data,
            model=model,
            uncertainty=uncertainty,
            db_collection=db_collection,
            overwrite=overwrite,
        ),
        seed,
    )
