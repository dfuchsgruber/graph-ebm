import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from graph_uq.cached import get_cached_dataset, load_cached_model
from graph_uq.config import Config
from graph_uq.config.data import (
    DataConfig,
    FeaturePerturbationsParameter,
    FeaturePerturbationType,
)
from graph_uq.config.evaluation import EvaluationConfig
from graph_uq.evaluation.evaluation import evaluate
from graph_uq.evaluation.registry import (
    CachedEvaluationResult,
    EvaluationRegistry,
    EvaluationRegistryKey,
)
from graph_uq.experiment import experiment
from graph_uq.uncertainty.build import get_uncertainty_models
from graph_uq.util.environment import setup_environment
from graph_uq.util.seed import set_seed
from graph_uq.util.seml import setup_experiment, to_json_serializable


@experiment.command
def clean_registry(evaluation: EvaluationConfig):
    """Clean the registry."""
    EvaluationRegistry(
        evaluation["registry"]["database_path"],
        evaluation["registry"]["lockfile_path"],
        evaluation["registry"]["storage_path"],
    ).clean()


@experiment.command
def list_registry(evaluation: EvaluationConfig):
    """Lists the registry."""
    EvaluationRegistry(
        evaluation["registry"]["database_path"],
        evaluation["registry"]["lockfile_path"],
        evaluation["registry"]["storage_path"],
    ).list()


def get_feature_perturbation_shifts(config: DataConfig) -> dict[Any, Any]:
    feature_perturbations = dict(
        ber_0=dict(
            type_=FeaturePerturbationType.BERNOULLI,
            p=0.0,
            transform=True,
        ),
        ber_1=dict(
            type_=FeaturePerturbationType.BERNOULLI,
            p=1.0,
            transform=True,
        ),
        ber_50=dict(
            type_=FeaturePerturbationType.BERNOULLI,
            p=0.5,
            transform=True,
        ),
        normal_0_1=dict(  # Far-OOD
            type_=FeaturePerturbationType.NORMAL,
            mean=0.0,
            std=1.0,
            transform=False,
        ),
    ) | (
        dict(
            ber_mean=dict(
                type_=FeaturePerturbationType.BERNOULLI,
                p=FeaturePerturbationsParameter.AVERAGE,
                transform=True,
            ),
            ber_mean_per_class=dict(
                type_=FeaturePerturbationType.BERNOULLI,
                p=FeaturePerturbationsParameter.AVERAGE_PER_CLASS,
                transform=True,
            ),
        )
        if config["categorical_features"]
        else dict(
            normal_mean=dict(
                type_=FeaturePerturbationType.NORMAL,
                mean=FeaturePerturbationsParameter.AVERAGE,
                std=FeaturePerturbationsParameter.AVERAGE,
                transform=True,
            ),
            normal_mean_per_class=dict(
                type_=FeaturePerturbationType.NORMAL,
                mean=FeaturePerturbationsParameter.AVERAGE_PER_CLASS,
                std=FeaturePerturbationsParameter.AVERAGE_PER_CLASS,
                transform=True,
            ),
        )
    )
    return feature_perturbations


@experiment.capture()  # type: ignore
def __(uncertainty, model, data, logging, trainer): ...


@experiment.automain
def main(
    _config: Config,
    _seed: int,
    force_reevaluate: bool = False,
    output_dir: str | None = None,
):
    """Trains a model and saves the weights in a cache."""
    setup_environment()
    setup_experiment()
    _ = set_seed(_seed)
    evaluation_registry = EvaluationRegistry(
        _config["evaluation"]["registry"]["database_path"],
        _config["evaluation"]["registry"]["lockfile_path"],
        _config["evaluation"]["registry"]["storage_path"],
    )

    assert _config["data"]["precomputed"] is not None
    assert _config["model"]["pretrained"] is not None
    assert _config["evaluation"]["precomputed"] is not None

    results_all = []
    for split_idx in range(_config["data"]["num_splits"]):
        cached_dataset = get_cached_dataset(
            _config["data"],
            _config["data"]["precomputed"],
            split_idx,
            get_feature_perturbation_shifts(_config["data"]),
        )

        for init_idx in range(_config["model"]["num_inits"]):
            model, model_config = load_cached_model(
                _config["model"],
                f"{_config['model']['pretrained']}_{split_idx}",
                init_idx,
                cached_dataset,
            )
            key = EvaluationRegistryKey(
                name=_config["evaluation"]["precomputed"] + f"_{split_idx}_{init_idx}"
            )
            uncertainty_models = get_uncertainty_models(_config["uncertainty"])

            logging.info(f"Split index: {split_idx}, Init index: {init_idx}:")
            logging.info(f"Dataset key: {_config['data']['precomputed']}")
            logging.info(f"Dataset key: {_config['model']['pretrained']}")
            logging.info(f"Evaluation key: {_config['evaluation']['precomputed']}")
            if key in evaluation_registry and not force_reevaluate:
                result, config = evaluation_registry[key]
                logging.warning(f"Already evaluated: {key}")
            else:
                with torch.no_grad():
                    model.eval()
                    result = evaluate(
                        _config["evaluation"],
                        cached_dataset.dataset,
                        model,
                        uncertainty_models,
                    )
                    evaluation_registry[key] = CachedEvaluationResult(
                        result=result,
                        config={
                            "data": cached_dataset.config,
                            "model": model_config["model"],
                            "uncertainty": _config["uncertainty"],
                            "evaluation": _config["evaluation"],
                        },
                    )

            results_all.append(result)

    # Aggregate results to log to MongoDB

    output = defaultdict(lambda: defaultdict(list))

    for result in results_all:
        for metric, value in result.metrics.items():
            output["metrics"][str(metric)].append(value)
        # for metric, value in result.uncertainties.items():
        #    output["uncertainties"][str(metric)].append(value)

    if _config["db_collection"] is None:
        results_aggregated = {
            metric: [result.metrics.get(metric, float("nan")) for result in results_all]
            for metric in set().union(*(result.metrics for result in results_all))
        }
        from graph_uq.summary import print_table as print_table_fn

        print_table_fn(results_aggregated)  # type: ignore

    output = to_json_serializable({k: dict(v) for k, v in output.items()})

    if output_dir is not None:
        output_path = Path(str(output_dir)) / f"{uuid4()}.json"
        with open(output_path, "w") as f:
            json.dump(
                {
                    "output": output,
                    "db_collection": _config["db_collection"],
                    "overwrite": _config["overwrite"],
                },
                f,
            )

        return str(output_path.absolute())
    else:
        return output
