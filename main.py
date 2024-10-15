import logging as _logging
from collections import defaultdict
from typing import Iterable

from typeguard import typechecked

from graph_uq.config import (
    Config,
    DataConfig,
    EvaluationConfig,
    LoggingConfig,
    PlottingConfig,
    TrainerConfig,
    UncertaintiesConfig,
)
from graph_uq.config.model import ModelConfig
from graph_uq.data.build import apply_distribution_shift_and_split, get_base_data
from graph_uq.evaluation.evaluation import evaluate
from graph_uq.experiment import experiment
from graph_uq.logging.build import get_logger
from graph_uq.model.base import Ensemble
from graph_uq.model.build import get_model
from graph_uq.training import train_model
from graph_uq.uncertainty.build import get_uncertainty_models
from graph_uq.util.environment import setup_environment
from graph_uq.util.seed import set_seed
from graph_uq.util.seml import setup_experiment


@typechecked
def main(
    config: Config,
    _seed: int,
) -> dict[str, Iterable[float | None | int]] | str:
    # Setup
    setup_environment()
    setup_experiment()
    rng = set_seed(_seed)  # noqa: F841
    logger = get_logger(config["logging"])

    logger.log_configuration(dict(**config))

    # Main
    metrics_train = defaultdict(list)
    results = []
    base_data = get_base_data(config["data"])

    for split_idx in range(config["data"]["num_splits"]):
        dataset = apply_distribution_shift_and_split(base_data, config["data"])
        for init_idx in range(config["model"]["num_inits"]):
            _logging.info(
                f"Split {split_idx + 1}/{config['data']['num_splits']}, Init {init_idx + 1}/{config['model']['num_inits']}"
            )
            # Train each model in the ensemble (even a single model is an ensemble with one member)
            models = []
            for _ in range(config["model"]["num_ensemble_members"]):
                model = get_model(config["model"], dataset.data_train)
                # Train model
                for metric, value in train_model(
                    config["trainer"], dataset.data_train, model, logger
                ).items():
                    metrics_train[metric].append(value)
                models.append(model.cpu())  # free up GPU memory by moving model to CPU
            model = Ensemble(config["model"], models)

            # Evaluate model
            uncertainty_models = get_uncertainty_models(config["uncertainty"])
            _logging.info("Evaluating...")
            result = evaluate(config["evaluation"], dataset, model, uncertainty_models)
            results.append(result)

    results_file = logger.log_results(results)
    _logging.info(f"Logged to {logger.dir}")
    logger.finish()
    return str(results_file)


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
