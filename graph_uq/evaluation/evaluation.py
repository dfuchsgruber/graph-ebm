import torch
from jaxtyping import Bool, jaxtyped
from torch import Tensor
from typeguard import typechecked

from graph_uq.config.data import DatasetSplit, DistributionType
from graph_uq.config.evaluation import EvaluationConfig
from graph_uq.data import Data, Dataset
from graph_uq.evaluation.calibration import evaluate_calibration
from graph_uq.evaluation.classificiation import evaluate_classification
from graph_uq.evaluation.latent_space import get_latent_embeddings
from graph_uq.evaluation.result import EvaluationResult
from graph_uq.metric import Metric
from graph_uq.model.base import BaseModel
from graph_uq.model.prediction import Prediction
from graph_uq.uncertainty.base import BaseUncertaintyModel


@jaxtyped(typechecker=typechecked)
def evaluate_classifier(
    config: EvaluationConfig,
    data: Data,
    model: BaseModel,
    prediction: Prediction,
    num_classes: int,
    mask: Bool[Tensor, " num_nodes"] | None = None,
) -> EvaluationResult:
    """Evaluates a classifier on a dataset

    Args:
        data (Data): the dataset to evaluate on
        model (BaseModel): the model to evaluate
        prediction (Prediction): the model predictions made on this dataset
        num_classes (int): the number of classes
        calibration_num_bins (int): the number of bins to use for calibration
        mask (Bool[Tensor, 'num_nodes'] | None): the mask of a subset of nodes to evaluate on

    Returns:
        EvaluationResult: the evaluation metrics
    """
    metrics = {}
    for split in DatasetSplit:
        mask_split = (
            mask & data.get_mask(split) if mask is not None else data.get_mask(split)
        )
        metrics |= {
            metric + Metric(dataset_split=split): value
            for metric, value in evaluate_classification(
                data, prediction, mask_split, num_classes
            ).items()
        }
        metrics |= {
            metric + Metric(dataset_split=split): value
            for metric, value in evaluate_calibration(
                data, prediction, mask_split, bins=config["calibration_num_bins"]
            ).items()
        }
    return EvaluationResult(metrics=metrics)


@typechecked
def transfer_model_to_device(model: BaseModel, use_gpu: bool) -> BaseModel:
    if use_gpu and torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()
    return model


@typechecked
def transfer_data_to_device(data: Data, use_gpu: bool) -> Data:
    if use_gpu and torch.cuda.is_available():
        data = data.cuda()
    else:
        data = data.cpu()
    return data


@torch.no_grad()
@typechecked
def evaluate(
    config: EvaluationConfig,
    dataset: Dataset,
    model: BaseModel,
    uncertainty_models: dict[str, BaseUncertaintyModel],
) -> EvaluationResult:
    """Runs the evaluation"""
    model.eval()
    model = transfer_model_to_device(model, config["use_gpu"])
    model.reset_cache()
    data_train = transfer_data_to_device(dataset.data_train, config["use_gpu"])
    result = EvaluationResult(dataset_node_keys={Metric(): data_train.node_keys.cpu()})
    result.masks |= {
        Metric(dataset_split=split): data_train.get_mask(split)
        for split in DatasetSplit
    }
    result.masks |= {
        Metric(
            dataset_distribution=distribution_type
        ): data_train.get_distribution_mask(distribution_type)
        for distribution_type in DistributionType
    }

    prediction: Prediction = model.predict(data_train)
    result += evaluate_classifier(
        config, data_train, model, prediction, data_train.num_classes_train
    )

    # Fit the uncertainty models to the training data.
    # Models need to fit the data after distributional shift should do so in their respective `__call__` method.
    for uncertainty_model in uncertainty_models.values():
        uncertainty_model.fit(data_train, model, prediction)

    # Evaluate the uncertainty models on the training data
    for uncertainty_model_name, uncertainty_model in uncertainty_models.items():
        for metric, uncertainty in uncertainty_model(
            data_train, model, prediction
        ).items():
            result += uncertainty_model.evaluate(
                data_train,
                model,
                prediction,
                uncertainty,
            ).extend_metrics(
                metric + Metric(uncertainty_model_name=uncertainty_model_name)
            )

    # Evaluate each distributional shift
    for distribution_shift_name, data in dataset.data_shifted.items():
        model.reset_cache()
        data = transfer_data_to_device(data, config["use_gpu"])
        result += EvaluationResult(
            dataset_node_keys={
                Metric(
                    distribution_shift_name=distribution_shift_name
                ): data.node_keys.cpu()
            },
            masks={
                Metric(
                    dataset_split=split, distribution_shift_name=distribution_shift_name
                ): data.get_mask(split)
                for split in DatasetSplit
            }
            | {
                Metric(
                    dataset_distribution=distribution_type,
                    distribution_shift_name=distribution_shift_name,
                ): data.get_distribution_mask(distribution_type)
                for distribution_type in DistributionType
            },
        )
        prediction = model.predict(data)

        # Evaluate the classifier on each distribution type (in-distribution, ood, both)
        for distribution_type in DistributionType:
            result += evaluate_classifier(
                config,
                data,
                model,
                prediction,
                data.num_classes,
                mask=data.get_distribution_mask(distribution_type),
            ).extend_metrics(
                Metric(
                    dataset_distribution=distribution_type,
                    distribution_shift_name=distribution_shift_name,
                )
            )

        # Visualize the latent space
        result += get_latent_embeddings(
            config["latent_space"], data, model, prediction
        ).extend_metrics(Metric(distribution_shift_name=distribution_shift_name))

        # Evaluate uncertainty metrics
        for uncertainty_model_name, uncertainty_model in uncertainty_models.items():
            for metric, uncertainty in uncertainty_model(
                data, model, prediction
            ).items():
                result += uncertainty_model.evaluate(
                    data,
                    model,
                    prediction,
                    uncertainty,
                ).extend_metrics(
                    metric
                    + Metric(
                        distribution_shift_name=distribution_shift_name,
                        uncertainty_model_name=uncertainty_model_name,
                    )
                )

    return result
