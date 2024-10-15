import torch
from jaxtyping import Bool, Float, Int, Shaped, jaxtyped
from torch import Tensor
from torchmetrics.functional import accuracy, f1_score
from typeguard import typechecked

from graph_uq.data.data import Data
from graph_uq.metric import Metric, MetricName
from graph_uq.model.prediction import Prediction


@jaxtyped(typechecker=typechecked)
def brier_score(
    probabilities: Float[Tensor, "num_nodes num_classes"],
    labels: Int[Tensor, "num_nodes"],
) -> Float[Tensor, ""]:
    """Computes the brier score for a set of probabilities and labels."""
    num_classes = int(max(probabilities.size(1), labels.max().item() + 1))
    differences = torch.zeros(
        probabilities.size(0),
        num_classes,
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    differences[:, : probabilities.size(1)] = probabilities
    differences[torch.arange(probabilities.size(0)), labels] -= 1
    return (differences**2).sum(dim=1).mean()


@jaxtyped(typechecker=typechecked)
def evaluate_classification(
    data: Data,
    prediction: Prediction,
    mask: Bool[Tensor, "num_nodes"],
    num_classes: int,
) -> dict[Metric, float | int | Shaped[Tensor, ""]]:
    """Evaluate a classifier.

    Args:
        data (Data): The data
        prediction (Prediction): The prediction of the model on the data
        mask (Bool[Tensor, 'num_nodes']): The mask to evaluate on
        compute_calibration (bool, optional): Whether to compute calibration metrics. Defaults to True.
        num_classes (int): The number of classes

    Returns:
        dict[Metric, float | int | Shaped[Tensor, '']]: The metrics
    """
    metrics = {}
    assert isinstance(data.y, torch.Tensor)
    for propagated in [False, True]:
        labels = prediction.get_predictions(propagated=propagated)
        # Accuracy and F1
        if labels is not None and (mask.sum() > 0):
            metrics |= {
                Metric(name=MetricName.ACCURACY, propagated=propagated): accuracy(
                    labels[mask],
                    data.y[mask],
                    task="multiclass",
                    num_classes=num_classes,
                ).item(),
                Metric(name=MetricName.F1, propagated=propagated): f1_score(
                    labels[mask],
                    data.y[mask],
                    task="multiclass",
                    num_classes=num_classes,
                ).item(),
            }
        else:
            metrics |= {
                Metric(name=MetricName.ACCURACY, propagated=propagated): float("nan"),
                Metric(name=MetricName.F1, propagated=propagated): float("nan"),
            }
        # Brier score
        probabilities = prediction.get_probabilities(propagated=propagated)
        if probabilities is not None and (mask.sum() > 0):
            probabilities = probabilities.mean(0)
            metrics[Metric(name=MetricName.BRIER_SCORE, propagated=propagated)] = (
                brier_score(probabilities[mask], data.y[mask]).item()
            )
        else:
            metrics[Metric(name=MetricName.BRIER_SCORE, propagated=propagated)] = float(
                "nan"
            )

    return metrics
