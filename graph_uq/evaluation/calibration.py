import numpy as np

from graph_uq.data.data import Data
from graph_uq.metric import Metric, MetricName
from graph_uq.model.prediction import Prediction

from jaxtyping import Float, Int, jaxtyped, Bool, Shaped
from typeguard import typechecked
from torch import Tensor

@jaxtyped(typechecker=typechecked)
def calibration_curve(probs: Float[Tensor, 'num_nodes num_classes'], y_true: Int[Tensor, 'num_nodes'], bins: int=10, 
        eps: float=1e-12, num_classes: int | None = None) -> tuple[Float[np.ndarray, 'bins_plus_one'], Float[np.ndarray, 'bins'], Float[np.ndarray, 'bins'], Float[np.ndarray, 'bins']]:
    """ Calculates the calibration curve for predictions.
    
    Args:
        probs (Float[Tensor, 'num_nodes num_classes']): Predicted probabilities.
        y_true (Int[Tensor, 'num_nodes']): True class labels.
        bins (int, optional): The number of bins to use. Defaults to 10.
        eps (float, optional): Epsilon to prevent division by zero. Defaults to 1e-12.
        num_classes (int | None, optional): The number of classes. Defaults to None.
    
    Returns:
        Float[np.ndarray, 'bins + 1']: Edges for the bins
        Float[np.ndarray, 'bins']: Average confidence per bin
        Float[np.ndarray, 'bins']: Average accuracy per bin
        Float[np.ndarray, 'bins']: Bin weights (i.e. fraction of samples that is in each bin)
    """
    n, c = probs.size()
    if num_classes is None:
        num_classes = max(c, y_true.max().item() + 1)
    max_prob, hard = probs.detach().cpu().max(dim=-1)
    y_true_one_hot = np.eye(num_classes)[y_true.detach().cpu().numpy()]
    
    bin_edges = np.linspace(0., 1., bins + 1)
    bin_width = 1 / bins
    digitized = np.digitize(max_prob.numpy(), bin_edges)
    digitized = np.maximum(np.minimum(digitized, bins), 1) - 1 # Push values outside the bins into the rightmost and leftmost bins
    
    bins_sum = np.bincount(digitized, minlength=bins, weights=max_prob.numpy())
    bins_size = np.bincount(digitized, minlength=bins)
    is_correct = y_true_one_hot[range(n), hard]
    
    bin_confidence = bins_sum / (bins_size + eps)
    bin_accuracy = np.bincount(digitized, minlength=bins, weights=is_correct) / (bins_size + eps)
    bin_weight = bins_size / bins_size.sum()
    
    return bin_edges, bin_confidence, bin_accuracy, bin_weight

@jaxtyped(typechecker=typechecked)
def expected_calibration_error(probs: Float[Tensor, 'num_nodes num_classes'], y_true: Int[Tensor, 'num_nodes'], bins: int=30, 
                               num_classes: int | None = None, eps: float=1e-12) -> float:
    """ Computes the expected calibration error as in [1].
    
    Args:
        probs (Float[Tensor, 'num_nodes num_classes']): Predicted probabilities.
        y_true (Int[Tensor, 'num_nodes']): True class labels.
        bins (int, optional): The number of bins to use. Defaults to 30.
        num_classes (int | None, optional): The number of classes. Defaults to None.
        eps (float, optional): Epsilon to prevent division by zero. Defaults to 1e-12.

    Returns:
        float: Expected calibration error.
    """
    _, bin_confidence, bin_accuracy, bin_weight = calibration_curve(probs, y_true, bins=bins, eps=eps, num_classes=num_classes)
    ece = (bin_weight * np.abs(bin_confidence - bin_accuracy)).sum()
    return ece



@jaxtyped(typechecker=typechecked)
def evaluate_calibration(data: Data, prediction: Prediction, mask: Bool[Tensor, 'num_nodes'], bins: int=20, eps: float=1e-12) -> dict[Metric, float | int | Shaped[Tensor, '']]:
    """Evaluate the calibration of the model on the data.

    Args:
        data (Data): The data
        prediction (Prediction): The prediction of the model on the data
        mask (Bool[Tensor, 'num_nodes']): The mask to evaluate on
        bins (int, optional): The number of bins to use. Defaults to 20.
        eps (float, optional): Epsilon to prevent division by zero. Defaults to 1e-12.

    Returns:
        dict[Metric, float | int | Shaped[Tensor, '']]: The metrics
    """
    metrics = {}
    for propagated in [False, True]:
        scores = prediction.get_probabilities(propagated=propagated)
        if scores is not None and (mask.sum() > 0):
            scores = scores.mean(0) # Average over MC samples
            metrics[Metric(name=MetricName.ECE, propagated=propagated)] = expected_calibration_error(scores[mask], data.y[mask], bins=bins, eps=eps,
                                                                                                     num_classes=data.num_classes)
    return metrics






