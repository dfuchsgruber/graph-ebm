from jaxtyping import Bool, Float, Shaped, jaxtyped
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch import Tensor
from typeguard import typechecked

from graph_uq.metric import Metric, MetricName


@jaxtyped(typechecker=typechecked)
def binary_classification(
    proxy: Float[Tensor, "num_nodes"], is_positive: Bool[Tensor, "num_nodes"]
) -> dict[Metric, float | int | Shaped[Tensor, ""]]:
    """Evaluates the binary classification performance of the proxy (auc roc, auc pr)."""
    metrics = {}
    if (
        is_positive.sum() > 0 and (~is_positive).sum() > 0
    ):  # If there are both positive and negative examples
        metrics[Metric(name=MetricName.AUCROC)] = roc_auc_score(
            is_positive.cpu().numpy().astype(int), proxy.cpu().numpy()
        )
        precision, recall, _ = precision_recall_curve(
            is_positive.cpu().numpy().astype(int), proxy.cpu().numpy()
        )
        metrics[Metric(name=MetricName.AUCPR)] = auc(recall, precision)
    return metrics
