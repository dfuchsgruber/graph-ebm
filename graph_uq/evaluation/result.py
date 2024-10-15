from dataclasses import asdict, dataclass, field, fields, is_dataclass

from jaxtyping import Bool, Shaped
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor

from graph_uq.metric import Metric


@dataclass
class Plot:
    """Dataclass for storing a matplotlib figure."""

    figure: Figure
    axes: Axes


@dataclass
class EvaluationResult:
    """Dataclass for storing the results of an evaluation run."""

    metrics: dict[Metric, float | int | Shaped[Tensor, ""]] = field(
        default_factory=dict
    )
    plots: dict[Metric, Plot] = field(default_factory=dict)
    embeddings: dict[Metric, Shaped[Tensor, "num_samples num_nodes num_classes"]] = (
        field(default_factory=dict)
    )
    uncertainties: dict[Metric, Shaped[Tensor, " num_nodes"]] = field(
        default_factory=dict
    )
    dataset_node_keys: dict[Metric, Shaped[Tensor, "num_nodes"]] = field(
        default_factory=dict
    )  # Which node idxs of the base dataset are in each shift
    masks: dict[Metric, Bool[Tensor, "num_nodes"]] = field(
        default_factory=dict
    )  # Which nodes are in the mask

    def __add__(self, other):
        """Merges two evaluation results."""
        if not isinstance(other, EvaluationResult):
            raise ValueError(f"Cannot add {type(other)} to {type(self)}")
        for f in fields(self):
            assert set(
                getattr(self, f.name).keys()
            ).isdisjoint(
                set(getattr(other, f.name).keys())
            ), f"Cannot add two evaluation results with overlapping {f.name}: {set(getattr(self, f.name).keys()) & set(getattr(other, f.name).keys())}"
        return EvaluationResult(
            **{
                f.name: {**getattr(self, f.name), **getattr(other, f.name)}
                for f in fields(self)
            }
        )

    # @typechecked
    # def log(self, logger: Logger, suffix: str | None = None, step: int | None = None):
    #     """ Logs the evaluation result. """
    #     suffix = f'/{suffix}' if suffix else ''
    #     logger.log({str(key) + suffix : value for key, value in self.metrics.items()}, step=step)
    #     for key, plot in self.plots.items():
    #         logger.log_image(plot.figure, f'plots/{str(key) + suffix}', step=step)

    def asdict(self) -> dict:
        """Returns the evaluation result as a dictionary."""
        # Merge all fields but plots, call astuple on the keys and transfer tensor valued values to cpu
        return dict(
            **{
                field.name: {
                    tuple((k, v) for k, v in asdict(key).items())
                    if is_dataclass(key)
                    else key: value.cpu() if isinstance(value, Tensor) else value
                    for key, value in getattr(self, field.name).items()
                }
                for field in fields(self)
                if field.name != "plots"
            }
        )

    def extend_metrics(self, metric: Metric) -> "EvaluationResult":
        """Extends all metrics in the result by adding the given metric."""
        return EvaluationResult(
            **{
                field.name: {
                    key + metric: value
                    for key, value in getattr(self, field.name).items()
                }
                for field in fields(self)
            }
        )
