from dataclasses import asdict, dataclass, fields
from enum import StrEnum, unique
from typing import Any, TypeAlias

import torch
from jaxtyping import Shaped

from graph_uq.config.data import DatasetSplit, DistributionShiftType, DistributionType
from graph_uq.config.evaluation import LatentSpaceVisualizationType
from graph_uq.config.plot import PlotType
from graph_uq.config.uncertainty import UncertaintyModelType


@unique
class UncertaintyMetric(StrEnum):
    OOD_DETECTION = "ood_detection"
    MISCLASSIFICATION_DETECTION = "misclassification_detection"


@unique
class MetricName(StrEnum):
    LOSS = "loss"
    ACCURACY = "accuracy"
    BRIER_SCORE = "brier_score"
    F1 = "f1"
    ECE = "ece"
    AUCROC = "auc_roc"
    AUCPR = "auc_pr"

    # Model specific losses
    UCE_LOSS = "uce"
    ENTROPY_REGULARIZATION_LOSS = "entropy_regularization"
    KL_DIVERGENCE = "kl_divergence"
    DISTIALLATION_LOSS = "distillation_loss"
    EDGE_RECONSTRUCTION_LOSS = "edge_reconstruction_loss"

    TRAIN_ENTROPY_REGULARIZATION_LOSS = "loss/train/entropy_regularization"
    VAL_ENTROPY_REGULARIZATION_LOSS = "loss/val/entropy_regularization"
    TEST_ENTROPY_REGULARIZATION_LOSS = "loss/test/entropy_regularization"
    ALL_ENTROPY_REGULARIZATION_LOSS = "loss/all/entropy_regularization"


MetricValue: TypeAlias = float | int | Shaped[torch.Tensor, ""]


@dataclass(unsafe_hash=True)
class Metric(dict):
    """Template for a metric that is not yet finalized with respect to its representation. This can be used for passing metrics through different
    levels of the pipeline and only logging them when the template is finalized"""

    name: MetricName | None = None
    distribution_shift_metric: UncertaintyMetric | None = None
    distribution_shift: DistributionShiftType | None = None
    distribution_shift_name: str | None = None
    dataset_split: DatasetSplit | None = None
    dataset_distribution: DistributionType | None = None
    propagated: bool | None = None
    targets_propagated: bool | None = None
    uncertainty_model_type: UncertaintyModelType | None = (
        None  # which uncertainty model was used to compute the metric
    )
    uncertainty_model_name: str | None = (
        None  # which uncertainty model was used to compute the metric
    )
    teacher: str | None = None  # which teacher was used to compute the metric
    plot_type: PlotType | None = None  # which type of plot was made
    uncertainty_diffusion: str | None = None
    uncertainty_diffusion_num_steps: int | None = None

    # based on latent embeddings
    embedding_type: LatentSpaceVisualizationType | None = None
    embedding_name: str | None = None
    embedding_idx: int | None = None

    # for composite metrics
    member_name: str | None = None

    suffix: str | None = None

    def __post_init__(self):
        # Subclassing dict is hacky, but we need it to make sure the metric is JSON serializable
        dict.__init__(self, **asdict(self))

    def asdict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def serialize(self) -> tuple[tuple[str, Any], ...]:
        """Serialize the metric to a tuple of key-value pairs."""
        return tuple(
            (k, str(v) if isinstance(v, str) else v) for k, v in self.asdict().items()
        )

    @classmethod
    def deserialize(cls, serialized: tuple[tuple[str, Any], ...]) -> "Metric":
        """Deserialize a serialized metric."""
        return cls(**{k: v for k, v in serialized})

    def __add__(self, __t: Any) -> "Metric":
        """Overload the + operator to allow for easy construction of metrics."""
        if isinstance(__t, Metric):
            new_fields = {}
            for field in fields(self):
                a, b = getattr(self, field.name), getattr(__t, field.name)
                if a is not None and b is not None and a != b:
                    raise ValueError(
                        f"Field {field.name} is already set to {a}, cannot set to {b}"
                    )
                elif a is not None:
                    new_fields[field.name] = a
                elif b is not None:
                    new_fields[field.name] = b
            return Metric(**new_fields)
        else:
            raise ValueError(f"Expected Metric, got {type(__t)}")

    def __sub__(self, __t: Any) -> "Metric":
        """Overload the - operator to allow for easy construction of metrics."""
        if isinstance(__t, Metric):
            new_fields = {}
            for field in fields(self):
                a, b = getattr(self, field.name), getattr(__t, field.name)
                if a is not None and b is not None and a != b:
                    raise ValueError(
                        f"Field {field.name} is already set to {a} in first metric, but to {b} in second metric"
                    )
                elif b is not None and a is None:
                    raise ValueError(
                        f"Field {field.name} is not set in first metric, but to {b} in second metric"
                    )
                elif a is not None and b is None:
                    new_fields[field.name] = a
            return Metric(**new_fields)
        else:
            raise ValueError(f"Expected Metric, got {type(__t)}")

    def __le__(self, __t: Any) -> bool:
        """Overload the <= operator to allow for easy construction of metrics."""
        if isinstance(__t, Metric):
            for field in fields(self):
                a, b = getattr(self, field.name), getattr(__t, field.name)
                if a is not None and a != b:
                    return False
            return True
        else:
            raise ValueError(f"Expected Metric, got {type(__t)}")

    def __repr__(self) -> str:
        keys = [
            getattr(self, field.name)
            for field in fields(self)
            if getattr(self, field.name) is not None
            and field.name not in ("propagated", "targets_propagated")
        ]
        if self.propagated == False:
            keys.append("unpropagated")  # type: ignore
        if self.targets_propagated == False:
            keys.append("targets_unpropagated")
        return "/".join(map(str, keys))

    def matches(self, other: Any) -> bool:
        """Check if this metric matches another metric, i.e. if all fields that are not None in the other metric match thid metric.

        Args:
            other (Any): The other metric to compare to.

        Returns:
            bool: True if the metrics match, False otherwise.
        """
        if not isinstance(other, Metric):
            raise ValueError(f"Expected Metric, got {type(other)}")
        for field in fields(other):
            if getattr(other, field.name) is not None and getattr(
                other, field.name
            ) != getattr(self, field.name):
                return False
        return True
