from enum import StrEnum, unique
from typing import Any, TypedDict

from graph_uq.experiment import experiment
from graph_uq.metric import DatasetSplit, Metric, MetricName


@unique
class TrainerType(StrEnum):
    SGD = "sgd"
    GPN = "gpn"
    BayesianSGD = "bayesian_sgd"
    NONE = "none"
    SGNN = "sgnn"


@unique
class LossFunctionType(StrEnum):
    CROSS_ENTROPY = "cross_entropy"
    CROSS_ENTROPY_AND_KL_DIVERGENCE = "cross_entropy_and_kl_divergence"
    BAYESIAN_RISK = "bayesian_risk"


@unique
class EdgeReconstructionLossType(StrEnum):
    CONTRASTIVE_WITH_MARGIN = "contrastive_with_margin"
    DOT_PRODUCT = "dot_product"
    ENERGY = "energy"


@unique
class GPNWarmup(StrEnum):
    FLOW = "flow"
    ENCODER = "encoder"


class EdgeReconstructionConfig(TypedDict):
    type_: EdgeReconstructionLossType | None | str
    weight: float
    num_samples: int
    margin: float
    embedding_layer: int
    embeddings_propagated: bool


class EarlyStoppingConfig(TypedDict):
    monitor: Metric | dict[str, Any]
    higher_is_better: bool
    patience: int
    min_delta: float
    save_model_state: bool


class TrainerConfig(TypedDict):
    type_: TrainerType | str
    loss_function_type: LossFunctionType | str
    edge_reconstruction_loss: EdgeReconstructionConfig
    min_epochs: int
    max_epochs: int
    learning_rate: float
    weight_decay: float
    train_with_propagated_prediction: bool
    commit_every_epoch: int
    log_every_epoch: int
    use_gpu: bool
    verbose: bool
    progress_bar: bool
    early_stopping: EarlyStoppingConfig
    kl_divergence_loss_weight: float
    # GPN
    learning_rate_flow: float
    weight_decay_flow: float
    entropy_regularization_loss_weight: float
    warmup: GPNWarmup | str
    num_warmup_epochs: int
    learning_rate_warmup: float
    weight_decay_warmup: float
    teacher: "TrainerConfig | None"


default_trainer_config = TrainerConfig(  # noqa: F841
    type_=TrainerType.SGD,
    loss_function_type=LossFunctionType.CROSS_ENTROPY,
    # Edge reconstruction
    edge_reconstruction_loss=EdgeReconstructionConfig(
        type_=None,
        weight=0.0,
        num_samples=100,
        margin=0.0,
        embedding_layer=-2,
        embeddings_propagated=True,
    ),
    min_epochs=1,
    max_epochs=10000,
    learning_rate=1e-3,
    weight_decay=1e-4,
    kl_divergence_loss_weight=0e0,
    train_with_propagated_prediction=True,
    commit_every_epoch=1,
    log_every_epoch=1,
    use_gpu=True,
    verbose=False,
    progress_bar=True,
    early_stopping=EarlyStoppingConfig(
        monitor=Metric(
            name=MetricName.LOSS,
            dataset_split=DatasetSplit.VAL,
            propagated=True,
        ),
        higher_is_better=False,
        patience=50,
        min_delta=1e-3,
        save_model_state=True,
    ),
    learning_rate_flow=1e-2,
    weight_decay_flow=0.0,
    entropy_regularization_loss_weight=1e-4,
    warmup=GPNWarmup.FLOW,
    num_warmup_epochs=5,
    learning_rate_warmup=1e-2,
    weight_decay_warmup=0.0,
    teacher=None,
)


@experiment.config
def _default_trainer_config():
    trainer = default_trainer_config  # noqa: F841
