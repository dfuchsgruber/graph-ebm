"""Config for the HEAT baseline."""

from enum import StrEnum, unique
from typing import TypedDict

from graph_uq.config.model import ModelConfig, ModelType, default_model_config
from graph_uq.config.uncertainty import UncertaintyModelType


@unique
class HeatProposalType(StrEnum):
    RANDOM_NORMAL = "random_normal"
    RANDOM_UNIFORM = "random_uniform"
    BASE_DIST = "base_dist"  # propose from the prior distribution
    BASE_DIST_TEMP = (
        "base_dist_temp"  # propose from the prior distribution with temperature
    )
    DATA = "data"  # propose from the data distribution


@unique
class HeatEBMType(StrEnum):
    GMM = "gmm"
    LOGITS = "logits"


class HeatOptimizerConfig(TypedDict):
    """Config for the optimizer."""

    lr: float
    betas: tuple[float, float]


class HeatSchedulerConfig(TypedDict):
    """Config for the scheduler."""

    milestones: list[int]
    gamma: float


class ContrasticeDivergenceLossConfig(TypedDict):
    """Config for the contrastive divergence loss."""

    eps_data: float
    l2_coef: float
    verbose: bool


class HeatConfig(TypedDict):
    """Config for the HEAT baseline."""

    optimizer: HeatOptimizerConfig
    loss: ContrasticeDivergenceLossConfig
    scheduler: HeatSchedulerConfig

    hidden_dims: list[int]

    verbose: bool
    use_gpu: bool
    num_epochs: int

    use_base_distribution: bool
    steps: int
    step_size_start: float
    step_size_end: float
    eps_start: float
    eps_end: float
    sgld_relu: bool
    proposal_type: str
    temperature: float
    temperature_prior: float
    sample_from_batch_statistics: bool
    train_max_iter: int | float
    diagonal_covariance: bool
    normalize: bool

    type_: UncertaintyModelType | str
    fit_to_propagated_embeddings: bool
    layer: int
    ebm_type_: HeatEBMType | str


class HeatCompositeConfig(TypedDict):
    ebms: dict[str, HeatConfig]
    beta: float


default_heat_optimizer_config: HeatOptimizerConfig = HeatOptimizerConfig(
    lr=1e-5, betas=(0, 0.999)
)

default_heat_scheduler_config: HeatSchedulerConfig = HeatSchedulerConfig(
    milestones=[10], gamma=0.1
)


default_heat_config = HeatConfig(
    type_=UncertaintyModelType.HEAT,
    optimizer=default_heat_optimizer_config,
    loss=ContrasticeDivergenceLossConfig(eps_data=1e-4, l2_coef=10, verbose=True),
    scheduler=default_heat_scheduler_config,
    hidden_dims=[
        64,
    ],
    verbose=True,
    use_gpu=True,
    num_epochs=500,
    use_base_distribution=True,
    steps=40,
    step_size_start=1e-4,
    step_size_end=1e-5,
    eps_start=5e-3,
    eps_end=5e-4,
    sgld_relu=True,
    proposal_type=HeatProposalType.BASE_DIST,
    temperature=1e0,
    temperature_prior=1e3,
    sample_from_batch_statistics=False,
    train_max_iter=float("inf"),
    diagonal_covariance=True,
    fit_to_propagated_embeddings=True,
    layer=-2,
    ebm_type_=HeatEBMType.GMM,
    normalize=False,
)
