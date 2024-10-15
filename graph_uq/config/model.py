from enum import StrEnum, unique
from pathlib import Path
from typing import Any, TypedDict

from graph_uq.config.registry import RegistryConfig
from graph_uq.experiment import experiment


@unique
class Activation(StrEnum):
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    TANH = "tanh"
    NONE = "none"


@unique
class ModelType(StrEnum):
    GCN = "gcn"
    GAT = "gat"
    SAGE = "sage"
    GIN = "gin"
    MLP = "mlp"
    NONE = "none"
    GPN = "gpn"
    BGCN = "bgcn"
    GDK = "gdk"
    SGNN = "sgnn"
    APPNP = "appnp"


@unique
class GPNEvidenceScale(StrEnum):
    LATENT_OLD = "latent-old"
    LATENT_NEW = "latent-new"
    LATENT_NEW_PLUS_CLASSES = "latent-new-plus-classes"
    NONE = "none"


class ModelConfig(TypedDict):
    name: str | None
    type_: ModelType | str
    num_inits: int
    init_idx: int | None
    residual: bool
    bias: bool
    activation: Activation | str
    leaky_relu_slope: float

    num_samples_eval: int
    num_samples_train: int

    dropout: float
    dropedge: float
    dropout_at_eval: bool
    dropedge_at_eval: bool

    num_ensemble_members: int
    inplace: bool
    hidden_dims: list[int]
    cached: bool
    add_self_loops: bool

    improved: bool
    normalize: bool

    num_heads: int
    concatenate: bool

    project: bool
    root_weight: bool

    eps_trainable: bool
    eps_init: float

    batch_norm: bool

    # Parameterizations
    spectral_normalization: bool
    bjorck_normalization: bool
    bjorck_orthonormal_scale: float
    bjorck_orthonormalization_num_iterations: int
    frobenius_normalization: bool
    frobenius_norm: float
    rescale: bool
    weight_scale: float
    initialization_scale: float
    # Bayesian NN
    prior_mean_bias: float
    prior_std_bias: float
    prior_mean_weight: float
    prior_std_weight: float
    mean_init_bias: float
    rho_init_bias: float
    mean_init_weight: float
    rho_init_weight: float
    pretrained: str | None
    registry: RegistryConfig
    # GPN
    flow_dim: int
    num_flow_layers: int
    evidence_scale: GPNEvidenceScale | str
    num_diffusion_steps: int
    alpha: float
    # GAT
    edge_dim: int | None
    fill_value: float
    # GDK
    cutoff: int
    sigma: float
    # SGNN
    teacher: "ModelConfig | None"
    backbone: "ModelConfig | None"
    gdk_prior: "ModelConfig | None"

    num_outputs: int | None

    prediction_registry: Any  # Deprecated, only for compatability


default_model_config: ModelConfig = ModelConfig(
    name=None,
    type_=ModelType.GCN,
    num_inits=1,
    init_idx=None,
    residual=False,
    bias=True,
    activation=Activation.RELU,
    leaky_relu_slope=0.2,
    # Monte Carlo methods
    num_samples_eval=1,  # how many samples to draw at eval
    num_samples_train=1,  # how many samples to draw at train
    # Monte Carlo dropout
    dropout=0.5,
    dropedge=0.0,
    dropout_at_eval=False,
    dropedge_at_eval=False,
    # Ensemble
    num_ensemble_members=1,
    inplace=False,
    hidden_dims=[
        64,
    ],
    batch_norm=False,
    cached=True,
    add_self_loops=True,
    # GCN
    improved=False,
    normalize=True,
    # GAT
    num_heads=8,
    concatenate=False,
    # SAGE
    project=False,
    root_weight=True,
    # GIN
    eps_trainable=False,
    eps_init=0.0,
    # Normalization of linear operators
    spectral_normalization=False,
    bjorck_normalization=False,
    bjorck_orthonormal_scale=1.0,
    bjorck_orthonormalization_num_iterations=1,
    frobenius_normalization=False,
    frobenius_norm=1.0,
    rescale=False,
    weight_scale=1.0,
    initialization_scale=1.0,
    # Bayesian linear layers
    prior_mean_bias=0.0,
    prior_std_bias=1.0,
    prior_mean_weight=0.0,
    prior_std_weight=1.0,
    mean_init_bias=0.0,
    rho_init_bias=-3.0,
    mean_init_weight=0.0,
    rho_init_weight=-3.0,
    pretrained=None,  # key of a pre-trained model to use,
    registry=RegistryConfig(
        database_path=str(Path("ceph") / "model" / "registry" / "db.json"),
        lockfile_path=str(Path("ceph") / "model" / "registry" / "db.json.lock"),
        storage_path=str(Path("ceph") / "model" / "registry" / "storage"),
    ),
    flow_dim=16,
    num_flow_layers=10,
    evidence_scale=GPNEvidenceScale.LATENT_NEW_PLUS_CLASSES,
    num_diffusion_steps=10,
    alpha=0.2,
    edge_dim=None,
    fill_value=0.0,
    cutoff=10,
    sigma=1.0,
    teacher=None,
    backbone=None,
    gdk_prior=None,
    prediction_registry=None,
    num_outputs=None,
)


@experiment.config
def _default_model_config():
    model = default_model_config  # noqa: F841
