from enum import StrEnum, unique
from pathlib import Path
from typing import TypedDict

from graph_uq.config.registry import RegistryConfig
from graph_uq.experiment import experiment


@unique
class Setting(StrEnum):
    INDUCTIVE = "inductive"
    TRANSDUCTIVE = "transductive"


@unique
class DistributionShiftType(StrEnum):
    LOC = "leave_out_classes"
    FEATURE_PERTURBATIONS = "feature_perturbations"
    PAGE_RANK_CENTRALITY = "page_rank_centrality"
    EGO_GRAPH_SIZE = "ego_graph_size"
    HOMOPHILY = "homophily"
    NONE = "none"


@unique
class LeaveOutClassesType(StrEnum):
    """How to pick the classes to be left out"""

    LAST = "last"
    FIRST = "first  "
    RANDOM = "random"
    FIXED = "fixed"
    HIGHEST_HOMOPHILY = "highest_homophily"
    LOWEST_HOMOPHILY = "lowest_homophily"


@unique
class FeaturePerturbationsParameter(StrEnum):
    """Method how to generate feature perturabtions."""

    AVERAGE = "average"
    AVERAGE_PER_CLASS = "average_per_class"


@unique
class FeaturePerturbationType(StrEnum):
    BERNOULLI = "bernoulli"
    NORMAL = "normal"
    NONE = "none"


@unique
class FeatureNormalization(StrEnum):
    L1 = "l1"
    L2 = "l2"
    NONE = "none"


@unique
class DatasetName(StrEnum):
    CORA = "cora"
    CORA_ML = "cora_ml"
    CORA_ML_LM = "cora_ml_lm"
    CITESEER = "citeseer"
    PUBMED = "pubmed"
    AMAZON_COMPUTERS = "amazon_computers"
    AMAZON_PHOTO = "amazon_photo"
    REDDIT = "reddit"
    OGBN_ARXIV = "ogbn_arxiv"
    COAUTHOR_CS = "coauthor_cs"
    COAUTHOR_PHYSICS = "coauthor_physics"


@unique
class DatasetSplit(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    ALL = "all"


@unique
class DistributionType(StrEnum):
    ID = "id"
    OOD = "ood"
    ALL = "all"


@unique
class DiffusionType(StrEnum):
    SYMMETRIC = "symmetric"
    STOCHASTIC = "stochastic"
    LABEL_PROPAGATION = "label_propagation"


class DistributionShiftConfig(TypedDict):
    """Distribution shift config."""

    type_: DistributionShiftType | str
    feature_perturbations: dict[str, dict[str, float]] | None
    left_out_classes: list[str | int] | None
    num_left_out_classes: int | None
    leave_out_classes_type: LeaveOutClassesType | str
    reorder_left_out_classes: bool
    num_ood: float
    teleport_probability: float
    num_iterations: int
    high_is_id: bool
    per_class: bool
    in_edges: bool
    percentile_ood: float
    num_hops: int


class DataConfig(TypedDict):
    """Data configuration."""

    root: str
    setting: Setting | str
    name: DatasetName | str
    categorical_features: bool
    feature_normalization: FeatureNormalization | str
    directed: bool
    largest_connected_component: bool
    remove_isolated_nodes: bool
    sentence_transformer: str | None
    select_classes: list[str] | None
    train_size: float | int
    test_size: float | int
    val_size: float | int
    allow_nodes_in_no_mask: bool
    allow_train_nodes_in_ood: bool
    num_splits: int
    split_idx: int | None
    precomputed: str | None
    max_num_split_attempts: int
    distribution_shift: DistributionShiftConfig
    registry: RegistryConfig


root = Path("/ceph/hdd/staff/fuchsgru/graph_uq_inductive_biases/data")
default_data_config = DataConfig(
    root=str(root),
    setting=Setting.INDUCTIVE,
    name=DatasetName.CORA_ML,
    categorical_features=True,
    # Transformations (Preprocessing)
    feature_normalization=FeatureNormalization.L2,
    directed=False,
    largest_connected_component=False,  # We run into issues for some datasets when this is True, as they are disconnected, e.g. Citeseer
    remove_isolated_nodes=False,
    sentence_transformer=None,
    select_classes=None,  # Subgraph of only some classes
    # Data split
    train_size=20,
    test_size=0.2,
    val_size=-1,
    allow_nodes_in_no_mask=True,
    allow_train_nodes_in_ood=False,
    num_splits=1,
    split_idx=None,
    # Distribution shift
    distribution_shift=DistributionShiftConfig(
        type_=DistributionShiftType.NONE,
        feature_perturbations=dict(),
        left_out_classes=None,
        num_left_out_classes=None,
        leave_out_classes_type=LeaveOutClassesType.LAST,
        reorder_left_out_classes=True,
        num_ood=0.5,
        teleport_probability=0.2,
        num_iterations=10,
        high_is_id=True,
        per_class=False,  # whether to use some proxy for splitting globally or per class
        in_edges=True,  # whether or not to use incoming edges for computing the shift
        percentile_ood=2,
        num_hops=2,
    ),
    precomputed=None,  # Path to precomputed, presplit and shifted data if used
    max_num_split_attempts=25,
    registry=RegistryConfig(
        database_path=str(Path(root) / "registry" / "db.json"),
        lockfile_path=str(Path(root) / "registry" / "db.json.lock"),
        storage_path=str(Path(root) / "registry" / "storage"),
    ),
)
del root


@experiment.config
def _default_data_config():
    data = default_data_config
