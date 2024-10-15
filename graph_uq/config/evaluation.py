from enum import StrEnum, unique
from pathlib import Path
from typing import Any, TypedDict

from graph_uq.config.registry import RegistryConfig
from graph_uq.experiment import experiment


@unique
class EvaluationType(StrEnum):
    CALIBRATION = "calibration"


@unique
class LatentSpaceVisualizationType(StrEnum):
    TSNE = "tsne"
    PCA = "pca"


class LatentSpaceConfig(TypedDict):
    first_embedding_layer: int
    last_embedding_layer: int | None
    propagation_types: list[bool]


class EvaluationConfig(TypedDict):
    use_gpu: bool
    calibration_num_bins: int
    latent_space: LatentSpaceConfig
    uncertainty_diffusion: dict[str, Any]
    precomputed: str | None
    registry: RegistryConfig


registry_root = Path("/ceph/hdd/staff/fuchsgru/graph_uq_inductive_biases/evaluation")
default_evaluation_config = EvaluationConfig(  # noqa: F841
    use_gpu=False,  # whether to use the GPU
    calibration_num_bins=20,  # the number of bins to use for calibration
    latent_space=LatentSpaceConfig(
        first_embedding_layer=1,  # the first embedding layer to visualize, we exclude 0 (the input)
        last_embedding_layer=None,  # all
        propagation_types=[True, False],  # which propagation types to visualize
    ),
    uncertainty_diffusion=dict(),
    precomputed=None,  # the precomputed evaluation
    registry=RegistryConfig(
        database_path=str(registry_root / "registry" / "db.json"),
        lockfile_path=str(registry_root / "registry" / "db.json.lock"),
        storage_path=str(registry_root / "registry" / "storage"),
    ),
)
del registry_root


@experiment.config
def _default_evaluation_config():
    evaluation = default_evaluation_config
