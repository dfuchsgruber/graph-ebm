from copy import deepcopy
from pathlib import Path

import torch
import torch_geometric.datasets as D
from jaxtyping import Bool, jaxtyped
from torch import Tensor
from typeguard import typechecked

from graph_uq.config.data import DataConfig, DatasetName

from .citation_full import CitationFull
from .data import Data
from .dataset import Dataset
from .transform import get_base_data_transform


@typechecked
def apply_distribution_shift_and_split(
    base_data: Data,
    config: DataConfig,
) -> Dataset:
    """Get the dataset dataset."""
    from .distribution_shift import apply_distribution_shift
    from .split import apply_dataset_split

    base_data = deepcopy(base_data)

    data_train, data_ood = apply_distribution_shift(base_data=base_data, config=config)
    dataset = Dataset(data_train=data_train, data_shifted=data_ood, data_base=base_data)
    apply_dataset_split(
        dataset,
        train_size=config["train_size"],
        val_size=config["val_size"],
        test_size=config["test_size"],
    )
    return dataset


@jaxtyped(typechecker=typechecked)
def get_ogbn_fixed_split(
    dataset,
    num_nodes: int,
) -> tuple[
    Bool[Tensor, " num_nodes"], Bool[Tensor, " num_nodes"], Bool[Tensor, " num_nodes"]
]:
    """Gets the fixed split from an ogbn dataset."""
    from ogb.nodeproppred import PygNodePropPredDataset

    assert isinstance(
        dataset, PygNodePropPredDataset
    ), f"Expected a `PygNodePropPredDataset` instance, got {type(dataset)}"
    split = dataset.get_idx_split()
    idx_train, idx_val, idx_test = split["train"], split["valid"], split["test"]  # type: ignore
    fixed_train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    fixed_train_mask[idx_train] = True  # type: ignore
    fixed_val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    fixed_val_mask[idx_val] = True  # type: ignore
    fixed_test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    fixed_test_mask[idx_test] = True  # type: ignore

    assert (
        fixed_train_mask.float() + fixed_val_mask.float() + fixed_test_mask.float()
        == 1.0
    ).all(), "The masks should be disjoint"

    return fixed_train_mask, fixed_val_mask, fixed_test_mask


@typechecked
def get_base_data(config: DataConfig) -> Data:
    transform = get_base_data_transform(config)
    match DatasetName(config["name"]):
        case DatasetName.CORA:
            data = CitationFull(
                Path(config["root"]),
                "Cora",
                transform=transform,
                sentence_transformer=config["sentence_transformer"],
            )[0]
        case DatasetName.CORA_ML | DatasetName.CORA_ML_LM:
            data = CitationFull(
                Path(config["root"]),
                "Cora_ML",
                transform=transform,
                sentence_transformer=config["sentence_transformer"],
            )[0]
        case DatasetName.CITESEER:
            data = CitationFull(
                Path(config["root"]),
                "CiteSeer",
                transform=transform,
                sentence_transformer=config["sentence_transformer"],
            )[0]
        case DatasetName.PUBMED:
            data = CitationFull(
                Path(config["root"]),
                "PubMed",
                transform=transform,
                sentence_transformer=config["sentence_transformer"],
            )[0]
        case DatasetName.AMAZON_COMPUTERS:
            data = D.Amazon(config["root"], "computers", transform=transform)[0]
        case DatasetName.AMAZON_PHOTO:
            data = D.Amazon(config["root"], "photo", transform=transform)[0]
        case DatasetName.REDDIT:
            data = D.Reddit(config["root"], transform=transform)[0]
        case DatasetName.OGBN_ARXIV:
            from ogb.nodeproppred import PygNodePropPredDataset

            dataset = PygNodePropPredDataset(
                name="ogbn-arxiv",
                root=str(Path(config["root"])),
                transform=None,  # type: ignore
            )
            data = dataset[0]
            data.train_mask, data.val_mask, data.test_mask = (  # type: ignore
                get_ogbn_fixed_split(dataset, data.x.size(0))  # type: ignore
            )
            data = transform(data)
        case DatasetName.COAUTHOR_CS:
            data = D.Coauthor(config["root"], "CS", transform=transform)[0]
        case DatasetName.COAUTHOR_PHYSICS:
            data = D.Coauthor(config["root"], "Physics", transform=transform)[0]
        case name:
            raise ValueError(f"Unknown dataset: {name}")
    assert isinstance(data, Data), f"Expected a Data instance, got {type(data)}"
    return data
