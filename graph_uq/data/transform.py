from copy import copy
from typing import Sequence

import torch
import torch_geometric.transforms as T
from jaxtyping import Bool, jaxtyped
from torch_geometric.data import Data as TgData
from typeguard import typechecked

from graph_uq.config.data import DataConfig, FeatureNormalization


@typechecked
class ToData(T.BaseTransform):
    """Transforms a `torch_geometric.data.Data` instance into a `graph_uq.data.Data` instance.

    This extends its functionality."""

    def __call__(self, data: TgData) -> TgData:
        from graph_uq.data.data import Data

        attributes = {k: v for k, v in data}  # type: ignore
        if "node_names" not in attributes:
            attributes["node_names"] = [f"_node_{i}" for i in range(data.num_nodes)]  # type: ignore
        if "class_names" not in attributes:
            attributes["class_names"] = [
                f"_class_{i}"
                for i in range(data.y.max().item() + 1)  # type: ignore
            ]
        if "feature_names" not in attributes:
            attributes["feature_names"] = [
                f"_feature_{i}" for i in range(data.num_features)
            ]
        if "node_keys" not in attributes:
            attributes["node_keys"] = torch.arange(data.num_nodes)  # type: ignore

        match attributes["y"].size():
            case (data.num_nodes, 1):
                attributes["y"] = attributes["y"].squeeze(-1)
            case (data.num_nodes,):
                pass
            case _:
                raise ValueError(
                    f"Expected y to have shape (num_nodes,) or (num_nodes, 1), got {attributes['y'].size()}"
                )

        return Data(**attributes)  # type: ignore


@typechecked
class NormalizeFeatures(T.BaseTransform):
    """Normalizes the features of a data instance

    Args:
        normalization (FeatureNormalization): The normalization to apply
    """

    def __init__(self, normalization: str) -> None:
        super().__init__()
        self.normalization = normalization

    def __call__(self, data: TgData) -> TgData:
        data = copy(data)  # shallow copy
        assert isinstance(data.x, torch.Tensor)
        match self.normalization:
            case FeatureNormalization.L1:
                data.x /= data.x.norm(p=1, dim=-1, keepdim=True) + 1e-12
            case FeatureNormalization.L2:
                data.x /= data.x.norm(p=2, dim=-1, keepdim=True) + 1e-12
        return data


@typechecked
class PermuteLabels(T.BaseTransform):
    """Reorders the labels of a data instance

    Args:
        order (list[int]): The order in which old classes are mapped. That is, all nodes with label order[i] will be assigned label i
    """

    def __init__(self, order: list[int]) -> None:
        super().__init__()
        assert len(order) == len(
            set(order)
        ), f"Mutiple defintions of classes in {order}"
        assert (
            set(order) == set(range(max(order) + 1))
        ), f"Order should specify class indices from 0, ..., num_classes - 1, not {order}"
        self.order = order

    def __call__(self, data: TgData) -> TgData:
        data = copy(data)  # shallow copy
        data.permute_labels(self.order)
        return data


@typechecked
class ReorderLeftOutClassLabels(PermuteLabels):
    """Transformation that reorders the labels such that

    Args:
        PermuteLabels (_type_): _description_
    """

    def __init__(self, left_out_class_labels: list[int], num_classes_all: int):
        super().__init__(
            list(
                sorted(range(num_classes_all), key=lambda x: x in left_out_class_labels)
            )
        )


@jaxtyped(typechecker=typechecked)
def node_idxs_by_classes(
    data: TgData, classes: Sequence[int | str]
) -> Bool[torch.Tensor, " num_nodes_with_class"]:
    """Returns a mask for the nodes that belong to the given classes.

    Args:
        data (TgData): The graph data.
        classes (Sequence[int | str]): The classes to include.

    Returns:
        torch.Tensor: The mask.
    """
    class_idxs = set(
        idx if isinstance(idx, int) else data.class_names.index(idx) for idx in classes
    )
    assert isinstance(data.y, torch.Tensor)
    return torch.tensor([idx in class_idxs for idx in data.y.tolist()])


@typechecked
class SubgraphByClasses(T.BaseTransform):
    """Extracts the sugraph of some classes."""

    def __init__(
        self, classes: Sequence[int | str], reorder_labels: bool = True
    ) -> None:
        super().__init__()
        self.classes = list(classes)
        self.reorder_labels = reorder_labels

    def __call__(self, data: TgData) -> TgData:
        """Extracts the subgraph."""
        from graph_uq.data.data import Data

        assert isinstance(data, Data), f"Expected a `Data` instance, got {type(data)}"
        return data.class_subgraph(self.classes, self.reorder_labels)


@typechecked
def get_base_data_transform(config: DataConfig) -> T.BaseTransform:
    """Get the data transform for the base data."""
    transforms: list[T.BaseTransform] = []
    transforms.append(ToData())
    if not config["directed"]:
        transforms.append(T.ToUndirected())
    if config["select_classes"]:
        transforms.append(SubgraphByClasses(config["select_classes"]))
    if config["largest_connected_component"]:
        transforms.append(T.LargestConnectedComponents())
    if config["remove_isolated_nodes"]:
        transforms.append(T.RemoveIsolatedNodes())
    return T.Compose(transforms)


@typechecked
def get_data_transform(feature_normalization: str) -> T.BaseTransform:
    """Get the data transform for the dataset.

    Args:
        feature_normalization (str): The feature normalization to apply

    Returns:
        T.BaseTransform: The data transform
    """
    transforms = []
    if feature_normalization != FeatureNormalization.NONE:
        transforms.append(NormalizeFeatures(feature_normalization))
    return T.Compose(transforms)
