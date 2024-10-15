import copy
import functools
import math
from collections import defaultdict
from typing import Any, Final, Protocol, Self

import numpy as np
import rich
import torch
import torch_geometric.nn as tgnn
import torch_scatter
from jaxtyping import Bool, Float, Int, Shaped, jaxtyped
from networkx.algorithms.shortest_paths.unweighted import (
    single_source_shortest_path_length,
)
from rich.table import Table
from torch import Tensor
from torch_geometric.data import Data as TGData
from torch_geometric.utils import add_remaining_self_loops, select, to_networkx
from typeguard import typechecked

from graph_uq.config.data import DatasetSplit, DiffusionType, DistributionType
from graph_uq.util.ppr import approximate_ppr_matrix, approximate_ppr_scores


def cached_tg_function(func):
    """Wrapper that caches method calls on a `Data` object."""

    @functools.wraps(func)
    def inner(data, *args, **kwargs):
        func(data, *args, **kwargs)
        assert isinstance(
            data, Data
        ), f"Cached function {func.__name__} can only be used on Data objects, not on {data}"
        key = f'{data.CACHE_PREFIX}_{"_".join(map(str, (func.__name__, *args, *(x for tpl in kwargs.items() for x in tpl))))}'
        if not hasattr(data, key):
            setattr(data, key, func(data, *args, **kwargs))
        return getattr(data, key)

    return inner


def cached_tg_property(func):
    """Wrapper that caches properties on a `Data` object."""
    return property(cached_tg_function(func))


class DataProtocol(Protocol):
    x: Float[Tensor, "num_nodes num_features"]
    y: Int[Tensor, "num_nodes"]
    edge_index: Int[Tensor, "2 num_edges"]


class Data(DataProtocol, TGData):
    """Wrapper around torch_geometric.data.Data that adds some convenience functions."""

    def __init__(self, *args, **kwargs):
        TGData.__init__(self, *args, **kwargs)

    CACHE_PREFIX: Final[str] = (
        "cached"  # The prefix should *not* start with `_`, `pygeo` doesn't like it...
    )

    def reset_cache(self):
        """Resets all cached attributes."""
        attributes_to_delete: set[str] = set()
        for key in self._store.keys():
            if self.is_cached_attr(key):
                attributes_to_delete.add(key)
        for key in attributes_to_delete:
            delattr(self, key)

    def __copy__(self) -> Self:
        other = super().__copy__()
        other.reset_cache()  # Reset the cache
        return other

    def __deepcopy__(self, memo: Any) -> Self:
        other = super().__deepcopy__(memo)  # type: ignore
        other.reset_cache()  # Reset the cache
        return other

    @typechecked
    def is_cached_attr(self, key: str) -> bool:
        """Checks whether the attribute is a cached attribute."""
        assert key in self._store.keys(), f"Attribute {key} not found"
        return key.startswith(self.CACHE_PREFIX)

    @typechecked
    def is_class_attr(self, key: str) -> bool:
        """Checks whether the attribute is a label attribute."""
        # TODO: the "clean" way is to use a `LabelStorage(BaseStorage)`
        # class
        assert key in self._store.keys(), f"Attribute {key} not found"
        value = getattr(self, key)

        if isinstance(value, (list, tuple)) and len(value) == self.num_classes:
            size = len(value)
        elif isinstance(value, torch.Tensor):
            size = value.size(self.__cat_dim__(key, value))
        else:
            return False

        if size != self.num_classes:
            return False
        if size not in (self.num_nodes, self.num_edges):
            return True
        elif "class" in key or "label" in key:
            return True
        else:
            return False  # ambigious

    @typechecked
    def assert_masks_valid(
        self,
        allow_nodes_in_no_mask: bool = True,
        allow_train_nodes_in_ood: bool = False,
    ):
        assert self.train_mask is not None, "training mask is None"
        assert self.val_mask is not None, "validation mask is None"
        assert self.test_mask is not None, "test mask is None"

        assert (
            (self.train_mask.long() + self.val_mask.long() + self.test_mask.long()) <= 1
        ).all(), "masks are not mutually exclusive"
        if not allow_nodes_in_no_mask:
            assert not (
                (self.train_mask.long() + self.val_mask.long() + self.test_mask.long())
                < 1
            ).any(), "not all nodes are assigned to a mask"

        if not allow_train_nodes_in_ood and hasattr(self, "ood_mask"):
            assert not any(
                self.train_mask & self.ood_mask
            ), "training and ood mask are not mutually exclusive"

    @property
    @jaxtyped(typechecker=typechecked)
    def split_labels(self) -> Shaped[np.ndarray, "num_nodes"]:
        labels = np.array(["None"] * self.num_nodes).astype(str)
        if self.has_split_masks:
            for split in [DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST]:
                labels[self.get_mask(split).cpu().numpy()] = split.name
        return labels

    @property
    @jaxtyped(typechecker=typechecked)
    def distribution_labels(self) -> Shaped[np.ndarray, "num_nodes"]:
        labels = np.array(["None"] * self.num_nodes).astype(str)
        if self.has_distribution_masks:
            for distribution_type in [DistributionType.ID, DistributionType.OOD]:
                labels[self.get_distribution_mask(distribution_type).cpu().numpy()] = (
                    distribution_type.name
                )
        return labels

    @typechecked
    def copy(self: Self) -> Self:
        return copy.copy(self)

    @cached_tg_property
    @typechecked
    def num_classes(self) -> int:
        assert isinstance(self.y, torch.Tensor)
        return int(self.y.max().item()) + 1

    @cached_tg_property
    @typechecked
    def num_classes_train(self) -> int:
        return self.y[self.get_mask(DatasetSplit.TRAIN)].max().item() + 1  # type: ignore

    # ********** Degree **********

    @cached_tg_function
    @jaxtyped(typechecker=typechecked)
    def node_degree(self, _in: bool) -> Int[Tensor, "num_nodes"]:
        return torch_scatter.scatter_add(
            torch.ones(self.edge_index.size(1), dtype=int),
            self.edge_index[1 if _in else 0],
        )

    @property
    @typechecked
    def num_nodes(self) -> int:
        return self.x.size(0)

    @property
    @typechecked
    def num_edges(self) -> int:
        return self.edge_index.size(1)

    @property
    @typechecked
    def num_input_features(self) -> int:
        return self.x.size(1)

    @cached_tg_function
    @typechecked
    def num_intra_inter_class_edges(self, intra: bool) -> int:
        """The number of intra-class edges, i.e. edges that connect nodes of the same class."""
        is_intra_edge = self.y[self.edge_index[0]] == self.y[self.edge_index[1]]
        return (
            is_intra_edge.long().sum().item()
            if intra
            else (~is_intra_edge).long().sum().item()
        )

    @cached_tg_function
    @jaxtyped(typechecker=typechecked)
    def num_edges_per_class(self, _in: bool) -> Int[Tensor, "num_classes"]:
        """The number of edges per class."""
        return torch_scatter.scatter_add(
            torch.ones(self.edge_index.size(1), dtype=int),
            self.y[self.edge_index[1 if _in else 0]],
            dim_size=self.num_classes,
        )

    @cached_tg_function
    @jaxtyped(typechecker=typechecked)
    def num_intra_inter_class_edges_per_class(
        self, intra: bool, _in: bool
    ) -> Int[Tensor, "num_classes"]:
        """The number of intra-class edges per class."""
        mask = (
            self.y[self.edge_index[0]] == self.y[self.edge_index[1]]
            if intra
            else self.y[self.edge_index[0]] != self.y[self.edge_index[1]]
        )
        return torch_scatter.scatter_add(
            mask.long(),
            self.y[self.edge_index[1 if _in else 0]],
            dim_size=self.num_classes,
        )

    @cached_tg_function
    @jaxtyped(typechecker=typechecked)
    def num_intra_inter_class_edges_per_node(
        self, intra: bool, _in: bool
    ) -> Int[Tensor, "num_nodes"]:
        """The number of inter-class edges per node."""
        mask = (
            self.y[self.edge_index[0]] == self.y[self.edge_index[1]]
            if intra
            else self.y[self.edge_index[0]] != self.y[self.edge_index[1]]
        )
        return torch_scatter.scatter_add(
            mask.long(), self.edge_index[1 if _in else 0], dim_size=self.num_nodes
        )

    @jaxtyped(typechecker=typechecked)
    def get_mask(self, which: str) -> Bool[Tensor, "num_nodes"]:
        match which:
            case DatasetSplit.TRAIN:
                return self.train_mask
            case DatasetSplit.VAL:
                return self.val_mask
            case DatasetSplit.TEST:
                return self.test_mask
            case DatasetSplit.ALL:
                return self.train_mask | self.val_mask | self.test_mask
            case which:
                raise ValueError(f"Unknown split {which}")

    @jaxtyped(typechecker=typechecked)
    def get_distribution_mask(self, which: str) -> Bool[Tensor, "num_nodes"]:
        match which:
            case DistributionType.ID:
                return ~self.ood_mask
            case DistributionType.OOD:
                return self.ood_mask
            case DistributionType.ALL:
                return torch.ones_like(self.train_mask)
            case which:
                raise ValueError(f"Unknown split {which}")

    def reset_masks(self):
        """Deletes all masks."""
        if hasattr(self, "ood_mask"):
            delattr(self, "ood_mask")
        if hasattr(self, "train_mask"):
            delattr(self, "train_mask")
        if hasattr(self, "val_mask"):
            delattr(self, "val_mask")
        if hasattr(self, "test_mask"):
            delattr(self, "test_mask")

    def node_keys_to_node_idxs(
        self, keys: Int[Tensor, " num_node_keys"]
    ) -> Int[Tensor, " num_node_keys"]:
        """Converts a list of node keys to node indices."""
        assert len(self.node_keys) == len(
            set(self.node_keys.tolist())
        ), "node keys (self) have duplicates"
        node_key_to_idx = {key: idx for idx, key in enumerate(self.node_keys.tolist())}
        return torch.tensor([node_key_to_idx[key] for key in keys.tolist()])

    @property
    def has_split_masks(self) -> bool:
        return (
            hasattr(self, "train_mask")
            and hasattr(self, "val_mask")
            and hasattr(self, "test_mask")
        )

    @property
    def has_distribution_masks(self) -> bool:
        return hasattr(self, "ood_mask")

    @jaxtyped(typechecker=typechecked)
    def transfer_mask_to(
        self: "Data", mask: Bool[Tensor, " num_nodes_src"], data_dst: "Data"
    ) -> Bool[Tensor, " num_nodes_dst"]:
        """Transfer a mask from one dataset to another using the `node_keys` attribute."""
        assert hasattr(
            self, "node_keys"
        ), "Source data does not have a node_idx attribute"
        assert hasattr(
            data_dst, "node_keys"
        ), "Destination data does not have a node_idx attribute"
        key_to_idx_src = {key: idx for idx, key in enumerate(self.node_keys.tolist())}
        return torch.tensor(
            [
                mask[key_to_idx_src[key]]
                if key in key_to_idx_src
                else False  # If dst is not a subset of src, then the mask is False
                for key in data_dst.node_keys.tolist()
            ],
            dtype=torch.bool,
            device=mask.device,
        )

    def __repr__(self) -> str:
        return f"Data(n={self.num_nodes}, m={self.num_edges}, d={self.num_node_features}, num_classes={self.num_classes})"

    def print_summary(self, title: str | None = None):
        """Prints a summary of the data."""
        table = Table(title=title)
        table.add_row("Nodes", str(self.num_nodes))
        table.add_row("Edges", str(self.num_edges))
        table.add_row("Input features", str(self.num_input_features))
        table.add_row("Classes", str(self.num_classes))
        if self.has_split_masks:
            table.add_row("Classes in train", str(self.num_classes_train))
            table.add_row(
                "Train nodes",
                f"{self.get_mask(DatasetSplit.TRAIN).sum().item()} ({self.get_mask(DatasetSplit.TRAIN).sum().item() / self.num_nodes:.2f})",
            )
            table.add_row(
                "Validation nodes",
                f"{self.get_mask(DatasetSplit.VAL).sum().item()} ({self.get_mask(DatasetSplit.VAL).sum().item() / self.num_nodes:.2f})",
            )
            table.add_row(
                "Test nodes",
                f"{self.get_mask(DatasetSplit.TEST).sum().item()} ({self.get_mask(DatasetSplit.TEST).sum().item() / self.num_nodes:.2f})",
            )
        if self.has_distribution_masks:
            table.add_row(
                "In-distribution nodes",
                f"{self.get_distribution_mask(DistributionType.ID).sum().item()} ({self.get_distribution_mask(DistributionType.ID).sum().item() / self.num_nodes:.2f})",
            )
            table.add_row(
                "Out-of-distribution nodes",
                f"{self.get_distribution_mask(DistributionType.OOD).sum().item()} ({self.get_distribution_mask(DistributionType.OOD).sum().item() / self.num_nodes:.2f})",
            )
        rich.print(table)

    @jaxtyped(typechecker=typechecked)
    def symmetric_diffusion(
        self,
        signal: Float[torch.Tensor, " num_nodes num_features"],
        k: int,
        normalize: bool = True,
        improved: bool = False,
        add_self_loops: bool = True,
        alpha: float = 0.0,
    ) -> list[Float[Tensor, " num_nodes num_features"]]:
        """Uses GCN-like diffusion to get a diffused node signal.

        $\alpha * h + (1 - \alpha)(D^{-0.5}AD^{-0.5})h$
        """

        edge_index, edge_weight = self.edge_index, self.edge_weight
        x = signal
        if normalize:
            edge_index, edge_weight = tgnn.conv.gcn_conv.gcn_norm(  # yapf: disable
                edge_index,
                edge_weight,
                x.size(0),
                improved=improved,
                add_self_loops=add_self_loops,
            )
        elif edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

        src, dst = edge_index
        result = [x]
        for _ in range(k):
            # Perform message passing
            messages = x[src] * edge_weight[:, None]  # type: ignore
            x = alpha * x + (1 - alpha) * torch_scatter.scatter_add(
                messages, dst, dim=0, dim_size=x.size(0)
            )
            result.append(x)
        return result

    @jaxtyped(typechecker=typechecked)
    def label_propagation_diffusion(
        self,
        signal: Float[torch.Tensor, "num_nodes num_features"],
        k: int,
        alpha: float = 0.5,
        add_self_loops: bool = True,
    ) -> list[Float[Tensor, "num_nodes num_features"]]:
        """Uses label propagation diffusion (using the row normalized adjacency matrix) to diffuse a node signal.

        $\alpha * h + (1 - \alpha)(D^{-1}A)h$

        Note that this does not represent a stochastic diffusion, as the diffusion operator is row-normalized, not
        column-normalized.

        Args:
            signal (Float[Tensor, 'num_nodes num_features']): The node signal to diffuse
            k (int): The number of diffusion steps
            cache (bool): Whether to cache the result
            alpha (float): The alpha parameter for the label propagation diffusion

        Returns:
            list[Float[Tensor, 'num_nodes num_features']]: The diffused node signals
        """
        (src, dst), edge_weight = self.stochastic_adjacency_edge_weights(
            row=True, add_self_loops=add_self_loops
        )
        edge_weight = edge_weight.to(signal.device)

        result = [signal]
        for _ in range(k):
            signal = (alpha * signal) + (
                (1 - alpha)
                * torch_scatter.scatter_add(
                    signal[dst] * edge_weight.view(-1, 1),
                    src,
                    dim_size=self.num_nodes,
                    dim=0,
                )
            )
            result.append(signal)
        return result

    @jaxtyped(typechecker=typechecked)
    def stochastic_diffusion(
        self,
        signal: Float[torch.Tensor, "num_nodes num_features"],
        k: int,
        alpha: float = 0.5,
        add_self_loops: bool = True,
    ) -> list[Float[Tensor, "num_nodes num_features"]]:
        """Uses stochastic diffusion (using the stochastic adjacency matrix) to diffuse a node signal.

        $\alpha * h + (1 - \alpha)({D^-1 A})^T h$
        """
        (src, dst), edge_weight = self.stochastic_adjacency_edge_weights(
            row=True, add_self_loops=add_self_loops
        )
        result = [signal]
        for _ in range(k):
            signal = (alpha * signal) + (
                (1 - alpha)
                * torch_scatter.scatter_add(
                    signal[src] * edge_weight.view(-1, 1),
                    dst,
                    dim_size=self.num_nodes,
                    dim=0,
                )
            )
            result.append(signal)
        return result

    @jaxtyped(typechecker=typechecked)
    def diffusion(
        self,
        signal: Shaped[Tensor, "num_nodes ..."],
        k: int,
        type_: DiffusionType | str,
        **kwargs,
    ) -> list[Shaped[Tensor, "num_nodes ..."]]:
        """Diffuses a signal on this graph."""
        match type_:
            case DiffusionType.SYMMETRIC:
                return self.symmetric_diffusion(signal, k, **kwargs)
            case DiffusionType.STOCHASTIC:
                return self.stochastic_diffusion(signal, k, **kwargs)
            case DiffusionType.LABEL_PROPAGATION:
                return self.label_propagation_diffusion(signal, k, **kwargs)
            case type_:
                raise ValueError(f"Unkown diffusion type {type_}")

    @cached_tg_function
    @jaxtyped(typechecker=typechecked)
    def get_diffused_nodes_features(
        self, k: int, diffusion_type: DiffusionType = DiffusionType.SYMMETRIC, **kwargs
    ) -> Float[Tensor, "num_nodes num_features"]:
        """Gets (and caches) nodes features after graph diffusion."""
        if k == 0:
            return self.x
        return self.diffusion(self.x, k, diffusion_type, **kwargs)[k]

    @cached_tg_function
    @jaxtyped(typechecker=typechecked)
    def get_log_appr_matrix(
        self,
        teleport_probability: float = 0.2,
        num_iterations: int = 10,
        add_self_loops: bool = True,
    ) -> Float[Tensor, "num_nodes num_nodes"]:
        """Lazily computes approximate log personalized page rank scores between all node pairs

        Args:
            teleport_probability (float): Teleport probability
            num_iterations (int): For how many iterations to do power iteration
            add_self_loops (bool): Whether to add self loops to the adjacency matrix. Default is True.

        Returns:
            Float[Tensor, 'num_nodes num_nodes']: The log APPR matrix log(Pi), where Pi_ij is the importance of node i to node j
        """
        log_appr_matrix = (
            approximate_ppr_matrix(
                *self.stochastic_adjacency_edge_weights(
                    row=True, add_self_loops=add_self_loops
                ),
                teleport_probability=teleport_probability,
                num_iterations=num_iterations,
                verbose=True,
                num_nodes=self.num_nodes,
            )
            .cpu()
            .log()
        )
        return log_appr_matrix

    @cached_tg_function
    @jaxtyped(typechecker=typechecked)
    def get_appr_scores(
        self,
        teleport_probability: float = 0.2,
        num_iterations: int = 10,
        add_self_loops: bool = True,
    ) -> Float[Tensor, "num_nodes"]:
        """Lazily computes APPR scores for every node as a centrality measure.

        Args:
            teleport_probability (float): Teleport probability
            num_iterations (int): For how many iterations to do power iteration
            add_self_loops (bool): Whether to add self loops to the adjacency matrix. Default is True.

        Returns:
            Float[Tensor, 'num_nodes num_nodes']: The log APPR matrix log(Pi), where
        """
        return approximate_ppr_scores(
            *self.stochastic_adjacency_edge_weights(
                row=True, add_self_loops=add_self_loops
            ),
            teleport_probability=teleport_probability,
            num_iterations=num_iterations,
            verbose=True,
            num_nodes=self.num_nodes,
        ).cpu()

    @jaxtyped(typechecker=typechecked)
    def get_shortest_path_distances_to(
        self, idxs: Int[Tensor, " n"], cutoff: int = 10
    ) -> Float[Tensor, "n num_nodes"]:
        """Lazily computes shortest path distances from all nodes to all nodes in idxs

        Args:
            idxs (Int[Tensor, 'n']): The indices of the nodes to compute shortest path distances to
            cutoff (int): Cutoff for shortest path computation

        Returns:
            Float[Tensor, 'num_nodes_train num_nodes']: The shortest path distances from all nodes to all nodes in idxs
        """

        edge_index, _ = add_remaining_self_loops(
            self.edge_index, edge_attr=self.edge_weight, num_nodes=self.num_nodes
        )
        nx_graph = to_networkx(Data(edge_index=edge_index, num_nodes=self.num_nodes))

        distances = []
        for idx in idxs.tolist():
            distances_to_idx = single_source_shortest_path_length(
                nx_graph, source=idx, cutoff=cutoff
            )
            distances.append(
                [
                    distances_to_idx.get(idx, float("inf"))
                    for idx in range(self.num_nodes)
                ]
            )
        distances = torch.Tensor(distances).to(self.edge_index.device).float()
        return distances

    @cached_tg_function
    @jaxtyped(typechecker=typechecked)
    def gdk(
        self, cutoff: int = 10, sigma: float = 1.0
    ) -> Float[Tensor, "num_nodes num_classes"]:
        """Computes the Graph Dirichlet Kernel (GDK)

        Args:
            cutoff (int): Cutoff for shortest path computation
            sigma (float): The sigma parameter for the Gaussian kernel

        Returns:
            Float[Tensor, 'num_nodes num_classes']: The GDK
        """
        idxs_train = torch.where(self.get_mask(DatasetSplit.TRAIN))[0]
        shortest_path_distances_to_train: Float[Tensor, "num_train num_nodes"] = (
            self.get_shortest_path_distances_to(idxs_train, cutoff=cutoff)
        )

        evidence = torch.exp(
            -0.5 * (shortest_path_distances_to_train**2) / (sigma**2)
        ) / (sigma * math.sqrt(2 * math.pi))
        evidence = torch_scatter.scatter_add(
            evidence, self.y[idxs_train], dim=0, dim_size=self.num_classes
        )
        return evidence.T

    @cached_tg_function
    @jaxtyped(typechecker=typechecked)
    def stochastic_adjacency_edge_weights(
        self, row: bool = True, add_self_loops: bool = True
    ) -> tuple[Int[Tensor, "2 num_edges"], Float[Tensor, "num_edges"]]:
        """Gets the weight of the stochastic adjacency matrix that is either row or column normalized.

        Parameters:
            row (bool): Whether to row normalize the adjacency matrix (column normalize otherwise).
            add_self_loops (bool): Whether to add self loops to the adjacency matrix.

        Returns:
            edge_index: Int[Tensor, '2 num_edges']: The edge index, where edge_index[0] are the source nodes and edge_index[1] are the target nodes
            edge_weights: Float[Tensor, 'num_edges']: The edge weights.
        """
        edge_index, edge_weights = self.edge_index, self.edge_weight
        if add_self_loops:
            edge_index, edge_weights = add_remaining_self_loops(
                self.edge_index, edge_attr=self.edge_weight, num_nodes=self.num_nodes
            )
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))

        degree = torch_scatter.scatter_add(
            edge_weights, edge_index[0 if row else 1], dim_size=self.num_nodes
        )
        if add_self_loops:
            assert (degree > 0).all(), "Found nodes without outgoing edges"

        edge_weights = (1 / degree)[edge_index[0 if row else 1]]
        if not add_self_loops:
            # Isolated nodes may have degree 0, we set the row in the adjacency to zero
            edge_weights = edge_weights = torch.nan_to_num(
                edge_weights, nan=0.0, posinf=0.0, neginf=0.0
            )

        # Assertions for normalization
        degree_normalized = torch_scatter.scatter_add(
            edge_weights, edge_index[0 if row else 1], dim_size=self.num_nodes
        )
        if add_self_loops:
            is_close = torch.isclose(
                degree_normalized, torch.ones_like(degree_normalized), atol=1e-3
            )
            assert is_close.all(), f"Stochastic adjacency matrix is not normalized {degree_normalized[~is_close]}"
        else:
            # Zero degree is also allowed for isolated nodes
            is_close = torch.isclose(
                degree_normalized, torch.ones_like(degree_normalized), atol=1e-3
            ) | torch.isclose(
                degree_normalized, torch.zeros_like(degree_normalized), atol=1e-3
            )
            assert is_close.all(), f"Stochastic adjacency matrix is not normalized {degree_normalized[~is_close]}"

        return edge_index, edge_weights

    @cached_tg_property
    @jaxtyped(typechecker=typechecked)
    def class_counts(self) -> Int[Tensor, "num_classes"]:
        return torch_scatter.scatter_add(
            torch.ones_like(self.y), self.y, dim_size=self.num_classes
        )

    @cached_tg_property
    @jaxtyped(typechecker=typechecked)
    def class_counts_train(self) -> Int[Tensor, "num_classes_train"]:
        y = self.y[self.get_mask(DatasetSplit.TRAIN)]
        return torch_scatter.scatter_add(
            torch.ones_like(y), y, dim_size=self.num_classes_train
        )

    @cached_tg_property
    @jaxtyped(typechecker=typechecked)
    def class_prior_probabilities_train(self) -> Float[Tensor, "num_classes_train"]:
        counts = self.class_counts_train
        if counts.sum() == 0:
            counts = torch.ones(
                self.num_classes_train, device=counts.device, dtype=torch.float
            )
        else:
            counts = counts.float()
        return counts / (counts.sum() + 1e-12)

    @cached_tg_function
    @jaxtyped(typechecker=typechecked)
    def get_ego_graph_sizes(self, num_hops: int) -> Int[Tensor, "num_nodes"]:
        """Gets the number of nodes in the ego graph of each node."""

        if self.edge_index.size(1) > 1_000_000:
            # Use a slower approach that is less demading on the memory
            adjacent = defaultdict(set)
            ego_graph_sizes = torch.zeros(self.num_nodes, dtype=int)
            for src, dst in self.edge_index.T.tolist():
                adjacent[src].add(dst)

            for node in range(self.num_nodes):
                closed = set()
                frontier = {node}
                for _ in range(num_hops):
                    frontier_new = {adj for node in frontier for adj in adjacent[node]}
                    closed |= frontier
                    frontier = frontier_new - closed
                ego_graph_sizes[node] = len(closed)
            return ego_graph_sizes
        else:
            adjacency = torch.sparse_coo_tensor(
                self.edge_index,
                torch.ones(self.num_edges),
                (self.num_nodes, self.num_nodes),
            ).to_sparse_csr()
            ego_graph_indicator = torch.sparse_coo_tensor(
                torch.arange(self.num_nodes).repeat(2).reshape(2, self.num_nodes),
                torch.ones(self.num_nodes),
                (self.num_nodes, self.num_nodes),
            ).to_sparse_csr()

            k_hop_adjacency = adjacency.clone()
            for _ in range(num_hops):
                k_hop_adjacency = adjacency @ k_hop_adjacency
                ego_graph_indicator += k_hop_adjacency

            indices = ego_graph_indicator.to_sparse_coo().indices()
            return torch_scatter.scatter_add(
                torch.ones(indices.shape[1], dtype=int),
                indices[0],
                dim_size=self.num_nodes,
            )

    @typechecked
    def permute_labels(
        self,
        permutation: Int[Tensor, "num_classes"],
    ):
        """Permutes the labels according to the given permutation."""
        assert set(permutation.tolist()) == set(
            range(self.num_classes)
        ), f"Permutation does not contain all classes but {set(permutation.tolist())}"

        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(
            permutation.size(0), dtype=permutation.dtype
        )
        self.y = inverse[self.y]

        for key in {k for k in list(self._store.keys()) if self.is_class_attr(k)}:
            value = getattr(self, key)
            if isinstance(value, (torch.Tensor, list, tuple)):
                cat_dim = self.__cat_dim__(key, value)
                setattr(self, key, select(value, permutation, dim=cat_dim))

    @typechecked
    def class_subgraph(
        self, classes: list[int | str], reorder_labels: bool = True
    ) -> Self:
        """Extracts the subgraph of some classes."""

        class_attributes = {
            k for k in list(self._store.keys()) if self.is_class_attr(k)
        }

        data = self.subgraph()
        if reorder_labels:
            permutation = torch.tensor(
                sorted(
                    range(data.y.max().item() + 1),
                    key=lambda x: classes.index(x) if x in classes else len(classes),
                ),
                dtype=int,
            )
            data.permute_labels(permutation)

        # Even with non-reordered labeles, the number of classes may have shrunk
        # if the tailing classes are not present in `data.y` anymore (`num_classes`
        # is dynamic and no fixed attribute...). Hence, we keep the attributes for
        # classes consistent
        subset = torch.arange(data.num_classes, dtype=int)
        for key in class_attributes:
            value = data[key]
            cat_dim = self.__cat_dim__(key, value)
            data[key] = select(value, subset, dim=cat_dim)
        return data
