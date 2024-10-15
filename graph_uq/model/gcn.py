import jaxtyping  # noqa: F401
import torch
import torch.nn as nn
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.inits import zeros
from typeguard import typechecked

from graph_uq.config.model import Activation, ModelConfig
from graph_uq.data import Data
from graph_uq.model.gnn import GNN, GNNBlock, GNNLayer
from graph_uq.model.nn import Linear
from graph_uq.model.prediction import LayerPrediction


class GCNPropagation(GNNLayer):
    """Base class for GCN propagation layers. Overrids the `propagate` method for GCN propagation."""

    # A global cache for the normalization coefficients that is shared across all instances of this class.
    cache: (
        tuple[
            Int[Tensor, "2 num_edges"],
            Float[Tensor, "num_edges *num_edge_features"],
            int,
            int,
        ]
        | None
    ) = None

    @typechecked
    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()

        add_self_loops = config["add_self_loops"]
        if add_self_loops is None:
            add_self_loops = config["normalize"]

        self.cached = config["cached"]
        self.add_self_loops = add_self_loops
        self.improved = config["improved"]
        self.normalize = config["normalize"]
        self.reset_cache()

    @jaxtyped(typechecker=typechecked)
    def gcn_propagate(
        self,
        x: Float[Tensor, "num_nodes dim"],
        edge_index: Int[Tensor, "2 num_edges"],
        edge_weight: Float[Tensor, " num_edges"] | None,
    ) -> Float[Tensor, "num_nodes dim"]:
        if self.normalize:
            if self.cache is None:
                edge_index, edge_weight = gcn_norm(  # type: ignore
                    edge_index,
                    edge_weight=edge_weight,
                    num_nodes=x.size(self.node_dim),
                    improved=self.improved,
                    add_self_loops=self.add_self_loops,
                )
                if self.cached:
                    self.cache = edge_index, edge_weight, x.size(0), edge_index.size(1)  # type: ignore

            else:
                edge_index, edge_weight, num_nodes, num_edges = self.cache
                assert x.size(0) == num_nodes, "Cached number of nodes mismatch"
                assert (
                    edge_index.size(1) == num_edges
                ), "Cached number of edges mismatch"

        return self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

    @jaxtyped(typechecker=typechecked)
    def message(
        self,
        x_j: Float[Tensor, "num_edges dim"],
        edge_weight: Float[Tensor, "num_edges"],
    ) -> Float[Tensor, "num_edges dim"]:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def reset_cache(self):
        self.cache = None

    def reset_parameters(self):
        super().reset_parameters()


class GCNLayer(GCNPropagation):
    """A GCN layer."""

    @typechecked
    def __init__(
        self, config: ModelConfig, in_dim: int, out_dim: int, bias: bool | None = None
    ) -> None:
        super().__init__(config)
        if bias is None:
            bias = config["bias"]
        self.lin = Linear(config, in_dim, out_dim, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()
        self.reset_cache()

    @jaxtyped(typechecker=typechecked)
    def forward(
        self,
        x: Float[Tensor, "num_nodes input_dim"],
        edge_index: Int[Tensor, "2 num_edges"],
        data: Data,
        edge_weight: Float[Tensor, "num_edges num_edge_features"] | None = None,
        unpropagated: bool = False,
    ) -> LayerPrediction:
        x = self.lin(x)
        if not unpropagated:
            x = self.gcn_propagate(x, edge_index, edge_weight=edge_weight)
        if self.bias is not None:
            x += self.bias
        return LayerPrediction(embeddings=[x])

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.bias is not None:
            zeros(self.bias)

    def reset_cache(self):
        return super().reset_cache()


class GCNBlock(GNNBlock):
    """GCN block that applies a convolution, activation, and dropout and residual connections."""

    @typechecked
    def __init__(self, config: ModelConfig, in_dim: int, out_dim: int):
        super().__init__(config, in_dim, out_dim)
        self.convolution = GCNLayer(config, in_dim, out_dim)


class GCN(GNN):
    """A GCN model."""

    @typechecked
    def make_block(
        self, config: ModelConfig, in_dim: int, out_dim: int, is_last_layer: bool
    ) -> GNNBlock:
        block = GCNBlock(config, in_dim, out_dim)
        if is_last_layer:
            block.activation = Activation.NONE
            block.dropout = 0.0
            block.batch_norm = None
        return block
