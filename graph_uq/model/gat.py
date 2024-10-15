from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    softmax,
)
from typeguard import typechecked

from graph_uq.config.model import Activation, ModelConfig
from graph_uq.data import Data
from graph_uq.model.gnn import GNN, GNNBlock, GNNLayer
from graph_uq.model.nn import Linear
from graph_uq.model.prediction import LayerPrediction


class GATv2Layer(GNNLayer):
    """GAT layer.
    Mostly taken from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gat_conv.html#GATConv
    And https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gatv2_conv.html#GATv2Conv"""

    def __init__(
        self,
        config: ModelConfig,
        in_dim: int,
        out_dim: int,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = config["num_heads"]
        self.concatenate = config["concatenate"]
        self.leaky_relu_slope = config["leaky_relu_slope"]
        self.dropout = config["dropout"]
        self.add_self_loops = add_self_loops
        self.edge_dim = config["edge_dim"]
        self.fill_value = config["fill_value"]
        bias = config["bias"]

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        self.lin = self.lin_src = self.lin_dst = None
        if isinstance(in_dim, int):
            self.lin = Linear(config, in_dim, self.num_heads * out_dim, bias=False)
        else:
            self.lin_src = Linear(in_dim[0], self.num_heads * out_dim, False)
            self.lin_dst = Linear(in_dim[1], self.num_heads * out_dim, False)

        # The learnable parameters to compute attention coefficients:
        self.att = nn.Parameter(torch.empty(1, self.num_heads, out_dim))

        if self.edge_dim is not None:
            self.lin_edge = Linear(
                config,
                self.edge_dim,
                self.num_heads * out_dim,
                bias=False,
                weight_initializer="glorot",
            )
            self.att_edge = nn.Parameter(torch.empty(1, self.num_heads, out_dim))
        else:
            self.lin_edge = None
            self.register_parameter("att_edge", None)

        if bias and self.concatenate:
            self.bias = nn.Parameter(torch.empty(self.num_heads * out_dim))
        elif bias and not self.concatenate:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_cache(self): ...

    def reset_parameters(self):
        super().reset_parameters()
        if self.lin is not None:
            self.lin.reset_parameters()
        if self.lin_src is not None:
            self.lin_src.reset_parameters()
        if self.lin_dst is not None:
            self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(
        self,
        x: Tensor | tuple[Tensor, Tensor | None],
        edge_index: Adj,
        data: Data,
        edge_weight: Tensor | None = None,
        size: Size = None,
        return_attention_weights=None,
        unpropagated: bool = False,
    ):
        # forward_type: (Union[Tensor, tuple[Tensor, Tensor | None]], Tensor, Tensor | None, Size, NoneType) -> Tensor  # noqa
        # forward_type: (Union[Tensor, tuple[Tensor, Tensor | None]], SparseTensor, Tensor | None, Size, NoneType) -> Tensor  # noqa
        # forward_type: (Union[Tensor, tuple[Tensor, Tensor | None]], Tensor, Tensor | None, Size, bool) -> tuple[Tensor, tuple[Tensor, Tensor]]  # noqa
        # forward_type: (Union[Tensor, tuple[Tensor, Tensor | None]], SparseTensor, Tensor | None, Size, bool) -> tuple[Tensor, SparseTensor]  # noqa
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            size ((int, int), optional): The shape of the adjacency matrix.
                (default: :obj:`None`)
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        # NOTE: attention weights will be returned whenever
        # `return_attention_weights` is set to a value, regardless of its
        # actual value (might be `True` or `False`). This is a current somewhat
        # hacky workaround to allow for TorchScript support via the
        # `torch.jit._overload` decorator, as we can only change the output
        # arguments conditioned on type (`None` or `bool`), not based on its
        # actual value.

        H, C = self.num_heads, self.out_dim

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"

            if self.lin is not None:
                x_src = x_dst = self.lin(x).view(-1, H, C)
            else:
                # If the module is initialized as bipartite, transform source
                # and destination node features separately:
                assert self.lin_src is not None and self.lin_dst is not None
                x_src = self.lin_src(x).view(-1, H, C)
                x_dst = self.lin_dst(x).view(-1, H, C)

        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"

            if self.lin is not None:
                # If the module is initialized as non-bipartite, we expect that
                # source and destination node features have the same shape and
                # that they their transformations are shared:
                x_src = self.lin(x_src).view(-1, H, C)
                if x_dst is not None:
                    x_dst = self.lin(x_dst).view(-1, H, C)
            else:
                assert self.lin_src is not None and self.lin_dst is not None

                x_src = self.lin_src(x_src).view(-1, H, C)
                if x_dst is not None:
                    x_dst = self.lin_dst(x_dst).view(-1, H, C)

        assert x_src is not None
        assert x_dst is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
                edge_index, edge_weight = add_self_loops(
                    edge_index,
                    edge_weight,
                    fill_value=self.fill_value,
                    num_nodes=num_nodes,
                )
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)  # type: ignore
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_weight' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form"
                    )

        # edge_updater_type: (x: PairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, x=(x_src, x_dst), edge_attr=edge_weight)

        if unpropagated:
            out = x_src
        else:
            # propagate_type: (x: PairTensor, alpha: Tensor)
            out = self.propagate(edge_index, x=(x_src, x_dst), alpha=alpha)

        if self.concatenate:
            out = out.view(-1, self.num_heads * self.out_dim)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return LayerPrediction(
            embeddings=[out],
            attention_weights=alpha if return_attention_weights else None,
        )

    def edge_update(
        self,
        x_j: Tensor,
        x_i: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        dim_size: Optional[int],
    ) -> Tensor:
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            assert edge_attr is not None
            edge_attr = edge_attr.view(-1, self.num_heads, self.out_dim)
            x = x + edge_attr

        x = F.leaky_relu(x, self.leaky_relu_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_dim}, "
            f"{self.out_dim}, num_heads={self.num_heads})"
        )


class GATBlock(GNNBlock):
    """GAT block that applies a convolution, activation, and dropout and residual connections."""

    @typechecked
    def __init__(self, config: ModelConfig, in_dim: int, out_dim: int):
        super().__init__(config, in_dim, out_dim)
        self.convolution = GATv2Layer(config, in_dim, out_dim)


class GAT(GNN):
    """A GAT model."""

    @typechecked
    def make_block(
        self, config: ModelConfig, in_dim: int, out_dim: int, is_last_layer: bool
    ) -> GNNBlock:
        block = GATBlock(config, in_dim, out_dim)
        if is_last_layer:
            block.activation = Activation.NONE
            block.dropout = 0.0
            block.batch_norm = None
        return block
