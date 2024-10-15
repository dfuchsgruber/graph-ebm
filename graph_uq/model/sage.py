from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.inits import zeros
from typeguard import typechecked

from graph_uq.config.model import Activation, ModelConfig
from graph_uq.data import Data
from graph_uq.experiment import experiment
from graph_uq.model.gnn import GNN, GNNBlock, GNNLayer
from graph_uq.model.nn import Linear
from graph_uq.model.prediction import LayerPrediction


class SAGELayer(GNNLayer):
    """A Graph SAGE layer."""

    @typechecked
    def __init__(
        self, config: ModelConfig, in_dim: int, out_dim: int, bias: bool | None = None
    ) -> None:
        super().__init__()
        bias = bias if bias is not None else config["bias"]
        self.project = config["project"]
        self.root_weight = config["root_weight"]
        self.normalize = config["normalize"]

        if self.project:
            self.lin = Linear(config, in_dim, out_dim, bias=True)
        else:
            self.register_parameter("lin", None)

        self.lin_l = Linear(config, in_dim, out_dim, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(config, in_dim, out_dim, bias=False)

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
        x0 = x
        if self.project:
            x = self.lin(x)

        if unpropagated:
            out = self.lin_r(x0)
        else:
            out = self.propagate(edge_index, x=(x, x))
            out = self.lin_l(out)

            if self.root_weight:
                out += self.lin_r(x0)

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        return LayerPrediction(embeddings=[out])

    def reset_parameters(self):
        if self.project:
            self.lin.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def reset_cache(self): ...


class SAGEBlock(GNNBlock):
    """SAGE block that applies a convolution, activation, and dropout and residual connections."""

    @typechecked
    def __init__(self, config: ModelConfig, in_dim: int, out_dim: int):
        super().__init__(config, in_dim, out_dim)
        self.convolution = SAGELayer(config, in_dim, out_dim)


class SAGE(GNN):
    """A Graph SAGE model."""

    @typechecked
    def make_block(
        self, config: ModelConfig, in_dim: int, out_dim: int, is_last_layer: bool
    ) -> GNNBlock:
        block = SAGEBlock(config, in_dim, out_dim)
        if is_last_layer:
            block.activation = Activation.NONE
            block.dropout = 0.0
            block.batch_norm = None
        return block
