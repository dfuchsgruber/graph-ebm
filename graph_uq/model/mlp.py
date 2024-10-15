from typing import List, Tuple

import torch
import torch.nn as nn
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


class MLPLayer(GNNLayer):
    def __init__(self, config: ModelConfig, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin = Linear(config, in_dim, out_dim, bias=config["bias"])
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
        return LayerPrediction(embeddings=[self.lin(x)])

    def reset_cache(self):
        pass

    def reset_parameters(self):
        self.lin.reset_parameters()


class LinearBlock(GNNBlock):
    """GCN block that applies a convolution, activation, and dropout and residual connections."""

    @typechecked
    def __init__(self, config: ModelConfig, in_dim: int, out_dim: int):
        super().__init__(config, in_dim, out_dim)
        self.convolution = MLPLayer(config, in_dim, out_dim)


class MLP(GNN):
    """A GCN model."""

    @typechecked
    def make_block(
        self, config: ModelConfig, in_dim: int, out_dim: int, is_last_layer: bool
    ) -> GNNBlock:
        block = LinearBlock(config, in_dim, out_dim)
        if is_last_layer:
            block.activation = Activation.NONE
            block.dropout = 0.0
            block.batch_norm = None
        return block
