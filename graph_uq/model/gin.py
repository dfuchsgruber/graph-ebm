import torch
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from typeguard import typechecked

from graph_uq.config.model import Activation, ModelConfig
from graph_uq.data import Data
from graph_uq.model.gnn import GNN, GNNBlock, GNNLayer
from graph_uq.model.nn import Linear
from graph_uq.model.prediction import LayerPrediction


class GINLayer(GNNLayer):
    """A GIN layer."""

    @typechecked
    def __init__(
        self,
        config: ModelConfig,
        in_dim: int,
        out_dim: int,
        bias: bool | None = None,
    ) -> None:
        super().__init__()
        self.eps_trainable = config["eps_trainable"]
        self.eps_init = config["eps_init"]

        # We realize the neural network as a linear layer
        self.nn = Linear(config, in_dim, out_dim, bias=bias)

        if self.eps_trainable:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer("eps", torch.empty(1))

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
        if unpropagated:
            out = (1 + self.eps) * x
        else:
            out = self.propagate(edge_index, x=x)
            out += (1 + self.eps) * x

        return LayerPrediction(embeddings=[self.nn(out)])

    def reset_parameters(self):
        self.nn.reset_parameters()
        self.eps.data.fill_(self.eps_init)

    def reset_cache(self): ...


class GINBlock(GNNBlock):
    """GIN block that applies a convolution, activation, and dropout and residual connections."""

    @typechecked
    def __init__(self, config: ModelConfig, in_dim: int, out_dim: int):
        super().__init__(config, in_dim, out_dim)
        self.convolution = GINLayer(config, in_dim, out_dim)


class GIN(GNN):
    """A GIN model."""

    @typechecked
    def make_block(
        self, config: ModelConfig, in_dim: int, out_dim: int, is_last_layer: bool
    ) -> GNNBlock:
        block = GINBlock(config, in_dim, out_dim)
        if is_last_layer:
            block.activation = Activation.NONE
            block.dropout = 0.0
            block.batch_norm = None
        return block
