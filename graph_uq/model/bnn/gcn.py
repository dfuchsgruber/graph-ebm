import jaxtyping  # noqa: F401
import torch
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from typeguard import typechecked

from graph_uq.config.model import Activation, ModelConfig
from graph_uq.data.data import Data
from graph_uq.model.bnn import (
    BayesianGNN,
    BayesianGNNBlock,
    BayesianGNNLayer,
    BayesianLinear,
)
from graph_uq.model.bnn.linear import BayesianTensor
from graph_uq.model.gcn import GCNPropagation
from graph_uq.model.prediction import LayerPrediction


class BayesianGCNLayer(GCNPropagation, BayesianGNNLayer):
    """GCN layer that uses a bayesian linear layer"""

    @typechecked
    def __init__(
        self,
        config: ModelConfig,
        in_dim: int,
        out_dim: int,
        bias: bool | None = None,
    ) -> None:
        super().__init__(config)
        self.lin = BayesianLinear(config, in_dim, out_dim, bias=False)
        if bias:
            self.bias = BayesianTensor(
                out_dim,
                prior_mean=config["prior_mean_bias"],
                prior_std=config["prior_std_bias"],
                mean_init=config["mean_init_bias"],
                rho_init=config["rho_init_bias"],
                std_init=0.0,
            )
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        if self.bias is not None:
            self.bias.reset_parameters()

    def reset_cache(self):
        return super().reset_cache()

    @property
    @jaxtyped(typechecker=typechecked)
    def kl_divergence(self) -> Float[Tensor, ""]:
        kl = self.lin.kl_divergence
        if self.bias is not None:
            kl += self.bias.kl_divergence
        return kl or torch.tensor(0.0)

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
            x += self.bias()
        return LayerPrediction(embeddings=[x])


class BayesianGCNBlock(BayesianGNNBlock):
    """GCN block that applies a convolution, activation, and dropout and residual connections."""

    @typechecked
    def __init__(self, config: ModelConfig, in_dim: int, out_dim: int):
        super().__init__(config, in_dim, out_dim)
        self.convolution = BayesianGCNLayer(config, in_dim, out_dim)


class BayesianGCN(BayesianGNN):
    """A Bayesian GCN model."""

    @typechecked
    def make_block(
        self, config: ModelConfig, in_dim: int, out_dim: int, is_last_layer: bool
    ) -> BayesianGCNBlock:
        block = BayesianGCNBlock(config, in_dim, out_dim)
        if is_last_layer:
            block.activation = Activation.NONE
            block.dropout = 0.0
            block.batch_norm = None
        return block
