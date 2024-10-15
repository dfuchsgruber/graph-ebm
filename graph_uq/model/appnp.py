"""APPNP model."""

import logging

import torch.nn as nn
from jaxtyping import Float, jaxtyped
from torch import Tensor
from torch_geometric.nn import APPNP as TgAPPNPPropagation
from typeguard import typechecked

from graph_uq.config.model import ModelConfig
from graph_uq.data.data import Data
from graph_uq.model.mlp import MLP
from graph_uq.model.prediction import LayerPrediction


class APPNPPropagation(TgAPPNPPropagation):
    """Propagation for the APPNP."""

    def __init__(self, config: ModelConfig):
        super().__init__(
            config["num_diffusion_steps"],
            config["alpha"],
            cached=config["cached"],
            add_self_loops=config["add_self_loops"],
            normalize=config["normalize"],
        )

    def reset_cache(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

    def reset_parameters(self):
        return super().reset_parameters()


class APPNP(MLP):
    """APPNP model."""

    @typechecked
    def __init__(self, config: ModelConfig, data: Data, *args, **kwargs):
        super().__init__(config, data, *args, **kwargs)
        self.diffusion = APPNPPropagation(config)

    @jaxtyped(typechecker=typechecked)
    def forward_impl(
        self, x: Tensor, data: Data, unpropagated: bool = False
    ) -> list[LayerPrediction]:
        layers = super().forward_impl(x, data, unpropagated)
        # Apply the APPNP propagation

        if not unpropagated:
            undiffused = layers[-1]
            assert undiffused.embeddings is not None
            diffused = LayerPrediction(
                embeddings=[self.diffusion(undiffused.embeddings[-1], data.edge_index)]
            )
        else:
            diffused = layers[-1]

        return layers[:-1] + [diffused]

    def reset_cache(self):
        super().reset_cache()
        self.diffusion.reset_cache()

    def reset_parameters(self):
        super().reset_parameters()
        self.diffusion.reset_parameters()
