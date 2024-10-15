import dataclasses

import torch
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from typeguard import typechecked

from graph_uq.data.data import Data
from graph_uq.model.bnn.linear import BayesianLinear
from graph_uq.model.gnn import GNN, GNNBlock, GNNLayer
from graph_uq.model.prediction import Prediction


class BayesianGNNLayer(GNNLayer):
    """Base class for Bayesian GNN layers."""

    def reset_cache(self):
        raise NotImplementedError

    def reset_parameters(self):
        raise NotImplementedError

    @property
    @typechecked
    def num_kl_terms(self) -> int:
        raise NotImplementedError

    @property
    @jaxtyped(typechecker=typechecked)
    def kl_divergence(self) -> Float[Tensor, ""]:
        raise NotImplementedError


class BayesianGNNBlock(GNNBlock):
    """Base Bayesian GNN block that applies a convolution, activation, and dropout and residual connections."""

    downsampling_cls = BayesianLinear

    @property
    def prediction_changes_at_eval(self) -> bool:
        return True

    @property
    @typechecked
    def num_kl_terms(self) -> int:
        num_terms = self.convolution.num_kl_terms
        if self.downsampling is not None:
            num_terms += self.downsampling.num_kl_terms
        return num_terms

    @property
    @jaxtyped(typechecker=typechecked)
    def kl_divergence(self) -> Float[Tensor, ""]:
        try:
            kl_divergence = self.convolution.kl_divergence
            if self.downsampling is not None:
                kl_divergence += self.downsampling.kl_divergence
            return kl_divergence
        except Exception as e:
            raise e


class BayesianGNN(GNN):
    """Bayesian GNN sekeleton class that relies on BayesianGNNLayer layers."""

    @jaxtyped(typechecker=typechecked)
    def forward(self, batch: Data) -> Prediction:
        kl_divergence = sum(block.kl_divergence for block in self.blocks)
        if isinstance(kl_divergence, Tensor):
            kl_divergence = kl_divergence.unsqueeze(0)

        prediction = super().forward(batch)
        return dataclasses.replace(prediction, kl_divergence=kl_divergence)
