# Partially taken from: https://github.com/stadlmax/Graph-Posterior-Network

import math

import torch.nn as nn
from jaxtyping import Float, jaxtyped
from torch import Tensor
from typeguard import typechecked

from graph_uq.config.model import GPNEvidenceScale


class Evidence(nn.Module):
    """layer to transform density values into evidence representations according to a predefined scale"""

    @typechecked
    def __init__(self, scale: GPNEvidenceScale | str, tau: float | None = None):
        super().__init__()
        self.tau = tau
        self.scale = scale

    def __repr__(self):
        return f"Evidence(tau={self.tau}, scale={self.scale})"

    @jaxtyped(typechecker=typechecked)
    def forward(
        self, log_q: Float[Tensor, "num_nodes num_classes"], dim: int, num_classes: int
    ) -> Float[Tensor, "num_nodes num_classes"]:
        scaled_log_q = log_q + self.log_scale(dim, num_classes)
        if self.tau is not None:
            scaled_log_q = self.tau * (scaled_log_q / self.tau).tanh()
        scaled_log_q = scaled_log_q.clamp(min=-30.0, max=30.0)
        return scaled_log_q

    @typechecked
    def log_scale(self, dim: int, num_classes) -> float:
        scale = 0

        match self.scale:
            case GPNEvidenceScale.LATENT_OLD:
                scale = 0.5 * (dim * math.log(2 * math.pi) + math.log(dim + 1))
            case GPNEvidenceScale.LATENT_NEW:
                scale = 0.5 * dim * math.log(4 * math.pi)
            case GPNEvidenceScale.LATENT_NEW_PLUS_CLASSES:
                scale = 0.5 * dim * math.log(4 * math.pi) + math.log(num_classes)
            case _:
                raise ValueError(f"Unknown evidence scale: {self.scale}")

        return scale
