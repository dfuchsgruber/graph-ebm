import torch
import torch.nn as nn
from jaxtyping import Float, jaxtyped
from torch import Tensor
from typeguard import typechecked

from graph_uq.config.model import ModelConfig
from graph_uq.model.parametrization import (
    bjorck_orthonormalize,
    forbenius_normalization,
    rescaling,
    spectral_norm,
)


@typechecked
def parametrize_linear(
    linear: nn.Module,
    config: ModelConfig,
    name: str = "weight",
) -> nn.Module:
    """Adds paramerization to a linear layer."""
    if config["spectral_normalization"]:
        linear = spectral_norm(linear, name=name, rescaling=config["weight_scale"])
    if config["bjorck_normalization"]:
        linear = bjorck_orthonormalize(
            linear,
            name,
            n_iter=config["bjorck_orthonormalization_num_iterations"],
            rescaling=config["bjorck_orthonormal_scale"],
        )
    if config["frobenius_normalization"]:
        linear = forbenius_normalization(
            linear, name, rescaling=config["frobenius_norm"]
        )
    if config["rescale"]:
        linear = rescaling(linear, name, rescaling=config["weight_scale"])
    if config["initialization_scale"] != 1.0:
        with torch.no_grad():
            getattr(linear, name).mul_(config["initialization_scale"])
    return linear


class Linear(nn.Module):
    """Wrapper for a linear layer that different normalization schemes"""

    @typechecked
    def __init__(
        self,
        config: ModelConfig,
        input_dim: int,
        output_dim: int,
        bias: bool | None = None,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.linear = parametrize_linear(
            nn.Linear(
                input_dim,
                output_dim,
                *args,
                bias=bias if bias is not None else config["bias"],
                **kwargs,
            ),
            name="weight",
            config=config,
        )

    @typechecked
    def get_weights(self) -> dict[str, nn.Parameter]:
        """Gets the weight matrix."""
        return {"weight": self.linear.weight}

    def reset_cache(self):
        """Clears and disables the cache."""
        pass

    def reset_parameters(self):
        """Resets the parameters."""
        self.linear.reset_parameters()

    @jaxtyped(typechecker=typechecked)
    def forward(
        self, x: Float[Tensor, "*batch input_dim"], *args, **kwargs
    ) -> Float[Tensor, "*batch output_dim"]:
        return self.linear(x)
