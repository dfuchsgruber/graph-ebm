import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, jaxtyped
from torch import Tensor
from typeguard import typechecked

from graph_uq.config.model import ModelConfig
from graph_uq.model.bnn.kl_divergence import kl_divergence_between_normals
from graph_uq.model.nn import parametrize_linear


@jaxtyped(typechecker=typechecked)
def kl_divergence_to_standard_normal(
    mean: Float[Tensor, "..."], std: Float[Tensor, "..."]
) -> Float[Tensor, ""]:
    """Computes the KL divergence of a diagonal normal to the standard normal prior."""
    kl = -0.5 * (std.log() * 2 - mean.pow(2) - std.pow(2) + 1).sum()
    return kl


class BayesianTensor(nn.Module):
    """A tensor in a BNN with a normal prior and a normal posterior."""

    def __init__(
        self,
        *shape,
        prior_mean=0.0,
        prior_std=1.0,
        mean_init: float = 0.0,
        rho_init: float = -3.0,
        std_init: float = 0.1,
    ):
        super().__init__()
        self.mean_init = mean_init
        self.rho_init = rho_init
        self.std_init = std_init
        self.prior_mean_value = prior_mean
        self.prior_std_value = prior_std

        self.mean = nn.Parameter(torch.Tensor(*shape))
        self.rho = nn.Parameter(torch.Tensor(*shape))
        self.register_buffer("eps", torch.Tensor(*shape), persistent=False)
        self.register_buffer("prior_mean", torch.Tensor(*shape), persistent=False)
        self.register_buffer("prior_std", torch.Tensor(*shape), persistent=False)
        self.reset_parameters()

    def reset_cache(self):
        self.eps.zero_()

    def reset_parameters(self):
        self.prior_mean.fill_(self.prior_mean_value)
        self.prior_std.fill_(self.prior_std_value)
        if self.std_init == 0.0:
            nn.init.constant_(self.mean, self.mean_init)
        else:
            nn.init.normal_(self.mean, mean=self.mean_init, std=self.std_init)
        nn.init.constant_(self.rho, self.rho_init)

    @property
    @jaxtyped(typechecker=typechecked)
    def kl_divergence(self) -> Float[Tensor, ""]:
        """Computes the KL divergence of the tensor to the prior."""
        kl = kl_divergence_between_normals(
            self.mean, self.rho.exp().log1p(), self.prior_mean, self.prior_std
        )
        return kl.mean()

    @jaxtyped(typechecker=typechecked)
    def forward(self) -> Float[Tensor, "..."]:
        """Samples a value from the posterior using the reparametrization trick."""
        sigma = self.rho.exp().log1p()
        eps = self.eps.normal_()
        return self.mean + sigma * eps


class BayesianLinear(nn.Module):
    """Bayesian linear layer that parametrizes its standard deviation sigma as as log(1 + exp(rho))"""

    @typechecked
    def __init__(
        self,
        config: ModelConfig,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.weight = parametrize_linear(
            BayesianTensor(
                out_dim,
                in_dim,
                prior_mean=config["prior_mean_weight"],
                prior_std=config["prior_std_weight"],
                mean_init=config["mean_init_weight"],
                rho_init=config["rho_init_weight"],
            ),
            config=config,
            name="mean",
        )

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

    @property
    @jaxtyped(typechecker=typechecked)
    def kl_divergence(self) -> Float[Tensor, ""] | None:
        """Computes the KL divergence of the weight and bias to the prior."""
        kl = self.weight.kl_divergence
        if self.bias is not None:
            kl += self.bias.kl_divergence
        return kl

    def reset_parameters(self):
        self.weight.reset_parameters()
        if self.bias is not None:
            self.bias.reset_parameters()

    def reset_cache(self): ...

    @jaxtyped(typechecker=typechecked)
    def forward(
        self, x: Float[Tensor, "*batch in_dim"]
    ) -> Float[Tensor, "*batch out_dim"]:
        weight = self.weight()
        bias = self.bias() if self.bias is not None else None
        return F.linear(x, weight, bias=bias)
