import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor
from typeguard import typechecked


@jaxtyped(typechecker=typechecked)
def sample_from_normal_with_reparametrization(
    mean: Float[Tensor, "*batch dim"], logsigma: Float[Tensor, "*batch dim"]
) -> Float[Tensor, "*batch dim"]:
    """Draws a sample from a normal distribution using reparametrization.

    Args:
        mean (Float[Tensor, '*batch dim']): The mean of the distribution.
        logsigma (Float[Tensor, '*batch dim']): The logarithm of the variance of the distribution.

    Returns:
        Float[Tensor, '*batch dim']: Samples from this distribution, differentiable w.r.t. to mean and logsigma.
    """
    sigma = torch.exp(logsigma)
    eps = torch.zeros_like(sigma, device=mean.device, requires_grad=False).normal_()
    return mean + eps * sigma


@jaxtyped(typechecker=typechecked)
def kl_divergence_to_diagonal_normal(
    mu: Float[Tensor, "*batch dim"],
    logsigma: Float[Tensor, "*batch dim"],
    prior_mu: Float[Tensor, "#batch dim"] | float,
    prior_sigma: Float[Tensor, "#batch dim"] | float,
) -> Float[Tensor, "#batch"]:
    """
    Computes the KL divergence between one diagonal Gaussian posterior and the diagonal Gaussian prior.

    Args:
        mu (Float[Tensor, '*batch dim']): Mean of the posterior Gaussian.
        logsigma (Float[Tensor, '*batch dim']): Variance of the posterior Gaussian.
        prior_mu (Float[Tensor, '#batch dim'] | float): Mean of the prior Gaussian.
        prior_sigma (Float[Tensor, '#batch dim'] | float): Variance of the prior Gaussian.

    Returns:
        Float[Tensor, '#batch']: The KL divergence between the posterior and the prior.
    """
    # See: https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/

    return 0.5 * torch.sum(
        logsigma.exp() / prior_sigma  # tr(prior_sigma^-1 * sigma0) as both are diagon
        + (
            (mu - prior_mu).pow(2) / (prior_sigma) ** 2
        )  # quadratic form, as sigma1 is diagonal we can express it that way
        - 1  # d
        +
        # ln(|sigma1| / |sigma0|) = ln(|sigma1|) - ln(|sigma0|)
        torch.log(prior_sigma)  # type: ignore
        - logsigma,
        dim=-1,
    )
