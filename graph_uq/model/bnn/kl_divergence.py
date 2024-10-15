from jaxtyping import Float, jaxtyped
from torch import Tensor
from typeguard import typechecked


@jaxtyped(typechecker=typechecked)
def kl_divergence_between_normals(
    mu_q: Float[Tensor, "..."],
    sigma_q: Float[Tensor, "..."],
    mu_p: Float[Tensor, "..."],
    sigma_p: Float[Tensor, "..."],
) -> Float[Tensor, "..."]:
    """Computes the KL divergence between two diagonal normals KL(q||p).

    Args:
        mu_q: Mean of the first normal.
        sigma_q: Standard deviation of the first normal.
        mu_p: Mean of the second normal.
        sigma_p: Standard deviation of the second normal.

    Returns:
        The KL divergence between the two normals.
    """
    kl = (
        sigma_p.log()
        - sigma_q.log()
        + 0.5 * ((sigma_q.pow(2) + (mu_q - mu_p).pow(2)) / sigma_p.pow(2) - 1)
    )
    return kl
