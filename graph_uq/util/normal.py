import logging

import numpy as np
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor
from typeguard import typechecked


@jaxtyped(typechecker=typechecked)
def cov_and_mean(
    x: Float[Tensor, "n d"],
    rowvar: bool = False,
    bias: bool = True,
    ddof: float | None = None,
    aweights: Float[Tensor, " n"] | None = None,
) -> tuple[Float[Tensor, "d d"], Float[Tensor, " d"]]:
    """Estimates covariance matrix like numpy.cov and also returns the weighted mean.

    Args:
        x (Float[Tensor, 'n d']): The tensor to estimate covariance of.
        rowvar (bool, optional): If given, columns are treated as observations. Defaults to False.
        bias (bool, optional): If given, use correction for the empirical covariance. Defaults to True.
        ddof (float | None, optional): Degrees of freedom. Defaults to None.
        aweights (Float[Tensor, 'n'] | None, optional): Weights for each observation. Defaults to None.

    Returns:
        tuple[Float[Tensor, 'd d'], Float[Tensor, 'd']]: Empirical covariance and mean.

    References:
    -----------
    From: https://github.com/pytorch/pytorch/issues/19037#issue-430654869
    """
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w / w_sum)[:, None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze(), avg


@jaxtyped(typechecker=typechecked)
def make_covariance_symmetric_and_psd(
    cov: Float[Tensor, "d d"],
    eps: float = 1e-8,
    tol: float = 1e-6,
    eps_multiplicator: float = 10.0,
) -> Float[Tensor, "d d"]:
    """Makes a covariance matrix symmetric and positive semi-definite.

    Args:
        cov (Float[Tensor, 'd d']): The covariance matrix.
        eps (float, optional): The epsilon to add to the diagonal. Defaults to 1e-12.
        tol (float, optional): The tolerance to check for symmetry. Defaults to 1e-6.
        eps_multiplicator (float, optional): The multiplicator for the epsilon. Defaults to 10.0.

    Returns:
        Float[Tensor, 'd d']: The symmetric and positive semi-definite covariance matrix.
    """
    while not (
        np.allclose(cov.numpy(), cov.numpy().T)
        and np.all(np.linalg.eigvalsh(cov.numpy()) > tol)
    ):
        # logging.warning(
        #     f"Covariance matrix was not symmetric or not positive semi-definite. Adding epsilon {eps} and making symmetric."
        # )
        cov += torch.eye(cov.numpy().shape[0]) * eps
        cov = (
            0.5 * (cov + cov.T)
        )  # Hacky way to make the matrix symmetric without changing its values too much (the diagonal stays intact for sure)
        eps *= eps_multiplicator
    return cov
