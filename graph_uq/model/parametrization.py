# The code for this module was taken by the latest version of pytorch's Spectral Normalization module and adapted.
# https://pytorch.org/docs/stable/_modules/torch/nn/utils/parametrizations.html#spectral_norm

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import spectral_norm as torch_spectral_norm


class _SpectralNorm(Module):
    def __init__(
        self,
        weight: torch.Tensor,
        rescaling: float = 1.0,
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError(
                "Expected n_power_iterations to be positive, but "
                "got n_power_iterations={}".format(n_power_iterations)
            )
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.register_buffer("u", None)
        self.register_buffer("v", None)
        self.rescaling = rescaling

        weight_mat = self._reshape_weight_to_matrix(weight)
        self._update_vectors(weight_mat)

    def __repr__(self):
        return f"SpectralNorm(rescaling={self.rescaling}, n_iter={self.n_power_iterations})"

    def _reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(
                self.dim, *[d for d in range(weight_mat.dim()) if d != self.dim]
            )
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    @torch.no_grad()
    def _update_vectors(self, weight_mat: torch.Tensor) -> None:
        # See original note at torch/nn/utils/spectral_norm.py
        # NB: If `do_power_iteration` is set, the `u` and `v` vectors are
        #     updated in power iteration **in-place**. This is very important
        #     because in `DataParallel` forward, the vectors (being buffers) are
        #     broadcast from the parallelized module to each module replica,
        #     which is a new module object created on the fly. And each replica
        #     runs its own spectral norm power iteration. So simply assigning
        #     the updated vectors to the module this function runs on will cause
        #     the update to be lost forever. And the next time the parallelized
        #     module is replicated, the same randomly initialized vectors are
        #     broadcast and used!
        #
        #     Therefore, to make the change propagate back, we rely on two
        #     important behaviors (also enforced via tests):
        #       1. `DataParallel` doesn't clone storage if the broadcast tensor
        #          is already on correct device; and it makes sure that the
        #          parallelized module is already on `device[0]`.
        #       2. If the out tensor in `out=` kwarg has correct shape, it will
        #          just fill in the values.
        #     Therefore, since the same power iteration is performed on all
        #     devices, simply updating the tensors in-place will make sure that
        #     the module replica on `device[0]` will update the _u vector on the
        #     parallized module (by shared storage).
        #
        #    However, after we update `u` and `v` in-place, we need to **clone**
        #    them before using them to normalize the weight. This is to support
        #    backproping through two forward passes, e.g., the common pattern in
        #    GAN training: loss = D(real) - D(fake). Otherwise, engine will
        #    complain that variables needed to do backward for the first forward
        #    (i.e., the `u` and `v` vectors) are changed in the second forward.
        if self.u is None or self.v is None:  # type: ignore[has-type]
            # randomly initialize `u` and `v`
            h, w = weight_mat.size()
            self.u = F.normalize(
                weight_mat.new_empty(h).normal_(0, 1), dim=0, eps=self.eps
            )
            self.v = F.normalize(
                weight_mat.new_empty(w).normal_(0, 1), dim=0, eps=self.eps
            )

        for _ in range(self.n_power_iterations):
            # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
            # are the first left and right singular vectors.
            # This power iteration produces approximations of `u` and `v`.
            self.u = F.normalize(
                torch.mv(weight_mat, self.v), dim=0, eps=self.eps, out=self.u
            )  # type: ignore[has-type]
            self.v = F.normalize(
                torch.mv(weight_mat.t(), self.u),  # type: ignore[has-type]
                dim=0,
                eps=self.eps,
                out=self.v,
            )  # type: ignore[has-type]
        # See above on why we need to clone
        self.u = self.u.clone(memory_format=torch.contiguous_format)
        self.v = self.v.clone(memory_format=torch.contiguous_format)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        weight_mat = self._reshape_weight_to_matrix(weight)
        if self.training:
            self._update_vectors(weight_mat)
        sigma = torch.dot(self.u, torch.mv(weight_mat, self.v))
        return weight / sigma * self.rescaling

    def right_inverse(self, value: torch.Tensor) -> torch.Tensor:
        # we may want to assert here that the passed value already
        # satisfies constraints
        return value


class BjorckOrthonormalization(nn.Module):
    """Approximation of orthonormal weight matrices using Björck Orthonormalization.

    Parameters:
    -----------
    n_iter : int, optional, default: 1
    """

    def __init__(self, n_iter: int = 1, rescaling: float = 1.0):
        super().__init__()
        self.n_iter = n_iter
        self.rescaling = rescaling

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n_iter):
            w = 1.5 * w - 0.5 * w @ (w.T @ w)
        return w * self.rescaling

    def __repr__(self):
        return f"BjorckOrthonormalization(rescaling={self.rescaling}, n_iter={self.n_iter})"


class ForbeniusNormalization(nn.Module):
    """Normalizes the forbenius norm of a matrix.

    Parameters:
    -----------
    rescaling : float, optional, default: 1.0
        The forbenius norm of the matrix.
    """

    def __init__(self, rescaling: float = 1.0):
        super().__init__()
        self.rescaling = rescaling

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        scale = torch.linalg.norm(w, ord="fro") * self.rescaling
        return w * scale

    def __repr__(self):
        return f"ForbeniusNormalization(rescaling={self.rescaling})"


class Rescaling(nn.Module):
    """Rescales the weight matrix by a scalar offset.

    Parameters:
    -----------
    rescaling : float, optional, default: 1.0
        The rescaling factor.
    """

    def __init__(self, rescaling: float = 1.0):
        super().__init__()
        self.rescaling = rescaling

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return w * self.rescaling

    def __repr__(self):
        return f"Rescaling(rescaling={self.rescaling})"


def spectral_norm(
    module: Module,
    name: str = "weight",
    rescaling: float = 1.0,
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    dim: Optional[int] = None,
) -> Module:
    r"""Applies spectral normalization to a parameter in the given module.

    .. math::
        \mathbf{W}_{SN} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})},
        \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm.

    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        rescaling (float, optional): rescaling of the matrix
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with a new parametrization registered to the specified
        weight

    .. note::
        This function is implemented using the new parametrization functionality
        in :func:`torch.nn.utils.parametrize.register_parametrization`. It is a
        reimplementation of :func:`torch.nn.utils.spectral_norm`.

    .. note::
        If the `_SpectralNorm` module, i.e., `module.parametrization.weight[idx]`,
        is in training mode on removal, it will perform another power iteration.
        If you'd like to avoid this iteration, set the module to eval mode
        before its removal.
    """
    module = torch_spectral_norm(
        module, name=name, n_power_iterations=n_power_iterations, eps=eps, dim=dim
    )
    parametrize.register_parametrization(module, name, Rescaling(rescaling))
    return module


def bjorck_orthonormalize(
    module: Module, name: str = "weight", n_iter: int = 1, rescaling: float = 1.0
):
    """Applies Björck Orthonormalization as parametrization to a module's parameter.

    Parameters:
    -----------
    module : nn.Module
        The module to apply parametrization to.
    weight : str
        The weight to within the module to apply the parametrization to.
    n_iter : int, optional, default: 1
        The number of iterations to perform.
    rescaling : float, optional, default: 1.0
        Rescaling applied to the matrix.

    Returns:
    --------
    module : nn.Module
        The module that parametrization applied to it.
    """
    parametrize.register_parametrization(
        module, name, BjorckOrthonormalization(n_iter=n_iter, rescaling=rescaling)
    )
    return module


def forbenius_normalization(
    module: Module, name: str = "weight", rescaling: float = 1.0
):
    """Normalizes the Forbenius norm of a module's parameter.

    Parameters:
    -----------
    module : nn.Module
        The module to apply parametrization to.
    weight : str
        The weight to within the module to apply the parametrization to.
    rescaling : float, optional, default: 1.0
        The desired forbenius norm.

    Returns:
    --------
    module : nn.Module
        The module that parametrization applied to it.
    """
    parametrize.register_parametrization(
        module, name, ForbeniusNormalization(rescaling=rescaling)
    )
    return module


def rescaling(module: Module, name: str = "weight", rescaling: float = 1.0):
    """Rescales weight of a module's parameter.

    Parameters:
    -----------
    module : nn.Module
        The module to apply parametrization to.
    weight : str
        The weight to within the module to apply the parametrization to.
    rescaling : float, optional, default: 1.0
        The desired scale.

    Returns:
    --------
    module : nn.Module
        The module that parametrization applied to it.
    """
    parametrize.register_parametrization(module, name, Rescaling(rescaling=rescaling))
    return module
