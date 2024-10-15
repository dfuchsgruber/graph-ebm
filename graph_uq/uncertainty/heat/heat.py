from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked

from graph_uq.config.heat import HeatProposalType
from graph_uq.uncertainty.heat.scorers.abstract_scorer import AbastractOODScorer

from .lib import requires_grad

NoneType = type(None)


class HybridEnergyModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: list[int] = [64],
        temperature: float = 2e-2,
        temperature_prior: float = 1e3,
        proposal_type: str = "random_normal",
        base_dist: AbastractOODScorer | None = None,
        use_base_dist: bool = False,
        sample_from_batch_statistics: bool = False,
        steps: int = 200,
        step_size_start: float = 1e-0,
        step_size_end: float = 1e-2,
        eps_start: float = 1e-1,
        eps_end: float = 1e-3,
        sgld_relu: bool = True,
        use_sgld: bool = True,
        train_max_iter: float | int | None = np.inf,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.temperature = temperature
        self.temperature_prior = temperature_prior
        self.proposal_type = proposal_type
        self.base_dist = AbastractOODScorer() if base_dist is None else base_dist
        self.use_base_dist = use_base_dist
        self.sample_from_batch_statistics = sample_from_batch_statistics
        self.steps = steps
        self.step_size_start = step_size_start
        self.step_size_end = step_size_end
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.sgld_relu = sgld_relu
        self.use_sgld = use_sgld
        self.train_max_iter = train_max_iter

        self._build_mlp(input_dim, hidden_dims)

    def _build_mlp(self, input_dim: int, hidden_dims: list[int]):
        in_dims, out_dims = [input_dim] + hidden_dims, hidden_dims + [1]
        layers = []
        for idx, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            layers.append(nn.Linear(in_dim, out_dim))
            if idx < len(hidden_dims):
                layers.append(nn.LeakyReLU(0.2))
        self.mlp = nn.Sequential(*layers)

    @jaxtyped(typechecker=typechecked)
    def energy_nn(
        self, z: Float[Tensor, "batch num_features"], labels: Optional[Tensor] = None
    ) -> Float[Tensor, "batch"]:
        return self.mlp(z).view(-1) / self.temperature

    def energy_prior(self, z: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        # prior_scores = - self.base_dist.log_prob(z, labels=labels)
        prior_scores = self.base_dist.energy(z, labels=labels)
        return prior_scores / self.temperature_prior

    def forward(
        self, z: Tensor, labels: Optional[Tensor] = None, nn_only: bool = False
    ) -> Tuple[Tensor]:
        energy_nn = self.energy_nn(z)  # .view(-1) placed inside energy_nn !

        energy_g = torch.zeros_like(energy_nn)
        if self.use_base_dist and (not nn_only):
            energy_g = self.energy_prior(z, labels=labels)

        assert energy_nn.shape == energy_g.shape
        energy = energy_nn + energy_g
        assert energy.ndim == 1 or (energy.ndim == 2 and energy.shape[1] == 1)

        return energy, energy_nn, energy_g  # type: ignore

    def energy(self, z: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        return self(z, labels=labels)[0]

    def log_prob(self, z: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        return -self(z, labels=labels)[0]

    @staticmethod
    def polynomial(
        t: int, T: int, init_val: float, end_val: float, power: float = 2.0
    ) -> float:
        return (init_val - end_val) * ((1 - t / T) ** power) + end_val

    def proposal_samples(
        self,
        sample_size: float,
        target: Optional[Tensor] = None,
        real_samples: Optional[Tensor] = None,
    ) -> tuple[Tensor | None, Tensor | None]:
        init_labels = None
        target = target if self.sample_from_batch_statistics else None
        dim = self.input_dim

        # drawing fresh samples
        if self.proposal_type == HeatProposalType.RANDOM_NORMAL:
            init_samples = torch.randn((sample_size, dim))  # type: ignore
        elif self.proposal_type == HeatProposalType.RANDOM_UNIFORM:
            init_samples = 2 * torch.rand((sample_size, dim)) - 4  # type: ignore
        elif self.proposal_type == HeatProposalType.BASE_DIST:
            assert (
                self.base_dist.has_sample
            ), f"base dist {self.base_dist.name} has not a sample method."
            init_labels, init_samples = self.base_dist.sample((sample_size,), target)
        elif self.proposal_type == HeatProposalType.BASE_DIST_TEMP:
            assert (
                self.base_dist.has_sample
            ), f"base dist {self.base_dist.name} has not a sample method."
            temp_scale = 100 * torch.rand((1,)).item()
            init_labels, init_samples = self.base_dist.sample(
                (sample_size,), target, temp_scale
            )
        elif self.proposal_type == HeatProposalType.DATA:
            init_labels = target
            init_samples = real_samples
        else:
            raise NotImplementedError

        # push to device
        init_samples = init_samples

        return init_labels, init_samples

    def negative_samples(
        self,
        init_samples: Optional[Tensor] = None,
        init_labels: Optional[Tensor] = None,
        steps: Optional[int] = None,
    ) -> Tensor:
        if self.use_sgld:
            assert init_samples is not None
            gen_z = self.sgld_samples(
                init_samples=init_samples, init_labels=init_labels, steps=steps
            )
        else:
            raise NotImplementedError

        return gen_z

    def sgld_samples(
        self,
        init_samples: Tensor,
        init_labels: Optional[Tensor] = None,
        steps: Optional[int] = None,
    ) -> Tensor:
        steps = self.steps if steps is None else steps
        is_training = self.training
        self.eval()
        requires_grad(self, False)

        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        if self.sgld_relu:
            init_samples = F.relu(init_samples)

        if self.base_dist.normalize:
            init_samples = F.normalize(init_samples, dim=1)

        z = init_samples.clone()
        z = z

        noise = torch.randn(z.shape, device=z.device)
        random_temperature = torch.FloatTensor(z.shape[0]).uniform_(1, 1).to(z.device)
        random_noise = torch.FloatTensor(z.shape[0]).uniform_(1, 1).to(z.device)

        for t in range(steps):
            z = z.detach().requires_grad_(True)
            noise.normal_(0, self.polynomial(t, steps, self.eps_start, self.eps_end))
            energy = self.energy(z, labels=init_labels)
            grad_Ez = torch.autograd.grad(energy.sum(), [z])[0]

            lr = (
                self.polynomial(t, steps, self.step_size_start, self.step_size_end)
                * random_temperature
            ).to(z.device)

            z.data -= torch.diag(lr).matmul(grad_Ez) + torch.diag(random_noise).matmul(
                noise
            )

            # Projection
            if self.sgld_relu:
                z = F.relu(z)

            if self.base_dist.normalize:
                z = F.normalize(z, dim=1)

        requires_grad(self, True)
        self.train(is_training)
        torch.set_grad_enabled(had_gradients_enabled)

        return torch.cat([z.detach(), init_samples.detach()])
