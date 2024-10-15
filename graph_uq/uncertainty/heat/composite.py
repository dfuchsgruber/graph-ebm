from copy import deepcopy
from typing import Any

import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor
from typeguard import typechecked

from graph_uq.config.heat import HeatCompositeConfig
from graph_uq.config.uncertainty import UncertaintyModelType
from graph_uq.data import Data
from graph_uq.metric import Metric
from graph_uq.model.base import BaseModel
from graph_uq.model.prediction import Prediction
from graph_uq.uncertainty.base import BaseUncertaintyModel

from .ebm import UncertaintyModelHeat


class UncertaintyModelHeatComposite(BaseUncertaintyModel):
    """HEAT."""

    uncertainty_model_type: UncertaintyModelType = UncertaintyModelType.HEAT_COMPOSITE

    def __init__(self, diffusion: dict[str, Any] = {}, **config: dict[str, Any]):
        super().__init__(diffusion=diffusion)
        self.config: HeatCompositeConfig = HeatCompositeConfig(**deepcopy(config))  # type: ignore
        self.members = {
            name: UncertaintyModelHeat(diffusion=diffusion, **config)  # type: ignore
            for name, config in self.config["ebms"].items()
        }

    def fit(self, data: Data, model: BaseModel, prediction: Prediction):
        for member in self.members.values():
            member.fit(data, model, prediction)

        # We learn the mean and std of energy scores of the members
        self.means, self.stds = {}, {}
        for name, member in self.members.items():
            uncertainties = member(data, model, prediction)
            for propagated in (True, False):
                energy = uncertainties[
                    Metric(
                        uncertainty_model_type=member.uncertainty_model_type,
                        propagated=propagated,
                    )
                ]
                mean = energy.mean().item()
                std = (energy - mean).std().item()

                self.means[(name, propagated)] = mean
                self.stds[(name, propagated)] = std

    @jaxtyped(typechecker=typechecked)
    @torch.no_grad()
    def __call__(
        self, data: Data, model: BaseModel, prediction: Prediction
    ) -> dict[Metric, Float[Tensor, " num_nodes"]]:
        """Computes the uncertainty estimates of this model (energy).

        Args:
            data (Data): The data for which to make uncertainty estimates
            model (BaseModel): The model for which to make uncertainty estimates
            prediction (Prediction): The prediction of the model on the data. For most estimators, this should be used instead using `model` and `data` directly.

        Returns:
            dict[Metric, Float[Tensor, 'num_nodes']]: The energy estimates
        """
        uncertainties = {}

        for name, member in self.members.items():
            uncertainties_member = {
                k + Metric(member_name=name): v
                for k, v in member(data, model, prediction).items()
            }

            overlap = uncertainties.keys() & uncertainties_member.keys()
            if overlap:
                raise RuntimeError(f"Overlapping keys: {overlap}")
            uncertainties |= uncertainties_member

        for propagated in (True, False):
            energies_scaled = []
            for name, member in self.members.items():
                mean, std = (
                    self.means[(name, propagated)],
                    self.stds[(name, propagated)],
                )

                energy = uncertainties[
                    member.energy_key
                    + Metric(
                        member_name=name,
                        propagated=propagated,
                    )
                ]
                energy = (energy - mean) / std
                energies_scaled.append(energy)
            energies_scaled = torch.stack(energies_scaled, dim=1)
            combined = (
                self.config["beta"]
                * torch.logsumexp(energies_scaled, dim=1)
                / self.config["beta"]
            )
            uncertainties |= {
                Metric(
                    propagated=propagated,
                    uncertainty_model_type=self.uncertainty_model_type,
                ): combined.cpu(),
            }
        return uncertainties
