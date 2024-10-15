import logging
from copy import deepcopy
from typing import Any

import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor
from typeguard import typechecked

from graph_uq.config.data import DatasetSplit
from graph_uq.config.heat import HeatConfig, HeatEBMType
from graph_uq.config.uncertainty import UncertaintyModelType
from graph_uq.data import Data
from graph_uq.metric import Metric
from graph_uq.model.base import BaseModel
from graph_uq.model.prediction import Prediction
from graph_uq.uncertainty.base import BaseUncertaintyModel

from .heat import HybridEnergyModel
from .scorers.energy_logits_scorer import EnergyLogitsScorer
from .scorers.ssd_scorer import SSDScorer
from .train import train_ebm


class UncertaintyModelHeat(BaseUncertaintyModel):
    """HEAT."""

    uncertainty_model_type: UncertaintyModelType = UncertaintyModelType.HEAT

    def __init__(self, diffusion: dict[str, Any] = {}, **config: dict[str, Any]):
        super().__init__(diffusion=diffusion)
        self.config: HeatConfig = HeatConfig(**deepcopy(config))  # type: ignore
        # Building of the heat model needs to be deferred to fitting, as only then will we know about the data...

    def fit(self, data: Data, model: BaseModel, prediction: Prediction):
        features = prediction.get_embeddings(
            propagated=self.config["fit_to_propagated_embeddings"],
            layer=self.config["layer"],
        )
        assert features is not None
        features = features.mean(0)

        match self.config["ebm_type_"]:
            case HeatEBMType.GMM:
                base_dist = SSDScorer(
                    use_gpu=self.config["use_gpu"],
                    diag_coefficient_only=self.config["diagonal_covariance"],
                )
            case HeatEBMType.LOGITS:
                base_dist = EnergyLogitsScorer()
                if not self.config["layer"] == -1:
                    logging.warning(
                        f"Using logits as base distribution, but layer is not -1, but {self.config['layer']}"
                    )

            case _:
                raise ValueError(f"Unknown EBM type {self.config['ebm_type_']}")

        self.ebm = HybridEnergyModel(
            input_dim=features.size(1),
            hidden_dims=self.config["hidden_dims"],
            base_dist=base_dist,
            temperature=self.config["temperature"],
            temperature_prior=self.config["temperature_prior"],
            proposal_type=self.config["proposal_type"],
            use_base_dist=self.config["use_base_distribution"],
            sample_from_batch_statistics=self.config["sample_from_batch_statistics"],
            steps=self.config["steps"],
            step_size_start=self.config["step_size_start"],
            step_size_end=self.config["step_size_end"],
            eps_start=self.config["eps_start"],
            eps_end=self.config["eps_end"],
            sgld_relu=self.config["sgld_relu"],
            use_sgld=True,
            train_max_iter=self.config["train_max_iter"],
        )

        train_ebm(
            self.ebm,
            features[data.get_mask(DatasetSplit.TRAIN)],
            data.y[data.get_mask(DatasetSplit.TRAIN)],
            self.config,  # type: ignore
        )

    @property
    def energy_key(self) -> Metric:
        """This key (plus propagation) will be the one aggregated by the combined HEAT."""
        return Metric(uncertainty_model_type=self.uncertainty_model_type)

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
        for propagated in (True, False):
            embeddings = prediction.get_embeddings(
                propagated=propagated, layer=self.config["layer"]
            )
            if embeddings is not None:
                embeddings = embeddings.mean(0)
                if self.config["use_gpu"] and torch.cuda.is_available():
                    embeddings = embeddings.cuda()
                # TODO: is there normalization needed here?
                energy, energy_correction, energy_prior = self.ebm(embeddings)
                uncertainties |= {
                    Metric(
                        propagated=propagated,
                        uncertainty_model_type=self.uncertainty_model_type,
                    ): energy.cpu(),
                    Metric(
                        propagated=propagated,
                        uncertainty_model_type=self.uncertainty_model_type,
                        suffix="_correction",
                    ): energy_correction.cpu(),
                    Metric(
                        propagated=propagated,
                        uncertainty_model_type=self.uncertainty_model_type,
                        suffix="_prior",
                    ): energy_prior.cpu(),
                }
        return uncertainties
