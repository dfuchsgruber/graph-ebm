from itertools import product
from typing import Any

import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor
from typeguard import typechecked

from graph_uq.config.data import DatasetSplit
from graph_uq.config.uncertainty import EnergyAggregation, UncertaintyModelType
from graph_uq.data import Data
from graph_uq.metric import Metric
from graph_uq.model.base import BaseModel
from graph_uq.model.prediction import Prediction
from graph_uq.uncertainty.base import BaseUncertaintyModel
from graph_uq.uncertainty.corrected_conditional_ebm import (
    UncertaintyModelCorrectedConditionalEBM,
)
from graph_uq.uncertainty.latent_space.gaussian_per_class import (
    UncertaintyModelGaussianPerClass,
)
from graph_uq.util import logsumexp


class UncertaintyModelMultiScaleCorrectedConditionalEBM(
    UncertaintyModelCorrectedConditionalEBM
):
    """GEBM: Uncertainty model that combines corrected conditional EBMs on multiple scales."""

    uncertainty_model_type: UncertaintyModelType = (
        UncertaintyModelType.MULTI_SCALE_CORRECTED_CONDITIONAL_EBM
    )

    @typechecked
    def __init__(
        self,
        tied_covariance: bool,
        covariance_type: str,
        embedding_layer: int,
        fit_to_embeddings_propagated: bool,
        fit_to_logits_propagated: bool,
        evidence_diffusion: dict[str, Any],
        energy_diffusion: dict[str, Any],
        fit_to: str = DatasetSplit.TRAIN,
        temperature: float = 1.0,
        lambda_correction=1.0,
        lambda_diffused_conditional_evidence: float = 1.0,
        lambda_diffused_energy: float = 1.0,
        lambda_undiffused_energy: float = 1.0,
        aggregation: str = EnergyAggregation.LOGSUMEXP,
        diffusion: dict[str, Any] = {},
    ):
        super().__init__(
            tied_covariance=tied_covariance,
            covariance_type=covariance_type,
            embedding_layer=embedding_layer,
            fit_to_embeddings_propagated=fit_to_embeddings_propagated,
            fit_to_logits_propagated=fit_to_logits_propagated,
            fit_to=fit_to,
            temperature=temperature,
            lambda_correction=lambda_correction,
            evidence_diffusion=evidence_diffusion,
            diffusion=diffusion,
        )
        assert evidence_diffusion is not None
        self.energy_diffusion_config = energy_diffusion
        self.lambda_diffused_conditional_evidence = lambda_diffused_conditional_evidence
        self.lambda_diffused_energy = lambda_diffused_energy
        self.lambda_undiffused_energy = lambda_undiffused_energy
        self.aggregation = aggregation

    @jaxtyped(typechecker=typechecked)
    def energy(
        self,
        prediction: Prediction,
        embeddings_propagated: bool,
        logits_propagated: bool,
    ) -> Float[Tensor, "num_nodes"]:
        """Computes the energy of the undiffused evidence.

        Args:
            prediction (Prediction): The prediction of the model
            propagated (bool): Whether the prediction is propagated or not

        Returns:
            Float[Tensor, "num_nodes"]: The undiffused energy
        """
        conditional_evidence = self.conditional_evidence(
            prediction,
            embeddings_propagated=embeddings_propagated,
            logits_propagated=logits_propagated,
        )
        energy = -self.temperature * torch.logsumexp(
            conditional_evidence / self.temperature, dim=-1
        )
        return energy

    @jaxtyped(typechecker=typechecked)
    def diffused_energy(
        self,
        data: Data,
        prediction: Prediction,
        embeddings_propagated: bool,
        logits_propagated: bool,
        propagated: bool,
    ) -> Float[Tensor, "num_nodes"]:
        """Diffuses the energy of the (undiffused) evidence.

        Args:
            prediction (Prediction): The prediction of the model
            propagated (bool): Whether the prediction is propagated or not

        Returns:
            Float[Tensor, "num_nodes"]: The diffused energy
        """
        energy = self.energy(
            prediction,
            embeddings_propagated=embeddings_propagated,
            logits_propagated=logits_propagated,
        )
        if self.energy_diffusion_config is None or not propagated:
            return energy
        diffused = data.diffusion(energy.unsqueeze(-1), **self.energy_diffusion_config)[
            -1
        ].squeeze(-1)
        return diffused

    @jaxtyped(typechecker=typechecked)
    def get_uncertainty(
        self,
        data: Data,
        model: BaseModel,
        prediction: Prediction,
        propagated: bool = True,
        embeddings_propagated: bool = False,
        logits_propagated: bool = False,
    ) -> Float[Tensor, " num_nodes"]:
        """Computes the uncertainty estimate."""
        terms = []
        if self.lambda_diffused_conditional_evidence > 0:
            diffused_evidence = self.diffused_conditional_evidence(
                data,
                prediction,
                embeddings_propagated=embeddings_propagated,
                logits_propagated=logits_propagated,
                propagated=propagated,
            )
            diffused_evidence_energy = -self.temperature * torch.logsumexp(
                diffused_evidence / self.temperature, dim=-1
            )
            terms.append(
                self.lambda_diffused_conditional_evidence * diffused_evidence_energy
            )
        if self.lambda_diffused_energy > 0:
            diffused_energy = self.diffused_energy(
                data,
                prediction,
                embeddings_propagated=embeddings_propagated,
                logits_propagated=logits_propagated,
                propagated=propagated,
            )
            terms.append(self.lambda_diffused_energy * diffused_energy)
        if self.lambda_undiffused_energy > 0:
            energy = self.energy(
                prediction,
                embeddings_propagated=embeddings_propagated,
                logits_propagated=logits_propagated,
            )
            terms.append(self.lambda_undiffused_energy * energy)
        suffix = ""
        if not embeddings_propagated:
            suffix += "_embeddings_unpropagated"
        if not logits_propagated:
            suffix += "_logits_unpropagated"

        match self.aggregation:
            case EnergyAggregation.LOGSUMEXP:
                total = torch.logsumexp(torch.stack(terms, dim=0), dim=0)
            case EnergyAggregation.SUM:
                total = sum(terms)
            case _:
                raise ValueError(f"Unknown aggregation method {self.aggregation}")
        return total  # type: ignore

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, data: Data, model: BaseModel, prediction: Prediction
    ) -> dict[Metric, Float[Tensor, " num_nodes"]]:
        """Computes the uncertainty estimates of this model.

        Args:
            data (Data): The data for which to make uncertainty estimates
            model (BaseModel): The model for which to make uncertainty estimates
            prediction (Prediction): The prediction of the model on the data. For most estimators, this should be used instead using `model` and `data` directly.

        Returns:
            dict[Metric, Float[Tensor, 'num_nodes']]: The uncertainty estimates
        """
        if not self._fitted:
            raise ValueError("Model not fitted")

        uncertainties = {}
        for propagated, embeddings_propagated, logits_propagated in product(
            (True, False), repeat=3
        ):
            suffix = ""
            if not embeddings_propagated:
                suffix += "_embeddings_unpropagated"
            if not logits_propagated:
                suffix += "_logits_unpropagated"

            uncertainties[
                Metric(
                    propagated=propagated,
                    uncertainty_model_type=self.uncertainty_model_type,
                    suffix=suffix if len(suffix) else None,
                )
            ] = self.get_uncertainty(
                data,
                model,
                prediction,
                propagated=propagated,
                embeddings_propagated=embeddings_propagated,
                logits_propagated=logits_propagated,
            )

        return uncertainties
