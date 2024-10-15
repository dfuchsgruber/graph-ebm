from itertools import product
from typing import Any

import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor
from typeguard import typechecked

from graph_uq.config.data import DatasetSplit
from graph_uq.config.uncertainty import UncertaintyModelType
from graph_uq.data import Data
from graph_uq.metric import Metric
from graph_uq.model.base import BaseModel
from graph_uq.model.prediction import Prediction
from graph_uq.uncertainty.base import BaseUncertaintyModel
from graph_uq.uncertainty.latent_space.gaussian_per_class import (
    UncertaintyModelGaussianPerClass,
)
from graph_uq.util import logsumexp


def confidence_interval(
    x: Float[Tensor, "*any"], dim: int = 0, quantile: float = 0.95
) -> tuple[Float[Tensor, "*any"], Float[Tensor, "*any"]]:
    x_sorted = torch.sort(x, dim=dim)
    alpha = (1 - quantile) / 2
    idxs = torch.arange(int(alpha * x.size(dim)), int((1 - alpha) * x.size(dim)))
    x_sorted = torch.index_select(x_sorted.values, dim, idxs)
    x_min, x_max = (
        x_sorted.min(dim=dim, keepdim=True).values,
        x_sorted.max(dim=dim, keepdim=True).values,
    )
    return x_min, x_max


class UncertaintyModelCorrectedConditionalEBM(BaseUncertaintyModel):
    """Uncertainty model that computes evidence-like conditional energy from logits and feature density.

    The evidence for each class is given by: e_c = l_c - lambda * Z^-1 * mahalanobis_distance(h, mu_c, Sigma_c)

    Where Z^-1 ensures that both terms are on the same scale.
    lambda interpolates from 0.0 (no correction) to inf (only feature density).
    """

    uncertainty_model_type: UncertaintyModelType = (
        UncertaintyModelType.CORRECTED_CONDITIONAL_EBM
    )

    @typechecked
    def __init__(
        self,
        tied_covariance: bool,
        covariance_type: str,
        embedding_layer: int,
        fit_to_embeddings_propagated: bool,
        fit_to_logits_propagated: bool,
        fit_to: str = DatasetSplit.TRAIN,
        temperature: float = 1.0,
        lambda_correction=1.0,
        evidence_diffusion: dict[str, Any] | None = None,
        diffusion: dict[str, Any] = {},
    ):
        super().__init__(diffusion=diffusion)
        self.feature_density = UncertaintyModelGaussianPerClass(
            tied_covariance=tied_covariance,
            covariance_type=covariance_type,
            embedding_layer=embedding_layer,
            embeddings_propagated=fit_to_embeddings_propagated,
            diffusion=diffusion,
            fit_to=fit_to,
        )
        self.tied_covariance = tied_covariance
        self.coveriance_type = covariance_type
        self.embedding_layer = embedding_layer
        self.fit_to_embeddings_propagated = fit_to_embeddings_propagated
        self.fit_to_logits_propagated = fit_to_logits_propagated
        self.fit_to = fit_to
        self.temperature = temperature
        self.lambda_correction = lambda_correction
        self.evidence_diffusion_config = evidence_diffusion
        self._fitted = False

    @typechecked
    def fit(self, data: Data, model: BaseModel, prediction: Prediction):
        """Fits the uncertainty model."""
        self.feature_density.fit(data, model, prediction)
        # Rescale the feature evidence (malahanobis distance)
        # such that it's scale matches the energy
        mahalanobis = self.feature_density.mahalanobis_distances(
            prediction.get_embeddings(
                layer=self.embedding_layer, propagated=self.fit_to_embeddings_propagated
            ).mean(0)  # type: ignore
        )
        self.mahalanoibs_scale = confidence_interval(mahalanobis, dim=0)
        self.logit_scale = confidence_interval(
            prediction.get_logits(propagated=self.fit_to_logits_propagated).mean(0),  # type: ignore
            dim=0,
        )
        self._fitted = True

    @jaxtyped(typechecker=typechecked)
    def correction(
        self,
        prediction: Prediction,
        embeddings_propagated: bool,
    ) -> Float[Tensor, "num_nodes num_classes_train"]:
        """Correction from the gaussian term.

        Args:
            prediction (Prediction): The prediction of the model
            embeddings_propagated (bool): Whether the embeddings_propagated are propagated or not

        Returns:
            Float[Tensor, "num_nodes num_classes_train"]: The correction
        """
        mahalanobis = self.feature_density.mahalanobis_distances(
            prediction.get_embeddings(
                layer=self.embedding_layer, propagated=embeddings_propagated
            ).mean(0)  # type: ignore
        )
        mahalanobis = (mahalanobis - self.mahalanoibs_scale[0]) / (
            self.mahalanoibs_scale[1] - self.mahalanoibs_scale[0]
        )
        mahalanobis *= self.logit_scale[1] - self.logit_scale[0]
        mahalanobis += self.logit_scale[0]
        return -mahalanobis

    @jaxtyped(typechecker=typechecked)
    def conditional_evidence(
        self,
        prediction: Prediction,
        embeddings_propagated: bool,
        logits_propagated: bool,
    ) -> Float[Tensor, "num_nodes num_classes_train"]:
        """Computes the conditional evidence of the prediction using the logits and a feature density correction term.

        Args:
            prediction (Prediction): The prediction of the model
            propagated (bool): Whether the prediction is propagated or not

        Returns:
            Float[Tensor, "num_nodes num_classes_train"]: The conditional evidence
        """
        correction = self.correction(
            prediction, embeddings_propagated=embeddings_propagated
        )
        logits = prediction.get_logits(propagated=logits_propagated).mean(0)  # type: ignore
        corrected = logits + self.lambda_correction * correction
        return corrected

    @jaxtyped(typechecker=typechecked)
    def diffused_conditional_evidence(
        self,
        data: Data,
        prediction: Prediction,
        embeddings_propagated: bool,
        logits_propagated: bool,
        propagated: bool,
    ) -> Float[Tensor, "num_nodes num_classes_train"]:
        """Computes the conditional evidence of the prediction using the logits and a feature density correction term and applies diffusion if needed.

        Args:
            prediction (Prediction): The prediction of the model
            embeddings_propagated (bool): Whether the embeddings are propagated
            logits_propagated (bool): Whether the logits are propagated
            propagated (bool): Whether the evidence is diffused (if a config is given)

        Returns:
            Float[Tensor, "num_nodes num_classes_train"]: The conditional evidence
        """
        conditional_evidence = self.conditional_evidence(
            prediction,
            embeddings_propagated=embeddings_propagated,
            logits_propagated=logits_propagated,
        )
        if self.evidence_diffusion_config is not None and propagated:
            conditional_evidence = data.diffusion(
                conditional_evidence, **self.evidence_diffusion_config
            )[-1]
        return conditional_evidence

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
            corrected = self.diffused_conditional_evidence(
                data,
                prediction,
                embeddings_propagated=embeddings_propagated,
                logits_propagated=logits_propagated,
                propagated=propagated,
            )

            energy = -self.temperature * logsumexp(
                corrected / self.temperature, dim=-1, keepdims=False
            )
            uncertainties[
                Metric(
                    propagated=propagated,
                    uncertainty_model_type=self.uncertainty_model_type,
                    embedding_idx=self.embedding_layer,
                    suffix=suffix if len(suffix) else None,
                )
            ] = energy

        return uncertainties
