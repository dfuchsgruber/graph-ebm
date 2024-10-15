from typing import Any, Dict

import torch
import torch_scatter
from jaxtyping import Float, jaxtyped
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal, _batch_mahalanobis
from typeguard import typechecked

from graph_uq.config.data import DatasetSplit
from graph_uq.config.uncertainty import CovarianceType, UncertaintyModelType
from graph_uq.data import Data
from graph_uq.metric import Metric
from graph_uq.model.base import BaseModel
from graph_uq.model.prediction import Prediction
from graph_uq.uncertainty.base import BaseUncertaintyModel
from graph_uq.util import logsumexp
from graph_uq.util.normal import cov_and_mean, make_covariance_symmetric_and_psd


class UncertaintyModelGaussianPerClass(BaseUncertaintyModel):
    """Uncertainty model that fits a Gaussian per class."""

    uncertainty_model_type: UncertaintyModelType = (
        UncertaintyModelType.GAUSSIAN_PER_CLASS
    )

    def __init__(
        self,
        tied_covariance: bool,
        covariance_type: str,
        embedding_layer: int,
        embeddings_propagated: bool,
        diffusion: dict[str, Any] = {},
        fit_to: str = DatasetSplit.TRAIN,
    ):
        super().__init__(diffusion=diffusion)
        self.tied_covariance = tied_covariance
        self.coveriance_type = covariance_type
        self.embedding_layer = embedding_layer
        self.embeddings_propagated = embeddings_propagated
        self.fit_to = fit_to
        self._fitted = False

    @jaxtyped(typechecker=typechecked)
    def mahalanobis_distances(self, x: Float[Tensor, "num_nodes num_features"]) -> Float[Tensor, "num_nodes num_classes"]:
        """Computes the Mahalanobis distances of the given data."""
        assert self._fitted, "Model must be fitted first"
        return torch.stack([
            _batch_mahalanobis(normal._unbroadcasted_scale_tril, x - normal.loc)
            for normal in self.normals
        ], dim=1)
        
    @typechecked
    def fit(self, data: Data, model: BaseModel, prediction: Prediction):
        """Fits the uncertainty model."""
        h = prediction.get_embeddings(
            propagated=self.embeddings_propagated, layer=self.embedding_layer
        )
        if h is None:
            return
        h = h.mean(0)  # ensemble mean
        h, y = h[data.get_mask(self.fit_to)], data.y[data.get_mask(self.fit_to)]

        class_counts = torch_scatter.scatter_add(torch.ones_like(y), y)
        self.class_prior = class_counts / class_counts.sum()

        covariances, means = zip(
            *[cov_and_mean(h[y == c]) for c in range(class_counts.size(0))]
        )
        if self.tied_covariance:
            covariances = [cov_and_mean(h)[0]] * len(covariances)
        match self.coveriance_type:
            case CovarianceType.DIAGONAL:
                covariances = [cov * torch.eye(h.size(1)) for cov in covariances]
            case CovarianceType.FULL:
                pass
            case CovarianceType.IDENTITY:
                covariances = [torch.eye(h.size(1))] * len(covariances)
            case CovarianceType.ISOTROPIC:
                covariances = [
                    torch.diag(cov).mean() * torch.eye(h.size(1)) for cov in covariances
                ]
            case _:
                raise ValueError(f"Unknown covariance type {self.coveriance_type}")

        covariances = [make_covariance_symmetric_and_psd(cov) for cov in covariances]
        self.normals = [
            MultivariateNormal(mean, cov, validate_args=False)
            for mean, cov in zip(means, covariances)
        ]
        self._fitted = True

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, data: Data, model: BaseModel, prediction: Prediction
    ) -> dict[Metric, Float[Tensor, "num_nodes"]]:
        """Computes the uncertainty estimates of this model.

        Args:
            data (Data): The data for which to make uncertainty estimates
            model (BaseModel): The model for which to make uncertainty estimates
            prediction (Prediction): The prediction of the model on the data. For most estimators, this should be used instead using `model` and `data` directly.

        Returns:
            dict[Metric, Float[Tensor, 'num_nodes']]: The uncertainty estimates, higher means more uncertain
        """
        assert self._fitted, "Model must be fitted first"
        uncertainties = {}
        for embeddings_propagated in (True, False):
            embeddings = prediction.get_embeddings(
                propagated=embeddings_propagated, layer=self.embedding_layer
            )
            if embeddings is None:
                continue
            embeddings = embeddings.mean(0)  # ensemble mean
            log_probs: Float[Tensor, "num_classes num_nodes"] = torch.stack(
                [normal.log_prob(embeddings) for normal in self.normals], dim=0
            )
            log_probs: Float[Tensor, "num_nodes"] = logsumexp(
                log_probs + self.class_prior.log()[:, None], dim=0, keepdims=False
            )
            uncertainties[
                Metric(
                    propagated=embeddings_propagated,
                    uncertainty_model_type=self.uncertainty_model_type,
                )
            ] = -log_probs
        return uncertainties

