"""Wrapper for the GEBM approach presented in the paper for simple usage."""

import torch
from jaxtyping import Bool, Float, Int, jaxtyped
from typeguard import typechecked

from graph_uq.config.default.model import default_model_config
from graph_uq.config.default.uncertainty import get_multi_scale_conditional_evidence_ebm
from graph_uq.data.data import Data
from graph_uq.model.base import BaseModel
from graph_uq.model.prediction import LayerPrediction, Prediction
from graph_uq.uncertainty.build import get_uncertainty_models
from graph_uq.uncertainty.multi_scale_corrected_conditional_ebm import (
    UncertaintyModelMultiScaleCorrectedConditionalEBM,
)


class _DummyModel(BaseModel):
    """Dummy model for the GEBM approach. It simply predicts pre-defined embeddings and logits"""

    @jaxtyped(typechecker=typechecked)
    def __init__(
        self,
        embeddings: Float[torch.Tensor, "num_nodes num_features"] | None = None,
        logits: Float[torch.Tensor, "num_nodes num_classes"] | None = None,
        embeddings_unpropagated: Float[torch.Tensor, "num_nodes num_features"]
        | None = None,
        logits_unpropagated: Float[torch.Tensor, "num_nodes num_classes"] | None = None,
    ):
        super().__init__(default_model_config)
        self.embeddings = embeddings
        self.embeddings_unpropagated = embeddings_unpropagated
        self.logits = logits
        self.logits_unpropagated = logits_unpropagated

    def forward(self, data: Data) -> Prediction:
        return Prediction(
            layers=[
                LayerPrediction(
                    embeddings=[
                        self.embeddings.unsqueeze(0)
                        if self.embeddings is not None
                        else None
                    ]
                ),
                LayerPrediction(
                    embeddings=[
                        self.logits.unsqueeze(0) if self.logits is not None else None
                    ]
                ),
            ],
            layers_unpropagated=[
                LayerPrediction(
                    embeddings=[
                        self.embeddings_unpropagated.unsqueeze(0)
                        if self.embeddings_unpropagated is not None
                        else None
                    ]
                ),
                LayerPrediction(
                    embeddings=[
                        self.logits_unpropagated.unsqueeze(0)
                        if self.logits_unpropagated is not None
                        else None
                    ]
                ),
            ],
        )


class GraphEBMWrapper:
    """Wrapper class for GEBM approach."""

    def __init__(
        self,
        covariance_type: str = "diagonal",
        tied_covariance: bool = False,
        gamma_correction: float = 1.0,
        lambda_independent_energy: float = 1.0,
        lambda_local_energy: float = 1.0,
        lambda_group_energy: float = 1.0,
        diffusion_type: str = "label_propagation",
        alpha: float = 0.5,
        num_diffusion_steps: int = 10,
        aggregation: str = "sum",
    ):
        """Initializes the GEBM wrapper.

        Args:
            covariance_type: Type of covariance matrix to use for gaussian regularization of energy. Default is 'diagonal'.
            tied_covariance: Whether to use tied covariance matrix for gaussian regularization of energy. Default is False.
            gamma_correction: Weight for the energy regularization term. Default is 1.0.
                The normalizer for the energy is always automatically inferred to match the distribution of logits and log densities.
            lambda_independent_energy: Weight for the independent energy regularization term. Default is 1.0.
            lambda_local_energy: Weight for the local energy regularization term. Default is 1.0.
            lambda_group_energy: Weight for the group energy regularization term. Default is 1.0.
            diffusion_type: Type of diffusion to use for the conditional evidence. Default is 'label_propagation'.
            alpha: Alpha parameter for the diffusion. Default is 0.5.
            num_diffusion_steps: Number of diffusion steps. Default is 10.
            aggregation: Aggregation function for energy types. Default is 'sum'.
        """
        config = get_multi_scale_conditional_evidence_ebm(
            covariance_type=covariance_type,
            tied_covariance=tied_covariance,
            lambda_correction=gamma_correction,
            lambda_undiffused_energy=lambda_independent_energy,
            lambda_diffused_conditional_evidence=lambda_local_energy,
            lambda_diffused_energy=lambda_group_energy,
            diffusion_type=diffusion_type,
            alpha=alpha,
            k=num_diffusion_steps,
            aggregation=aggregation,
        )
        uncertainty_model = get_uncertainty_models({"": config})[""]
        assert isinstance(
            uncertainty_model, UncertaintyModelMultiScaleCorrectedConditionalEBM
        )
        self._uncertainty_model = uncertainty_model
        self._fitted = False

    @torch.no_grad()
    @jaxtyped(typechecker=typechecked)
    def fit_from_model(self, data: Data, model: BaseModel):
        """Fits the estimator from model and data by getting predictions from the model.

        Args:
            data: The data to fit the model to.
            model: The model to fit the model to.
        """
        prediction = model(data)
        self._uncertainty_model.fit(data, model, prediction)
        self._fitted = True

    @jaxtyped(typechecker=typechecked)
    def fit(
        self,
        logits: Float[torch.Tensor, "num_nodes num_classes"],
        embeddings: Float[torch.Tensor, "num_nodes num_features"],
        edge_index: Int[torch.Tensor, "2 num_edges"],
        y: Int[torch.Tensor, "num_nodes"],
        mask: Bool[torch.Tensor, "num_nodes"],
        edge_weight: Float[torch.Tensor, "num_edges"] | None = None,
    ):
        """Fits the model to the given logits and embeddings, both of which should be predicted in the presence of the structure during fitting.

        Args:
            logits: The logits. Should be predicted in the presence of the structure.
            embeddings: The embeddings of the model. Should be predicted in the presence of the structure.
            edge_index: The edge index of the graph.
            y: The labels.
            mask: The mask for the training data, i.e. to which nodes the model is fit to.
            edge_weight: The edge weights. Default is None.
        """
        dummy_data = Data(  # type: ignore
            x=embeddings,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            train_mask=mask,
            val_mask=~mask,
            test_mask=torch.zeros_like(mask),
        )
        self.fit_from_model(
            dummy_data,  # type: ignore
            _DummyModel(embeddings=embeddings, logits=logits),
        )

    @torch.no_grad()
    def get_uncertainty_from_model(
        self, data: Data, model: BaseModel
    ) -> Float[torch.Tensor, "num_nodes"]:
        """Computes the uncertainty estimate from the model and data.

        Args:
            data: The data to compute the uncertainty estimate for.
            model: The model to compute the uncertainty estimate for.

        Returns:
            The uncertainty estimate.
        """
        assert self._fitted, "The model has not been fitted yet."
        prediction = model(data)
        return self._uncertainty_model.get_uncertainty(
            data,
            model,
            prediction,
            propagated=True,
            embeddings_propagated=False,
            logits_propagated=False,
        )

    @jaxtyped(typechecker=typechecked)
    def get_uncertainty(
        self,
        logits_unpropagated: Float[torch.Tensor, "num_nodes num_classes"],
        embeddings_unpropagated: Float[torch.Tensor, "num_nodes embedding_dim"],
        edge_index: Int[torch.Tensor, "2 num_edges"],
        edge_weight: Float[torch.Tensor, "num_edges"] | None = None,
    ) -> Float[torch.Tensor, "num_nodes"]:
        """Computes the uncertainty estimate.

        Args:
            logits_unpropagated: The logits in the absence of structure.
            embeddings_unpropagated: The embeddings in the absence of structure.
            edge_index: The edge index of the graph.
            y: The labels.
            edge_weight: The edge weights. Default is None.

        Returns:
            The uncertainty estimate.
        """
        dummy_data = Data(  # type: ignore
            x=embeddings_unpropagated,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        return self.get_uncertainty_from_model(
            dummy_data,  # type: ignore
            _DummyModel(
                embeddings_unpropagated=embeddings_unpropagated,
                logits_unpropagated=logits_unpropagated,
            ),
        )
