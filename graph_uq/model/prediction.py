from dataclasses import dataclass, fields
from typing import Callable, Self, Sequence

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from typeguard import typechecked

from graph_uq.metric import *
from graph_uq.util import apply_to_nested_tensors


@dataclass
class LayerPrediction:
    """Prediction for a single layer."""

    embeddings: (
        list[Float[Tensor, "num_samples num_nodes embedding_dim"] | None] | None
    ) = None
    attention_weights: Float[Tensor, "num_samples num_edges num_heads"] | None = None
    output_idx: int = -1

    @property
    @jaxtyped(typechecker=typechecked)
    def output(self) -> Float[Tensor, "num_samples num_nodes embedding_dim"] | None:
        return self.embeddings[self.output_idx] if self.embeddings is not None else None

    @classmethod
    @typechecked
    def collate(cls, predictions: Sequence[Self]) -> Self:
        """Collates predictions at the same layer."""

        prediction = cls()
        if len(predictions) == 0:
            return prediction
        assert (
            len(set(prediction.output_idx for prediction in predictions)) > 0
        ), "Can not collate predictions with different output_idxs"
        prediction.output_idx = predictions[0].output_idx

        for field in fields(cls):
            if field.name in ("embeddings", "output_idx"):
                continue
            attributes = [getattr(prediction, field.name) for prediction in predictions]
            if all(attribute is None for attribute in attributes):
                continue
            elif all(attribute is not None for attribute in attributes):
                setattr(prediction, field.name, torch.cat(attributes, dim=0))
            else:
                raise RuntimeError(
                    f"Cant collate {field.name} as it is inhomogenous for Nones among predictions."
                )

        # collate the embeddings
        for attribute in ("embeddings",):
            embeddings = [getattr(prediction, attribute) for prediction in predictions]
            if all(embedding is None for embedding in embeddings):
                continue
            elif not all(embedding is not None for embedding in embeddings):
                raise RuntimeError(
                    f"Cant collate {attribute} as it is inhomogenous for Nones among predictions."
                )
            # collate the embeddings of each layer
            embeddings = [
                torch.cat(embeddings_layer, dim=0)
                for embeddings_layer in zip(*embeddings)
            ]
            setattr(prediction, attribute, embeddings)

        return prediction

    def apply_to_tensors(self, func: Callable[[Tensor], Tensor]) -> "LayerPrediction":
        """Applies a function to all tensors in the prediction"""
        return LayerPrediction(
            **{
                field.name: apply_to_nested_tensors(
                    getattr(self, field.name, None), func
                )
                for field in fields(self)
            }
        )


class PredictionAttributeNotFound(BaseException):
    """Exception that is raised if a prediction attribute is not found."""


@dataclass
class Prediction:
    """Class for the model predictions."""

    logits: Float[Tensor, "num_samples num_nodes num_classes"] | None = None
    logits_unpropagated: Float[Tensor, "num_samples num_nodes num_classes"] | None = (
        None
    )

    layers: list[LayerPrediction] | None = None
    layers_unpropagated: list[LayerPrediction] | None = None

    probabilities: Float[Tensor, "num_samples num_nodes num_classes"] | None = None
    probabilities_unpropagated: (
        Float[Tensor, "num_samples num_nodes num_classes"] | None
    ) = None

    # KL divergence to prior is only predicted by a Bayesian model
    kl_divergence: Float[Tensor, " num_samples"] | None = None
    num_kl_terms: Int[Tensor, ""] | None = None

    # Evidential methods
    alpha: Float[Tensor, "num_samples num_nodes num_classes"] | None = None
    alpha_unpropagated: Float[Tensor, "num_samples num_nodes num_classes"] | None = None
    log_beta: Float[Tensor, "num_samples num_nodes num_classes"] | None = None
    log_beta_unpropagated: Float[Tensor, "num_samples num_nodes num_classes"] | None = (
        None
    )

    # SGNN
    teacher_probabilities: Float[Tensor, "num_samples num_nodes num_classes"] | None = (
        None
    )
    alpha_prior: Float[Tensor, "num_samples num_nodes num_classes"] | None = None

    def __repr__(self) -> str:
        super().__repr__()
        sizes = {
            attribute: apply_to_nested_tensors(
                getattr(self, attribute, None), lambda tensor: tuple(tensor.size())
            )
            for attribute in self.__annotations__
            if getattr(self, attribute, None) is not None
        }
        return f'Prediction({", ".join([f"{attribute}={size}" for attribute, size in sizes.items()])})'

    @property
    def num_classes(self) -> int:
        """For how many classes predictions are made."""
        # It is cheaper to not "compute" `self.get_predictions()` right away but instead infer the number
        # of classes from the "raw" data tensors
        for attribute in (
            "logits",
            "logits_unpropagated",
            "probabilities",
            "probabilities_unpropagated",
            "aleatoric_confidence",
            "total_confidence",
            "epistemic_confidence",
            "alpha",
            "alpha_unpropagated",
            "beta",
            "beta_unpropagated",
        ):
            value = getattr(self, attribute, None)
            if value is not None:
                return value.size(-1)
        prediction = self.get_predictions()
        if prediction is not None:
            return int((prediction.max() + 1).item())
        raise ValueError(
            "Can not infer the number of predicted classes from the prediction"
        )

    @jaxtyped(typechecker=typechecked)
    def get_predictions(
        self, propagated: bool = True
    ) -> Int[Tensor, " num_nodes"] | None:
        probabilities = self.get_probabilities(propagated=propagated)
        if probabilities is not None:
            return probabilities.mean(0).max(dim=-1)[1].long()
        return None

    @jaxtyped(typechecker=typechecked)
    def get_probabilities(
        self, propagated: bool = True
    ) -> Float[Tensor, "num_samples num_nodes num_classes"] | None:
        probabilities = (
            self.probabilities if propagated else self.probabilities_unpropagated
        )
        if probabilities is not None:
            return probabilities
        logits = self.get_logits(propagated=propagated)
        if logits is not None:
            return F.softmax(logits, dim=-1)
        return None

    @jaxtyped(typechecker=typechecked)
    def get_logits(
        self, propagated: bool = True
    ) -> Float[Tensor, "num_samples num_nodes num_classes"] | None:
        """Gets the logits of a prediction

        Args:
            propagated (bool): Whether to return the propagated version

        Returns:
            Float[Tensor, 'num_samples num_nodes num_classes'] | None: the logits
        """
        if propagated:
            if self.logits is not None:
                return self.logits
        else:
            if self.logits_unpropagated is not None:
                return self.logits_unpropagated
        logits = self.get_embeddings(propagated=propagated, layer=-1)
        if logits is not None:
            return logits
        return None

    @jaxtyped(typechecker=typechecked)
    def get_embeddings(
        self, propagated: bool = True, layer: int = -1
    ) -> Float[Tensor, "num_samples num_nodes embedding_dim"] | None:
        """Gets the embeddings of a prediction

        Args:
            propagated (bool): Whether to return the propagated version
            layer (int): The layer of the embeddings to return

        Returns:
            Float[Tensor, 'num_samples num_nodes embedding_dim'] | None: the embeddings
        """
        layer_prediction = None
        if propagated:
            if self.layers is not None:
                layer_prediction = self.layers[layer]
        else:
            if self.layers_unpropagated is not None:
                layer_prediction = self.layers_unpropagated[layer]
        if layer_prediction is not None and layer_prediction.embeddings is not None:
            return layer_prediction.embeddings[layer_prediction.output_idx]
        return None

    @typechecked
    def num_embeddings(self, propagated: bool = True) -> int:
        """Gets the number of embeddings of a prediction (i.e. the number of layers)"""
        if propagated:
            return len(self.layers) if self.layers is not None else 0
        else:
            return (
                len(self.layers_unpropagated)
                if self.layers_unpropagated is not None
                else 0
            )

    @jaxtyped(typechecker=typechecked)
    def get_evidence(
        self, propagated: bool = True
    ) -> Float[Tensor, "num_samples num_nodes num_classes"] | None:
        """Gets the evidence."""
        return self.alpha if propagated else self.alpha_unpropagated

    @classmethod
    @typechecked
    def collate(cls, predictions: list["Prediction"]) -> "Prediction":
        """Collates multiple predictions into one by concatenating tensors along the first axis ('num_samples')

        Args:
            predictions (list[&#39;Prediction&#39;]): predictions to concatenate

        Returns:
            Prediction: the collated prediction
        """
        prediction = cls()
        for field in fields(cls):
            if field.name in ("layers", "layers_unpropagated"):
                continue
            attributes = [getattr(prediction, field.name) for prediction in predictions]
            if all(attribute is None for attribute in attributes):
                continue
            elif all(attribute is not None for attribute in attributes):
                setattr(prediction, field.name, torch.cat(attributes, dim=0))
            else:
                raise RuntimeError(
                    f"Cant collate {field.name} as it is inhomogenous for Nones among predictions."
                )

        # Collate the embeddings
        for attribute in ("layers", "layers_unpropagated"):
            layers = [getattr(prediction, attribute) for prediction in predictions]
            if all(prediction_layer is None for prediction_layer in layers):
                continue
            elif not all(prediction_layer is not None for prediction_layer in layers):
                raise RuntimeError(
                    f"Cant collate {attribute} as it is inhomogenous for Nones among predictions."
                )
            # Collate the prediction of each layer
            layers = [
                LayerPrediction.collate(predictions_layer)
                for predictions_layer in zip(*layers)
            ]
            setattr(prediction, attribute, layers)

        return prediction
