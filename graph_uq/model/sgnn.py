from abc import abstractmethod
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import dropout_edge
from typeguard import typechecked

from graph_uq.config.model import Activation, ModelConfig
from graph_uq.data.data import Data
from graph_uq.experiment import experiment
from graph_uq.model.base import BaseModel
from graph_uq.model.nn import Linear
from graph_uq.model.prediction import Prediction
from graph_uq.util import apply_to_optional_tensors


class SGNN(BaseModel):
    @typechecked
    def __init__(self, config: ModelConfig, data: Data):
        from graph_uq.model.build import get_model

        super().__init__(config)
        if config["teacher"] is not None:
            self.teacher = get_model(config["teacher"], data)
        else:
            self.teacher = None
        if config["gdk_prior"] is not None:
            self.gdk_prior = get_model(config["gdk_prior"], data)
        else:
            self.gdk_prior = None
        assert config["backbone"] is not None
        self.backbone = get_model(config["backbone"], data)
        self.cached = config["cached"]
        self.reset_cache()
        self.dropout = config["dropout"]

    @property
    def prediction_changes_at_eval(self) -> bool:
        return (
            self.dropout > 0
            or (self.teacher is not None and self.teacher.prediction_changes_at_eval)
            or (
                self.gdk_prior is not None and self.gdk_prior.prediction_changes_at_eval
            )
        )

    def reset_cache(self):
        super().reset_cache()
        self.cached_prior_evidence = self.register_buffer("cached_prior_evidence", None)
        self.cached_teacher_probabilities = self.register_buffer(
            "cached_prior_evidence", None
        )
        if self.gdk_prior is not None:
            self.gdk_prior.reset_cache()
        if self.teacher is not None:
            self.teacher.reset_cache()
        self.backbone.reset_cache()

    @jaxtyped(typechecker=typechecked)
    def get_teacher_probabilities(
        self, batch: Data
    ) -> Float[Tensor, "num_samples num_nodes num_classes"] | None:
        """Gets the probabilities of the teacher network if used.

        Args:
            batch (Data): The batch of data.

        Returns:
            Float['num_samples num_nodes num_classes'] | None: The teacher probabilities.
        """
        if self.cached_teacher_probabilities is None and self.teacher is not None:
            with torch.no_grad():
                self.teacher.eval()
                teacher_prediction: Prediction = self.teacher(batch)
                teacher_probabilities = teacher_prediction.get_probabilities(
                    propagated=True
                )
                if self.cached:
                    self.cached_teacher_probabilities = teacher_probabilities
        else:
            teacher_probabilities = self.cached_teacher_probabilities

        return teacher_probabilities

    def get_evidence_prior(
        self, batch: Data
    ) -> Float[Tensor, "num_samples num_nodes num_classes"] | None:
        """Gets the evidence prior if used.

        Args:
            batch (Data): The batch of data.

        Returns:
            Float['num_samples num_nodes num_classes'] | None: The evidence prior.
        """
        if self.cached_prior_evidence is None and self.gdk_prior is not None:
            with torch.no_grad():
                self.gdk_prior.eval()
                prior_prediction: Prediction = self.gdk_prior(batch)
                prior_evidence = prior_prediction.alpha
                if self.cached:
                    self.cached_prior_evidence = prior_evidence
        else:
            prior_evidence = self.cached_prior_evidence
        return prior_evidence

    @typechecked
    def forward(self, batch: Data, propagated: bool = True) -> Prediction:
        evidence = self.backbone(batch).get_logits(propagated=propagated).exp()
        alpha = evidence + 1.0
        return Prediction(
            teacher_probabilities=self.get_teacher_probabilities(batch),
            alpha_prior=self.get_evidence_prior(batch),
            alpha=alpha,
            probabilities=alpha / alpha.sum(dim=-1, keepdim=True),
        )
