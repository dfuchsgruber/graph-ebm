import torch
import torch.nn.functional as F
from jaxtyping import Float, jaxtyped
from torch import Tensor
from typeguard import typechecked

from graph_uq.config.trainer import EdgeReconstructionConfig, EdgeReconstructionLossType
from graph_uq.data.data import Data
from graph_uq.model.prediction import Prediction
from graph_uq.util.edge_sampler import EdgeSampler


class EdgeReconstructionTrainerMixin:
    """Base class that trainers that use edge reconstruction can inherit from"""

    def __init__(self, config: EdgeReconstructionConfig, *args, **kwargs):
        self.edge_reconstruction_loss_weight = config["weight"]
        if config["type_"] is not None and config["weight"] > 0.0:
            self.edge_reconstruction = EdgeReconstruction(config, *args, **kwargs)
        else:
            self.edge_reconstruction = None

    @property
    @typechecked
    def has_edge_reconstruction_loss(self) -> bool:
        return self.edge_reconstruction is not None

    @jaxtyped(typechecker=typechecked)
    def edge_reconstruction_loss(
        self, data: Data, prediction: Prediction
    ) -> Float[Tensor, ""]:
        assert self.edge_reconstruction is not None
        assert self.has_edge_reconstruction_loss
        return self.edge_reconstruction.loss(data, prediction)

    def reset_cache(self):
        if self.has_edge_reconstruction_loss:
            assert self.edge_reconstruction is not None
            self.edge_reconstruction.reset_cache()


class EdgeReconstruction:
    """Stateful module that computes a loss on the edge reconstruction"""

    def __init__(self, config: EdgeReconstructionConfig):
        self.type_ = config["type_"]
        self.num_samples = config["num_samples"]
        self.margin = config["margin"]
        self.embedding_layer = config["embedding_layer"]
        self.embeddings_propagated = config["embeddings_propagated"]
        self.edge_sampler = EdgeSampler()

    def reset_cache(self):
        self.edge_sampler.reset_cache()

    @jaxtyped(typechecker=typechecked)
    def sample_triplet_features(
        self, data: Data, prediction: Prediction, num: int
    ) -> tuple[
        Float[Tensor, "num_sampled num_features"],
        Float[Tensor, "num_sampled num_features"],
        Float[Tensor, "num_sampled num_features"],
    ]:
        """Samples triplets of features: anchor, positive, negative"""
        triplets = self.edge_sampler.sample(data, num)
        h = prediction.get_embeddings(
            propagated=self.embeddings_propagated, layer=self.embedding_layer
        )
        assert h is not None, "No embeddings found"
        assert (
            h.size(0) == 1
        ), "Reconstruction loss with an average embedding is a bad idea..."
        h = h.mean(0)
        return (
            torch.index_select(h, 0, triplets[0]),
            torch.index_select(h, 0, triplets[1]),
            torch.index_select(h, 0, triplets[2]),
        )

    @jaxtyped(typechecker=typechecked)
    def energy(self, data: Data, prediction: Prediction) -> Float[Tensor, ""]:
        """Computes an energy based loss on the edge reconstruction"""
        anchors, positives, negatives = self.sample_triplet_features(
            data, prediction, self.num_samples
        )
        distances_pos = torch.norm(anchors - positives, dim=-1, p=2.0)
        distances_neg = torch.norm(anchors - negatives, dim=-1, p=2.0)
        return (distances_pos.pow(2) + (-distances_neg).exp()).mean()

    @jaxtyped(typechecker=typechecked)
    def contrastive_with_margin(
        self, data: Data, prediction: Prediction
    ) -> Float[Tensor, ""]:
        """Computes a contrastive loss with margin on the edge reconstruction"""
        anchors, positives, negatives = self.sample_triplet_features(
            data, prediction, self.num_samples
        )
        distances_pos = torch.norm(anchors - positives, dim=-1, p=2.0)
        distances_neg = torch.norm(anchors - negatives, dim=-1, p=2.0)
        return torch.clamp(distances_pos - distances_neg + self.margin, min=0.0).mean()

    @jaxtyped(typechecker=typechecked)
    def dot_product(self, data: Data, prediction: Prediction) -> Float[Tensor, ""]:
        """Computes a dot product and binary cross entropy based loss on the edge reconstruction"""
        anchors, positives, negatives = self.sample_triplet_features(
            data, prediction, self.num_samples
        )
        logits_pos = torch.sum(anchors * positives, dim=-1)
        logits_neg = torch.sum(anchors * negatives, dim=-1)
        proxy = torch.cat((logits_pos, logits_neg), 0)
        labels = torch.cat(
            (torch.ones_like(logits_pos), torch.zeros_like(logits_neg)), 0
        ).float()
        return F.binary_cross_entropy_with_logits(proxy, labels, reduction="mean")

    def loss(self, data: Data, prediction: Prediction) -> Float[Tensor, ""]:
        """Computes the loss"""
        match self.type_:
            case EdgeReconstructionLossType.ENERGY:
                return self.energy(data, prediction)
            case EdgeReconstructionLossType.CONTRASTIVE_WITH_MARGIN:
                return self.contrastive_with_margin(data, prediction)
            case EdgeReconstructionLossType.DOT_PRODUCT:
                return self.dot_product(data, prediction)
            case _:
                raise RuntimeError(
                    f"Unknown edge reconstruction loss type {self.type_}"
                )
