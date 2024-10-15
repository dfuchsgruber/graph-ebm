import dataclasses
from abc import abstractmethod

import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import dropout_edge
from typeguard import typechecked

from graph_uq.config.model import Activation, ModelConfig
from graph_uq.data.data import Data
from graph_uq.model.base import BaseModel
from graph_uq.model.nn import Linear
from graph_uq.model.prediction import LayerPrediction, Prediction


class GNNLayer(MessagePassing):
    """Base class for GNN layers."""

    @property
    def output_idx(self) -> int:
        """The index of the output to use as input to the next layer."""
        return -1

    def reset_cache(self):
        raise NotImplementedError

    def reset_parameters(self):
        super().reset_parameters()

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def forward_impl(
        self,
        x: Float[Tensor, "num_nodes input_dim"],
        edge_index: Int[Tensor, "2 num_edges"],
        data: Data,
        edge_weight: Float[Tensor, "num_edges num_edge_features"] | None = None,
        unpropagated: bool = False,
    ) -> LayerPrediction:
        raise NotImplementedError

    @jaxtyped(typechecker=typechecked)
    def forward(
        self,
        x: Float[Tensor, "num_nodes input_dim"],
        edge_index: Int[Tensor, "2 num_edges"],
        data: Data,
        edge_weight: Float[Tensor, "num_edges num_edge_features"] | None = None,
        unpropagated: bool = False,
    ) -> LayerPrediction:
        return self.forward_impl(
            x, edge_index, data, edge_weight=edge_weight, unpropagated=unpropagated
        )


class GNNBlock(nn.Module):
    """Base GNN block that applies a convolution, activation, and dropout and residual connections."""

    downsampling_cls = Linear

    def __init__(self, config: ModelConfig, in_dim: int, out_dim: int):
        super().__init__()
        self.activation = Activation(config["activation"])
        self.residual = config["residual"]
        self.dropout = config["dropout"]
        self.dropout_at_eval = config["dropout_at_eval"]
        self.dropedge = config["dropedge"]
        self.dropedge_at_eval = config["dropout_at_eval"]
        self.leaky_relu_slope = config["leaky_relu_slope"]
        if self.residual and in_dim != out_dim:
            self.downsampling = self.downsampling_cls(config, in_dim, out_dim)
        else:
            self.downsampling = None
        if config["batch_norm"]:
            self.batch_norm = nn.BatchNorm1d(out_dim)
        else:
            self.batch_norm = None

        # needs to be initialized in subclasses
        self.convolution: GNNLayer = ...  # type: ignore

    @property
    def prediction_changes_at_eval(self) -> bool:
        return self.dropout > 0

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def forward(
        self,
        x: Float[Tensor, "num_nodes input_dim"],
        batch: Data,
        unpropagated: bool = False,
    ) -> LayerPrediction:
        edge_index, edge_weight = batch.edge_index, getattr(batch, "edge_weight", None)
        if self.dropedge:
            edge_index, edge_mask = dropout_edge(
                batch.edge_index,
                p=self.dropedge,
                training=self.training or self.dropedge_at_eval,
            )
            if edge_weight is not None:
                edge_weight = edge_weight[edge_mask]

        layer_prediction: LayerPrediction = self.convolution(
            x, edge_index, batch, edge_weight=edge_weight, unpropagated=unpropagated
        )
        h = layer_prediction.output
        assert h is not None

        if self.batch_norm:
            h = self.batch_norm(h)

        # Apply activation and dropout
        match self.activation:
            case Activation.RELU:
                h = F.relu(h)
            case Activation.LEAKY_RELU:
                h = F.leaky_relu(h, negative_slope=self.leaky_relu_slope)
            case Activation.TANH:
                h = F.tanh(h)

        if self.dropout:
            h = F.dropout(
                h, p=self.dropout, training=self.training or self.dropout_at_eval
            )

        # Apply residual connection
        if self.residual:
            if self.downsampling is not None:
                h = h + self.downsampling(x)
            else:
                h = h + x
        return dataclasses.replace(layer_prediction, embeddings=[h])

    def reset_cache(self):
        self.convolution.reset_cache()
        if self.residual and self.downsampling:
            self.downsampling.reset_cache()

    def reset_parameters(self):
        self.convolution.reset_parameters()
        if self.residual and self.downsampling:
            self.downsampling.reset_parameters()


class GNN(BaseModel):
    """Base GNN sekeleton class that relies on GNNLayer layers."""

    @typechecked
    def __init__(self, config: ModelConfig, data: Data, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.blocks = nn.ModuleList()
        in_dim = data.num_input_features
        for layer_idx, out_dim in enumerate(
            list(config["hidden_dims"])
            + [
                data.num_classes_train
                if config["num_outputs"] is None
                else config["num_outputs"]
            ]
        ):
            self.blocks.append(
                self.make_block(
                    config, in_dim, out_dim, layer_idx == len(config["hidden_dims"])
                )
            )
            in_dim = out_dim

    @abstractmethod
    @typechecked
    def make_block(
        self, config: ModelConfig, in_dim: int, out_dim: int, is_last_layer: bool
    ) -> GNNBlock:
        """Make a layer of the GNN."""
        raise NotImplementedError

    def reset_cache(self):
        for block in self.blocks:
            block.reset_cache()

    def reset_parameters(
        self,
    ):
        for block in self.blocks:
            block.reset_parameters()

    @property
    def prediction_changes_at_eval(self) -> bool:
        return any(block.prediction_changes_at_eval for block in self.blocks)

    @jaxtyped(typechecker=typechecked)
    def forward_impl(
        self,
        x: Float[Tensor, "num_nodes num_input_features"],
        data: Data,
        unpropagated: bool = False,
    ) -> list[LayerPrediction]:
        predictions = []
        for block in self.blocks:
            predictions_block = block(x, data, unpropagated)
            x = predictions_block.output
            predictions.append(predictions_block)
        return predictions

    @jaxtyped(typechecker=typechecked)
    def forward(self, batch: Data) -> Prediction:
        predictions = self.forward_impl(batch.x, batch)
        if not self.training:
            predictions_unpropagated = self.forward_impl(
                batch.x, batch, unpropagated=True
            )
        else:
            predictions_unpropagated = None
        return Prediction(
            layers=[
                layer_prediction.apply_to_tensors(lambda t: t.unsqueeze(0))
                for layer_prediction in predictions
            ],
            layers_unpropagated=[
                layer_prediction.apply_to_tensors(lambda t: t.unsqueeze(0))
                for layer_prediction in predictions_unpropagated
            ]
            if predictions_unpropagated is not None
            else None,
        )
