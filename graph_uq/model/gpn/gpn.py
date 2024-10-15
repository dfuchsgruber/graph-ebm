# Partially taken from: https://github.com/stadlmax/Graph-Posterior-Network


import torch.nn as nn
import torch_geometric.nn as tgnn
from typeguard import typechecked

from graph_uq.config.model import ModelConfig
from graph_uq.data import Data
from graph_uq.model.base import BaseModel
from graph_uq.model.gpn.evidence import Evidence
from graph_uq.model.gpn.normalizing_flow import BatchedNormalizingFlowDensity
from graph_uq.model.prediction import LayerPrediction, Prediction


class GraphPosteriorNetwork(BaseModel):
    """Graph Posterior Network model"""

    @typechecked
    def __init__(self, config: ModelConfig, data: Data):
        super().__init__(config)
        self.dropout = config["dropout"]
        self._build_mlp(config, data.num_features)
        self.flow = BatchedNormalizingFlowDensity(
            c=data.num_classes_train,
            dim=config["flow_dim"],
            flow_length=config["num_flow_layers"],
        )

        self.evidence = Evidence(config["evidence_scale"])

        self.propagation = tgnn.APPNP(
            K=config["num_diffusion_steps"],
            alpha=config["alpha"],
            add_self_loops=config["add_self_loops"],
            cached=True,
        )
        self.flow_dim = config["flow_dim"]
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, (nn.Linear, nn.BatchNorm1d)):
                layer.reset_parameters()
        self.flow.reset_parameters()
        self.propagation.reset_parameters()

    def reset_cache(self):
        self.propagation._cached_edge_index = None
        self.propagation._cached_adj_t = None

    @typechecked
    def _build_mlp(
        self,
        config,
        num_input_features: int,
    ):
        """Builds the MLP that embeds as an input for normalizing flows"""
        layers = []
        in_dim = num_input_features
        for layer_idx, out_dim in enumerate(config["hidden_dims"]):
            layers.append(nn.Linear(in_dim, out_dim))
            if config["batch_norm"]:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=config["inplace"]))
            if self.dropout:
                layers.append(nn.Dropout(self.dropout, inplace=config["inplace"]))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, config["flow_dim"]))
        self.mlp = nn.Sequential(*layers)

    @typechecked
    def forward(self, data: Data) -> Prediction:
        z = self.mlp(data.x)

        # Compute flow probabilities
        p_c = data.class_prior_probabilities_train[
            None, -1
        ].log()  # [num_nodes num_classes]
        log_q_ft_per_class = self.flow.log_prob(z) + p_c

        # Scale with certainty budget
        beta_features = self.evidence(
            log_q_ft_per_class, dim=self.flow_dim, num_classes=p_c.size(-1)
        ).exp()
        # Compute posterior
        alpha_features = 1.0 + beta_features

        # Propagated pseudo-counts and posterior
        beta = self.propagation(beta_features, data.edge_index)
        alpha = 1.0 + beta

        # Aggregate into prediction result
        alpha = alpha.unsqueeze(0)
        alpha_features = alpha_features.unsqueeze(0)
        beta = beta.unsqueeze(0)
        beta_features = beta_features.unsqueeze(0)
        soft = alpha / alpha.sum(-1, keepdim=True)
        soft_unpropagated = alpha_features / alpha_features.sum(-1, keepdim=True)

        return Prediction(
            alpha=alpha,
            alpha_unpropagated=alpha_features,
            log_beta=beta.log(),
            log_beta_unpropagated=beta_features.log(),
            probabilities=soft,
            probabilities_unpropagated=soft_unpropagated,
            layers_unpropagated=[LayerPrediction(embeddings=[z.unsqueeze(0)])],
        )

    @property
    @typechecked
    def flow_parameters(self) -> list[nn.Parameter]:
        """Returns the parameters of the normalizing flow"""
        flow_params = list(self.flow.named_parameters())
        flow_param_weights = [p[1] for p in flow_params]
        return flow_param_weights

    @property
    @typechecked
    def non_flow_parameters(self) -> list[nn.Parameter]:
        """Returns the parameters of the model that are not part of the normalizing flow"""
        flow_params = list(self.flow.named_parameters())
        flow_param_names = [f"flow.{p[0]}" for p in flow_params]
        all_params = list(self.named_parameters())
        params = [p[1] for p in all_params if p[0] not in flow_param_names]
        return params

    @property
    def prediction_changes_at_eval(self) -> bool:
        return self.dropout > 0
