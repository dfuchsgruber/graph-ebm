from typeguard import typechecked

from graph_uq.config.model import ModelConfig, ModelType
from graph_uq.data import Data
from graph_uq.model.appnp import APPNP
from graph_uq.model.base import BaseModel
from graph_uq.model.bnn.gcn import BayesianGCN
from graph_uq.model.gat import GAT
from graph_uq.model.gcn import GCN
from graph_uq.model.gdk import GDK
from graph_uq.model.gin import GIN
from graph_uq.model.gpn import GraphPosteriorNetwork
from graph_uq.model.mlp import MLP
from graph_uq.model.none import NoModel
from graph_uq.model.sage import SAGE
from graph_uq.model.sgnn import SGNN


@typechecked
def get_model(config: ModelConfig, data: Data, *args, **kwargs) -> BaseModel:
    """Get a model.

    Args:
        data (Data): The data
        type_ (str): The type of model to get

    Returns:
        BaseModel: The model
    """
    match ModelType(config["type_"]):
        case ModelType.GCN:
            return GCN(config, data, *args, **kwargs)
        case ModelType.GAT:
            return GAT(config, data, *args, **kwargs)
        case ModelType.SAGE:
            return SAGE(config, data, *args, **kwargs)
        case ModelType.GIN:
            return GIN(config, data, *args, **kwargs)
        case ModelType.MLP:
            return MLP(config, data, *args, **kwargs)
        case ModelType.NONE:
            return NoModel(config, *args, **kwargs)
        case ModelType.GPN:
            return GraphPosteriorNetwork(config, data, *args, **kwargs)
        case ModelType.BGCN:
            return BayesianGCN(config, data, *args, **kwargs)
        case ModelType.GDK:
            return GDK(config, data, *args, **kwargs)
        case ModelType.SGNN:
            return SGNN(config, data, *args, **kwargs)
        case ModelType.APPNP:
            return APPNP(config, data, *args, **kwargs)
        case type_:
            raise ValueError(f"Unknown model type: {type_}")
