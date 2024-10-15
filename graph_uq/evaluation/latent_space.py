import torch
from jaxtyping import Float, jaxtyped
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import Tensor
from typeguard import typechecked

from graph_uq.config.evaluation import LatentSpaceConfig
from graph_uq.data.data import Data
from graph_uq.evaluation.result import EvaluationResult
from graph_uq.metric import Metric
from graph_uq.model.base import BaseModel
from graph_uq.model.prediction import Prediction


@jaxtyped(typechecker=typechecked)
def tsne_embedding(
    embeddings: Float[Tensor, "num_nodes embedding_dim"], perplexity: float
) -> tuple[Float[Tensor, "num_nodes 2"], dict]:
    """Computes a 2D embedding of the given embeddings using t-SNE."""
    return torch.from_numpy(
        TSNE(n_components=2, perplexity=perplexity).fit_transform(
            embeddings.detach().cpu().numpy()
        )
    ).float(), {}


@jaxtyped(typechecker=typechecked)
def pca_embedding(
    embeddings: Float[Tensor, "num_nodes embedding_dim"],
) -> tuple[Float[Tensor, "num_nodes 2"], dict]:
    """Computes a 2D embedding of the given embeddings using PCA."""
    pca = PCA(n_components=2)
    embeddings_2d = torch.from_numpy(
        pca.fit_transform(embeddings.detach().cpu().numpy())
    ).float()
    return embeddings_2d, {"explained_variance_ratio": pca.explained_variance_ratio_}


@jaxtyped(typechecker=typechecked)
def get_latent_embeddings(
    config: LatentSpaceConfig,
    data: Data,
    model: BaseModel,
    prediction: Prediction,
) -> EvaluationResult:
    if config["last_embedding_layer"] is None:
        last_embedding_layer = prediction.num_embeddings(propagated=False) - 1
    embeddings_dict = {}
    for propagated in config["propagation_types"]:
        for embedding_idx in range(
            config["first_embedding_layer"], last_embedding_layer + 1
        ):
            embeddings = prediction.get_embeddings(
                propagated=propagated, layer=embedding_idx
            )
            if embeddings is not None:
                embeddings_dict[
                    Metric(propagated=propagated, embedding_idx=embedding_idx)
                ] = embeddings.cpu()
    return EvaluationResult(embeddings=embeddings_dict)
