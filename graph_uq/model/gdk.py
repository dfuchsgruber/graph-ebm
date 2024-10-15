from jaxtyping import jaxtyped
from typeguard import typechecked

from graph_uq.config.model import ModelConfig
from graph_uq.data.data import Data
from graph_uq.experiment import experiment
from graph_uq.model.base import BaseModel
from graph_uq.model.prediction import Prediction


class GDK(BaseModel):
    """Parameterless baseline using the Graph Dirichlet Kernel (GDK)"""

    @typechecked
    def __init__(
        self,
        config: ModelConfig,
        data: Data,
        *args,
        **kwargs,
    ):
        super().__init__(config, *args, **kwargs)
        self.cutoff = config["cutoff"]
        self.sigma = config["sigma"]
        self.cached = config["cached"]

    def reset_cache(self): ...

    def reset_parameters(
        self,
    ): ...

    @property
    def prediction_changes_at_eval(self) -> bool:
        return False

    @jaxtyped(typechecker=typechecked)
    def forward(self, batch: Data) -> Prediction:
        alpha = batch.gdk(cutoff=self.cutoff, sigma=self.sigma) + 1
        soft = alpha / (alpha.sum(dim=1, keepdim=True))
        return Prediction(
            alpha=alpha.unsqueeze(0),
            probabilities=soft.unsqueeze(0),
        )
