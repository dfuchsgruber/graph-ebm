import torch
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from typeguard import typechecked

from .abstract_scorer import AbastractOODScorer


class EnergyLogitsScorer(AbastractOODScorer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "EnergyLogits"
        self.is_fitted = False

    @torch.no_grad()
    @jaxtyped(typechecker=typechecked)
    def _score_batch(
        self,
        features: Float[Tensor, "batch num_classes"],
    ) -> Float[Tensor, "batch"]:
        z = features
        if self.use_react:
            z = z.clip(max=self.react_threshold)

        return self.energy(z)

    @torch.no_grad()
    def _fit(
        self,
        features: Float[Tensor, "num_train num_features"],
        targets: Int[Tensor, "num_train"],
        verbose: bool = False,
    ): ...

    @jaxtyped(typechecker=typechecked)
    def energy(
        self,
        logits: Float[Tensor, "batch num_classes"],
        labels: Int[Tensor, "batch"] | None = None,
    ) -> Tensor:
        return -torch.logsumexp(logits, dim=1)

    @property
    def has_sample(self) -> bool:
        return False
