from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from typeguard import typechecked


class AbastractOODScorer(nn.Module):
    def __init__(
        self,
        name: str = "",
        layer_name: str = "layer4",
        normalize: bool = False,
        pooling: str = "avg",
        features_dataset: bool = False,
        max_fit_iter: float = np.inf,
        use_react: bool = False,
        input_preprocessing: bool = False,
        eps: float = 0.005,
        react_p: float = 0.95,
        **kwargs,
    ):
        super().__init__()
        assert pooling in ["avg", "std", "avg_std", "token"]

        self.is_fitted = False
        self.name = name
        self.normalize = normalize
        self.layer_name = layer_name
        self.pooling = pooling
        self.features_dataset = features_dataset
        self.use_react = use_react
        self.input_preprocessing = input_preprocessing
        self.eps = eps
        self.react_p = react_p
        self.react_threshold = None
        self.max_fit_iter = max_fit_iter
        self.max_fit_iter_react = 20

    @jaxtyped(typechecker=typechecked)
    def fit_react(self, features: Float[Tensor, "batch num_features"]):
        self.react_threshold = torch.quantile(features, self.react_p).item()

        del features
        torch.cuda.empty_cache()

    @jaxtyped(typechecker=typechecked)
    def _fit(
        self,
        features: Float[Tensor, "batch num_features"],
        targets: Int[Tensor, "batch"],
        verbose: bool = False,
    ):
        pass

    @jaxtyped(typechecker=typechecked)
    def fit(
        self,
        features: Float[Tensor, "batch num_features"],
        targets: Int[Tensor, "batch"],
        verbose: bool = False,
    ):
        if self.use_react:
            self.fit_react(features)

        self._fit(features, targets=targets, verbose=verbose)
        self.is_fitted = True

    def energy(self, z: Tensor, labels: Optional[Tensor] = None) -> torch.Tensor:
        return torch.zeros((z.shape[0],), device=z.device)

    @property
    def has_sample(self) -> bool:
        return False

    @jaxtyped(typechecker=typechecked)
    def _score_batch(
        self,
        features: Float[Tensor, "batch num_features"],
    ) -> torch.Tensor:
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def score_batch(
        self,
        features: Float[Tensor, "batch num_features"],
    ) -> torch.Tensor:
        return self._score_batch(features)

    # def get_scores(
    #     self,
    #     model: torch.nn.Module,
    #     ood_loader: torch.utils.data.DataLoader,
    #     max_iter: float = np.inf,
    #     force_tqdm_disable: bool = True,
    # ) -> torch.Tensor:
    #     tqdm_disable = os.getenv("TQDM_DISABLE") and force_tqdm_disable
    #     model.eval()
    #     data, _ = next(iter(ood_loader))
    #     is_score_batch_tuple = isinstance(self._score_batch(model, data), tuple)
    #     scores = []
    #     norms = []
    #     if is_score_batch_tuple:
    #         scores_nn = []
    #         scores_g = []

    #     for i, (data, _) in enumerate(tqdm(ood_loader, disable=tqdm_disable)):
    #         data = data.to("cuda", non_blocking=True)
    #         _score = self.score_batch(model, data)

    #         # *** This part is only to log the norm..  *** #

    #         z = lib.get_features(model, data, layer_names=[self.layer_name])["avg"][-1]

    #         if self.use_react:
    #             z = z.clip(max=self.react_threshold)

    #         norms.append(z.norm(2, 1).mean().cpu().detach())
    #         # ****** *** *** *** *** *** *** *** ***  **** #

    #         if is_score_batch_tuple:
    #             scores.append(_score[0].detach().squeeze().cpu())
    #             scores_nn.append(_score[1].detach().squeeze().cpu())
    #             scores_g.append(_score[2].detach().squeeze().cpu())

    #         else:
    #             _s = _score.detach().squeeze().cpu()
    #             if _s.dim() == 0:
    #                 _s = _s.unsqueeze(0)
    #             scores.append(_s)

    #         if i >= max_iter:
    #             break

    #     scores = torch.cat(scores)
    #     norms = torch.stack(norms)
    #     if is_score_batch_tuple:
    #         return (
    #             scores.numpy(),
    #             torch.cat(scores_nn).numpy(),
    #             torch.cat(scores_g).numpy(),
    #             norms.numpy(),
    #         )
    #     else:
    #         return (
    #             scores.numpy(),
    #             torch.zeros_like(scores).numpy(),
    #             torch.zeros_like(scores).numpy(),
    #             norms.numpy(),
    #         )
