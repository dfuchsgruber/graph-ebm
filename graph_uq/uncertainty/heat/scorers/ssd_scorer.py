import numpy as np
import torch
import torch.nn.functional as F
import torch_scatter
from jaxtyping import Float, Int
from torch import Tensor

from graph_uq.uncertainty.heat.distributions import (
    Categorical,
    MixtureSameFamily,
    MultivariateNormal,
    _batch_mahalanobis,
)

from .abstract_scorer import AbastractOODScorer


class SSDScorer(AbastractOODScorer):
    def __init__(
        self,
        force_fit_base_dist: bool = True,
        use_simplified_mahalanobis_score: bool = False,
        use_gpu: bool = True,
        diag_coefficient_only: bool = True,
        name: str = "SSD",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = name
        self.num_classes = None
        self.force_fit_base_dist = force_fit_base_dist
        self.use_simplified_mahalanobis_score = use_simplified_mahalanobis_score
        self.use_gpu = use_gpu
        self.diag_coefficient_only = diag_coefficient_only

        # Internal attributes
        self.global_mean = 0
        self.global_std = 1
        self.means = []
        self.dist = None
        self.comp = None
        self.mix = None
        self.precision = None

    def reset(self):
        self.global_mean = 0
        self.global_std = 1
        self.means = []
        self.dist = None
        self.comp = None
        self.mix = None
        self.precision = None

    @property
    def cache_name(self) -> str:
        str_cache = ""
        str_normalize = "_normalized" if self.normalize else ""
        str_pooling = "_" + self.pooling + "_pooling"

        return str_cache + str_normalize + str_pooling

    @property
    def has_sample(self) -> bool:
        return True

    def sample(
        self,
        sample_shape: int,
        target: Tensor | None = None,
        temp: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        assert self.dist is not None
        rand_labels, rand_imgs = self.dist.sample(sample_shape, target, temp)  # type: ignore
        rand_labels = rand_labels if self.use_simplified_mahalanobis_score else None
        return rand_labels, rand_imgs  # type: ignore

    @torch.no_grad()
    def _fit(
        self,
        features: Float[Tensor, "num_train num_features"],
        targets: Int[Tensor, "num_train"],
        verbose: bool = False,
    ):
        self.reset()

        self.num_classes = int(targets.max().item() + 1)

        z = features

        if self.use_react:
            z = z.clip(max=self.react_threshold)

        if self.normalize:
            z = F.normalize(z, dim=1)

        class_features = [z[targets == c] for c in range(self.num_classes)]
        class_count = torch_scatter.scatter_add(
            torch.ones_like(targets), targets, dim=0, dim_size=self.num_classes
        )
        self.means = torch_scatter.scatter_mean(
            z, targets, dim=0, dim_size=self.num_classes
        )
        Z = torch.cat(
            [class_features[c] - self.means[c] for c in range(self.num_classes)]
        )
        del class_features
        cov = np.cov(Z.T.cpu().numpy(), bias=True)
        # solved singular values pb in cholesky (see https://scicomp.stackexchange.com/questions/30631/how-to-find-the-nearest-a-near-positive-definite-from-a-given-matrix)
        cov += 1e-12 * np.eye(cov.shape[0])
        cov = torch.from_numpy(cov).float().to(features.device, non_blocking=True)
        if self.diag_coefficient_only:
            cov *= torch.eye(cov.size(0), device=features.device)
        L = torch.linalg.cholesky(cov)
        del cov
        self.mix = Categorical(class_count)
        self.comp = MultivariateNormal(
            loc=self.means, scale_tril=L.float(), validate_args=False
        )
        self.dist = MixtureSameFamily(self.mix, self.comp, validate_args=False)
        torch.cuda.empty_cache()

    def _maha_score(self, z: Tensor) -> Tensor:
        return (
            _batch_mahalanobis(
                self.comp._unbroadcasted_scale_tril,  # type: ignore
                z.unsqueeze(1) - self.comp.loc,  # type: ignore
            )
            .min(dim=1)
            .values
        )

    def energy(self, z: Tensor, labels: Tensor | None = None) -> Tensor:
        return -self.dist.log_prob(z, labels=labels)  # type: ignore

    def _score_batch(
        self, features: Float[Tensor, "batch num_features"]
    ) -> torch.Tensor:
        z = features

        if self.use_react:
            z = z.clip(max=self.react_threshold)

        if self.normalize:
            z = F.normalize(z, dim=1)

        score = self._maha_score(z)
        assert len(score) == len(z)
        return score
