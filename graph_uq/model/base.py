from abc import abstractmethod

import torch.nn as nn
from typeguard import typechecked

from graph_uq.config.model import ModelConfig
from graph_uq.data.data import Data
from graph_uq.model.prediction import Prediction


class BaseModel(nn.Module):
    """Base class for models."""

    @typechecked
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.name = config["name"]
        self.num_samples_train = config["num_samples_train"]
        self.num_samples_eval = config["num_samples_eval"]

    def reset_cache(self): ...

    def reset_parameters(self): ...

    @abstractmethod
    def forward(self, batch: Data) -> Prediction:
        raise NotImplementedError

    @property
    @abstractmethod
    def prediction_changes_at_eval(self) -> bool:
        raise NotImplementedError

    def predict(
        self,
        batch: Data,
        num_samples: int | None = None,
    ) -> Prediction:
        num_samples = num_samples or (
            self.num_samples_train if self.training else self.num_samples_eval
        )
        if num_samples == 1:
            prediction = self(batch)
            return prediction
        else:
            prediction = Prediction.collate([self(batch) for _ in range(num_samples)])
            return prediction


class Ensemble(BaseModel):
    """Wrapper for an ensemble of models."""

    def __init__(self, config: ModelConfig, models: list[BaseModel]):
        super().__init__(config)
        self.models: list[BaseModel] = nn.ModuleList(models)  # type: ignore

    def reset_parameters(self):
        for model in self.models:
            model.reset_parameters()

    def reset_cache(self):
        for model in self.models:
            model.reset_cache()

    @property
    def prediction_changes_at_eval(self) -> bool:
        return any(model.prediction_changes_at_eval for model in self.models)

    @typechecked
    def predict(self, batch: Data) -> Prediction:
        return Prediction.collate([model.predict(batch) for model in self.models])  # type: ignore
