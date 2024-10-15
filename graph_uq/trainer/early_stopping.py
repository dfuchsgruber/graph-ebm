from typing import Any

import torch.nn as nn
from typeguard import typechecked

from graph_uq.config.trainer import EarlyStoppingConfig
from graph_uq.metric import Metric


class EarlyStopping:
    """Monitor for early stopping."""

    @typechecked
    def __init__(self, config: EarlyStoppingConfig):
        self.monitor = (
            Metric(**config["monitor"])
            if isinstance(config["monitor"], dict)
            else config["monitor"]
        )
        self.higher_is_better = config["higher_is_better"]
        self.patience = config["patience"]
        self.min_delta = config["min_delta"]
        self.save_model_state = config["save_model_state"]
        self.reset()

    def reset(self):
        """Resets the monitor."""
        if self.higher_is_better:
            self.best = -float("inf")
        else:
            self.best = float("inf")
        self.best_epoch = -1
        self.best_state = None
        self.epochs_without_improvement = 0

    @typechecked
    def step(
        self, metrics: dict[Metric, Any], epoch: int, model: nn.Module | None = None
    ):
        if self.monitor not in metrics:
            raise RuntimeError(
                f"Metric to be monitored {self.monitor} is not computed in an epoch."
            )
        value = metrics[self.monitor]

        if (self.higher_is_better and value > (self.best + self.min_delta)) or (
            not self.higher_is_better and value < (self.best - self.min_delta)
        ):
            self.best = value
            self.best_epoch = epoch
            if model is not None and self.save_model_state:
                self.best_state = {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                }
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

    @property
    def should_stop(self) -> bool:
        return self.epochs_without_improvement > self.patience
