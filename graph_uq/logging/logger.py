import pickle
from pathlib import Path
from typing import Any

import torch
from jaxtyping import Shaped
from matplotlib.figure import Figure
from torch import Tensor
from typeguard import typechecked

from graph_uq.config import Config
from graph_uq.data.data import Data
from graph_uq.evaluation.result import EvaluationResult
from graph_uq.summary import print_table as print_table_fn
from graph_uq.util.index import multilevel_indexed
from graph_uq.util.seml import to_json_serializable


class Logger:
    """Base class for loggers."""

    @property
    def dir(self) -> Path:
        raise NotImplementedError

    def finish(self): ...

    @typechecked
    def log(
        self,
        metrics: dict[str, float | int | Shaped[Tensor, ""]],
        step: int | None = None,
        commit: bool | None = None,
    ): ...

    @typechecked
    def log_figure(
        self,
        figure: Figure,
        key: str,
        step: int | None = None,
        commit: bool | None = None,
    ): ...

    @typechecked
    def log_configuration(self, config: dict[str, Any]):
        torch.save(config, self.dir / "config.pt")

    @typechecked
    def log_results(
        self, results: list[EvaluationResult], print_table: bool = True
    ) -> Path:
        results_aggregated = {
            metric: [result.metrics.get(metric, float("nan")) for result in results]
            for metric in set().union(*(result.metrics for result in results))
        }
        if print_table:
            print_table_fn(results_aggregated)  # type: ignore

        # Log aggregated results in a compressed pickle file
        results_file: Path = self.dir / "results_aggregated.pkl"
        with open(results_file, "wb") as f:
            pickle.dump(
                to_json_serializable(
                    multilevel_indexed(
                        [
                            (metric.asdict(), values)
                            for metric, values in results_aggregated.items()
                        ]
                    )
                ),
                f,
            )

        return results_file
