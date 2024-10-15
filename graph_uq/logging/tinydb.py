import os
from os import PathLike
from pathlib import Path
from uuid import uuid4

from tinydb import TinyDB
from torch import Tensor
from typeguard import typechecked

from graph_uq.experiment import experiment
from graph_uq.logging.logger import Logger
from graph_uq.util.seml import to_json_serializable


class TinyDBLogger(Logger):
    """Uses TinyDB for logging."""

    @property
    def dir(self) -> Path:
        return self._dir

    def __init__(self, output_base_dir: PathLike | str):
        self._dir = Path(output_base_dir) / str(uuid4())
        os.makedirs(self._dir, exist_ok=True)
        self.db = TinyDB(self._dir / "logs.json")

    @typechecked
    def log(
        self,
        metrics: dict[str, float | int | Tensor],
        step: int | None = None,
        commit: bool | None = None,
    ):
        super().log(metrics, step, commit)
        for key, value in metrics.items():
            self.db.insert(dict(key=key, value=to_json_serializable(value), step=step))
