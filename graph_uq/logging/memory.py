import json
import os
from collections import defaultdict
from os import PathLike
from pathlib import Path
from uuid import uuid4

from torch import Tensor
from typeguard import typechecked

from graph_uq.logging.logger import Logger
from graph_uq.util.seml import to_json_serializable


class MemoryLogger(Logger):
    """Uses the memory for logging and only saves upon calling the `finish` command."""

    @property
    def dir(self) -> Path:
        return self._dir

    def __init__(self, output_base_dir: PathLike | str):
        self._dir = Path(output_base_dir) / str(uuid4())
        os.makedirs(self._dir, exist_ok=True)
        self.logs = defaultdict(list)

    @typechecked
    def log(
        self,
        metrics: dict[str, float | int | Tensor],
        step: int | None = None,
        commit: bool | None = None,
    ):
        super().log(metrics, step, commit)
        for key, value in metrics.items():
            self.logs[key].append(to_json_serializable(value))

    def finish(self):
        super().finish()
        with open(self._dir / "logs.json", "w") as f:
            json.dump(self.logs, f)
