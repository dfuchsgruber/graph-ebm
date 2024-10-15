import os
from os import PathLike
from pathlib import Path

import wandb
from matplotlib.figure import Figure
from torch import Tensor
from typeguard import typechecked

from graph_uq.config import Config
from graph_uq.config.logging import WandbConfig
from graph_uq.experiment import experiment
from graph_uq.logging.logger import Logger


class WandbLogger(Logger):
    """Logger class for wandb."""

    @experiment.capture()  # type: ignore
    def __init__(self, config: WandbConfig, _config: Config | None = None):
        dir = config["dir"]
        cache_dir = config["cache_dir"]

        if dir is not None:
            os.makedirs(dir, exist_ok=True)
        if cache_dir is not None:
            self._set_cache_dir(cache_dir)
        wandb_run = wandb.init(
            config=_config,  # type: ignore
            id=config["id"],
            entity=config["entity"],
            project=config["project"],
            group=config["group"],
            mode=config["mode"],
            name=config["name"],
            tags=config["tags"],
            dir=dir,
            resume=(config["mode"] == "online") and "allow",
            settings=wandb.Settings(log_internal=str(config["log_internal_dir"])),
        )
        if wandb_run is not None:
            experiment.info["wandb"] = dict(
                dir=wandb_run.dir,
                entity=wandb_run.entity,
                group=wandb_run.group,
                id=wandb_run.id,
                name=wandb_run.name,
                notes=wandb_run.notes,
                path=wandb_run.path,
                project=wandb_run.project,
                resumed=wandb_run.resumed,
                start_time=wandb_run.start_time,
                tags=wandb_run.tags,
                url=wandb_run.url,
            )
        self.run = wandb_run

    def _set_cache_dir(self, cache_dir: PathLike | str):
        """Sets all cache directories W&B uses to stop it from flooding ./cache and .local/share with artifacts..."""
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["WANDB_ARTIFACT_LOCATION"] = str(cache_dir)
        os.environ["WANDB_ARTIFACT_DIR"] = str(cache_dir)
        os.environ["WANDB_CACHE_DIR"] = str(cache_dir)
        os.environ["WANDB_CONFIG_DIR"] = str(cache_dir)
        os.environ["WANDB_DATA_DIR"] = str(cache_dir)

    @property
    def dir(self) -> Path | None:
        if wandb.run is None:
            return None
        path = Path(wandb.run.dir) / "metrics"
        os.makedirs(path, exist_ok=True)
        return path

    def _clean(self):
        import subprocess

        subprocess.run(["wandb", "sync", "--clean-force"], check=True)
        subprocess.run(["wandb", "artifact", "cache", "cleanup", "0GB"], check=True)

    def finish(self):
        super().finish()
        if self.run is not None:
            self.run.finish()
        self._clean()

    @typechecked
    def log(
        self,
        metrics: dict[str, float | int | Tensor],
        step: int | None = None,
        commit: bool | None = None,
    ):
        super().log(metrics, step=step, commit=commit)
        if self.run is not None:
            self.run.log(metrics, step=step, commit=commit)

    @typechecked
    def log_figure(
        self,
        figure: Figure,
        key: str,
        step: int | None = None,
        commit: bool | None = None,
    ):
        super().log_figure(figure, key, step=step, commit=commit)
        if self.run is not None:
            self.run.log({key: wandb.Image(figure)}, step=step, commit=commit)
