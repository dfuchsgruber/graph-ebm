from contextlib import ContextDecorator
from typing import Any
from uuid import uuid4

from sacred.experiment import Run

from graph_uq.config import (  # needs to be imported first to set up the default configuration
    setup,
    setup_all,
)
from graph_uq.experiment import experiment

setup()  # basic setup of the configs: needed such that imported methods know "some configuration"


class experiment_run(ContextDecorator):
    """Decorator that sets up a configuration for debugging."""

    def __init__(
        self,
        *setup_args,
        named_configs: list[str] = [],
        config_overrides: dict[str, Any] | None = None,
        _all=False,
        **setup_kwargs,
    ):
        self.setup_args = setup_args
        self.named_configs = named_configs
        self.config_overrides = config_overrides
        self.setup_kwargs = setup_kwargs
        self._all = _all
        self._experiment_default_command = None

    def __enter__(self) -> Run:
        def mock_command():
            pass

        mock_command.__name__ = f"mock_command_{uuid4()}"
        self._experiment_default_command = experiment.default_command
        experiment.main(mock_command)

        return create_run(
            *self.setup_args,
            named_configs=self.named_configs,
            config_overrides=self.config_overrides,
            _all=self._all,
            **self.setup_kwargs,
        )

    def __exit__(self, *exc):
        experiment.default_command = self._experiment_default_command


def create_run(
    *setup_args,
    named_configs: list[str] = [],
    config_overrides: dict[str, Any] | None = None,
    _all=False,
    **setup_kwargs,
) -> Run:
    """Sets up a configuration for testing."""
    setup_kwargs = setup_kwargs or {}
    if _all:
        setup_all()
    else:
        setup(*setup_args, **setup_kwargs)
    experiment.main(lambda: None)
    return experiment._create_run(
        named_configs=named_configs, config_updates=config_overrides
    )
