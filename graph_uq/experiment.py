from contextlib import ContextDecorator
from typing import Any, Callable, Iterable, Self

from sacred.run import Run
from seml.experiment.experiment import Experiment as BaseExperiment


class Experiment(BaseExperiment):
    """Singleton experiment."""

    def __new__(cls: type[Self]) -> Self:
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
        return cls.instance

    class RunContext(ContextDecorator):
        """Decorator that sets up a run with custom overrides."""

        def __init__(
            self,
            experiment: "Experiment",
            *named_configs: Any,
            enter_hooks: Iterable[Callable[["Experiment"], Any]] = [],
            exit_hooks: Iterable[Callable[["Experiment"], Any]] = [],
            **overrides: Any,
        ) -> None:
            self.experiment = experiment
            self.named_configs = named_configs
            self.overrides = overrides
            self._saved_experiment_default_command = None
            self.enter_hooks = list(enter_hooks)
            self.exit_hooks = list(exit_hooks)

        def __enter__(self) -> Run:
            from uuid import uuid4

            def mock_command():
                pass

            mock_command.__name__ = f"_mock_command_{uuid4()}"
            self._saved_experiment_default_command = self.experiment.default_command
            self.experiment.main(mock_command)
            for hook in self.enter_hooks:
                hook(self.experiment)
            return self.experiment._create_run(
                named_configs=self.named_configs, config_updates=self.overrides
            )

        def __exit__(self, *exc):
            self.experiment.default_command = self._saved_experiment_default_command
            for hook in self.exit_hooks:
                hook(self.experiment)

    def create_run(
        self,
        *named_configs: Any,
        enter_hooks: Iterable[Callable[["Experiment"], Any]] = [],
        exit_hooks: Iterable[Callable[["Experiment"], Any]] = [],
        **overrides: Any,
    ) -> RunContext:
        """Context decorator that creates a run with custom overrides.

        Args:
            exit_hooks (Iterable[Callable[[&quot;Experiment&quot;], Any]]): hooks to be executed when the run context is exited
            enter_hooks (Iterable[Callable[[&quot;Experiment&quot;], Any]], optional): hooks to be executed when the run context is entered. Defaults to [].

        Returns:
            RunContext: the context manager
        """
        return self.RunContext(
            self,
            *named_configs,
            **overrides,
            enter_hooks=enter_hooks,
            exit_hooks=exit_hooks,
        )


experiment = Experiment()
