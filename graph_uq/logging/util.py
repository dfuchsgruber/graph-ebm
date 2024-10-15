# Code by https://github.com/martenlienen?tab=repositories

from typing import Any, Iterable

import rich
from rich.syntax import Syntax
from typeguard import typechecked


@typechecked
def print_config(config: dict) -> None:
    rich.print(Syntax(config, "yaml"))  # type: ignore


@typechecked
def print_table(data: dict[Any, Iterable[float | None]], title: str | None = None):
    """Prints some data over multiple runs as a table."""
    import numpy as np
    import rich
    import torch
    from rich.table import Table

    table = Table("Metric", "Mean", "Std", title=title)
    for key in sorted(data.keys(), key=str):
        values = data[key]
        values = [
            v.item() if isinstance(v, torch.Tensor) and len(v.size()) == 0 else v
            for v in values
        ]
        values = [float(value) for value in values if isinstance(value, float)]
        table.add_row(
            str(key),
            f"{np.mean(values) if len(values) > 0 else np.nan:.2f}",
            f"{np.std(values) if len(values) > 1 else np.nan:.2f}",
        )
    rich.print(table)
