import rich
from rich.table import Table
import torch
import numpy as np
from collections import defaultdict

from typing import Any, Iterable
from typeguard import typechecked

from graph_uq.metric import Metric
from graph_uq.config.data import DatasetSplit

@typechecked
def print_table(data: dict[Metric, Iterable[float | None]], title: str | None = None,
    print_splits: Iterable[str] = [DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.ALL]):
    """ Prints some data over multiple runs as a table. """
    
    # Group by dataset split
    grouped = defaultdict(dict)
    for key, values in data.items():
        split = key.dataset_split
        metric = Metric() + key
        metric.dataset_split = None
        grouped[metric][split] = values

    table = Table('Metric ' + '/'.join(map(str, print_splits)), 'Mean', 'Std', title=title)
    for key in sorted(grouped.keys(), key=str):
        values = grouped[key]

        values = [[v.item() if isinstance(v, torch.Tensor) and len(v.size()) == 0 else v for v in values.get(split, [])] for split in print_splits]
        values = [[float(value) for value in values_split if isinstance(value, float)] for values_split in values]
        means = [np.mean(values_split) if len(values_split) > 0 else np.nan for values_split in values]
        stds = [np.std(values_split) if len(values_split) > 1 else np.nan for values_split in values]
        
        table.add_row(str(key), 
            '/'.join([f'{mean:.2f}' for mean in means],),
            '/'.join([f'{std:.2f}' for std in stds],)
        )
    rich.print(table)