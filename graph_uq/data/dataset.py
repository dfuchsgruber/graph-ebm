from dataclasses import dataclass, field
import rich

from graph_uq.data.data import Data


@dataclass
class Dataset:
    """ A dataset consists of in-distribution and out-of-distribution data. """
    data_base: Data # The base data from which the dataset was created
    data_train: Data
    data_shifted: dict[str, Data] = field(default_factory=dict)

    def __post_init__(self):
        # Verify that the node idxs of training set are a subset of the ood datasets
        for name, dataset in self.data_shifted.items():
            assert set(self.data_train.node_keys.tolist()).issubset(set(dataset.node_keys.tolist())), f'Node idxs of training set are not a subset of the {name} shift'
    
    def print_summary(self):
        rich.print('Dataset summary')
        self.data_base.print_summary(title='Base Data')
        self.data_train.print_summary(title='Train Data')
        for name, data_ood in self.data_shifted.items():
            data_ood.print_summary(title=f'{name} Shift')
        
        
        

