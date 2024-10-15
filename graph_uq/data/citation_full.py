import torch_geometric.datasets as D
from pathlib import PurePath
import torch

from graph_uq.data.io import read_npz_and_transform_texts

class CitationFull(D.CitationFull):
    """ Wrapper around `CitationFull` that allows for a custom word embedding. """
    
    def __init__(self, *args, sentence_transformer: str | None = None, **kwargs):
        self.sentence_transformer = sentence_transformer
        super().__init__(*args, **kwargs)
        
    @property
    def processed_file_names(self) -> str:
        path = PurePath(super().processed_file_names)
        stem, suffix = path.stem, path.suffix
        if self.sentence_transformer is not None:
            stem += f'_{self.sentence_transformer.replace("/", "_")}'
        return stem + suffix

    def process(self):
        data = read_npz_and_transform_texts(self.raw_paths[0], to_undirected=self.to_undirected,
                                            sentence_transformer=self.sentence_transformer)
        data = data if self.pre_transform is None else self.pre_transform(data)
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])
        
    
    