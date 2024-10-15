from typing import Mapping

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import to_undirected as to_undirected_fn
from typeguard import typechecked

from graph_uq.data.transformer import transform_text_with_sentence_transformer


@typechecked
def read_npz_and_transform_texts(
    path: str, to_undirected: bool = True, sentence_transformer: str | None = None
) -> Data:
    """Reads an npz and transforms the text attributes with a sentence transformer.

    Args:
        path (str): The path to the npz file.
        to_undirected (bool, optional): Whether to convert the graph to undirected. Defaults to True.
        sentence_transformer (str | None, optional): The name of the sentence transformer to use. Defaults to None.

    Returns:
        Data: The graph data.
    """
    with np.load(path, allow_pickle=True) as f:
        return parse_npz_and_transform_texts(
            f, to_undirected=to_undirected, sentence_transformer=sentence_transformer
        )


@typechecked
def parse_npz_and_transform_texts(
    f: Mapping, to_undirected: bool = True, sentence_transformer: str | None = None
) -> Data:
    """Parses an npz and transforms the text attributes with a sentence transformer.

    Args:
        f (dict[str, Any]): The npz file.
        to_undirected (bool, optional): Whether to convert the graph to undirected. Defaults to True.
        sentence_transformer (str | None, optional): The name of the sentence transformer to use. Defaults to None.

    Returns:
        Data: The graph data.
    """
    attributes = {}

    if sentence_transformer is not None:
        assert (
            "attr_text" in f
        ), f"The npz file does not contain text attributes: {list(f.keys())}"
        x = transform_text_with_sentence_transformer(
            f["attr_text"].tolist(), sentence_transformer
        )
        x = torch.from_numpy(x).to(torch.float)
        attributes["feature_names"] = [
            f"_{sentence_transformer}_feature_{idx}" for idx in range(x.size(1))
        ]
    else:
        x = sp.csr_matrix(
            (f["attr_data"], f["attr_indices"], f["attr_indptr"]), f["attr_shape"]
        ).todense()
        x = torch.from_numpy(x).to(torch.float)
        x[x > 0] = 1
        if "idx_to_attr" in f:
            idx_to_feature_name = f["idx_to_attr"].tolist()
            attributes["feature_names"] = [
                idx_to_feature_name[idx] for idx in range(x.size(1))
            ]

    adj = sp.csr_matrix(
        (f["adj_data"], f["adj_indices"], f["adj_indptr"]), f["adj_shape"]
    ).tocoo()
    row = torch.from_numpy(adj.row).to(torch.long)
    col = torch.from_numpy(adj.col).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = remove_self_loops(edge_index)
    if to_undirected:
        edge_index = to_undirected_fn(edge_index, num_nodes=x.size(0))

    y = torch.from_numpy(f["labels"]).to(torch.long)
    if "idx_to_class" in f:
        idx_to_class = f["idx_to_class"].tolist()
        attributes["class_names"] = [
            idx_to_class[idx]
            for idx in range(y.max().item() + 1)  # type: ignore
        ]

    if "idx_to_node" in f:
        idx_to_node = f["idx_to_node"].tolist()
        attributes["node_names"] = [idx_to_node[idx] for idx in range(x.size(0))]

    return Data(x=x, edge_index=edge_index, y=y, **attributes)
