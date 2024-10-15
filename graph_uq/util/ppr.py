import numpy as np
import scipy.sparse as sp
import torch
import torch_scatter
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from tqdm import tqdm
from typeguard import typechecked


@jaxtyped(typechecker=typechecked)
def assert_adjacency_is_row_normalized(
    edge_index: Int[Tensor, "2 num_edges"],
    edge_weights: Float[Tensor, " num_edges"],
    num_nodes: int,
):
    """Asserts that the adjacency matrix is row-normalized"""

    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1

    idx_src, idx_target = edge_index
    # Check for a stochastic matrix
    sums = torch_scatter.scatter_add(edge_weights, idx_src, dim=-1, dim_size=num_nodes)

    if torch.isclose(sums, torch.tensor(0.0, dtype=sums.dtype), atol=1e-3).any():
        raise ValueError(
            f"Found nodes without outgoing edges at indices {torch.where(torch.isclose(sums, torch.tensor(0.0), atol=1e-3))[0]}"
        )
    assert torch.isclose(sums, torch.tensor(1.0, dtype=sums.dtype), atol=1e-3).all(), (
        f"Expected semi-stochastic matrix for PPR approximation but got {sums[~torch.isclose(sums, torch.tensor(1.0), atol=1e-3)]}"
        + f" at indices {torch.where(~torch.isclose(sums, torch.tensor(1.0), atol=1e-3))[0]}"
    )


@torch.no_grad()
@jaxtyped(typechecker=typechecked)
def approximate_ppr_matrix(
    edge_index: Int[Tensor, "2 num_edges"],
    edge_weights: Float[Tensor, " num_edges"],
    teleport_probability: float = 0.2,
    num_iterations: int = 10,
    verbose: bool = True,
    num_nodes: int | None = None,
) -> Float[Tensor, "num_nodes num_nodes"]:
    """approximates the ppr matrix by doing power iteration.

    Args:
        edge_index: edge index tensor, where the first row contains the source nodes and the second row the target nodes
        edge_weights: edge weights tensor, where each entry corresponds to the weight of the edge in edge_index.
            the weights should be row-normalized, i.e. the sum of the weights of the outgoing edges of each node should be 1.
        teleport_probability: probability of teleporting to a random node
        num_iterations: number of iterations for power iteration
        verbose: whether to print progress
        num_nodes: number of nodes in the graph

    Returns:
        Float[Tensor, 'num_nodes num_nodes']: the approximate ppr matrix

    """
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1

    assert_adjacency_is_row_normalized(edge_index, edge_weights, num_nodes)
    A = sp.coo_matrix(
        (edge_weights.cpu().numpy(), edge_index.cpu().numpy()),
        shape=(num_nodes, num_nodes),
    )

    Pi = np.ones((num_nodes, num_nodes)) / num_nodes
    pbar = range(num_iterations)
    if verbose:
        pbar = tqdm(pbar)
    for it in pbar:
        new = (1 - teleport_probability) * (A.T @ Pi) + (
            teleport_probability / num_nodes
        ) * np.eye(num_nodes)
        diff = np.linalg.norm(new - Pi)
        if verbose:
            pbar.set_description(f"APPR residuals: {diff:.5f}")  # type: ignore
        Pi = new
    return torch.tensor(Pi)


@torch.no_grad()
@jaxtyped(typechecker=typechecked)
def approximate_ppr_scores(
    edge_index: Int[Tensor, "2 num_edges"],
    edge_weights: Float[Tensor, " num_edges"],
    teleport_probability: float = 0.2,
    num_iterations: int = 10,
    verbose: bool = True,
    num_nodes: int | None = None,
) -> Float[Tensor, " num_nodes"]:
    """Computes (approximate) per-node ppr centrality scores"""

    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1

    assert_adjacency_is_row_normalized(edge_index, edge_weights, num_nodes)
    A = sp.coo_matrix(
        (edge_weights.cpu().numpy(), edge_index.cpu().numpy()),
        shape=(num_nodes, num_nodes),
    )

    page_rank_scores = np.ones(num_nodes) / num_nodes
    pbar = range(num_iterations)
    if verbose:
        pbar = tqdm(pbar)
    for it in pbar:
        prev = page_rank_scores.copy()  # type: ignore
        page_rank_scores = (1 - teleport_probability) * (A.T @ page_rank_scores) + (
            teleport_probability
        ) / num_nodes
        diff = np.linalg.norm(prev - page_rank_scores)
        if verbose:
            pbar.set_description(f"APPR residuals: {diff:.5f}")  # type: ignore
        assert np.allclose(page_rank_scores.sum(), 1.0, atol=1e-3)
    return torch.tensor(page_rank_scores)
