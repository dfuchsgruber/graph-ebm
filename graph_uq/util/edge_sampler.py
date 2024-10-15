import numpy as np
import scipy.sparse as sp
import torch

from graph_uq.data import Data

from jaxtyping import jaxtyped, Int
from typeguard import typechecked
from torch import Tensor

class EdgeSampler:
    """ Samples edge anchor pairs, i.e. anchor, positive and negative, where positives are connected
    to the anchor and negatives are not."""

    def __init__(self, seed: int=1337, cached: bool=True, max_attempts: int=100):
        self.rng = np.random.RandomState(seed=seed)
        self.cached = cached
        self.max_attempts = max_attempts
        self.reset_cache()
        
    def reset_cache(self):
        self.cache = None

    @jaxtyped(typechecker=typechecked)
    def sample(self, data: Data, num: int) -> Int[Tensor, '3 num_sampled']:
        """ Samples anchors """
        
        A = self.cache
        if A is None:
            A = sp.coo_matrix((np.ones(data.edge_index.size(1)), data.edge_index.detach().cpu().numpy()), 
                              shape=(data.num_nodes, data.num_nodes), dtype=bool).tocsr()
            if self.cached:
                self.cache = A
           
        # Sample positive links (which implies the anchors)
        for _ in range(self.max_attempts):
            idx_pos = self.rng.choice(data.edge_index.size(1), size=num, replace=False)
            pos = data.edge_index[:, idx_pos]
            anchors = pos[0, :].detach().cpu().numpy()
            if (A[anchors, :].sum(1) < data.num_nodes).all():
                break
        else:
            raise RuntimeError(f'Could not sample edges such that all anchors have non-existent links')
        
        # Sample negative endpoints
        neg_endpoints = self.rng.choice(data.num_nodes, size=len(anchors), replace=True)
        for _ in range(self.max_attempts):
            is_edge = np.array(A[anchors, :].todense()[np.arange(num), neg_endpoints]).flatten()
            
            if not is_edge.any():
                break
            else:
                neg_endpoints[is_edge] = self.rng.choice(data.num_nodes, size=int(is_edge.sum()), replace=True)
        else:
            raise RuntimeError(f'Could not sample negative endpoints...')
            
        for u, v in zip(anchors, neg_endpoints):
            if A[u, v]:
                raise RuntimeError(f'Invalid negative endpoint {v} for anchor {u}')
        
        neg_endpoints = torch.Tensor(neg_endpoints).view((1, -1)).long().to(pos.device)
        return torch.cat((pos, neg_endpoints), 0)
    
