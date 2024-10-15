import torch
from torch import Tensor
import logging

from jaxtyping import Bool, jaxtyped
from typeguard import typechecked

class MaskSamplingException(Exception):
    """ Exception raised when trying to sample more elements than available in a mask """

@jaxtyped(typechecker=typechecked)
def sample_from_mask(mask: Bool[Tensor, 'num'], size: int | float,
                     generator: torch.Generator | None = None, allow_empty_mask: bool=False) -> Bool[Tensor, 'num']:
    """ Samples indices from a mask of possible indices.
    
    Args:
        mask (Bool[Tensor, &#39;num_nodes&#39;]): All available indices 
        size (int | float): How many (absolute or as a fraction) to sample
        generator (torch.Generator | None): A random number generator for sampling
        allow_empty_mask (bool): Whether to allow empty masks
        
    Returns:
        (Bool[Tensor, &#39;num_nodes&#39;]): Sampled indices
    """
    mask_size = mask.sum().item()
    
    result = torch.zeros_like(mask)
    if mask_size == 0:
        if allow_empty_mask:
            return result
        else:
            raise MaskSamplingException('Trying to sample from an empty mask')

    if isinstance(size, float):
        size = int(size * mask_size)
    elif not isinstance(size, int):
        raise ValueError(f'Size to sample must either be int or float, but got {type(size)}')
    if size == 0:
        logging.warn(f'Sampling {size} (type {type(size)}) from a mask of size {mask_size}')
        return result
    elif size > mask_size:
        raise MaskSamplingException(f'Trying to sample {size} indices from a mask containing {mask_size} elements')
    else:
        indices = torch.where(mask)[0]
        indices = indices[torch.randperm(indices.size(0), generator=generator)[:size]]
        result[indices] = True
        return result
    
    
        
    