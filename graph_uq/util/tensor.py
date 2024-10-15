import torch

from typing import Callable, Any, Iterable, Tuple, Sequence, Dict
from jaxtyping import jaxtyped, Shaped

from typeguard import typechecked

def apply_to_nested(x: Any, fn: Callable, filter_: Callable=lambda _: True) -> Any:
    """ Applies a function to (nested) torch.Tensor instances. """
    if isinstance(x, Tuple):
        return tuple((apply_to_nested(t, fn, filter_) for t in x))
    elif isinstance(x, Sequence):
        return [apply_to_nested(t, fn, filter_) for t in x]
    elif isinstance(x, Dict):
        return {k : apply_to_nested(v, fn, filter_) for k, v in x.items()}
    elif filter_(x):
        return fn(x)
    return x

def apply_to_tensor(x: Any, fn: Callable) -> Any:
    """ Applies a function if the input is a torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return fn(x)
    else:
        return x
    
def apply_to_nested_tensors(x: Any, fn: Callable, filter_: Callable=lambda _: True) -> Any:
    """ Applies a function to (nested) inputs that are torch.Tensor instances."""
    return apply_to_nested(x, lambda t: apply_to_tensor(t, fn), filter_=filter_)

@typechecked
def apply_to_optional_tensors(fn: Callable, tensors: Sequence[torch.Tensor | None]) -> Any | None:
    """ Applies a function to a sequence of optional tensors. """
    if not (all(tensor is None for tensor in tensors) or all(tensor is not None for tensor in tensors)):
        raise ValueError(f'Sequence of optional tensors must be homogenous w.r.t. being None')
    if len(tensors) == 0 or tensors[0] is None:
        return None
    else:
        return fn(tensors)
    
@typechecked
def collate_dict_of_tensors(x: Iterable[dict[Any, torch.Tensor | None]], collate_fn: lambda tensors: torch.cat(tensors, dim=0)) -> dict[Any, torch.Tensor | None]:
    """ Collates a dictionary of tensors by concatenating them.
    
    Args:
        x (Iterable[dict[Any, torch.Tensor | None]]): the dictionary of tensors to collate
        collate_fn (lambda tensors: torch.cat(tensors, dim=0)): the function to use to collate the tensors

    Returns:
        dict[Any, torch.Tensor | None]: the collated dictionary of tensors
    """
    if len(x) == 0:
        raise ValueError(f'Cannot collate an empty sequence of dictionaries')
    # Assert that all dictionaries have the same keys
    keys = set(x[0].keys())
    for d in x:
        if set(d.keys()) != keys:
            raise ValueError(f'All dictionaries must have the same keys')
    # Assert that for each key, all values are either None or a tensor consistently
    for key in keys:
        if not all((d[key] is None) == (d[key] is None) for d in x):
            raise ValueError(f'All values for key {key} must be either None or a tensor')
    # Collate the tensors for each key if they are not None, otherwise None
    return {key : collate_fn([d[key] for d in x if d[key] is not None]) if any(d[key] is not None for d in x) else None for key in keys}

@jaxtyped(typechecker=typechecked)
def logsumexp(input: Shaped[torch.Tensor, '...'], dim=0, weights: Shaped[torch.Tensor, '...'] | None = None, keepdims: bool = True) -> Shaped[torch.Tensor, '...']:
    """ Computes the logsumexp of a tensor along a dimension.

    Args:
        input (Shaped[torch.Tensor, &#39;...&#39;]): The input tensor.
        dim (int, optional): The dimension. Defaults to 0.
        weights (Shaped[torch.Tensor, &#39;...&#39;] | None, optional): Optional weights. Defaults to None.
        keepdims (bool, optional): Whether to keep dimensions. Defaults to True.

    Returns:
        _type_: _description_
    """
    max = input.max(dim=dim, keepdims=True).values
    shifted = (input - max).exp()
    if weights is not None:
        shifted *= weights
    return shifted.sum(dim=dim, keepdims=keepdims).log() + input.max(dim=dim, keepdims=keepdims).values
