import torch
import torch.nn as nn
from jaxtyping import Float, jaxtyped
from typeguard import typechecked


@jaxtyped(typechecker=typechecked)
def get_features(
    model, data, *args, **kwargs
) -> Float[torch.Tensor, "batch num_features"]:
    raise RuntimeError(f"Implement feature getting for HEAT")


def requires_grad(model: nn.Module, flag: bool = True):
    for p in model.parameters():
        p.requires_grad = flag
