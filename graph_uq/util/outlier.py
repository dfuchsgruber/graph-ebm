import numpy as np

from jaxtyping import Shaped, jaxtyped, Bool
from typeguard import typechecked
from numpy import ndarray

@jaxtyped(typechecker=typechecked)
def is_outlier(array: Shaped[ndarray, 'n'], quantile: float=0.95) -> Bool[ndarray, 'n']:
    """ Detects outliers as samples that are not within a certain quantile of the data. 
    
    Parameters:
    -----------
    array : ndarray, shape [N]
        The array to find outliers in.
    quantile : float
        How much of the data to include.

    Returns:
    --------
    outliers : ndarray, shape [N]
        Array that masks all outliers, i.e. `outliers[i] == True` if a point is identified as outlier.    
    """
    finite = np.isfinite(array)
    array_finite = array[finite]
    idxs = np.argsort(array_finite)
    delta = (1 - quantile) / 2
    upper, lower = int((quantile + delta) * array_finite.shape[0]), int(delta * array_finite.shape[0])
    idxs = idxs[lower:upper]
    # Translate to the whole array
    idxs = np.arange(array.shape[0])[finite][idxs]
    is_outlier = np.ones_like(array, dtype=bool)
    is_outlier[idxs] = False
    return is_outlier