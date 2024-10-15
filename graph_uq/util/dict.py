"""Utils for dicts."""

from copy import deepcopy


def merge_dicts(*dicts):
    """Recursively merge dictionaries."""
    result = {}
    for d in dicts:
        for k, v in d.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = merge_dicts(result[k], v)
            else:
                result[k] = deepcopy(v)
    return result
