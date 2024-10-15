from typing import List, Dict, Tuple, Any, Sequence
from typeguard import typechecked
from collections import Counter, defaultdict

@typechecked
def groupby(iterable: Sequence, key=lambda x: x):
    """ `groupby` function that does not require sorted keys. """
    grouped = defaultdict(list)
    for idx, x in enumerate(iterable):
        grouped[key(x)].append(idx)
    for k, idxs in grouped.items():
        yield k, (iterable[idx] for idx in idxs)

@typechecked
def multilevel_indexed(pairs: list[tuple[Dict, Any]], depth: int = 0, **kwargs) -> Any:
    """ Creates a multilevel index for key, value pairs where the levels of the key are given as a dict
    
    E.g. for pairs
    ```python
    [
        ({'first' : 3, 'second' : 2}, 25),
        ({'first' : 3, 'second' : 4}, 25),
    ]
    ```
    This function will output:
    ```python
    {
        'first' : {
            3 : {
                'second' : {
                    2 : 25,
                    4 : 25,
                }
            }
        }
    }
    ```
    """
    if depth > 1000:
        raise RecursionError
    
    if len(pairs) == 0:
        return {}
    elif all(len(key) == 0 for key, _ in pairs):
        assert len(pairs) == 1, f'Found duplicate keys for {kwargs}'
        return pairs[0][1] # just return the value

    levels = set.union(*(set(pair[0].keys()) for pair in pairs))
    
    # Pick the key that gives the largest group
    counters = {
        level : Counter(pair[0].get(level, None) for pair in pairs)
        for level in levels
    }
    key = max(counters, key=lambda level: max(counters[level].values()))
    
    return {
        key : {
            value : multilevel_indexed(list(group), depth=depth+1, **(kwargs | {key : value}))
            for value, group in groupby(pairs, lambda pair: pair[0].pop(key, None))
        }
    }