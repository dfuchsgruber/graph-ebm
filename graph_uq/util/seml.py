import logging
from dataclasses import asdict, is_dataclass

import numpy as np
import sacred.run
import torch

from graph_uq.experiment import experiment


@experiment.capture()  # type: ignore
def setup_experiment(_run: sacred.run.Run | None = None):
    """Sets up the Sacred experiment."""
    null_logger = logging.getLogger("sacred")
    null_logger.addHandler(logging.NullHandler())
    null_logger.propagate = False
    _run.run_logger = null_logger  # type: ignore


def to_json_serializable(result):
    """Makes a (nested) object JSON serializable."""
    match result:
        case object if is_dataclass(object):
            return to_json_serializable(asdict(object))
        case torch.Tensor():
            return to_json_serializable(result.tolist())
        case np.ndarray():
            return to_json_serializable(result.tolist())
        case [*_] | tuple():
            return [to_json_serializable(i) for i in result]
        case {**items}:
            return {k: to_json_serializable(v) for k, v in items.items()}
        case bool():
            return bool(result)
        case float():
            return float(result)
        case int():
            return int(result)
        case str():
            return str(result)
        case None:
            return None
        case scalar:
            raise ValueError(f"Can not serialize {scalar} of type {type(scalar)}")
