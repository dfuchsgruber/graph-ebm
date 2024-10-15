from enum import StrEnum, unique
from typing import Any, TypeAlias

from graph_uq.experiment import experiment


@unique
class PlotType(StrEnum):
    UNCERTAINTY_DISTRIBUTION = "uncertainty_distribution"


PlottingConfig: TypeAlias = dict[str, Any]


@experiment.config
def default_plotting_config():
    plot = dict()
