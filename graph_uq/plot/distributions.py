from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Bool, Float, Int, jaxtyped
from typeguard import typechecked

from graph_uq.evaluation.result import Plot
from graph_uq.experiment import experiment
from graph_uq.util import is_outlier


@jaxtyped(typechecker=typechecked)
def plot_uncertainty_distribution_at(
    ax,
    uncertainty: Float[np.ndarray, "num_nodes"],
    labels: Int[np.ndarray, "num_nodes"],
    label_names: dict = {},
    kwargs_hist: dict = {},
    outlier_quantile: float = 0.95,
    legend: bool = True,
    legend_kwargs: dict = {},
):
    """Plots the distribution of uncertainty values for each label.

    Args:
        ax (_type_): axes to plot on
        uncertainty (Float[np.ndarray, &#39;num_nodes&#39;]): The uncertainty values
        labels (Int[np.ndarray, &#39;num_nodes&#39;]): The labels of the nodes
        label_names (Dict, optional): Names for the labels. Defaults to {}.
        kwargs_hist (Dict, optional): _description_. Defaults to {}.
        outlier_quantile (float, optional): _description_. Defaults to .95.
        legend (bool, optional): _description_. Defaults to True.
        legend_kwargs (Dict, optional): _description_. Defaults to {}.
    """
    for label in sorted(np.unique(labels)):
        uncertainty_label = uncertainty[labels == label]
        if outlier_quantile is not None:
            uncertainty_label = uncertainty_label[
                ~is_outlier(uncertainty_label, quantile=outlier_quantile)
            ]

        ax.hist(uncertainty_label, **kwargs_hist, label=label_names.get(label))
        if legend:
            ax.legend(**legend_kwargs)


@jaxtyped(typechecker=typechecked)
def plot_uncertainty_distribution(
    uncertainty: Float[np.ndarray, "num_nodes"],
    labels: Int[np.ndarray, "num_nodes"],
    label_names: dict = {},
    kwargs_hist: dict = {},
    outlier_quantile: float = 0.95,
    legend: bool = True,
    legend_kwargs: dict = {},
    subplots_kwargs: dict = {},
    density: bool = True,
    bins: int | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    alpha: float = 0.5,
    title: str | None = None,
) -> Plot:
    """Plots the distribution of uncertainty values for each label.

    Args:
        uncertainty (Float[np.ndarray, &#39;num_nodes&#39;]): The uncertainty values
        labels (Int[np.ndarray, &#39;num_nodes&#39;]): The labels of the nodes
        label_names (Dict, optional): Names for the labels. Defaults to {}.
        kwargs_hist (Dict, optional): _description_. Defaults to {}.
        outlier_quantile (float, optional): _description_. Defaults to .95.
        legend (bool, optional): _description_. Defaults to True.
        legend_kwargs (Dict, optional): _description_. Defaults to {}.
        subplots_kwargs (Dict, optional): _description_. Defaults to {}.
        density (bool, optional): Whether to have a density plot. Defaults to True.
        bins (int | None, optional): Number of bins. Defaults to None.
        x_label (str | None, optional): Label for the x-axis. Defaults to None.
        y_label (str | None, optional): Label for the y-axis. Defaults to None.
        alpha (float, optional): Alpha for the plot. Defaults to 0.5.
        title (str | None, optional): Title for the plot. Defaults to None.

    Returns:
        Plot: The plot
    """
    fig, ax = plt.subplots(**subplots_kwargs)
    kwargs_hist.setdefault("density", density)
    kwargs_hist.setdefault("bins", bins)
    kwargs_hist.setdefault("alpha", alpha)
    plot_uncertainty_distribution_at(
        ax,
        uncertainty,
        labels,
        label_names,
        kwargs_hist,
        outlier_quantile,
        legend,
        legend_kwargs,
    )
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)
    plt.close(fig)
    return Plot(fig, ax)
