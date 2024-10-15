from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Bool, Float, Int, jaxtyped
from typeguard import typechecked

from graph_uq.evaluation.result import Plot
from graph_uq.experiment import experiment
from graph_uq.util import is_outlier


@jaxtyped(typechecker=typechecked)
def plot_2d_embeddings_at(
    ax,
    embeddings: Float[np.ndarray, "num_nodes 2"],
    labels: Int[np.ndarray, "num_nodes"],
    label_names: dict = {},
    hue_labels: Int[np.ndarray, "num_nodes"] | None = None,
    hue_label_to_name_and_marker: dict | None = None,
    kwargs_scatter: dict = {},
    legend: bool = True,
    legend_kwargs: dict = {},
):
    """Plots 2d embeddings.

    Args:
        ax (_type_): axes to plot on
        embeddings (Float[np.ndarray, &#39;num_nodes 2&#39;]): The embeddings
        labels (Int[np.ndarray, &#39;num_nodes&#39;]): The labels of the nodes
        label_names (Dict, optional): Names for the labels. Defaults to {}.
        hue_labels (Int[np.ndarray, &#39;num_nodes&#39;], optional): The labels to use for the hue. Defaults to None.
        hue_label_to_name_and_marker (Dict, optional): Maps hue labels to names and markers. Defaults to None.
        kwargs_scatter (Dict, optional): Keyword arguments to  `ax.scatter`. Defaults to {}.
        legend (bool, optional): Whether to use a legend. Defaults to True.
        legend_kwargs (Dict, optional): Keyword arguments to `ax.legend`. Defaults to {}.
    """

    for label in sorted(np.unique(labels)):
        embeddings_label = embeddings[labels == label]
        if hue_labels is not None:
            for hue_label in sorted(np.unique(hue_labels)):
                embeddings_label_hue = embeddings_label[
                    hue_labels[labels == label] == hue_label
                ]
                ax.scatter(
                    embeddings_label_hue,
                    **kwargs_scatter,
                    label=label_names.get(label),
                    marker=hue_label_to_name_and_marker.get(hue_label, (_, "o"))[1],  # type: ignore
                )
        else:
            ax.scatter(
                embeddings_label[:, 0],
                embeddings_label[:, 1],
                **kwargs_scatter,
                label=label_names.get(label),
            )
        if legend and len(ax.get_legend_handles_labels()[0]) > 0:
            ax.legend(**legend_kwargs)


@jaxtyped(typechecker=typechecked)
def plot_2d_embeddings(
    embeddings: Float[np.ndarray, "num_nodes 2"],
    labels: Int[np.ndarray, "num_nodes"],
    label_names: dict = {},
    kwargs_scatter: dict = {},
    hue_labels: Int[np.ndarray, "num_nodes"] | None = None,
    hue_label_to_name_and_marker: dict | None = None,
    legend: bool = True,
    legend_kwargs: dict = {},
    subplots_kwargs: dict = {},
    markersize: float | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    alpha: float = 0.5,
    title: str | None = None,
) -> Plot:
    """Plots the distribution of uncertainty values for each label.

    Args:
        embeddings (Float[np.ndarray, &#39;num_nodes 2&#39;]): The embeddings
        labels (Int[np.ndarray, &#39;num_nodes&#39;]): The labels of the nodes
        label_names (Dict, optional): Names for the labels. Defaults to {}.
        hue_labels (Int[np.ndarray, &#39;num_nodes&#39;] | None, optional): Labels for the hue. Defaults to None.
        hue_label_to_name_and_marker (Dict | None, optional): Mapping from hue label to name and marker. Defaults to None.
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
    kwargs_scatter.setdefault("alpha", alpha)
    kwargs_scatter.setdefault("s", markersize)
    plot_2d_embeddings_at(
        ax,
        embeddings,
        labels,
        label_names=label_names,
        kwargs_scatter=kwargs_scatter,
        legend=legend,
        legend_kwargs=legend_kwargs,
        hue_label_to_name_and_marker=hue_label_to_name_and_marker,
        hue_labels=hue_labels,
    )
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)
    plt.close(fig)
    return Plot(fig, ax)
