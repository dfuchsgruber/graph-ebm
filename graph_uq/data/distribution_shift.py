import logging
from typing import Any, Iterable

import torch
import torch_scatter
from jaxtyping import Bool, Shaped, jaxtyped
from torch import Tensor
from typeguard import typechecked

from graph_uq.config.data import (
    DataConfig,
    DistributionShiftType,
    DistributionType,
    FeaturePerturbationsParameter,
    FeaturePerturbationType,
    LeaveOutClassesType,
    Setting,
)
from graph_uq.data.data import Data
from graph_uq.data.split import sample_stratified_mask
from graph_uq.data.transform import get_data_transform
from graph_uq.util.normal import cov_and_mean, make_covariance_symmetric_and_psd


@jaxtyped(typechecker=typechecked)
def get_id_and_ood_data(
    data_id: Data, ood_mask: Bool[Tensor, " num_nodes"], config: DataConfig
) -> tuple[Data, Data]:
    """Split the data into in-distribution and out-of-distribution data.

    Args:
        data_id (Data): The data to split
        ood_mask (Bool[Tensor, &#39;num_nodes&#39;]): A mask of out-of-distribution nodes
        setting (str): Which setting to use

    Returns:
        Data: The in-distribution data, potentially reduced to only the in-distribution nodes
        Data: The base for the out-of-distribution data
    """
    assert not hasattr(
        data_id, "ood_mask"
    ), "Data already has an out-of-distribution mask"
    data_id.ood_mask = ood_mask
    data_id, data_ood = data_id.copy(), data_id.copy()  # shallow copy

    if Setting(config["setting"]) == Setting.INDUCTIVE:
        # Remove out-of-distribution nodes from the id graph
        data_id = data_id.subgraph(~ood_mask)

    data_id = get_data_transform(config["feature_normalization"])(data_id)
    data_ood = get_data_transform(config["feature_normalization"])(data_ood)
    return data_id, data_ood


@typechecked
def get_left_out_classes(
    base_data: Data,
    left_out_classes: Iterable[int | str] | None,
    num_left_out_classes: int | None,
    leave_out_classes_type: str,
    in_edges: bool,
) -> list[int]:
    """Get the classes to leave out.

    Args:
        base_data (Data): The base data
        left_out_classes (_type_): The classes to leave out
        num_left_out_classes (_type_): The number of classes to leave out
        leave_out_classes_type (str): The type of leave out classes
        in_edges (bool): Whether to use incoming edges for computing the shift (default: True)

    Returns:
        Iterable[int]: The classes to leave out
    """
    match leave_out_classes_type:
        case LeaveOutClassesType.FIRST:
            assert isinstance(
                num_left_out_classes, int
            ), "num_left_out_classes must be set."
            assert (
                num_left_out_classes < base_data.num_classes
            ), "num_left_out_classes must be smaller than the number of classes."
            return list(range(num_left_out_classes))
        case LeaveOutClassesType.LAST:
            assert (
                num_left_out_classes < base_data.num_classes
            ), "num_left_out_classes must be smaller than the number of classes."
            return list(
                range(
                    base_data.num_classes - num_left_out_classes, base_data.num_classes
                )
            )
        case LeaveOutClassesType.FIXED:
            assert isinstance(
                left_out_classes, Iterable
            ), "left_out_classes must be set."
            assert all(
                c in range(base_data.num_classes) for c in left_out_classes
            ), "Invalid classes to leave out"
            return list(left_out_classes)  # type: ignore
        case LeaveOutClassesType.RANDOM:
            return torch.randperm(base_data.num_classes)[:num_left_out_classes].tolist()
        case (
            LeaveOutClassesType.LOWEST_HOMOPHILY
            | LeaveOutClassesType.HIGHEST_HOMOPHILY
        ):
            homophily = (
                base_data.num_intra_inter_class_edges_per_class(
                    intra=True, _in=in_edges
                ).float()
                / base_data.num_edges_per_class(_in=in_edges).float()
            )
            return torch.argsort(
                homophily,
                descending=leave_out_classes_type
                == LeaveOutClassesType.HIGHEST_HOMOPHILY,
            )[:num_left_out_classes].tolist()
        case _:
            raise ValueError(
                f"Unknown leave out classes type: {leave_out_classes_type}"
            )


@typechecked
def apply_distribution_shift_leave_out_classes(
    base_data: Data, config: DataConfig
) -> tuple[Data, dict[str, Data]]:
    """Apply a distribution shift by leaving out classes.

    Args:
        base_data (Data): The base data

    Returns:
        Data: The in-distribution data, potentially reduced to the subset of classes that are not left out
        dict[str, Data]: The out-of-distribution data objects
    """
    left_out_classes = get_left_out_classes(
        base_data,
        config["distribution_shift"]["left_out_classes"],
        config["distribution_shift"]["num_left_out_classes"],
        config["distribution_shift"]["leave_out_classes_type"],
        config["distribution_shift"]["in_edges"],
    )
    base_data = base_data.copy()

    logging.info(f"Leaving out classes: {left_out_classes}")
    ood_mask = torch.zeros(
        base_data.num_nodes,
        dtype=torch.bool,
        device=base_data.edge_index.device,  # type: ignore
    )
    for y in left_out_classes:
        ood_mask |= base_data.y == y
    base_data.permute_labels(
        torch.tensor(
            [i for i in range(base_data.num_classes) if i not in left_out_classes]
            + list(left_out_classes),
            dtype=int,  # type: ignore
        )
    )

    data_id, data_ood = get_id_and_ood_data(base_data, ood_mask=ood_mask, config=config)
    # assert data_id. < data_ood.num_classes_in_masks, 'The number of classes in the in-distribution data must be smaller than the number of classes in the out-of-distribution data.'
    return data_id, {
        "loc": data_ood,
    }


@typechecked
def apply_feature_perturbation_bernoulli(data_ood: Data, config: dict[str, Any]):
    """Applies a bernoulli feature perturbation in-place."""
    match config["p"]:
        case int() | float():
            data_ood.x[data_ood.ood_mask] = data_ood.x[data_ood.ood_mask].bernoulli_(
                config["p"]
            )
        case FeaturePerturbationsParameter.AVERAGE:
            p = (
                (data_ood.x[data_ood.get_distribution_mask(DistributionType.ID)] > 0)
                .float()
                .mean(dim=0)
            )
            data_ood.x[data_ood.ood_mask] = (
                data_ood.x[data_ood.ood_mask].uniform_() < p[None, :]
            ).float()
        case FeaturePerturbationsParameter.AVERAGE_PER_CLASS:
            p = torch_scatter.scatter_mean(
                (
                    data_ood.x[data_ood.get_distribution_mask(DistributionType.ID)] > 0
                ).float(),
                data_ood.y[data_ood.get_distribution_mask(DistributionType.ID)],  # type: ignore
                dim=0,
                dim_size=data_ood.num_classes,
            )
            data_ood.x[data_ood.ood_mask] = (
                data_ood.x[data_ood.ood_mask].uniform_()
                < p[data_ood.y[data_ood.ood_mask]]  # type: ignore
            ).float()
        case _:
            raise ValueError(f"Unknown p type: {config['p']}")


@typechecked
def apply_feature_perturbation_normal(data_ood: Data, config: dict[str, Any]):
    """Applies a normal feature perturbation in-place."""

    x_ood = data_ood.x[data_ood.get_distribution_mask(DistributionType.OOD)].normal_(
        0, 1.0
    )

    match config["std"]:
        case int() | float():
            x_ood *= config["std"]
        case FeaturePerturbationsParameter.AVERAGE:
            covariance, _ = cov_and_mean(
                data_ood.x[data_ood.get_distribution_mask(DistributionType.ID)]
            )
            L = torch.linalg.cholesky(
                make_covariance_symmetric_and_psd(covariance), upper=False
            )
            x_ood = x_ood @ L
        case FeaturePerturbationsParameter.AVERAGE_PER_CLASS:
            # There is no scatter covariance, so we iterate over the classes
            for class_idx in range(data_ood.num_classes):
                covariance, _ = cov_and_mean(
                    data_ood.x[
                        data_ood.get_distribution_mask(DistributionType.ID)
                        & (data_ood.y == class_idx)
                    ]
                )
                L = torch.linalg.cholesky(
                    make_covariance_symmetric_and_psd(covariance), upper=False
                )
                data_ood.x[data_ood.ood_mask & (data_ood.y == class_idx)] = (
                    data_ood.x[data_ood.ood_mask & (data_ood.y == class_idx)] @ L
                )
        case _:
            raise ValueError(f"Unknown std type: {config['std']}")

    match config["mean"]:
        case int() | float():
            x_ood += config["mean"]
        case FeaturePerturbationsParameter.AVERAGE:
            x_ood += data_ood.x[
                data_ood.get_distribution_mask(DistributionType.ID)
            ].mean(dim=0, keepdim=True)
        case FeaturePerturbationsParameter.AVERAGE_PER_CLASS:
            mean_per_class = torch_scatter.scatter_mean(
                data_ood.x[data_ood.get_distribution_mask(DistributionType.ID)],
                data_ood.y[data_ood.get_distribution_mask(DistributionType.ID)],
                dim=0,
                dim_size=data_ood.num_classes,
            )
            x_ood += mean_per_class[data_ood.y[data_ood.ood_mask]]
        case _:
            raise ValueError(f"Unknown mean type: {config['mean']}")

    p_replacement = config.get("p_replacement", 1.0)
    mask = (
        torch.rand_like(
            data_ood.x[data_ood.get_distribution_mask(DistributionType.OOD)]
        )
        <= p_replacement
    )
    data_ood.x[data_ood.get_distribution_mask(DistributionType.OOD)] = torch.where(
        mask,
        x_ood,
        data_ood.x[data_ood.get_distribution_mask(DistributionType.OOD)],
    )


@typechecked
def apply_feature_perturbation(
    data_config: DataConfig, data_ood: Data, config: dict[str, Any]
) -> Data:
    """Apply a feature perturbation to the out-of-distribution data.

    Args:
        data_ood (Data): The out-of-distribution data
        config (dict[str, Any]): The configuration of the feature perturbation

    Returns:
        Data: The out-of-distribution data with the feature perturbation applied
    """
    data_ood = data_ood.copy()  # shallow_copy
    assert data_ood.x is not None
    data_ood.x = data_ood.x.clone()
    match FeaturePerturbationType(config["type_"]):
        case FeaturePerturbationType.BERNOULLI:
            apply_feature_perturbation_bernoulli(data_ood, config)
        case FeaturePerturbationType.NORMAL:
            apply_feature_perturbation_normal(data_ood, config)
        case FeaturePerturbationType.NONE:
            pass
        case type_:
            raise ValueError(f"Unknown feature perturbation type: {type_}")

    if config.get("transform", False):
        data_ood = get_data_transform(data_config["feature_normalization"])(data_ood)

    return data_ood


@typechecked
def apply_distribution_shift_feature_perturbations(
    base_data: Data,
    config: DataConfig,
) -> tuple[Data, dict[str, Data]]:
    """Apply a distribution shift by perturbing the features of out-of-distribution nodes.

    Args:
        base_data (Data): The base data
        feature_perturbations (dict[str, dict]): The feature perturbations to apply
        num_ood (float | int): The number of out-of-distribution nodes

    Returns:
        Data: The in-distribution data, potentially reduced to exclude the out-of-distribution nodes
        dict[str, Data]: The out-of-distribution data objects, one for each feature perturbation
    """
    ood_mask = sample_stratified_mask(
        base_data, config["distribution_shift"]["num_ood"]
    )
    data_id, data_ood = get_id_and_ood_data(base_data, ood_mask=ood_mask, config=config)
    if config["distribution_shift"]["feature_perturbations"] is None:
        return data_id, {}
    return data_id, {
        name: apply_feature_perturbation(config, data_ood, ood_config)
        for name, ood_config in config["distribution_shift"][
            "feature_perturbations"
        ].items()
    }


@typechecked
def apply_distribution_shift_none(
    base_data: Data, config: DataConfig
) -> tuple[Data, dict[str, Data]]:
    """Applies no distribution shift

    Args:
        base_data (Data): The base data

    Returns:
        Data: The in-distribution data, potentially reduced to exclude the out-of-distribution nodes
        dict[str, Data]: The out-of-distribution data objects, one for each feature perturbation
    """
    ood_mask = torch.zeros(base_data.num_nodes, dtype=bool)  # type: ignore
    data_id, data_ood = get_id_and_ood_data(base_data, config=config, ood_mask=ood_mask)
    return data_id, {}


@jaxtyped(typechecker=typechecked)
def apply_distribution_shift_by_score(
    base_data: Data,
    scores: Shaped[Tensor, " num_nodes"],
    config: DataConfig,
    ood_name: str,
) -> tuple[Data, dict[str, Data]]:
    """Applies a distribution shift based on a score.

    Args:
        base_data (Data): The base data
        scores (Shaped[Tensor, &#39;num_nodes&#39;]): The scores to use for the distribution shift
        percentile_ood (float): Fraction of nodes to consider as out-of-distribution
        high_is_id (bool): Whether high scores should be considered in-distribution
        ood_name (str): The name of the out-of-distribution data
        per_class (bool): Whether to use some proxy for splitting globally or per class

    Returns:
        Data: The in-distribution data, potentially reduced to exclude the out-of-distribution nodes
        dict[str, Data]: The out-of-distribution data objects
    """
    idxs_sorted = torch.argsort(
        scores, descending=not config["distribution_shift"]["high_is_id"]
    )
    ood_mask = torch.zeros(base_data.num_nodes, dtype=bool)  # type: ignore
    if config["distribution_shift"]["per_class"]:
        for class_idx in range(base_data.num_classes):
            class_idxs_sorted = idxs_sorted[base_data.y[idxs_sorted] == class_idx]
            ood_mask[
                class_idxs_sorted[
                    : int(
                        float(class_idxs_sorted.size(0))
                        * config["distribution_shift"]["percentile_ood"]
                    )
                ]
            ] = True
    else:
        ood_mask[
            idxs_sorted[
                : int(
                    idxs_sorted.size(0) * config["distribution_shift"]["percentile_ood"]
                )
            ]
        ] = True
    data_id, data_ood = get_id_and_ood_data(base_data, config=config, ood_mask=ood_mask)
    return data_id, {ood_name: data_ood}


@typechecked
def apply_distribution_shift_page_rank_centrality(
    base_data: Data,
    config: DataConfig,
    ood_name: str = "prc",
) -> tuple[Data, dict[str, Data]]:
    """Separates the in-distribution and out-of-distribution data based on the PageRank centrality.

    Args:
        base_data (Data): The base data
        percentile_ood (float): Fraction of nodes to consider as out-of-distribution
        teleport_probability (float): teleport probability for the PageRank computation
        num_iterations (int): number of iterations for the PageRank computation
        high_is_id (bool): Whether high PageRank centrality should be considered in-distribution
        ood_name (str): The name of the out-of-distribution data

    Returns:
        Data: The in-distribution data, potentially reduced to exclude the out-of-distribution nodes
        dict[str, Data]: The out-of-distribution data objects
    """
    scores = base_data.get_appr_scores(
        teleport_probability=config["distribution_shift"]["teleport_probability"],
        num_iterations=config["distribution_shift"]["num_iterations"],
    )
    return apply_distribution_shift_by_score(
        base_data, scores, config, ood_name=ood_name
    )


@typechecked
def apply_distribution_shift_ego_graph_size(
    base_data: Data, config: DataConfig, ood_name: str = "ego"
) -> tuple[Data, dict[str, Data]]:
    """Separates the in-distribution and out-of-distribution data based on the size of the ego graph.

    Args:
        base_data (Data): The base data
        num_hops (int): The number of hops for the ego graph
        ood_name (str, optional): The name of the ood dataset. Defaults to 'ego'.

    Returns:
        Data: The in-distribution data, potentially reduced to exclude the out-of-distribution nodes
        dict[str, Data]: The out-of-distribution data objects
    """
    scores = base_data.get_ego_graph_sizes(
        num_hops=config["distribution_shift"]["num_hops"]
    )
    return apply_distribution_shift_by_score(
        base_data, scores, config, ood_name=ood_name
    )


@typechecked
def apply_distribution_shift_homophily(
    base_data: Data, config: DataConfig, ood_name: str = "homo"
) -> tuple[Data, dict[str, Data]]:
    """Separates the in-distribution and out-of-distribution data based on the homophily of a node.

    Args:
        base_data (Data): The base data
        in_edges (bool): whether to consider incoming or outgoing edges
        ood_name (str, optional): name of the ood data. Defaults to 'homo'.

    Returns:
        Data: The in-distribution data, potentially reduced to exclude the out-of-distribution nodes
        dict[str, Data]: The out-of-distribution data objects
    """
    homophily = base_data.num_intra_inter_class_edges_per_node(
        intra=True, _in=config["distribution_shift"]["in_edges"]
    ) / base_data.node_degree(config["distribution_shift"]["in_edges"])
    return apply_distribution_shift_by_score(
        base_data, homophily, config, ood_name=ood_name
    )


@typechecked
def apply_distribution_shift(
    base_data: Data,
    config: DataConfig,
) -> tuple[Data, dict[str, Data]]:
    """Apply a distribution shift.

    Args:
        base_data (Data): The base data
        type_ (str): The type of distribution shift to apply

    Returns:
        Data: The in-distribution data, potentially reduced to exclude the out-of-distribution nodes
        dict[str, Data]: The out-of-distribution data objects, one for each distribution shift
    """
    match DistributionShiftType(config["distribution_shift"]["type_"]):
        case DistributionShiftType.LOC:
            return apply_distribution_shift_leave_out_classes(
                base_data=base_data, config=config
            )
        case DistributionShiftType.FEATURE_PERTURBATIONS:
            return apply_distribution_shift_feature_perturbations(
                base_data=base_data, config=config
            )
        case DistributionShiftType.NONE:
            return apply_distribution_shift_none(base_data=base_data, config=config)
        case DistributionShiftType.PAGE_RANK_CENTRALITY:
            return apply_distribution_shift_page_rank_centrality(
                base_data=base_data, config=config
            )
        case DistributionShiftType.EGO_GRAPH_SIZE:
            return apply_distribution_shift_ego_graph_size(
                base_data=base_data, config=config
            )
        case DistributionShiftType.HOMOPHILY:
            return apply_distribution_shift_homophily(
                base_data=base_data, config=config
            )
        case type_:
            raise ValueError(f"Unknown distribution shift type: {type_}")
