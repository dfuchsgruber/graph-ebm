from graph_uq.config.data import (
    DistributionShiftType,
    FeaturePerturbationsParameter,
    FeaturePerturbationType,
    LeaveOutClassesType,
)
from graph_uq.experiment import experiment


@experiment.named_config
def feature_perturbations_template():
    """Generates the split for feature perturbations but does not actually apply any."""
    data = dict(  # noqa: F841
        distribution_shift=dict(
            type_=DistributionShiftType.FEATURE_PERTURBATIONS,
            feature_perturbations=dict(
                base=dict(
                    type_=FeaturePerturbationType.NONE,
                )
            ),
        )
    )


@experiment.named_config
def feature_perturbations(data):
    """Applies i.i.d. bernoulli (p in {0, 1, 0.01, 0.5}) and normal (mean=0, std=1) feature perturbations to 50% of data."""
    ...

    data["distribution_shift"] |= dict(
        type_=DistributionShiftType.FEATURE_PERTURBATIONS,
        feature_perturbations=dict(
            ber_0=dict(
                type_=FeaturePerturbationType.BERNOULLI,
                p=0.0,
                transform=True,
            ),
            ber_1=dict(
                type_=FeaturePerturbationType.BERNOULLI,
                p=1.0,
                transform=True,
            ),
            ber_50=dict(
                type_=FeaturePerturbationType.BERNOULLI,
                p=0.5,
                transform=True,
            ),
            normal_0_1=dict(  # Far-OOD
                type_=FeaturePerturbationType.NORMAL,
                mean=0.0,
                std=1.0,
                transform=False,
            ),
        )
        | (
            dict(
                ber_mean=dict(
                    type_=FeaturePerturbationType.BERNOULLI,
                    p=FeaturePerturbationsParameter.AVERAGE,
                    transform=True,
                ),
                ber_mean_per_class=dict(
                    type_=FeaturePerturbationType.BERNOULLI,
                    p=FeaturePerturbationsParameter.AVERAGE_PER_CLASS,
                    transform=True,
                ),
            )
            if data["categorical_features"]
            else dict(
                normal_mean=dict(
                    type_=FeaturePerturbationType.NORMAL,
                    mean=FeaturePerturbationsParameter.AVERAGE,
                    std=FeaturePerturbationsParameter.AVERAGE,
                    transform=True,
                ),
                normal_mean_per_class=dict(
                    type_=FeaturePerturbationType.NORMAL,
                    mean=FeaturePerturbationsParameter.AVERAGE_PER_CLASS,
                    std=FeaturePerturbationsParameter.AVERAGE_PER_CLASS,
                    transform=True,
                ),
            )
        ),
    )


@experiment.named_config
def leave_out_classes():
    """The last classes in the dataset are OOD and the rest is ID."""
    data = dict(  # noqa: F841
        distribution_shift=dict(
            type_=DistributionShiftType.LOC,
            leave_out_classes_type=LeaveOutClassesType.LAST,
        ),
    )


@experiment.named_config
def ego_graph_size_shift():
    """50% of nodes with large ego graph size are ID and the rest is OOD."""
    data = dict(  # noqa: F841
        distribution_shift=dict(
            type_=DistributionShiftType.EGO_GRAPH_SIZE,
            percentile_ood=0.5,
            num_hops=2,
            high_is_id=True,
            per_class=True,
        )
    )


@experiment.named_config
def homophily_shift():
    """50% of high homophily nodes are ID and the rest is OOD."""
    data = dict(  # noqa: F841
        distribution_shift=dict(
            type_=DistributionShiftType.HOMOPHILY,
            percentile_ood=0.5,
            high_is_id=True,
            per_class=True,
        )
    )


@experiment.named_config
def page_rank_centrality_shift():
    """50% of nodes with high page rank(alpha=0.2, k=10) centrality are ID and the rest is OOD."""
    data = dict(  # noqa: F841
        distribution_shift=dict(
            type_=DistributionShiftType.PAGE_RANK_CENTRALITY,
            percentile_ood=0.5,
            teleport_probability=0.2,
            num_iterations=10,
            high_is_id=True,
            per_class=True,
        )
    )


@experiment.named_config
def inverse_page_rank_centrality_shift():
    """50% of nodes with high page rank(alpha=0.2, k=10) centrality are ID and the rest is OOD."""
    data = dict(  # noqa: F841
        distribution_shift=dict(
            type_=DistributionShiftType.PAGE_RANK_CENTRALITY,
            percentile_ood=0.5,
            teleport_probability=0.2,
            num_iterations=10,
            high_is_id=False,
            per_class=True,
        )
    )


@experiment.named_config
def no_distribution_shift():
    """No distribution shift."""
    data = dict(  # noqa: F841
        distribution_shift=dict(
            type_=DistributionShiftType.NONE,
        ),
    )
