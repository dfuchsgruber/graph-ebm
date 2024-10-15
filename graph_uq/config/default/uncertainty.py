import itertools

from graph_uq.config.data import DiffusionType
from graph_uq.config.uncertainty import (
    CovarianceType,
    EnergyAggregation,
    UncertaintyModelType,
)
from graph_uq.experiment import experiment


def get_latent_space_denisty_grid(
    layers=(-1, -2),
    propagation_types=(True, False),
    covariance_types=(
        CovarianceType.DIAGONAL,
        CovarianceType.FULL,
        CovarianceType.IDENTITY,
        CovarianceType.ISOTROPIC,
    ),
    tied_covariances=(True, False),
    **kwargs,
):
    """Builds the latent space density uncertainty model config. This is in an extra method so that sacred does not capture the locals."""
    uncertainty = dict()
    for layer_idx, propagated in itertools.product(layers, propagation_types):
        # add gaussian per class
        for covariance_type, tied in itertools.product(
            covariance_types, tied_covariances
        ):
            name = f"gaussian_per_class_layer_{layer_idx}_{covariance_type}"
            if tied:
                name += "_tied"
            if not propagated:
                name += "_unpropagated_embeddings"
            uncertainty[name] = dict(
                type_=UncertaintyModelType.GAUSSIAN_PER_CLASS,
                covariance_type=covariance_type,
                tied_covariance=tied,
                embedding_layer=layer_idx,
                embeddings_propagated=propagated,
                **kwargs,
            )
    return uncertainty


@experiment.named_config
def latent_space_density_grid():
    uncertainty = get_latent_space_denisty_grid(layers=(-2,))


@experiment.named_config
def latent_space_density():
    uncertainty = get_latent_space_denisty_grid(
        layers=(-2,),
        propagation_types=(True,),
        covariance_types=(CovarianceType.DIAGONAL,),
        tied_covariances=(False,),
    )


@experiment.named_config
def gpn_latent_space_density_grid():
    uncertainty = get_latent_space_denisty_grid(
        layers=(-1,), propagation_types=(False,)
    )


@experiment.named_config
def gpn_latent_space_density():
    uncertainty = get_latent_space_denisty_grid(
        layers=(-1,),
        propagation_types=(True,),
        covariance_types=(CovarianceType.DIAGONAL,),
        tied_covariances=(False),
    )


@experiment.named_config
def mc_uncertainty():
    uncertainty = dict(
        mutual_information=dict(type_=UncertaintyModelType.MUTUAL_INFORMATION),
        variance=dict(type_=UncertaintyModelType.PREDICTED_VARIANCE),
    )


@experiment.named_config
def evidence_uncertainty():
    uncertainty = dict(
        evidence=dict(type_=UncertaintyModelType.EVIDENCE),
    )


@experiment.named_config
def logit_uncertainty():
    uncertainty = dict(
        energy=dict(type_=UncertaintyModelType.ENERGY, temperature=1.0),
        gnn_safe=dict(type_=UncertaintyModelType.GNN_SAFE),
    )


@experiment.named_config
def softmax_uncertainty():
    uncertainty = dict(
        entropy=dict(type_=UncertaintyModelType.ENTROPY),
        max_score=dict(type_=UncertaintyModelType.MAX_SCORE),
    )


@experiment.named_config
def uncertainty_diffusion_grid():
    evaluation = dict(
        uncertainty_diffusion=dict(
            symmetric=dict(
                type_=DiffusionType.SYMMETRIC,
                steps=[1, 2, 5, 10],
            ),
            stochastic_alpha_50=dict(
                type_=DiffusionType.STOCHASTIC,
                steps=[1, 2, 5, 10],
                alpha=0.5,
            ),
            stochastic_alpha_20=dict(
                type_=DiffusionType.STOCHASTIC,
                steps=[1, 2, 5, 10],
                alpha=0.2,
            ),
            stochastic_alpha_90=dict(
                type_=DiffusionType.STOCHASTIC,
                steps=[1, 2, 5, 10],
                alpha=0.9,
            ),
            label_propagation_50=dict(
                type_=DiffusionType.LABEL_PROPAGATION,
                steps=[1, 2, 5, 10],
                alpha=0.5,
            ),
            label_propagation_20=dict(
                type_=DiffusionType.LABEL_PROPAGATION,
                steps=[1, 2, 5, 10],
                alpha=0.2,
            ),
            label_propagation_90=dict(
                type_=DiffusionType.LABEL_PROPAGATION,
                steps=[1, 2, 5, 10],
                alpha=0.9,
            ),
        )
    )


@experiment.named_config
def data_uncertainty():
    # Uncertainties from just the dataset
    uncertainty = dict(
        appr_distance=dict(
            type_=UncertaintyModelType.APPR_DISTANCE,
            alpha=0.2,
            num_steps=10,
        ),
        appr_diffusion=dict(
            type_=UncertaintyModelType.APPR_DIFFUSION,
            alpha=0.2,
            num_steps=10,
        ),
        feature_distance=dict(
            type_=UncertaintyModelType.FEATURE_DISTANCE,
            num_diffusion_steps=0,
        ),
        feature_distance_diffused=dict(
            type_=UncertaintyModelType.FEATURE_DISTANCE,
            num_diffusion_steps=10,
        ),
    )


@experiment.named_config
def diffused_gpc():
    uncertainty = get_latent_space_denisty_grid(
        layers=(-1, -2),
        propagation_types=(True, False),
        covariance_types=(CovarianceType.DIAGONAL,),
        tied_covariances=(False, True),
        diffusion=dict(
            symmetric=dict(
                type_=DiffusionType.SYMMETRIC,
                steps=[2, 5],
            ),
            stochastic_50=dict(
                type_=DiffusionType.STOCHASTIC,
                steps=[2, 5],
                alpha=0.5,
            ),
            label_propagation_50=dict(
                type_=DiffusionType.STOCHASTIC,
                steps=[2, 5],
                alpha=0.5,
            ),
            stochastic_20=dict(
                type_=DiffusionType.STOCHASTIC,
                steps=[2, 5],
                alpha=0.2,
            ),
            label_propagation_20=dict(
                type_=DiffusionType.STOCHASTIC,
                steps=[2, 5],
                alpha=0.2,
            ),
        ),
    )


def get_multi_scale_conditional_evidence_ebm_grid(
    embeddings_propagations=(
        True,
        False,
    ),
    logit_propagations=(True, False),
    layers=(
        -1,
        -2,
    ),
    correction_weights=(
        0.0,
        0.5,
        1.0,
    ),
    weights=(
        # diffused evidence, diffused energy, undiffused energy
        (1.0, 1.0, 1.0),
        (1.0, 0, 0),
        (0, 1.0, 0),
        (0, 0, 1.0),
    ),
):
    uncertainties = dict()
    for embeddings_propagated, logits_propagated, layer, correction_weight, (
        lambda_diffused_conditional_evidence,
        lambda_diffused_energy,
        lambda_undiffused_energy,
    ) in itertools.product(
        embeddings_propagations, logit_propagations, layers, correction_weights, weights
    ):
        name = f"multi_scale_conditional_evidence_ebm_layer_{layer}_{int(10 * correction_weight)}"
        name += "_weights_" + "_".join(
            map(
                lambda s: str(int(10 * s)),
                (
                    lambda_diffused_conditional_evidence,
                    lambda_diffused_energy,
                    lambda_undiffused_energy,
                ),
            )
        )
        if not embeddings_propagated:
            name += "_fit_to_unpropagated_embeddings"
        if not logits_propagated:
            name += "_fit_to_unpropagated_logits"
        uncertainties[name] = dict(
            type_=UncertaintyModelType.MULTI_SCALE_CORRECTED_CONDITIONAL_EBM,
            covariance_type="diagonal",
            tied_covariance=False,
            embedding_layer=layer,
            fit_to_embeddings_propagated=embeddings_propagated,
            fit_to_logits_propagated=logits_propagated,
            lambda_correction=correction_weight,
            lambda_diffused_conditional_evidence=lambda_diffused_conditional_evidence,
            lambda_diffused_energy=lambda_diffused_energy,
            lambda_undiffused_energy=lambda_undiffused_energy,
            evidence_diffusion={
                "type_": "label_propagation",
                "alpha": 0.5,
                "k": 10,
            },
            energy_diffusion={
                "type_": "label_propagation",
                "alpha": 0.5,
                "k": 10,
            },
        )
    return uncertainties


@experiment.named_config
def multi_scale_conditional_evidence_ebm_grid():
    uncertainty = get_multi_scale_conditional_evidence_ebm_grid(
        embeddings_propagations=(
            True,
            False,
        ),
        logit_propagations=(True, False),
        layers=(
            -1,
            -2,
        ),
        correction_weights=(
            0.0,
            0.5,
            1.0,
        ),
        weights=(
            # diffused evidence, diffused energy, undiffused energy
            (1.0, 1.0, 1.0),
            (1.0, 0, 0),
            (0, 1.0, 0),
            (0, 0, 1.0),
        ),
    )


def get_multi_scale_conditional_evidence_ebm(
    covariance_type: str = "diagonal",
    tied_covariance: bool = False,
    embedding_layer: int = -2,
    fit_to_embeddings_propagated: bool = True,
    fit_to_logits_propagated: bool = True,
    lambda_correction: float = 1.0,
    lambda_diffused_conditional_evidence: float = 1.0,
    lambda_diffused_energy: float = 1.0,
    lambda_undiffused_energy: float = 1.0,
    diffusion_type: str = "label_propagation",
    alpha: float = 0.5,
    k: int = 10,
    aggregation: str = EnergyAggregation.SUM,
):
    if alpha is not None:
        alpha_dict = dict(alpha=alpha)
    else:
        alpha_dict = dict()
    return dict(
        type_=UncertaintyModelType.MULTI_SCALE_CORRECTED_CONDITIONAL_EBM,
        covariance_type=covariance_type,
        tied_covariance=tied_covariance,
        embedding_layer=embedding_layer,
        fit_to_embeddings_propagated=fit_to_embeddings_propagated,
        fit_to_logits_propagated=fit_to_logits_propagated,
        lambda_correction=lambda_correction,
        lambda_diffused_conditional_evidence=lambda_diffused_conditional_evidence,
        lambda_diffused_energy=lambda_diffused_energy,
        lambda_undiffused_energy=lambda_undiffused_energy,
        evidence_diffusion={
            "type_": diffusion_type,
            "k": k,
        }
        | alpha_dict,
        energy_diffusion={
            "type_": diffusion_type,
            "k": k,
        }
        | alpha_dict,
        aggregation=aggregation,
    )


@experiment.named_config
def multi_scale_conditional_evidence_ebm():
    uncertainty = dict(
        msceebm=get_multi_scale_conditional_evidence_ebm(),
        msceebm_no_correction=get_multi_scale_conditional_evidence_ebm(
            lambda_correction=0.0
        ),
        msceebm_conditional_evidence_only=get_multi_scale_conditional_evidence_ebm(
            lambda_diffused_energy=0.0, lambda_undiffused_energy=0.0
        ),
        msceebm_diffused_energy_only=get_multi_scale_conditional_evidence_ebm(
            lambda_diffused_conditional_evidence=0.0, lambda_undiffused_energy=0.0
        ),
        mscceebm_undiffused_energy_only=get_multi_scale_conditional_evidence_ebm(
            lambda_diffused_conditional_evidence=0.0, lambda_diffused_energy=0.0
        ),
    )
