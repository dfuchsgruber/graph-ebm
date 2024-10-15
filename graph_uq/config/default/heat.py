import itertools

from graph_uq.config.heat import (
    HeatEBMType,
    HeatProposalType,
    default_heat_config,
)
from graph_uq.config.uncertainty import UncertaintyModelType
from graph_uq.experiment import experiment

default_heat_gmm_config = default_heat_config | dict(
    type_=UncertaintyModelType.HEAT,
    ebm_type_=HeatEBMType.GMM,
    fit_to_propagated_embeddings=True,
    layer=-2,
    optimizer=dict(
        lr=1e-5,
        betas=(0, 0.999),
    ),
    loss=dict(eps_data=1e-4, l2_coef=10, verbose=False),
    scheduler=dict(milestones=[10], gamma=0.1),
    hidden_dims=[
        64,
    ],
    verbose=False,
    use_gpu=True,
    num_epochs=1000,
    use_base_distribution=True,
    steps=40,
    step_size_start=1e-4,
    step_size_end=1e-5,
    eps_start=5e-3,
    eps_end=5e-4,
    sgld_relu=True,
    proposal_type=HeatProposalType.BASE_DIST,
    temperature=1e0,
    temperature_prior=1e3,
    sample_from_batch_statistics=False,
    train_max_iter=100000,
    normalize=False,
)

default_heat_logits_config = default_heat_config | dict(
    type_=UncertaintyModelType.HEAT,
    ebm_type_=HeatEBMType.LOGITS,
    fit_to_propagated_embeddings=True,
    layer=-1,
    optimizer=dict(
        lr=1e-5,
        betas=(0, 0.999),
    ),
    loss=dict(eps_data=1e-4, l2_coef=1e-1, verbose=False),
    scheduler=dict(milestones=[10], gamma=0.1),
    hidden_dims=[
        64,
    ],
    verbose=False,
    use_gpu=True,
    num_epochs=1000,
    use_base_distribution=True,
    steps=50,
    step_size_start=1e-1,
    step_size_end=1e-2,
    eps_start=1e-1,
    eps_end=2e-1,
    sgld_relu=True,
    proposal_type=HeatProposalType.RANDOM_NORMAL,
    temperature=1e0,
    temperature_prior=1e0,
    sample_from_batch_statistics=False,
    train_max_iter=100000,
    normalize=False,
)


@experiment.named_config
def heat():
    uncertainty = dict(
        heat_combined=dict(
            type_=UncertaintyModelType.HEAT_COMPOSITE,
            ebms=dict(
                gmm=default_heat_gmm_config,
                logits=default_heat_logits_config,
            ),
            beta=-1.0,
        ),
    )
