from graph_uq.config.model import GPNEvidenceScale, ModelType, default_model_config
from graph_uq.config.trainer import (
    GPNWarmup,
    LossFunctionType,
    TrainerType,
    default_trainer_config,
)
from graph_uq.experiment import experiment


@experiment.named_config
def gcn():
    model = dict(
        type_=ModelType.GCN,
        name="gcn",
        hidden_dims=[
            64,
        ],
    )


@experiment.named_config
def gcn_large_with_bn():
    model = dict(
        type_=ModelType.GCN,
        name="gcn_large_with_bn",
        hidden_dims=[256, 256],
        batch_norm=True,
    )
    trainer = dict(
        early_stopping=dict(
            patience=200,
        )
    )


@experiment.named_config
def gat():
    model = dict(
        type_=ModelType.GAT,
        name="gat",
        hidden_dims=[
            64,
        ],
    )


@experiment.named_config
def sage():
    model = dict(
        type_=ModelType.SAGE,
        name="sage",
        hidden_dims=[
            64,
        ],
    )


@experiment.named_config
def sage_no_normalize():
    model = dict(
        type_=ModelType.SAGE,
        name="sage",
        hidden_dims=[
            64,
        ],
        normalize=False,
    )


@experiment.named_config
def gin():
    model = dict(
        type_=ModelType.GIN,
        name="gin",
        hidden_dims=[
            64,
        ],
    )


@experiment.named_config
def mlp():
    model = dict(
        type_=ModelType.MLP,
        name="mlp",
        hidden_dims=[
            64,
        ],
    )


@experiment.named_config
def no_model():
    model = dict(
        type_=ModelType.NONE,
        name="none",
        num_inits=1,
    )
    trainer = dict(
        type_=TrainerType.NONE,
    )


@experiment.named_config
def ensemble():
    model = dict(
        num_ensemble_members=10,
    )


@experiment.named_config
def mc_dropout():
    model = dict(
        num_samples_eval=50,  # how many samples to draw at eval
        num_samples_train=1,  # how many samples to draw at train
        dropout=0.5,
        dropout_at_eval=True,
    )


@experiment.named_config
def mc_dropedge():
    model = dict(
        num_samples_eval=50,  # how many samples to draw at eval
        num_samples_train=1,  # how many samples to draw at train
        dropedge=0.5,
        dropedge_at_eval=True,
    )


@experiment.named_config
def appnp():
    model = dict(
        name="appnp",
        type_=ModelType.APPNP,
        num_diffusion_steps=10,
        alpha=0.1,
        hidden_dims=[
            64,
        ],
    )


@experiment.named_config
def gpn():
    model = dict(
        name="gpn",
        type_=ModelType.GPN,
        num_diffusion_steps=10,
        alpha=0.1,
        flow_dim=16,
        num_flow_layers=10,
        evidence_scale=GPNEvidenceScale.LATENT_NEW,
        dropout=0.5,
        inplace=False,
        batch_norm=False,  # Only used for OGBN arxiv
        # Following appendix D of https://arxiv.org/pdf/2110.14012.pdf
    )
    trainer = dict(
        type_=TrainerType.GPN,
        weight_decay=1e-3,
        learning_rate=1e-3,
        weight_decay_flow=0.0,
        learning_rate_flow=1e-2,
        weight_decay_warmup=0.0,
        learning_rate_warmup=1e-2,
        entropy_regularization_loss_weight=1e-4,
        num_warmup_epochs=5,
        warmup=GPNWarmup.FLOW,
    )


@experiment.named_config
def gpn_large_with_bn():
    model = dict(
        name="gpn_large_with_bn",
        type_=ModelType.GPN,
        num_diffusion_steps=10,
        alpha=0.1,
        flow_dim=16,
        num_flow_layers=10,
        evidence_scale=GPNEvidenceScale.LATENT_NEW,
        dropout=0.5,
        inplace=False,
        batch_norm=True,
        hidden_dims=[256, 256],
        # Following appendix D of https://arxiv.org/pdf/2110.14012.pdf
    )
    trainer = dict(
        type_=TrainerType.GPN,
        weight_decay=1e-3,
        learning_rate=1e-3,
        weight_decay_flow=0.0,
        learning_rate_flow=1e-2,
        weight_decay_warmup=0.0,
        learning_rate_warmup=1e-2,
        entropy_regularization_loss_weight=1e-4,
        num_warmup_epochs=5,
        warmup=GPNWarmup.FLOW,
        early_stopping=dict(
            patience=200,
        ),
    )


@experiment.named_config
def bgcn():
    model = dict(
        num_samples_eval=50,  # how many samples to draw at eval
        name="bgcn",
        type_=ModelType.BGCN,
        hidden_dims=[256, 256],
        batch_norm=False,
    )
    trainer = dict(
        type_=TrainerType.BayesianSGD,
        kl_divergence_loss_weight=1e-1,
        learning_rate=1e-2,
        weight_decay=0.0,
        loss_function_type=LossFunctionType.CROSS_ENTROPY_AND_KL_DIVERGENCE,
    )


@experiment.named_config
def bgcn_large_with_bn():
    model = dict(
        num_samples_eval=50,  # how many samples to draw at eval
        name="bgcn",
        type_=ModelType.BGCN,
        hidden_dims=[256, 256],
        batch_norm=True,
    )
    trainer = dict(
        type_=TrainerType.BayesianSGD,
        kl_divergence_loss_weight=1e-1,
        learning_rate=1e-2,
        weight_decay=0.0,
        loss_function_type=LossFunctionType.CROSS_ENTROPY_AND_KL_DIVERGENCE,
        early_stopping=dict(
            patience=200,
        ),
    )


@experiment.named_config
def gdk():
    model = dict(
        cutoff=10,
        sigma=1.0,
        type_=ModelType.GDK,
        name="gdk",
    )
    trainer = dict(
        type_=TrainerType.NONE,
    )


@experiment.named_config
def sgcn():
    model = dict(
        type_=ModelType.SGNN,
        name="gdk_sgcn",
        teacher=default_model_config
        | dict(
            type_=ModelType.GCN,
            name="gcn",
            hidden_dims=[
                64,
            ],
        ),
        gdk_prior=default_model_config
        | dict(
            cutoff=10,
            sigma=1.0,
            type_=ModelType.GDK,
            name="gdk",
        ),
        backbone=default_model_config
        | dict(
            type_=ModelType.GCN,
            name="gcn",
            hidden_dims=[
                16,
            ],
        ),
    )
    trainer = dict(
        type_=TrainerType.SGNN,
        teacher=default_trainer_config
        | dict(
            type_=TrainerType.SGD,
            loss_function_type=LossFunctionType.CROSS_ENTROPY,
        ),
        loss_function_type=LossFunctionType.BAYESIAN_RISK,
        kl_divergence_loss_weight=1e-1,
        weight_decay=0.0005,
        learning_rate=1e-2,
    )


@experiment.named_config
def sgcn_large_with_bn():
    model = dict(
        type_=ModelType.SGNN,
        name="gdk_sgcn",
        teacher=default_model_config
        | dict(
            type_=ModelType.GCN,
            name="gcn",
            hidden_dims=[256, 256],
            batch_norm=True,
        ),
        gdk_prior=default_model_config
        | dict(
            cutoff=10,
            sigma=1.0,
            type_=ModelType.GDK,
            name="gdk",
        ),
        backbone=default_model_config
        | dict(
            type_=ModelType.GCN,
            name="gcn",
            hidden_dims=[64, 64],
        ),
    )
    trainer = dict(
        type_=TrainerType.SGNN,
        teacher=default_trainer_config
        | dict(
            type_=TrainerType.SGD,
            loss_function_type=LossFunctionType.CROSS_ENTROPY,
        ),
        loss_function_type=LossFunctionType.BAYESIAN_RISK,
        kl_divergence_loss_weight=1e-1,
        weight_decay=0.0005,
        learning_rate=1e-2,
        early_stopping=dict(
            patience=200,
        ),
    )
