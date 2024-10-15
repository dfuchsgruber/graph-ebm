from graph_uq.config.trainer import EdgeReconstructionLossType
from graph_uq.experiment import experiment

@experiment.named_config
def edge_reconstruction():
    trainer = dict(
        edge_reconstruction_loss = dict(
            type_ = EdgeReconstructionLossType.ENERGY,
            weight = 1.0,
            num_samples = 100,
        ),
    )