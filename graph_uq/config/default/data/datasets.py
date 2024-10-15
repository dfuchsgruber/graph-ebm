from graph_uq.config.data import DatasetName
from graph_uq.experiment import experiment


@experiment.named_config
def cora_ml():
    data = dict(
        name=DatasetName.CORA_ML,
        distribution_shift=dict(
            num_left_out_classes=3,
        ),
        categorical_features=True,
    )


@experiment.named_config
def citeseer():
    data = dict(
        name=DatasetName.CITESEER,
        distribution_shift=dict(
            num_left_out_classes=2,
        ),
        categorical_features=True,
    )


@experiment.named_config
def pubmed():
    data = dict(
        name=DatasetName.PUBMED,
        distribution_shift=dict(
            num_left_out_classes=1,
        ),
        categorical_features=False,
    )


@experiment.named_config
def amazon_computers():
    data = dict(
        name=DatasetName.AMAZON_COMPUTERS,
        distribution_shift=dict(
            num_left_out_classes=4,
        ),
        categorical_features=True,
    )


@experiment.named_config
def amazon_photo():
    data = dict(
        name=DatasetName.AMAZON_PHOTO,
        distribution_shift=dict(
            num_left_out_classes=3,
        ),
        categorical_features=True,
    )


@experiment.named_config
def reddit():
    data = dict(
        name=DatasetName.REDDIT,
        distribution_shift=dict(
            num_left_out_classes=10,
        ),
    )


@experiment.named_config
def cora():
    data = dict(
        name=DatasetName.CORA,
        distribution_shift=dict(
            num_left_out_classes=25,
        ),
        categorical_features=True,
    )


@experiment.named_config
def cora_ml_lm():
    data = dict(
        name=DatasetName.CORA_ML_LM,
        distribution_shift=dict(
            num_left_out_classes=3,
        ),
        categorical_features=False,
        sentence_transformer="sentence-transformers/all-MiniLM-L6-v2",
    )


@experiment.named_config
def coauthor_cs():
    data = dict(
        name=DatasetName.COAUTHOR_CS,
        distribution_shift=dict(
            num_left_out_classes=5,
        ),
        categorical_features=True,
    )


@experiment.named_config
def coauthor_physics():
    data = dict(
        name=DatasetName.COAUTHOR_PHYSICS,
        distribution_shift=dict(
            num_left_out_classes=2,
        ),
        categorical_features=True,
    )


@experiment.named_config
def ogbn_arxiv():
    data = dict(
        name=DatasetName.OGBN_ARXIV,
        distribution_shift=dict(
            num_left_out_classes=15,
        ),
        categorical_features=False,
        # use the pre-defined split
        train_size=None,
        val_size=None,
        test_size=None,
    )
