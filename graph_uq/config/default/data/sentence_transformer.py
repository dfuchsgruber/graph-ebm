from graph_uq.experiment import experiment

@experiment.named_config
def sentence_transformer():
    data = dict(
        sentence_transformer = 'sentence-transformers/all-MiniLM-L6-v2'
    )