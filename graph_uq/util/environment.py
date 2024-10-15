def setup_environment():
    """ Sets up the environment for running the code. This includes setting the number of threads, setting the number of open files, and setting multiprocessing to use the filesystem."""
    import os
    import warnings
    os.environ['WANDB__SERVICE_WAIT'] = '300'

    # Stop pytorch-lightning from pestering us about things we already know
    warnings.filterwarnings(
        "ignore",
        "There is a wandb run already in progress",
        module="pytorch_lightning.loggers.wandb",
    )

    # Fixes a weird pl bug with dataloading and multiprocessing
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    