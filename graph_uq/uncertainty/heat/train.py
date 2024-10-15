import logging
from typing import Any

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from tqdm import tqdm
from typeguard import typechecked

from graph_uq.config.heat import HeatConfig
from graph_uq.uncertainty.heat.heat import HybridEnergyModel
from graph_uq.uncertainty.heat.losses import ContrastiveDivergenceLoss


@jaxtyped(typechecker=typechecked)
def train_ebm(
    ebm: HybridEnergyModel,
    features_train: Float[Tensor, "num_train num_features"],
    classes_train: Int[Tensor, "num_train"],
    _config: dict[str, Any],
):
    config: HeatConfig = HeatConfig(**_config)  # type: ignore
    if config["use_gpu"] and torch.cuda.is_available():
        ebm = ebm.cuda()
        features_train = features_train.to("cuda")
        classes_train = classes_train.to("cuda")

    ebm.base_dist.fit(
        features_train, classes_train, verbose=config["verbose"]
    )  # TODO: the fitting is not implemented yet

    optimizer = torch.optim.Adam(ebm.parameters(), **config["optimizer"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **config["scheduler"])
    loss_function = ContrastiveDivergenceLoss(config["loss"])

    with torch.enable_grad():
        ebm.train()

        if config["verbose"]:
            logging.info("Training EBM.")

        iterator = tqdm(
            range(config["num_epochs"]),
            desc="Training EBM",
            disable=not config["verbose"],
        )
        for epoch in iterator:
            # Training loop
            ...
            optimizer.zero_grad(set_to_none=True)

            if ebm.base_dist.normalize:
                features_train = F.normalize(features_train)

            loss, logs, _ = loss_function(ebm, features_train, classes_train)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if config["verbose"]:
                iterator.set_description(
                    f"Epoch {epoch+1}/{config['num_epochs']} - Loss: {loss.item():.4f}"
                )

    if config["verbose"]:
        logging.info("Trained EBM.")
