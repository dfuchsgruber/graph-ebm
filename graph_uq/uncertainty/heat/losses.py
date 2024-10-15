import torch
from torch import nn

from graph_uq.config.heat import ContrasticeDivergenceLossConfig
from graph_uq.uncertainty.heat.heat import HybridEnergyModel


class ContrastiveDivergenceLoss(nn.Module):
    def __init__(self, config: ContrasticeDivergenceLossConfig):
        super().__init__()
        self.l2_coef = config["l2_coef"]
        self.eps_data = config["eps_data"]
        self.verbose = config["verbose"]

    def forward(
        self,
        ebm: HybridEnergyModel,
        real_z: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sample_size = real_z.shape[0]

        init_labels, init_samples = ebm.proposal_samples(
            sample_size, target=target, real_samples=real_z
        )
        if init_labels is not None:
            init_labels = init_labels.to(real_z.device)
        if init_samples is not None:
            init_samples = init_samples.to(real_z.device)
        gen_z = ebm.negative_samples(init_samples, init_labels)

        real_energy, real_energy_nn, real_energy_g = ebm(
            real_z + torch.randn_like(real_z) * self.eps_data, nn_only=not self.verbose
        )
        gen_energy, gen_energy_nn, gen_energy_g = ebm(
            gen_z + torch.randn_like(gen_z) * self.eps_data, nn_only=not self.verbose
        )

        cdiv_loss = real_energy_nn.mean() - gen_energy_nn.mean()
        l2_reg = real_energy_nn.pow(2).mean() + gen_energy_nn.pow(2).mean()
        loss = cdiv_loss + self.l2_coef * l2_reg

        logs = {
            "energy IN": real_energy.mean().detach(),
            "engergy Gen": gen_energy.mean().detach(),
            "energy_nn IN": real_energy_nn.mean().detach(),
            "energy_nn Gen": gen_energy_nn.mean().detach(),
            "energy_prior IN": real_energy_g.mean().detach(),
            "energy_prior Gen": gen_energy_g.mean().detach(),
            "loss": loss.detach(),
            "cd_loss": cdiv_loss.detach(),
            "l2_loss": (self.l2_coef * l2_reg).detach(),
        }

        return loss, logs, gen_z  # type: ignore
