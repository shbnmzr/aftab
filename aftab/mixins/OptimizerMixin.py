import torch


class OptimizerMixin:
    def __init__(self):
        super().__init__()

    def __build_quantile_optimizers(self):
        quantile_value_optimizer = torch.optim.RAdam(
            list(self._network.phi.parameters())
            + list(self._network.quantile_value.parameters()),
            lr=self.lr,
            eps=self.optimizer_epsilon,
        )
        fraction_proposal_optimizer = torch.optim.RAdam(
            self._network.fraction_proposal.parameters(),
            lr=self.fraction_proposal_lr,
            eps=self.optimizer_epsilon,
        )
        return fraction_proposal_optimizer, quantile_value_optimizer

    def make_optimizer(self):
        return self.optimizer_instance(
            self._network.parameters(),
            lr=self.lr,
            eps=self.optimizer_epsilon,
            betas=(self.optimizer_first_beta, self.optimizer_second_beta),
            weight_decay=self.optimizer_weight_decay,
        )
