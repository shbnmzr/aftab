import torch
from hl_gauss_pytorch import HLGaussLoss
from ..functions import mse_loss


class LossMixin:
    def __init__(self):
        self.hl_gauss = None
        if self.__uses_hl_gauss():
            self.__initialize_hl_gauss()
        super().__init__()

    def __uses_hl_gauss(self):
        return self.network in {"distributional", "distributional-duelling"}

    def __initialize_hl_gauss(self):
        if self.v_max <= self.v_min:
            raise ValueError("Expected `v_max` to be greater than `v_min`.")

        if self.bins <= 0:
            raise ValueError("Expected `bins` to be a positive integer.")

        bin_width = (self.v_max - self.v_min) / self.bins
        sigma = bin_width * self.hl_gauss_smoothing_ratio

        self.hl_gauss = HLGaussLoss(
            min_value=self.v_min,
            max_value=self.v_max,
            num_bins=self.bins,
            sigma=sigma,
            clamp_to_range=True,
        ).to(self.device)

    def get_loss(
        self, mini_batch_observations, mini_batch_actions, mini_batch_targets
    ) -> torch.Tensor:
        mini_batch_observations = mini_batch_observations.float()

        if self.__uses_hl_gauss():
            q_logits = self._network.get_q_logits(mini_batch_observations)
            q_taken_logits = q_logits[
                torch.arange(q_logits.size(0), device=q_logits.device),
                mini_batch_actions,
            ]
            return self.hl_gauss(q_taken_logits, mini_batch_targets)

        q_values = self._network.get_q(mini_batch_observations)
        q_taken = q_values.gather(1, mini_batch_actions.unsqueeze(1)).squeeze(1)
        return mse_loss(q_taken, mini_batch_targets)
