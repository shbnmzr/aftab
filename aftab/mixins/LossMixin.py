import torch


class LossMixin:
    def __init__(self):
        super().__init__()

    def get_loss(
        self, mini_batch_observations, mini_batch_actions, mini_batch_targets
    ) -> torch.Tensor:
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            q_values = self.get_q_values(
                float_observations=mini_batch_observations.float(),
                gradient=True,
            )
            q_taken = q_values.gather(1, mini_batch_actions.unsqueeze(1)).squeeze()
            loss = self._network.loss(q_taken, mini_batch_targets)
        return loss
