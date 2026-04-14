import torch


class QValueMixin:
    def __init__(self):
        super().__init__()

    def get_q_values(self, float_observations, no_grad: bool = False):
        if no_grad:
            with torch.no_grad():
                q_values = self._network(float_observations)
        else:
            q_values = self._network(float_observations)

        return q_values
