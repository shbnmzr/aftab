import torch


class QValueMixin:
    def __init__(self):
        super().__init__()

    def get_q_values(
        self,
        float_train_observations: torch.Tensor = None,
        float_test_observations: torch.Tensor = None,
        float_observations: torch.Tensor = None,
        gradient: bool = False,
    ):
        with torch.set_grad_enabled(gradient):
            if float_observations is not None:
                return self._network.get_q(float_observations)
            test_q_values = self._network.get_q(float_test_observations)
            train_q_values = self._network.get_q(float_train_observations)
            return {"test": test_q_values, "train": train_q_values}
