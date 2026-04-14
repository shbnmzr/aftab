import torch


class DummyPassMixin:
    def __init__(self):
        super().__init__()

    def get_dummy_sample(self):
        if not hasattr(self, "stack_number"):
            raise AttributeError("Expected `stack_number` to be defined.")

        if not hasattr(self, "device"):
            raise AttributeError("Expected `device` to be defined.")

        batch_size = 1
        picture_size = 84

        return torch.randn(
            batch_size,
            self.stack_number,
            picture_size,
            picture_size,
            device=self.device,
        )

    @torch.no_grad()
    def perform_dummy_pass(self):
        """
        Runs a forward pass with a dummy input to initialize lazy modules
        before training or compilation.
        """
        if not hasattr(self, "_network"):
            raise AttributeError("Expected `_network` to be defined.")

        dummy_input = self.get_dummy_sample()
        self._network(dummy_input)
