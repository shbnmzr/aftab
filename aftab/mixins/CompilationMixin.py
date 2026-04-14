import torch


class CompilationMixin:
    def __init__(self):
        super().__init__()

    def compile_network(self):
        if not getattr(self, "should_compile", False):
            return

        if not hasattr(self, "_network"):
            raise AttributeError(
                "Expected `_network` to be defined before compilation."
            )

        self._network = torch.compile(self._network)
