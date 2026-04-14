import torch


class CompilationMixin:
    def __init__(self):
        super().__init__()

    def compile_network(self):
        if not self.should_compile:
            return
        self._network = torch.compile(self._network)
