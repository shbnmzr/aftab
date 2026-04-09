import torch
from ..modules import LayerNorm2d
from ..constants import ModuleType


class ZetaEncoder(torch.nn.Module):
    def __init__(self, *, activation: ModuleType = torch.nn.ReLU):
        super().__init__()
        self.stream = torch.nn.Sequential(
            torch.nn.Conv2d(4, 48, 4, 2, 1),
            LayerNorm2d(48),
            activation(),
            torch.nn.Conv2d(48, 48, 4, 2, 1),
            LayerNorm2d(48),
            activation(),
            torch.nn.Conv2d(48, 48, 4, 2, 1),
            LayerNorm2d(48),
            activation(),
            torch.nn.Flatten(),
        )

    def forward(self, x):
        return self.stream(x)
