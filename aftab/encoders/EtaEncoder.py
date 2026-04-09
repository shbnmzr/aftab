import torch
from ..modules import LayerNorm2d
from ..constants import ModuleType


class EtaEncoder(torch.nn.Module):
    def __init__(self, *, activation: ModuleType = torch.nn.ReLU):
        super().__init__()
        self.stream = torch.nn.Sequential(
            torch.nn.Conv2d(4, 64, 4, 4, 0),
            LayerNorm2d(64),
            activation(),
            torch.nn.Conv2d(64, 128, 3, 1, 0),
            LayerNorm2d(128),
            activation(),
            torch.nn.Flatten(),
        )

    def forward(self, x):
        return self.stream(x)
