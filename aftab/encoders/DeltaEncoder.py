import torch
from ..common import LayerNorm2d


class DeltaEncoder(torch.nn.Module):
    def __init__(self, *, activation=torch.nn.ReLU):
        super().__init__()
        self.stream = torch.nn.Sequential(
            torch.nn.Conv2d(4, 24, 9, 4, 0),
            LayerNorm2d(24),
            activation(),
            torch.nn.Conv2d(24, 48, 5, 2, 0),
            LayerNorm2d(48),
            activation(),
            torch.nn.Conv2d(48, 96, 3, 1, 0),
            LayerNorm2d(96),
            activation(),
            torch.nn.Flatten(),
        )

    def forward(self, x):
        return self.stream(x)
