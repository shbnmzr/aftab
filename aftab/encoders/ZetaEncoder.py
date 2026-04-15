import torch
from ..modules import EncoderBlock
from ..constants import ModuleType


class ZetaEncoder(torch.nn.Module):
    def __init__(self, *, activation: ModuleType = torch.nn.ReLU):
        super().__init__()
        self.stream = torch.nn.Sequential(
            EncoderBlock(
                4, 48, kernel_size=4, stride=2, padding=1, activation=activation
            ),
            EncoderBlock(
                48, 48, kernel_size=4, stride=2, padding=1, activation=activation
            ),
            EncoderBlock(
                48, 48, kernel_size=4, stride=2, padding=1, activation=activation
            ),
            torch.nn.Flatten(),
        )

    def forward(self, x):
        return self.stream(x)
