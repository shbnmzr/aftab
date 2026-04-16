import torch
from ..constants import ModuleType


class Stream(torch.nn.Module):
    def __init__(
        self,
        *,
        # this is just here to keep the consistency. it doesn't do anything in this block.
        input_dimension: int = 3136,
        hidden_dimension: int = 512,
        output_dimension: int,
        activation: ModuleType = torch.nn.ReLU,
    ):
        super().__init__()
        self.stream = torch.nn.Sequential(
            torch.nn.LazyLinear(hidden_dimension),
            torch.nn.LayerNorm(hidden_dimension),
            activation(),
            torch.nn.Linear(hidden_dimension, output_dimension),
        )

    def forward(self, features):
        return self.stream(features)
