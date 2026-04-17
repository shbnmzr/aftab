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
        normalization: bool = True,
    ):
        super().__init__()
        self.mu = torch.nn.LazyLineaR(hidden_dimension)
        self.nu = torch.nn.Linear(hidden_dimension, output_dimension)
        self.xi = torch.nn.LayerNorm(hidden_dimension)
        self.activation = activation()
        self.normalization = normalization

    def forward(self, features):
        features = self.mu(features)
        features = self.xi(features)

        if self.normalization:
            features = self.activation(features)

        features = self.nu(features)
        return features
