import torch
import math
from ..constants import ModuleType

class CosineEmbeddingModule(torch.nn.Module):
    def __init__(
        self,
        *,
        embedding_dimension: int,
        feature_dimension: int,
        activation: ModuleType = torch.nn.ReLU,
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        pi_indices = math.pi * torch.arange(1, embedding_dimension + 1).float()
        self.register_buffer("pi_indices", pi_indices)
        self.linear = torch.nn.Linear(embedding_dimension, feature_dimension)
        self.activation = activation()

    def forward(self, fractions: torch.Tensor) -> torch.Tensor:
        cos = torch.cos(fractions.unsqueeze(-1) * self.pi_indices.view(1, 1, -1))
        return self.activation(self.linear(cos))