import torch
from ..constants import ModuleType


class ViTStream(torch.nn.Module):
    def __init__(
        self,
        embedding_dimension: int,
        hidden_dimension: int,
        activation: ModuleType = torch.nn.GELU,
    ):
        super().__init__()
        self.stream = torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_dimension),
            torch.nn.Linear(embedding_dimension, hidden_dimension),
            activation(),
            torch.nn.Linear(hidden_dimension, embedding_dimension),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stream(x)
