import torch
from ..modules import Stream
from .BaseNetwork import BaseNetwork


class DistributionalPQNNetwork(BaseNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.q = Stream(
            input_dimension=self.feature_dimension,
            hidden_dimension=self.embedding_dimension,
            output_dimension=self.action_dimension * self.bins,
        )

    def get_q_logits(self, states: torch.Tensor) -> torch.Tensor:
        features = self.get_features(states)
        logits = self.q(features)
        return logits.reshape(-1, self.action_dimension, self.bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_q_logits(x)
