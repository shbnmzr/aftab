import torch
from typing import Type
from ..encoders import NatureDQNEncoder
from ..common import LinearEpsilon
from ..modules import Stream
from ..functions import mse_loss


class PQNAgent(torch.nn.Module):
    def __init__(
        self,
        action_dimension,
        encoder_instance: Type[torch.nn.Module] = NatureDQNEncoder,
    ):
        super().__init__()
        self.epsilon_greedy = True

        self.epsilon = LinearEpsilon()
        self.phi = encoder_instance()
        self.q = Stream(output_dim=action_dimension)

    def no_epsilon_greedy(self):
        self.epsilon_greedy = False

    def normalize_observations(self, x: torch.Tensor) -> torch.Tensor:
        return x / 255.0

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize_observations(x)
        features = self.phi(x)
        return features

    def get_q(self, states: torch.Tensor) -> torch.Tensor:
        features = self.get_features(states)
        return self.q(features)

    def loss(self, q: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return mse_loss(q, target)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_q(x)
