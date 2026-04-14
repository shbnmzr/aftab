import torch
from typing import Type
from ..agents import PQNAgent


class NetworkMixin:
    def __init__(self):
        super().__init__()

    def make_network(
        self, action_dimension: int, encoder_instance: Type[torch.nn.Module]
    ) -> Type[torch.nn.Module]:
        return PQNAgent(
            action_dimension=action_dimension, encoder_instance=encoder_instance
        )
