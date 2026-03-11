from .PQNAgent import PQNAgent
from ..encoders import AlphaEncoder


class AlphaAgent(PQNAgent):
    def __init__(self, action_dim: int):
        super().__init__(action_dim=action_dim)
        self.phi = AlphaEncoder()
