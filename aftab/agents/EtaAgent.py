from .PQNAgent import PQNAgent
from ..encoders import EtaEncoder


class EtaAgent(PQNAgent):
    def __init__(self, action_dim: int):
        super().__init__(action_dim=action_dim)
        self.phi = EtaEncoder()
