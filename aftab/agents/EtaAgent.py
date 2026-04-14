from .PQNAgent import PQNAgent
from ..encoders import EtaEncoder


class EtaAgent(PQNAgent):
    def __init__(self, action_dimension: int):
        super().__init__(action_dimension=action_dimension)
        self.phi = EtaEncoder()
