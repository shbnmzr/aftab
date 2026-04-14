from .PQNAgent import PQNAgent
from ..encoders import ZetaEncoder


class ZetaAgent(PQNAgent):
    def __init__(self, action_dimension: int):
        super().__init__(action_dimension=action_dimension)
        self.phi = ZetaEncoder()
