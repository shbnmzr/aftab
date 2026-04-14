from .PQNAgent import PQNAgent
from ..encoders import AlphaEncoder


class AlphaAgent(PQNAgent):
    def __init__(self, action_dimension: int):
        super().__init__(action_dimension=action_dimension)
        self.phi = AlphaEncoder()
