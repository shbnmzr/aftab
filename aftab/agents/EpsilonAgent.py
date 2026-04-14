from .PQNAgent import PQNAgent
from ..encoders import EpsilonEncoder


class EpsilonAgent(PQNAgent):
    def __init__(self, action_dimension: int):
        super().__init__(action_dimension=action_dimension)
        self.phi = EpsilonEncoder()
