from .PQNAgent import PQNAgent
from ..encoders import GammaEncoder


class GammaAgent(PQNAgent):
    def __init__(self, action_dimension: int):
        super().__init__(action_dimension=action_dimension)
        self.phi = GammaEncoder()
