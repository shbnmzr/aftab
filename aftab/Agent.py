import torch
from baloot import acceleration_device
from .helpers import panic_if_encoder_identifier_is_illegal


class Agent:
    def __init__(self, encoder_identifier: str = "gamma"):
        self.device = acceleration_device()
        torch.set_float32_matmul_precision("high")
        panic_if_encoder_identifier_is_illegal(encoder_identifier)

    def train(self, environment):
        pass
