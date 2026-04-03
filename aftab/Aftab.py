import torch
from baloot import acceleration_device


class Aftab:
    def __init__(
        self,
        encoder_identifier: str = "gamma",
        frameskip: int = 4,
        num_minibatches: int = 32,
        epochs: int = 2,
        gamma: float = 0.99,
        lmbda: float = 0.65,
        lr: float = 0.00025,
        logging_interval: int = 10,
    ):
        self.device = acceleration_device()
        self.frameskip = frameskip
        self.lr = lr
        self.lmbda = lmbda
        self.gamma = gamma
        self.epochs = epochs
        self.num_minibatches = num_minibatches
        self.logging_interval = logging_interval

    def train(self, environment):
        torch.set_float32_matmul_precision("high")
