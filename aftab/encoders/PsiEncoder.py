import torch
from thriftynet import ThriftyEncoder
from ..modules import LayerNorm2d


class PsiEncoder(ThriftyEncoder):
    def __init__(self):
        super().__init__(
            filters=64,
            iterations=20,
            kernel_size=3,
            normalization=LayerNorm2d,
            activation=torch.nn.ReLU,
        )
