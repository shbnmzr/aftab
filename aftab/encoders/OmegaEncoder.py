from .ThriftyEncoder import ThriftyEncoder


class OmegaEncoder(ThriftyEncoder):
    def __init__(self):
        super().__init__(filters=128, kernel_size=3, iterations=20)
