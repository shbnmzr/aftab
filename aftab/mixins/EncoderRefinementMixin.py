from ..maps import encoders_map


class EncoderRefinementMixin:
    def __init__(self):
        super().__init__()

        if isinstance(self.encoder, str):
            module = encoders_map.get(self.encoder)
            self.encoder = module
