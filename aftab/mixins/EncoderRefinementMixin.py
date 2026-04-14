from .InvokableMixin import InvokableMixin
from ..maps import encoders_map


class EncoderRefinementMixin(InvokableMixin):
    def __init__(self):
        super().__init__()

        if not isinstance(self.encoder, str):
            return

        module = encoders_map.get(self.encoder)
        self.encoder = module
