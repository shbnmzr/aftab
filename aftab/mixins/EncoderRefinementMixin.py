from .InvokableMixin import InvokableMixin
from ..maps import encoders_map


class EncoderRefinementMixin(InvokableMixin):
    def __init__(self):
        super().__init__()

        if not isinstance(self.encoder, str):
            return

        try:
            self.encoder = encoders_map[self.encoder]
        except KeyError as exc:
            raise ValueError(
                f"Unknown encoder key: {self.encoder!r}. "
                f"Expected one of: {tuple(encoders_map.keys())}"
            ) from exc
