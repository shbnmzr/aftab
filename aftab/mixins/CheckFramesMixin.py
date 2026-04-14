from .InvokableMixin import InvokableMixin


class CheckFramesMixin(InvokableMixin):
    ACCEPTABLE_FRAMES_IDX = {
        "pilot": 50_000_000,
        "ablation": 50_000_000,
        "full": 200_000_000,
    }

    def __init__(self):
        super().__init__()

        if not isinstance(self.frames, str):
            return

        try:
            self.frames = self.ACCEPTABLE_FRAMES_IDX[self.frames]
        except KeyError as exc:
            raise ValueError(
                f"Invalid value for `frames`: {self.frames!r}. "
                f"Expected one of {tuple(self.ACCEPTABLE_FRAMES_IDX)}."
            ) from exc
