from baloot import acceleration_device


class AccelerationDeviceMixin:
    def __init__(self):
        super().__init__()
        self.device = acceleration_device()
