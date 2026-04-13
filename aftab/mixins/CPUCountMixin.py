import os


class CPUCountMixin:
    def __init__(self):
        super().__init__()
        self.cpu_count = os.cpu_count()
