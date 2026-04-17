from baloot import funnel
from types import SimpleNamespace


class TrainingResultsMixin:
    def __init__(self):
        super().__init__()

    def __make_log_filename(self, **kwargs):
        filename = "_".join(f"{k}-{v}" for k, v in kwargs.items())
        return f"{filename}.pkl"

    def __build_log_payload(self):
        duration = self.results.duration or 0
        return {
            "training_reward": self.results.rewards.train,
            "test_reward": self.results.rewards.test,
            "loss": self.results.loss,
            "duration_seconds": duration,
            "duration_hours": duration / 3600,
        }

    def flush_results(self):
        self.results = SimpleNamespace()
        self.results.rewards = SimpleNamespace()
        self.results.rewards.train = []
        self.results.rewards.test = []
        self.results.loss = []
        self.results.duration = 0.0

    def save(self, **kwargs) -> None:
        filename = self.__make_log_filename(**kwargs)
        payload = self.__build_log_payload()
        funnel(filename, payload)
