import torch
from ..functions import lambda_returns


class LambdaReturnsMixin:
    def __init__(self):
        super().__init__()

    def get_returns(
        self,
        float_observations=None,
        batch_q=None,
        batch_rewards=None,
        batch_terminations=None,
        next_q_values=None,
        next_q=None,
    ):
        with torch.no_grad():
            if next_q is not None:
                pass
            elif next_q_values is not None:
                final_next_q = next_q_values.max(dim=-1).values
                if batch_q is not None:
                    max_q_seq = batch_q.max(dim=-1).values
                    next_q = torch.cat([max_q_seq, final_next_q.unsqueeze(0)])[1:]
                else:
                    next_q = final_next_q
            else:
                if float_observations is None or batch_q is None:
                    raise ValueError(
                        "`next_q`, `next_q_values`, or both `float_observations` "
                        "and `batch_q` must be provided."
                    )
                next_q = self._network(float_observations).max(dim=-1).values
                max_q_seq = batch_q.max(dim=-1).values
                next_q = torch.cat([max_q_seq, next_q.unsqueeze(0)])[1:]

            returns = lambda_returns(
                batch_rewards,
                batch_terminations,
                next_q,
                self.gamma,
                self.lmbda,
            )
        return returns
