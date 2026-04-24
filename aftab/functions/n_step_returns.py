import torch


def n_step_returns(
    self,
    batch_rewards: torch.Tensor,
    batch_terminations: torch.Tensor,
    q_seq_for_bootstrap: torch.Tensor,
    gradient: bool,
):
    with torch.set_gradient_enabled(gradient):
        steps = getattr(self, "steps")
        T = batch_rewards.size(0)
        targets = torch.zeros_like(q_seq_for_bootstrap)
        for t in range(T):
            accumulated = torch.zeros_like(q_seq_for_bootstrap[t])
            current_gamma = 1.0
            m = min(steps, T - t)
            for k in range(m):
                idx = t + k
                accumulated = accumulated + current_gamma * batch_rewards[
                    idx
                ].unsqueeze(-1)
                not_done = (1.0 - batch_terminations[idx]).unsqueeze(-1)
                current_gamma = current_gamma * self.gamma * not_done

            accumulated = accumulated + current_gamma * q_seq_for_bootstrap[t + m - 1]
            targets[t] = accumulated
        return targets
