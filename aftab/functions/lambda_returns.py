import torch


@torch.jit.script
def lambda_returns(
    rewards,
    terminations,
    next_q,
    gamma: float,
    lmbda: float,
):
    trajectory_length = rewards.size(0)
    output = torch.zeros_like(rewards)
    not_done_last = 1.0 - terminations[-1]
    accumulated_return = rewards[-1] + gamma * not_done_last * next_q[-1]
    output[-1] = accumulated_return
    for t in range(trajectory_length - 2, -1, -1):
        not_done = 1.0 - terminations[t]
        bootstrap = next_q[t]
        mix = (1.0 - lmbda) * bootstrap + lmbda * accumulated_return
        accumulated_return = rewards[t] + gamma * not_done * mix
        output[t] = accumulated_return
    return output
