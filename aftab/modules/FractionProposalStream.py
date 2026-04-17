import torch
from .Stream import Stream


class FractionProposalStream(torch.nn.Module):
    def __init__(self, *, number_quantiles: int, embedding_dimension: int):
        super().__init__()
        self.number_quantiles = number_quantiles
        self.mu = Stream(
            input_dimension=None,
            output_dimension=number_quantiles,
            hidden_dimension=embedding_dimension,
        )

    def forward(self, features):
        q_logits = self.mu(features)
        q_probs = torch.nn.functional.softmax(q_logits, dim=-1)
        log_q_probabilities = torch.log(q_probs.clamp(min=1e-8))

        tau_0 = torch.zeros((features.size(0), 1), device=features.device)
        tau_1_to_N = torch.cumsum(q_probs, dim=-1)
        tau = torch.cat([tau_0, tau_1_to_N], dim=-1)
        assert tau.shape[-1] == self.number_quantiles + 1

        tau_hat = (tau[:, :-1] + tau[:, 1:]) / 2.0
        tau_hat = tau_hat.detach()

        entropy = -torch.sum(q_probs * log_q_probabilities, dim=-1, keepdim=True)

        return tau, tau_hat, q_probs, entropy
