import torch


class FractionProposalStream(torch.nn.Module):
    def __init__(self, feature_dimension: int, number_quantiles: int):
        super().__init__()
        self.mu = torch.nn.Linear(feature_dimension, number_quantiles)

    def forward(self, x):
        logits = self.mu(x)
        probs = torch.softmax(logits, dim=-1)
        taus = torch.cumsum(probs, dim=-1)
        taus = torch.cat([torch.zeros_like(taus[:, :1]), taus], dim=1)
        tau_hats = (taus[:, :-1] + taus[:, 1:]) / 2.0
        return taus, tau_hats
