import torch
from .BaseNetwork import BaseNetwork
from ..modules import QuantileStream, FractionProposalStream


class QuantileHead(torch.nn.Module):
    def __init__(self, action_dimension: int, embedding_dimension: int):
        super().__init__()
        self.value_stream = QuantileStream(
            action_dimension=1,
            embedding_dimension=embedding_dimension,
        )
        self.advantage_stream = QuantileStream(
            action_dimension=action_dimension,
            embedding_dimension=embedding_dimension,
        )

    def forward(self, features: torch.Tensor, tau_hats: torch.Tensor):
        value_quantiles = self.value_stream(features, tau_hats)
        advantage_quantiles = self.advantage_stream(features, tau_hats)
        return (
            value_quantiles
            + advantage_quantiles
            - advantage_quantiles.mean(dim=2, keepdim=True)
        )


class DuellingFQFNetwork(BaseNetwork):
    def __init__(
        self,
        quantile_embedding_dimension: int,
        number_quantiles: int,
        entropy_coefficient: float = 1e-3,
        fraction_proposal_coefficient: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.entropy_coefficient = entropy_coefficient
        self.fraction_proposal_coefficient = fraction_proposal_coefficient
        self.action_dimension = kwargs["action_dimension"]

        self.fraction_proposal = FractionProposalStream(
            number_quantiles=number_quantiles,
            embedding_dimension=quantile_embedding_dimension,
        )
        self.quantile_value = QuantileHead(
            action_dimension=self.action_dimension,
            embedding_dimension=quantile_embedding_dimension,
        )

    def forward(self, x: torch.Tensor):
        features = self.get_features(x)
        taus, tau_hats, q_probs, entropy = self.fraction_proposal(features)
        quantiles = self.quantile_value(features, tau_hats)
        return {
            "features": features,
            "quantiles": quantiles,
            "taus": taus,
            "tau_hats": tau_hats,
            "q_probs": q_probs,
            "entropy": entropy,
        }

    def get_q(self, x: torch.Tensor) -> torch.Tensor:
        output = self.forward(x)
        quantiles = output["quantiles"]
        q_probs = output["q_probs"]
        return (quantiles * q_probs.unsqueeze(-1)).sum(dim=1)
