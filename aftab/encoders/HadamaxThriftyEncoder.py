import torch
from typing import Type
from ..modules import LayerNorm2d


class HadamaxThriftyEncoder(torch.nn.Module):
    def __init__(
        self,
        *,
        filters: int,
        iterations: int,
        kernel_size: int,
        activation: Type[torch.nn.Module] = torch.nn.GELU,
    ):
        super().__init__()
        self.filters = filters
        self.iterations = iterations

        self.conv = torch.nn.Conv2d(
            filters,
            filters * 2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

        self.activation = activation()
        alpha = torch.zeros(iterations, 2)
        alpha[:, 0] = 0.1
        alpha[:, 1] = 0.9
        self.alpha = torch.nn.Parameter(alpha)

        pool_every = max(1, iterations // 5)
        self.pool_strategy = [
            t % pool_every == pool_every - 1 for t in range(iterations)
        ]
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_pool = torch.nn.AdaptiveMaxPool2d((1, 1))
        self.normalizations = torch.nn.ModuleList(
            [LayerNorm2d(self.filters) for _ in range(self.iterations)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.filters > x.size(1):
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, self.filters - x.size(1)))

        for t in range(self.iterations):
            conv_out = self.conv(x)
            conv_1, conv_2 = conv_out.chunk(2, dim=1)
            a_t = self.activation(conv_1) * self.activation(conv_2)

            x = self.alpha[t, 0] * a_t + self.alpha[t, 1] * x
            x = self.normalizations[t](x)

            if self.pool_strategy[t]:
                x = self.pool(x)

        x = self.global_pool(x)
        return x.flatten(1)
